import pandas as pd
import baostock as bs
import function_book as f
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import cmath
import csv

def get_stock_code_by_name(stock_name_list,isreturn = False):

    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    data_list = []

    for stock_name_str in stock_name_list:

        rs = bs.query_stock_basic(code_name=stock_name_str)  # 支持模糊查询

        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    bs.logout()

    if isreturn == True:
        return result
    else:
        print("! save tmt_stock csv file.")


def loading_prices(stocks_id_str,start_date,end_date):
        k_data_df = pd.DataFrame()
        rs = bs.query_history_k_data_plus(stocks_id_str,"date,close",start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
        k_data_df = k_data_df.append(rs.get_data())
        k_data_df.set_index(keys = 'date',inplace=True)
        k_data_df['close'] = k_data_df['close'].astype('float')
        k_data_df.rename(columns={"close": stocks_id_str},inplace=True)
        return k_data_df

def load_stocks_prices(stocks_id,start_date,end_date):
    history_df = pd.DataFrame()
    lg = bs.login()
    for i in stocks_id:
        k_data_df = loading_prices(i,start_date,end_date)
        history_df = pd.concat([history_df,k_data_df],axis=1)
    bs.logout()
    return history_df

def get_table_for_reg(fundid,report_date_list,i,fund_value,hs300):

    start_date = report_date_list[i-1]
    end_date = report_date_list[i+1]
    print( "start_date:",report_date_list[i-1],"report_date:",report_date_list[i],"endstart_date:",report_date_list[i+1])
    filename =fundid + '_' +report_date_list[i]

    fund_inv_details =pd.read_excel(filename+'.xlsx')
    fund_inv_details.rename(columns = {'股票名称':"code_name",
                                  '占净值\n比例':"ratio",
                                  '持股数\n（万股）':"num_shares",
                                  '持仓市值\n（万元）':"fund_value"},inplace=True)

    inv_stocks_details = get_stock_code_by_name(fund_inv_details['code_name'],isreturn = True)
    inv_stocks_detail = inv_stocks_details[inv_stocks_details['ipoDate']<start_date]
    inv_stocks_details = pd.merge(fund_inv_details[['code_name','ratio','num_shares','fund_value']],inv_stocks_details,on= 'code_name')
    inv_stocks_details['ratio'] = inv_stocks_details['ratio'].apply(lambda x:float(x.replace('%','')))
    inv_stocks_details[['num_shares','fund_value']] = inv_stocks_details[['num_shares','fund_value']] *10000

    inv_stocks_details.to_csv(filename+'.csv',index = False )

    stocks_id = inv_stocks_details['code']
    history_df = load_stocks_prices(stocks_id,start_date,end_date)

    for i in range(len(inv_stocks_details)):
        code_str = inv_stocks_details.loc[i,'code']
        num_shares = inv_stocks_details.loc[i,'num_shares']
        history_df[code_str] = history_df[code_str].apply(lambda x:x*num_shares)

    if len(inv_stocks_details['ipoDate']>start_date) != 0 :
        print(inv_stocks_details[inv_stocks_details['ipoDate']>start_date])
        history_df.fillna(method = 'backfill',inplace = True,axis=0)

    history_df['stocks_unit_value']=history_df.apply(lambda x:x.sum(),axis=1)

    history_df.index.name = 'date'

    net_value = pd.merge(history_df[['stocks_unit_value']],fund_value,'inner',on = 'date')
    net_value = pd.merge(net_value,hs300,'inner',on = 'date')

    net_value['hs300_daily_return'] = 100 * (net_value['sh.000300']-net_value['sh.000300'].shift(1))/net_value['sh.000300'].shift(1)
    net_value['stocks_daily_return'] = 100 * (net_value['stocks_unit_value']-net_value['stocks_unit_value'].shift(1))/net_value['stocks_unit_value'].shift(1)

    net_value.to_csv("reg_"+filename+'.csv',index = True ,index_label = 'date')

    # stocks_details['fund_value'].sum()/(stocks_details['fund_value'][0]/(stocks_details['ratio'][0]/100))

def reg_sk(mydf,x_list,comments,disc,fundid,report_date,ratio):
    if comments == 'c':
        mylist = [ratio]
        #disc = 0.98 #0.985,0.99
        for i in range(0,63,1):
            mylist.append(disc  * mylist[-1])

        mydf['coef'] = mydf['dist'].apply(lambda x:mylist[int(x)])
        mydf['c_stocks_daily_return'] = mydf['coef']  * mydf['stocks_daily_return']
        mydf['c_hs300_daily_return'] = (1-mydf['coef'])  * mydf['hs300_daily_return']

    diabetes_X  = mydf[x_list]
    diabetes_y  = mydf['daily_return']
    #size = test_size

    diabetes_X_train = diabetes_X.loc[:report_date]
    diabetes_X_test = diabetes_X[report_date:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y.loc[:report_date]
    diabetes_y_test = diabetes_y[report_date:]

    # Create linear regression object
    model = linear_model.LinearRegression()

    # Train the model using the training sets
    model.fit(diabetes_X_train, diabetes_y_train)

    diabetes_y_fit = model.predict(diabetes_X_train)
    d = diabetes_y_train - diabetes_y_fit
    train_var = cmath.sqrt(sum(d*d)).real/len(d)
    train_r2 = model.score( diabetes_X_train, diabetes_y_train , sample_weight=None)

    diabetes_y_pred = model.predict(diabetes_X_test)
    d = diabetes_y_test - diabetes_y_pred
    test_var = cmath.sqrt(sum(d*d)).real/len(d)
    test_r2 = model.score( diabetes_X_test, diabetes_y_test, sample_weight=None)

    size = int(0.5*len(diabetes_y_test))
    d = diabetes_y_test[:size] - diabetes_y_pred[:size]
    test_var1 = cmath.sqrt(sum(d*d)).real/len(d)
    d = diabetes_y_test[size:] - diabetes_y_pred[size:]
    test_var2 = cmath.sqrt(sum(d*d)).real/len(d)

    with open('result.csv', "a", newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(['fundid','date',    'reg',         'train_r2','test_r2','train_var','test_var','test_var1','test_var2','ratio','disc','comments','coef','intercept'])
        writer.writerow([fundid ,report_date,'daily_return', train_r2,  test_r2,  train_var,  test_var,  test_var1,  test_var2, ratio, disc,comments,model.coef_,model.intercept_])
        f.close()
