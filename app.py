from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_jsonpify import jsonpify
from flask_cors import CORS

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn; seaborn.set()
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
import matplotlib.pylab as plt
# %matplotlib inline
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 15,6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

import pandas as pd
from datetime import datetime
import pmdarima as pm

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyodbc

def getDatetime(date):
#     https://www.programiz.com/python-programming/datetime/strptime
    return datetime.strptime(date, '%d/%m/%Y %H:%M')

# def colorCode(volume):
# #     75k per hour
#     if volume>=treshold:
#         return "Red"
# #     around 80%
#     elif volume>=(treshold*0.8):
#         return "Yellow"
#     return "Green"

def get_sql_query(spark: SparkSession, statement: str):
    connection = 'jdbc:sqlserver://is484dbserver.database.windows.net:1433;database=is484db;'
    username = 'user'
    password = 'Password123'
    
    table = "(" + statement + ")table_a"

    
    query_result = spark.read \
                    .format("jdbc") \
                    .option("url", "jdbc:sqlserver://is484dbserver.database.windows.net:1433;database=is484db;") \
                    .option("driver","com.microsoft.sqlserver.jdbc.SQLServerDriver") \
                    .option("dbtable", table) \
                    .option("user", "user") \
                    .option("password", "Password123") \
                    .load()

    return query_result
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return('hello')


@app.route('/prediction') 
def prediction():
    print("Value Recieved Making Prediction")
    # dfmain=''
    # check=False
    # for i in range (23,27):
    #     filename='SFE_gen_vol_Aug2021/sfe_'+str(i)+"08.csv"
    #     if check==False:
    #         dfmain=pd.read_csv(filename)
    #         check=True
    #     else:
    #         tempdf=pd.read_csv(filename)
    #         dfmain=pd.concat([dfmain,tempdf])
    spark = SparkSession\
        .builder\
        .master('local[*]')\
        .appName('velociti_test')\
        .config("spark.driver.extraClassPath","C:/drivers/sqljdbc_9.4/enu/mssql-jdbc-9.4.0.jre8.jar")\
        .getOrCreate()
    query = 'SELECT TOP(1) * FROM ' + 'dbo.config'
    config_df = get_sql_query(spark, query)
        
    config_df2 = config_df.collect()
    config_row = config_df2[0]
    msgtype = config_row.asDict()['msgtype'].upper()
    threshold = config_row.asDict()['threshold']
    processingtime = config_row.asDict()['processingtime']
    timestep = config_row.asDict()['no_of_days_for_prediction']
    loaddatetime = config_row.asDict()['loaddatetime']
    print("==loading data==")
    query = (f"SELECT * FROM dbo.sfe WHERE msgtype = '{msgtype}' and CAST('{loaddatetime}' as date) between dateAdd(DD, -7, CAST('{loaddatetime}' as date)) and  CAST('{loaddatetime}' as date)")
    dfmain_sfe = get_sql_query(spark, query)
    dfmain_sfe = dfmain_sfe.toPandas()
    dfmain_sfe["datetimecreated"]=dfmain_sfe.datetimecreated.apply(getDatetime)
    sfe_count=dfmain_sfe.groupby(['datetimecreated']).msgid.count()
    dfmain_sfe["processingtime"] = pd.to_numeric(dfmain_sfe["processingtime"], downcast="float") 
    dfmain_sfe["CPU"] = pd.to_numeric(dfmain_sfe["CPU"], downcast="float") 
    dfmain_sfe["RAM"] = pd.to_numeric(dfmain_sfe["RAM"], downcast="float") 
    sfe_df=pd.DataFrame(dfmain_sfe.groupby('datetimecreated')["processingtime","CPU","RAM"].mean()) 
    sfe_df['Volume']=sfe_count

    query = (f"SELECT * FROM dbo.secore WHERE msgtype = '{msgtype}' and CAST('{loaddatetime}' as date) between dateAdd(DD, -7, CAST('{loaddatetime}' as date)) and  CAST('{loaddatetime}' as date)")
    dfmain_secore = get_sql_query(spark, query)
    dfmain_secore = dfmain_secore.toPandas()
    dfmain_secore["datetimecreated"]=dfmain_secore.datetimecreated.apply(getDatetime)
    secore_count=dfmain_secore.groupby(['datetimecreated']).msgid.count()
    dfmain_secore["processingtime"] = pd.to_numeric(dfmain_secore["processingtime"], downcast="float") 
    dfmain_secore["CPU"] = pd.to_numeric(dfmain_secore["CPU"], downcast="float") 
    dfmain_secore["RAM"] = pd.to_numeric(dfmain_secore["RAM"], downcast="float") 
    secore_df=pd.DataFrame(dfmain_secore.groupby('datetimecreated')["processingtime","CPU","RAM"].mean()) 
    secore_df['Volume']=secore_count

    query = (f"SELECT * FROM dbo.cdi WHERE msgtype = '{msgtype}' and CAST('{loaddatetime}' as date) between dateAdd(DD, -7, CAST('{loaddatetime}' as date)) and  CAST('{loaddatetime}' as date)")
    dfmain_CDI = get_sql_query(spark, query)
    dfmain_CDI = dfmain_CDI.toPandas()
    dfmain_CDI["datetimecreated"]=dfmain_CDI.datetimecreated.apply(getDatetime)
    CDI_count=dfmain_CDI.groupby(['datetimecreated']).msgid.count()
    dfmain_CDI["processingtime"] = pd.to_numeric(dfmain_CDI["processingtime"], downcast="float") 
    dfmain_CDI["CPU"] = pd.to_numeric(dfmain_CDI["CPU"], downcast="float") 
    dfmain_CDI["RAM"] = pd.to_numeric(dfmain_CDI["RAM"], downcast="float") 
    cdi_df=pd.DataFrame(dfmain_CDI.groupby('datetimecreated')["processingtime","CPU","RAM"].mean()) 
    cdi_df['Volume']=CDI_count


    # sfe_df=pd.DataFrame(dfmain.groupby('datetimecreated')["processingtime"].mean())
    print("==SFE MODEL==")
    vol= sfe_df['Volume'].resample('1min').mean()
    cpu= sfe_df['CPU'].resample('1min').mean()
    ram= sfe_df['RAM'].resample('1min').mean()
    vol.fillna(0,inplace=True)
    cpu.fillna(0,inplace=True)
    ram.fillna(0,inplace=True)
    print("predicting sfe vol")
    model = pm.auto_arima(vol, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    n_periods = 180
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = vol.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(vol.values , index=vol.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(0)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputVol=pd.concat([df,forecast])
    outputVol['Variable']="Volume"
    outputVol.reset_index(inplace=True)
    treshold=600
    outputVol["Color"]=outputVol["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    print("predicting sfe ram")
    model = pm.auto_arima(ram, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = ram.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(ram.values , index=ram.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(2)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputRam=pd.concat([df,forecast])
    outputRam['Variable']="Ram"
    outputRam.reset_index(inplace=True)
    treshold=600
    outputRam["Color"]=outputRam["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    print("predicting sfe cpu")
    model = pm.auto_arima(cpu, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = cpu.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(cpu.values , index=cpu.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(2)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputCPU=pd.concat([df,forecast])
    outputCPU['Variable']="CPU"
    outputCPU.reset_index(inplace=True)
    treshold=600
    outputCPU["Color"]=outputCPU["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    final_SFE_df=pd.concat([outputVol,outputRam,outputCPU])
    final_SFE_df.reset_index(drop=True,inplace=True)
    final_SFE_df.rename(columns={'index': "Time"}, inplace=True)
    final_SFE_df['Time']=final_SFE_df.Time.astype(str)
    print ("storing to sfe ouput db")
    spark = SparkSession.builder.appName("velociti").getOrCreate()
    # DB connection strings 
    server = 'is484dbserver.database.windows.net'  
    database = 'is484db'  
    username = 'user'  
    password = 'Password123'   
    cnxn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=tcp:is484dbserver.database.windows.net,1433;Database=is484db;Uid=user;Pwd={Password123};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;') 
    cursor = cnxn.cursor()
    startTime=datetime.now() 

    cursor.execute('TRUNCATE TABLE dbo.sfe_output')

    for index,row in final_SFE_df.iterrows(): 
        cursor.execute('INSERT INTO dbo.sfe_output([Time],[Value],[Status],[Color],[Variable]) values (?,?,?,?,?)',  
                        row['Time'],  
                        row['Value'],  
                        row['Status'], 
                        row['Color'],
                        row['Variable']) 
    endTime=datetime.now() 
    print(endTime-startTime) 

    cnxn.commit() 
    cursor.close() 
    cnxn.close()

    print("==SECORE MODEL==")
    vol= secore_df['Volume'].resample('1min').mean()
    cpu= secore_df['CPU'].resample('1min').mean()
    ram= secore_df['RAM'].resample('1min').mean()
    vol.fillna(0,inplace=True)
    cpu.fillna(0,inplace=True)
    ram.fillna(0,inplace=True)
    print("predicting secore vol")
    model = pm.auto_arima(vol, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    n_periods = 180
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = vol.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(vol.values , index=vol.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(0)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputVol=pd.concat([df,forecast])
    outputVol['Variable']="Volume"
    outputVol.reset_index(inplace=True)
    treshold=600
    outputVol["Color"]=outputVol["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    print("predicting sfe ram")
    model = pm.auto_arima(ram, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = ram.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(ram.values , index=ram.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(2)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputRam=pd.concat([df,forecast])
    outputRam['Variable']="Ram"
    outputRam.reset_index(inplace=True)
    treshold=600
    outputRam["Color"]=outputRam["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    print("predicting secore cpu")
    model = pm.auto_arima(cpu, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = cpu.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(cpu.values , index=cpu.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(2)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputCPU=pd.concat([df,forecast])
    outputCPU['Variable']="CPU"
    outputCPU.reset_index(inplace=True)
    treshold=600
    outputCPU["Color"]=outputCPU["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    final_SECORE_df=pd.concat([outputVol,outputRam,outputCPU])
    final_SECORE_df.reset_index(drop=True,inplace=True)
    final_SECORE_df.rename(columns={'index': "Time"}, inplace=True)
    final_SECORE_df['Time']=final_SECORE_df.Time.astype(str)
    print ("storing to SECORE ouput db")
    spark = SparkSession.builder.appName("velociti").getOrCreate()
    # DB connection strings 
    server = 'is484dbserver.database.windows.net'  
    database = 'is484db'  
    username = 'user'  
    password = 'Password123'   
    cnxn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=tcp:is484dbserver.database.windows.net,1433;Database=is484db;Uid=user;Pwd={Password123};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;') 
    cursor = cnxn.cursor()
    startTime=datetime.now() 

    cursor.execute('TRUNCATE TABLE dbo.secore_output')

    for index,row in final_SECORE_df.iterrows(): 
        cursor.execute('INSERT INTO dbo.secore_output([Time],[Value],[Status],[Color],[Variable]) values (?,?,?,?,?)',  
                        row['Time'],  
                        row['Value'],  
                        row['Status'], 
                        row['Color'],
                        row['Variable']) 
    endTime=datetime.now() 
    print(endTime-startTime) 

    cnxn.commit() 
    cursor.close() 
    cnxn.close()

    print("==CDI MODEL==")
    vol= cdi_df['Volume'].resample('1min').mean()
    cpu= cdi_df['CPU'].resample('1min').mean()
    ram= cdi_df['RAM'].resample('1min').mean()
    vol.fillna(0,inplace=True)
    cpu.fillna(0,inplace=True)
    ram.fillna(0,inplace=True)
    print("predicting cdi vol")
    model = pm.auto_arima(vol, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    n_periods = 180
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = vol.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(vol.values , index=vol.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(0)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputVol=pd.concat([df,forecast])
    outputVol['Variable']="Volume"
    outputVol.reset_index(inplace=True)
    treshold=600
    outputVol["Color"]=outputVol["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    print("predicting cdi ram")
    model = pm.auto_arima(ram, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = ram.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(ram.values , index=ram.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(2)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputRam=pd.concat([df,forecast])
    outputRam['Variable']="Ram"
    outputRam.reset_index(inplace=True)
    treshold=600
    outputRam["Color"]=outputRam["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    print("predicting cdi cpu")
    model = pm.auto_arima(cpu, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    # Forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = cpu.index[-1] + pd.to_timedelta(np.arange(1,n_periods+1), 'm')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)

    df=pd.DataFrame(cpu.values , index=cpu.index, columns=["Value"])
    # last histrocal date
    lastDate=df.index[-1].date()
    lastDate=datetime.combine(lastDate, datetime.min.time())
    # filter last day
    df=df[df.index>=lastDate]
    df['Status']="Historical"

    # round forecast result
    fc=fc.round(2)
    forecast=pd.DataFrame(fc, index=index_of_fc, columns=["Value"])
    forecast['Status']="Predicted"
    # concat
    outputCPU=pd.concat([df,forecast])
    outputCPU['Variable']="CPU"
    outputCPU.reset_index(inplace=True)
    treshold=600
    outputCPU["Color"]=outputCPU["Value"].apply(lambda x: 'Red' if x>treshold  else 'Yellow' if (treshold*0.8)<x<treshold else 'Green')

    final_CDI_df=pd.concat([outputVol,outputRam,outputCPU])
    final_CDI_df.reset_index(drop=True,inplace=True)
    final_CDI_df.rename(columns={'index': "Time"}, inplace=True)
    final_CDI_df['Time']=final_CDI_df.Time.astype(str)
    print ("storing to CDI ouput db")
    spark = SparkSession.builder.appName("velociti").getOrCreate()
    # DB connection strings 
    server = 'is484dbserver.database.windows.net'  
    database = 'is484db'  
    username = 'user'  
    password = 'Password123'   
    cnxn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=tcp:is484dbserver.database.windows.net,1433;Database=is484db;Uid=user;Pwd={Password123};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;') 
    cursor = cnxn.cursor()
    startTime=datetime.now() 

    cursor.execute('TRUNCATE TABLE dbo.cdi_output')

    for index,row in final_CDI_df.iterrows(): 
        cursor.execute('INSERT INTO dbo.cdi_output([Time],[Value],[Status],[Color],[Variable]) values (?,?,?,?,?)',  
                        row['Time'],  
                        row['Value'],  
                        row['Status'], 
                        row['Color'],
                        row['Variable']) 
    endTime=datetime.now() 
    print(endTime-startTime) 

    cnxn.commit() 
    cursor.close() 
    cnxn.close()

    df_list = final_SFE_df.values.tolist()
    JSONP_data = jsonpify(df_list)
    return JSONP_data


if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5001,debug=True)  # Enable reloader and debugger

