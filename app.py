# Users Dashboard 
# -*- coding: utf-8 -*-
import base64
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import pathlib
from dash.dependencies import Input, Output, State
from scipy import stats
import xlrd
import plotly.express as px
import dash_table
import pyrebase
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os 
import threading
import sklearn
import xlrd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import warnings
import time 
import schedule
warnings.filterwarnings("ignore")

credential_path = './fire.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
cred = credentials.Certificate(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
firebase_admin.initialize_app(cred)

# ###################################################### Talking to Firebase 
def Nitrous():
    db = firestore.client()
    Actions = list(db.collection(u'Nitrous').document(u'Actions').collection(u'Actions').stream())
    Actions = list(map(lambda x: x.to_dict(), Actions))
    df=pd.DataFrame(Actions)
    #Merging 
    df.loc[df['username'].isnull(),'username'] = df['userName']
    df=df.drop(['formattedDate','endMonth','endDay','endYear','id','status','value','userName','factor','status'],1)
    df.dropna()
    #Renaming
    df["username"] = df["username"].str.lower()
    df['powerBar'] = df['powerBar'].replace([1.0],'Int.Comm')
    df['powerBar'] = df['powerBar'].replace([2.0],'Ext.Comm')
    df['powerBar'] = df['powerBar'].replace([3.0],'Learn')
    df['powerBar'] = df['powerBar'].replace([4.0],'Tech')
    df['powerBar'] = df['powerBar'].replace([5.0],'Reletive')
    df['powerBar'] = df['powerBar'].replace([6.0],'Teach')
    df['powerBar'] = df['powerBar'].replace([0],'Break')
    df['username'] = df['username'].replace('HEGAZY','hegazy')
    df['username'] = df['username'].replace('hegazi','hegazy')
    df['username'] = df['username'].replace( 'Nada hashim','nada hashim')
    df['username'] = df['username'].replace( 'nahla.khaled','nahla khaled')
    df['username'] = df['username'].replace('amansour','mansour' )
    # Replacing -ve value with 120 mins
    df['duration']=df['duration'].mask(df['duration'] < 0, 120)
    df=df[42:]
    df.dropna(subset=['powerBar'], inplace = True)

    df=df.reset_index(drop=True)
    users = df['username'].unique()
    dataframes=[]
    banned_list =['hossary','mostafa','mostaf']
    clean = pd.DataFrame(np.zeros((15, 9)))
    clean.columns = ["Int. Comm","Ext. Comm","Learn","Tech","Reletive","Break","Teach","Total Hours","UserName"]
    i=-1
    Names=[]
    for j, user in enumerate( users) :
        if not user in banned_list :
            i+=1

            Names.append(user)
            hk = df.groupby('username') 
            df_user=hk.get_group(user)
            df_user=df_user.reset_index(drop=True)

            df_user=df_user.rename(columns={'startYear':"year", 'startMonth':"month", 'startDay':"day", 'startHour':"hour", 'startMin':"minute"})
            df_user['year'] = df_user['year'].replace([20],2020)

            dataframes.append(df_user)
            # to export  to excel 
            #dataframes[i].to_excel(str(user)+'_df.xlsx')
    N=len(users)-len(banned_list)
    x=pd.to_datetime(dataframes[1][['year', 'month', 'day', 'hour', 'minute']])
    g=pd.DataFrame(x,columns=['date'])
    dataframes[1]=pd.concat([g,dataframes[1]],axis=1)
    dataframes[1]['week_number_of_year'] = dataframes[1]['date'].dt.week
    num=max(dataframes[1]['week_number_of_year'].values)-min(dataframes[1]['week_number_of_year'].values)
    num=num+1

    dataforml=[0] * N

    l=[]
    for v in range (35,35+num):
        l.append(v)
    for k in range(N-1):
        dataforml[k] = pd.DataFrame(np.zeros((num, 9)), index =l)
        dataforml[k].columns = ["Int.Comm","Ext.Comm","Learn","Tech","Reletive","Break","Teach","Total Hours","UserName"]
        dataforml[k]['UserName'][:k+num]=Names[k]
        dataforml[k].index.name = 'week_number_of_year'



    for k in range(N-1):

        gk = dataframes[k].groupby('powerBar')

        try :

            h=gk.get_group('Int.Comm') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Int.Comm"]=v['duration']
        except:
            pass



        try:
            h=gk.get_group('Ext.Comm') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Ext.Comm"]=v['duration']
        except:
            pass

        try:
            h=gk.get_group('Learn') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Learn"]=v['duration']
        except:
            pass
        try:

            h=gk.get_group('Tech') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Tech"]=v['duration']

        except:
            pass

        try:
            h=gk.get_group('Reletive') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Reletive"]=v['duration']
        except:
            pass
        try:

            h=gk.get_group('Break') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Break"]=v['duration']

        except:
            pass
        try:
            h=gk.get_group('Teach') 
            h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
            h['week_number_of_year'] = h['date'].dt.week 
            v=h.groupby(["week_number_of_year"]).sum()/60
            dataforml[k]["Teach"]=v['duration']

        except:
            pass

        dataframes[k].dropna(inplace=True)
        dataforml[k].fillna(0, inplace=True)

        dataforml[k]['Total Hours']=dataforml[k].iloc[:,:7].sum(axis=1)

    #ML Part

    data=pd.read_excel('Nitrous.xlsx')
    Features=data.iloc[:,:8].values
    Labels=data.iloc[:,9:].values
    #Setting 20 % of the Data for Testing 
    X_train, x_test_old, y_train, y_test_old = train_test_split(Features, Labels, test_size=0.20, random_state=4)
    # Standardization
    scaler = StandardScaler()

    X_train=scaler.fit_transform(X_train)

    regressor = RandomForestRegressor(random_state=1)
    regressor.fit(X_train,y_train)

    Preds=[0] * N
    Final=[0]*N
    for k in range(N-1):

        X_pred=scaler.transform(dataforml[k].drop(['UserName'],axis=1))
        y_pred=regressor.predict(X_pred)
        Preds[k]=pd.DataFrame(y_pred,index=l)

        Preds[k].columns =["Work Ethics","Student Mentality","Self Management","Technical Skills","Interpersonal","LeaderShip"]
        Final[k]=pd.concat([dataforml[k],Preds[k]],axis=1)
        Final[k]=Final[k].round(decimals=2)


    current_states=[0]*N
    for k in range (N-1):
        current_states[k]= Final[k].iloc[:,9:].sum(axis=0)+50
    currentweek=[0]*N    
    for k in range(N-1):
        currentweek[k]= pd.DataFrame(np.zeros((8, 2)))
        currentweek[k].columns = ["topics","duration"]
        currentweek[k]['duration']= pd.DataFrame(Final[k].iloc[-1,:8].values)
        currentweek[k]['topics']=pd.DataFrame(['Int.Comm','Ext.Comm','Learn','Tech','Reletive','Break','Teach','Total Hours'])
    for k in range (len(Names)):
        try:
            doc_ref = db.collection(u'Nitrous').document(u'Users').collection(dataframes[k]['userID'].values[0]).document(u'info')
            doc_ref.update({
                            u'class1': int(current_states[k][0]),
                            u'class2': int(current_states[k][1]),
                            u'class3': int(current_states[k][2]),
                            u'class4': int(current_states[k][3]),
                            u'class5': int(current_states[k][4]),
                            u'class6': int(current_states[k][5]),
            })
        except:
            print ('This User '+str(Names[k]+' has no actions'))
        
    ########################################################## Dash        
        







    df=Final[0].iloc[:,:8]
    table = go.Figure(data=[go.Table(header=dict(values=list(df.columns),fill_color='#ee1a24',align='center'),cells=dict(values=[df['Int.Comm'],df['Ext.Comm'],df['Learn'], df['Tech'], df['Reletive'],df['Break'],df['Teach'],df['Total Hours']],fill_color='lavender',align='center'))])

    sunburst =px.sunburst(dataframes[0], path=['year','month','title', 'topic'], values='duration')

    q = pd.DataFrame(dict( r= current_states[0],theta=["Work Ethics","Student Mentality","Self Management","Technical Skills","Interpersonal","LeaderShip"]))
    radar = px.line_polar(q, r='r', theta='theta', line_close=True)
    radar.update_traces(fill='toself')


    line = go.Figure()
    line.add_trace(go.Scatter(x=dataforml[0].index, y=Final[0]['Work Ethics'].values          ,mode='lines+markers',name='Work Ethics'))
    line.add_trace(go.Scatter(x=dataforml[0].index, y=Final[0]['Student Mentality'].values    ,mode='lines+markers',name='Student Mentality'))
    line.add_trace(go.Scatter(x=dataforml[0].index, y=Final[0]['Self Management'].values      ,mode='lines+markers',name='Self Management'))
    line.add_trace(go.Scatter(x=dataforml[0].index, y=Final[0]['Technical Skills'].values     ,mode='lines+markers',name='Technical Skills'))
    line.add_trace(go.Scatter(x=dataforml[0].index, y=Final[0]['Interpersonal'].values        ,mode='lines+markers',name='Interpersonal'))
    line.add_trace(go.Scatter(x=dataforml[0].index, y=Final[0]['LeaderShip'].values           ,mode='lines+markers',name='LeaderShip'))


    app.layout=html.Div([

        html.Div([html.A([html.H2('Nitrous Dashboard'),html.Img(src='/assets/nitrous-logo.png')], href='http://projectnitrous.com/')],className="banner"),
        html.Div([dcc.Dropdown(id='demo-dropdown',
                               options=[{'label':name, 'value':i} for i,name in enumerate( Names)],value=  0),
                                        ],style={'margin-bottom': '10px','textAlign':'center','width':'220px','margin':'auto'}),



         html.Div([html.Div(dcc.Graph(id="Radar",figure=radar))],className="five columns"),
         html.Div([html.Div(dcc.Graph(id="SunBurst",figure=sunburst))],className="five columns"),
         html.Div([html.Div(dcc.Graph(id="Line",figure=line))],className="ten columns"),
         html.Div([html.Div(dcc.Graph(id="Table",figure=table))],className="ten columns")




      ])



    @app.callback(dash.dependencies.Output('Table','figure'),
                 [dash.dependencies.Input('demo-dropdown','value')]
                 )
    def update_fig(input_value):
        df=Final[input_value].iloc[:,:8]
        table = go.Figure(data=[go.Table(header=dict(values=list(df.columns),font=dict(color='white'),fill_color='#ee1a24',align='center'),cells=dict(values=[df['Int.Comm'],df['Ext.Comm'],df['Learn'], df['Tech'], df['Reletive'],df['Break'],df['Teach'],df['Total Hours']],fill_color='lavender',align='center'))])
        return table

    @app.callback(dash.dependencies.Output('Radar','figure'),
                 [dash.dependencies.Input('demo-dropdown','value')]
                 )
    def update_fig(input_value):
        q = pd.DataFrame(dict( r= current_states[input_value],theta=["Work Ethics","Student Mentality","Self Management","Technical Skills","Interpersonal","LeaderShip"]))
        radar = px.line_polar(q, r='r', theta='theta', line_close=True)
        radar.update_traces(fill='toself')    
        return radar        

    @app.callback(dash.dependencies.Output('SunBurst','figure'),
                 [dash.dependencies.Input('demo-dropdown','value')]
                 )
    def update_fig(input_value):
        sunburst =px.sunburst(dataframes[input_value], path=['year','month','title', 'topic'], values='duration')
        return sunburst        

    @app.callback(dash.dependencies.Output('Line','figure'),
                 [dash.dependencies.Input('demo-dropdown','value')]
                 )
    def update_fig(input_value):
        line = go.Figure()
        line.add_trace(go.Scatter(x=dataforml[input_value].index, y=Final[input_value]['Work Ethics'].values          ,mode='lines+markers',name='Work Ethics'))
        line.add_trace(go.Scatter(x=dataforml[input_value].index, y=Final[input_value]['Student Mentality'].values    ,mode='lines+markers',name='Student Mentality'))
        line.add_trace(go.Scatter(x=dataforml[input_value].index, y=Final[input_value]['Self Management'].values      ,mode='lines+markers',name='Self Management'))
        line.add_trace(go.Scatter(x=dataforml[input_value].index, y=Final[input_value]['Technical Skills'].values     ,mode='lines+markers',name='Technical Skills'))
        line.add_trace(go.Scatter(x=dataforml[input_value].index, y=Final[input_value]['Interpersonal'].values        ,mode='lines+markers',name='Interpersonal'))
        line.add_trace(go.Scatter(x=dataforml[input_value].index, y=Final[input_value]['LeaderShip'].values           ,mode='lines+markers',name='LeaderShip'))
        return line  

app =dash.Dash()
app = dash.Dash(__name__)
server = app.server
app.title = 'Nitrous Users Dashboard'
Nitrous() 

  


if __name__ == '__main__':
    app.run_server(debug=True)
    
schedule.every().day.at("00:00").do(Nitrous)
while True: 

    schedule.run_pending() 
    time.sleep(1)   