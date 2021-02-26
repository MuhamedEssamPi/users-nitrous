# Admin Dashboard 
# -*- coding: utf-8 -*-
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
df.dropna(subset=['powerBar'], inplace = True)
# Starting since 2021
df=df.loc[df['startYear'].isin([21])] 
df.dropna(subset=['powerBar'], inplace = True)
df=df.reset_index(drop=True)

users = df['username'].unique()
dataframes=[]
banned_list =['hossary','mostafa','mostaf','yyyy','gracegarcia']
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
        df_user['year'] = df_user['year'].replace([21],2021)
        
        dataframes.append(df_user)

users=Names
N=len(users)
for z in range(N):
    x=pd.to_datetime(dataframes[1][['year', 'month', 'day', 'hour', 'minute']])
    g=pd.DataFrame(x,columns=['date'])
    dataframes[z]=pd.concat([g,dataframes[z]],axis=1)
    dataframes[z]['week_number_of_year'] = dataframes[z]['date'].dt.week
    dataframes[z]=dataframes[z].dropna()
    
from datetime import datetime
b=datetime.now()
num=int(b.strftime("%V"))

dataforml=[0] * N

l=[]
for v in range (0,num):
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
app =dash.Dash()
app = dash.Dash(__name__)
server = app.server
app.title = 'Nitrous Admin Dashboard'

df.dropna(inplace=True)
print('in Nitrous')
fig1 =px.sunburst(df, path=['startYear','startMonth','title', 'topic'], values='duration')
fig2 =px.sunburst(df, path=['startYear','startMonth','username'], values='duration')



categories = ["Work Ethics","Student Mentality","Self Management","Technical Skills","Interpersonal","LeaderShip"]

radar = go.Figure()
for k in range(N-1):
    radar.add_trace(go.Scatterpolar(
          r=current_states[k],
          theta=categories,
          fill='toself',
          name=Names[k] ))

radar.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 100]
    )),
  showlegend=True
)
radar.update_layout(
    title={
        'text': "Cumulative Users Power Bar Performance",
        'y':0.95,
        'x':0.48,
        'xanchor': 'center',
        'yanchor': 'top'
    })


fig1.update_layout(
    title={
        'text': "Cumulative Users Time Distribution",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })

fig2.update_layout(
    title={
        'text':'Documentation Time',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })

app.layout=html.Div([

    html.Div([html.A([html.H2('Nitrous Dashboard'),html.Img(src='/assets/nitrous-logo.png')], href='http://projectnitrous.com/')],className="banner"),



     html.Div([html.Div(dcc.Graph(id="Radar",figure=radar))],className="twelve columns"),
     html.Div([html.Div(dcc.Graph(id="Pie1",figure=fig1))],className="five columns"),
     html.Div([html.Div(dcc.Graph(id="Pie2",figure=fig2))],className="five columns"),

#     html.Div([html.Div(dcc.Graph(id="Violin"))],className="ten columns"),
#     html.Div([html.Div(dcc.Graph(id="Table"))],className="ten columns")


  ])




  


while( __name__ == '__main__'):
    app.run_server(debug=True)