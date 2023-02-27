"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from datetime import datetime

import numpy as np
from datetime import datetime as dt
from datetime import date, timedelta
import os
from statistics import mean
from random import randint, shuffle
import pandas as pd
import plotly.graph_objs as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash, dash_table
from collections import OrderedDict
from dash import Dash, dash_table
import pandas as pd
from collections import OrderedDict
from dash import Dash, dcc, html, Output, Input       # pip install dash
import dash_bootstrap_components as dbc               # pip install dash-bootstrap-components
import plotly.express as px                     # pip install pandas; pip install plotly express
import pygsheets
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

import dash.dash_table.FormatTemplate as FormatTemplate
from dash.dash_table.Format import Sign

################################################################
################################################################

import pygsheets

import datetime as dt
gc = pygsheets.authorize(service_file='client_secrets.json')
sh = gc.open_by_key('1fJ4v3ZkUVTF5I3WC2XJ_bhk3VMeFpUr38PG3x8L4AEU')
worksheet1 = sh.worksheet('title', 'Hoja 1')
worksheet1 = sh.worksheet('title', 'Consolidado_')

sheetData = worksheet1.get_all_records()
print('Desde Metadata!!!')
DFMetadata = pd.DataFrame(sheetData)
print('DFMetadata= ' )
print(DFMetadata)
print(DFMetadata.columns)


#DFMetadata= DFMetadata.drop(columns=['Responsable'])
print(DFMetadata.columns)


def Generador_DFMetadata(DFMetadata):


    DFMetadata['C']=DFMetadata['Indice'].astype(str).str.count('.') 
    DFMetadata['C']= DFMetadata['C']-2
    DFMetadata['C']= DFMetadata['C'].astype(str)

    #DFMetadata['N2']= DFMetadata['N2'].astype(int)
    #DFMetadata['N3']= DFMetadata['N3'].astype(int)
    DFMetadata['C']= ""


    DFMetadata['C']= DFMetadata['C'].astype(str)

    DFMetadata.loc[(DFMetadata['N2'] ==0 ) , 'C'] = "1"
    DFMetadata.loc[(DFMetadata['N3'] ==0 ) , 'C'] = "2"
    #DFMetadata.loc[(DFMetadata['C'] ==2 ) & ( DFMetadata['N3']==3) , 'C'] = 2
    dft=DFMetadata[DFMetadata['C']=='2']
    print("*_"*30)
    print(dft[['Actividad','C']])
    print("*_"*30)


    #DFMetadata['PAvance']=
    DFMetadata.loc[DFMetadata['PAvance'] == 0, 'PAvance'] =None
    DFMetadata['Flag']=0
    DFMetadata['hoy'] = dt.datetime.now()


    print(DFMetadata[['hoy','Inicio','Fin']] )

    DFMetadata['hoy']       = pd.to_datetime(DFMetadata['hoy']    ,format="%d/%m/%Y"  ,infer_datetime_format=True,dayfirst=True)
    DFMetadata['Inicio_']   = pd.to_datetime(DFMetadata['Inicio'] ,format="%d/%m/%Y"  ,infer_datetime_format=True,dayfirst=True)
    DFMetadata['Fin_']      = pd.to_datetime(DFMetadata['Fin']    ,format="%d/%m/%Y"  ,infer_datetime_format=True,dayfirst=True)


    DFMetadata['hoy_'] = DFMetadata['hoy'].dt.strftime('%d/%m/%Y')
    DFMetadata['hoy_']      = DFMetadata['hoy'].dt.strftime('%Y-%m-%d')
    DFMetadata['Fin_']      = DFMetadata['Fin_'].dt.strftime('%Y-%m-%d')

    print(DFMetadata['hoy_'])
    print(DFMetadata[['ind', 'hoy_','Inicio_','Fin_']] )

    #
    DFMetadata['Inicio_']   = DFMetadata['Inicio_'].dt.strftime('%Y-%m-%d')
    #DFMetadata['Inicio_'] = DFMetadata['Inicio_'].dt.strftime('%d/%m/%Y')


    #DFMetadata['Flag']= DFMetadata['Inicio_']- DFMetadata['hoy_']

    #DFMetadata['Flag'] = (DFMetadata['Inicio_'] - DFMetadata['hoy_']).dt.days
    #DFMetadata['Flag'] = DFMetadata['Inicio_'].sub(DFMetadata['hoy_'], axis=0)
    # semaforo solo a las tareas que se iniciarion 



    #DFMetadata['Inicio_T'] = pd.to_datetime(DFMetadata['Inicio_'],unit='s')
    #DFMetadata['hoy_T'] = pd.to_datetime(DFMetadata['hoy'],unit='s')
    DFMetadata['Flag2']=0
    #DFMetadata.loc[(DFMetadata['Inicio_T'] < DFMetadata['hoy_T']) , 'Flag2']=1


    DFMetadata4=(DFMetadata[(DFMetadata['N_']=="Artistas")  ])
    print(DFMetadata4[['C','Indice', 'hoy' , 'Inicio', 'Fin' ,'Flag','Flag2']])

    print(DFMetadata[['C','Indice', 'hoy' , 'Inicio_', 'Fin_' ,'Flag','Flag2']])

    #DFMetadata[['N1', 'N2', 'N3','N4']] = DFMetadata['Indice'].astype(str).str.split('.', expand=True)

    print(DFMetadata[['C','Indice','N1', 'N2', 'N3','PAvance']])


    #DFMetadata['PAvance_']= DFMetadata['PAvance'].str.replace(',', '').astype('Int32')
    #DFMetadata['PAvance_']= DFMetadata['PAvance'].astype('Int32')

    DFMetadata['PAvance_'] = pd.to_numeric(DFMetadata['PAvance'], errors='coerce')
    DFMetadata['N2'] = pd.to_numeric(DFMetadata['N2'], errors='coerce')
    DFMetadata['PAvance_'] = DFMetadata['PAvance_'].fillna(0)
    DFMetadata['P_']=0
    #DFMetadata['PAvance_']= DFMetadata['PAvance_']/100


    ###print("______________________________________________________________________________________________fase 1")

    for j in (DFMetadata['Intervención'].unique().tolist()):
        ###print('*'*30)
        ###print('j= ', j)
        #for i in (DFMetadata['N1'].unique().tolist()):

        l=DFMetadata[DFMetadata['Intervención']==j ]['N1'].unique().tolist()
        ###print('l=', l)
        for i in (l):
            ###print('i= ', i)
            ###print(DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0)& (DFMetadata['Intervención'] == j), 'PAvance_'].mean())
        
            ###print(DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0)& (DFMetadata['Intervención'] == j)][['ind', 'PAvance','Inicio_', 'Fin_']])
            dfmin=(DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0)& (DFMetadata['Intervención'] == j)][['Inicio_']])
            dfmax=(DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0)& (DFMetadata['Intervención'] == j)][['Fin_']])
            dfmin=dfmin.dropna()
            dfmax=dfmax.dropna()

            ###print('min =',dfmin.min().values )
            ###print('max =',dfmax.max().values )

            #DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'P_']
            ###print('0=', DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] ==0)& (DFMetadata['Intervención'] == j)][['ind', 'PAvance','Inicio', 'Fin']])

            DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'P_']=DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0)& (DFMetadata['Intervención'] == j), 'PAvance_'].mean().round(2)
            DFMetadata['P_'] = DFMetadata['P_'].round(2)
            DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'PAvance']=DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'P_']

            #DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'P_']=DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0)& (DFMetadata['Intervención'] == j), 'PAvance_'].mean()


            #DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'Inicio'] = dfmin.min().values[0].dt.strftime('%Y-%m-%d')
            #DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'Fin'   ] = dfmax.max().values[0].dt.strftime('%Y-%m-%d')

            DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'Inicio'] = datetime.strptime(str(dfmin.min().values[0]), '%Y-%m-%d')
            DFMetadata.loc[(DFMetadata['N1'] == i) & (DFMetadata['N2'] == 0) & (DFMetadata['Intervención'] == j), 'Fin'   ] = datetime.strptime(str(dfmax.max().values[0]), '%Y-%m-%d')


            ###print('1=', DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] ==0)& (DFMetadata['Intervención'] == j)][['ind', 'PAvance','Inicio', 'Fin']])



    DFMetadata['Inicio']   = pd.to_datetime(DFMetadata['Inicio'] ,format="%d/%m/%Y"  ,infer_datetime_format=True,dayfirst=True)
    DFMetadata['Fin']      = pd.to_datetime(DFMetadata['Fin']    ,format="%d/%m/%Y"  ,infer_datetime_format=True,dayfirst=True)

    DFMetadata['Fin']      = DFMetadata['Fin'].dt.strftime('%Y-%m-%d')
    DFMetadata['Inicio']      = DFMetadata['Inicio'].dt.strftime('%Y-%m-%d')
    DFMetadata['P_'] = DFMetadata['P_'].round(2)

    ###print("______________________________________________________________________________________________Fase 2")
    for j in (DFMetadata['Intervención'].unique().tolist()):

        ###print('j= ', j)
        #print('-->', DFMetadata['N2'].unique().tolist())

        l=DFMetadata[DFMetadata['Intervención']==j ]['N2'].unique().tolist()
        ###print('l=', l)
        for i in (l):
            ###print('<>'*10)
            ###print('i======= ', i)
            ###print('mean= ', DFMetadata.loc[(DFMetadata['N2'] ==i) & (DFMetadata['N3'] !=0)& (DFMetadata['Intervención'] == j), 'PAvance_'].mean())
            ###print('df= ',DFMetadata.loc[(DFMetadata['N2'] ==i) & (DFMetadata['N3'] !=0)& (DFMetadata['Intervención'] == j)][['ind','PAvance_']])
            DFMetadata.loc[(DFMetadata['N2'] == i) & (DFMetadata['N3'] == 0) & (DFMetadata['Intervención'] == j), 'P_']=DFMetadata.loc[(DFMetadata['N2'] ==i) & (DFMetadata['N3'] !=0)& (DFMetadata['Intervención'] == j), 'PAvance_'].mean().round(2)
            DFMetadata.loc[(DFMetadata['N2'] == i) & (DFMetadata['N3'] == 0) & (DFMetadata['Intervención'] == j), 'PAvance']=DFMetadata.loc[(DFMetadata['N2'] == i) & (DFMetadata['N3'] == 0) & (DFMetadata['Intervención'] == j), 'P_']
            
            dfmin=(DFMetadata.loc[(DFMetadata['N2'] ==i) & (DFMetadata['N3'] !=0)& (DFMetadata['Intervención'] == j)][['Inicio']])
            dfmax=(DFMetadata.loc[(DFMetadata['N2'] ==i) & (DFMetadata['N3'] !=0)& (DFMetadata['Intervención'] == j)][['Fin']])
            ###print('dfmin= ', dfmin)
            ###print('dfmax= ', dfmax)
            dfmin=dfmin.dropna()
            dfmax=dfmax.dropna()
            ###print('min =',dfmin.min().values[0] )
            ###print('max =',dfmax.max().values[0] )
            DFMetadata.loc[(DFMetadata['N2'] == i) & (DFMetadata['N3'] == 0) & (DFMetadata['Intervención'] == j), 'Inicio'] = datetime.strptime(str(dfmin.min().values[0]), '%Y-%m-%d')
            DFMetadata.loc[(DFMetadata['N2'] == i) & (DFMetadata['N3'] == 0) & (DFMetadata['Intervención'] == j), 'Fin'   ] = datetime.strptime(str(dfmax.max().values[0]), '%Y-%m-%d')

            ###print('<>'*10)




    ###print("______________________________________________________________________________________________")
    import numpy as np
    DFMetadata['Inicio']   = pd.to_datetime(DFMetadata['Inicio'] ,format="%Y-%m-%d"  ,infer_datetime_format=True,dayfirst=True)
    DFMetadata['Fin']      = pd.to_datetime(DFMetadata['Fin']    ,format="%Y-%m-%d"  ,infer_datetime_format=True,dayfirst=True)
    DFMetadata['hoy_']      = pd.to_datetime(DFMetadata['hoy_']    ,format="%Y-%m-%d"  ,infer_datetime_format=True,dayfirst=True)

    DFMetadata['Fin']      = DFMetadata['Fin'].dt.strftime('%Y-%m-%d')
    DFMetadata['Inicio']      = DFMetadata['Inicio'].dt.strftime('%Y-%m-%d')
    DFMetadata['P_'] = DFMetadata['P_'].round(2)
    print(DFMetadata[['Inicio', 'Fin','hoy_']])
    #DFMetadata['diff'] = (DFMetadata['Fin'] - DFMetadata['hoy_'])

    DFMetadata['Fin__'] = pd.to_datetime(DFMetadata['Fin'])
    DFMetadata['diff'] = (DFMetadata['Fin__'] - dt.datetime.now()).dt.days  #method 11
    #DFMetadata["diff"] = (DFMetadata["Fin"] - DFMetadata["hoy_"])/np.timedelta64(1, 'D')

    global gcols
    gcols=0
    #data.loc[3, ['Price']] = [65]

    #DFMetadata['PAvance'] = DFMetadata['PAvance'].round(2)
    print(DFMetadata[['C','Indice','N1', 'N2', 'N3','PAvance','PAvance_', 'P_']])


    DFMetadata.loc[(DFMetadata['Inicio'] < DFMetadata['hoy']) , 'Flag'] = 1
    DFMetadata['Pt_str0']=DFMetadata['PAvance'].astype(str) +" %"
    DFMetadata.loc[(DFMetadata['PAvance'] == "") , 'Pt_str0'] = ""
    
    DFMetadata['Alerta'] =""

    DFMetadata.loc[(DFMetadata['diff'] > 0) & (DFMetadata['diff'] <= 5)& (DFMetadata['PAvance'] != 100) , 'Alerta'] =  "Quedan <= 5 dias"


    # Pendietne enproceso cumplio ""
    """
    Estado
    Pendietne --> comenzo el proceso pero no han iniciado
    en rpoceso --> comenzo el proceso y ya hay avance   -- fecha fin aun no alcanzadsoa
    cumplio 
    no cumplio 
    """

    #DFMetadata.loc[(DFMetadata['Flag'] == 1 ) , 'PAvance']="En Implementación"

    DFMetadata['P_'] = DFMetadata['P_'].round(2)


    return DFMetadata

DFMetadata= Generador_DFMetadata(DFMetadata)

def Generador_DFMetadata2(DFMetadata):



    DFMetadata2=DFMetadata.copy()

    DFMetadata2['Pt_']=0

    print('*'*80)
    print(DFMetadata2)
    print('*'*80)


    for j in (DFMetadata2['Intervención'].unique().tolist()):
        #print('j= ', j)
        #for i in (DFMetadata['N1'].unique().tolist()):
        #print('i= ', i)
        #print(DFMetadata.loc[(DFMetadata['N1'] ==i) & (DFMetadata['N2'] !=0), 'PAvance_'].mean())
        #print(DFMetadata2.loc[ (DFMetadata2['N2'] ==0)& (DFMetadata2['Intervención'] == j), 'P_'].mean())
        DFMetadata2.loc[ (DFMetadata2['N2'] == 0) & (DFMetadata2['Intervención'] == j), 'Pt_'] =    DFMetadata2.loc[ (DFMetadata2['N2'] ==0)& (DFMetadata2['Intervención'] == j), 'P_'].mean()


    #Estado		Porcentaje de avance de la medida	Alerta

    DFMetadata2['Estado']=''
    DFMetadata2.loc[ (DFMetadata2['Pt_'] < 100) , 'Estado'] ='En implementación' 
    DFMetadata2['Pt_'] = DFMetadata2['Pt_'].round(2)

    DFMetadata2['Alerta']=''
    DFMetadata2['Comentario']='Avance con respecto a lo planeado'

    #DFMetadata2['Comentario']

    #'filter_query': '{PAvance} >= 70 && {PAvance} <= 100',
    #'filter_query': '{PAvance} >= 11 && {PAvance} <= 69',
    #'filter_query': '{PAvance} >= 0 && {PAvance} <= 10',
    DFMetadata2['PA']= DFMetadata2['PAvance_'].astype(int)

    DFMetadata2.loc[(DFMetadata2['Pt_'] >= 0)  & (DFMetadata2['Pt_'] <=10)  , 'Comentario'] = "Alerta"
    DFMetadata2.loc[(DFMetadata2['Pt_'] >= 11) & (DFMetadata2['Pt_'] <=69)  , 'Comentario'] = "En proceso"
    DFMetadata2.loc[(DFMetadata2['Pt_'] >= 70) & (DFMetadata2['Pt_'] <=100) , 'Comentario'] = "Óptimo"


    """
    Alerta (Todo Rojo):
    - Hito retrasado
    - Actividad en riesgo( no mencionar la tarea - solo el retraso de la actividad )
    
 
    Comentario: 
    - Especificar los dias de retraso en meses y dias 
    - motivo del retraso , actividad secuencial , actividad paralela - % buffer 
      comprometio el cronograma? 


    """



    DFMetadata3=(DFMetadata2[(DFMetadata2['N2']==0) & (DFMetadata2['N1']== 1) ])
    print(DFMetadata3[['Estado','Intervención','Pt_','Alerta']])

    print(DFMetadata3)
    DFMetadata3['Pt_str']=DFMetadata3['Pt_'].astype(str) +" %"




    return DFMetadata3

DFMetadata3= Generador_DFMetadata2(DFMetadata)


sh = gc.open_by_key('1mSPf0eP3sUKkbYgyp8hKTjLWLwDqDdIAyHKhwB28rQ0')
worksheet1 = sh.worksheet('title', 'Hoja 1')
sheetData = worksheet1.get_all_records()
DFBD = pd.DataFrame(sheetData)

DFBD['Act']="Agua y Saneamiento"

DFBD.loc[DFBD['ACTIVIDAD_ACCION_OBRA'] == 5006300, 'Act'] = "Agua"
DFBD.loc[DFBD['ACTIVIDAD_ACCION_OBRA'] == 5006299, 'Act'] = "Agua"

DFBD.loc[DFBD['ACTIVIDAD_ACCION_OBRA'] == 5006300, 'Act'] = "Agua"
DFBD.loc[DFBD['ACTIVIDAD_ACCION_OBRA'] == 5006299, 'Act'] = "Agua"


#agua  

print('Act df2 = ', (DFBD['Act'].unique().tolist()))


df2 = DFBD.groupby('TIPO_GOBIERNO_NOMBRE')['MONTO_PIM'].sum()
print(df2)
df2 = DFBD.groupby('TIPO_GOBIERNO_NOMBRE')['MONTO_DEVENGADO', 'MONTO_PIM'].sum()
print(df2)
df2['Avance%']= (df2['MONTO_DEVENGADO'] / df2['MONTO_PIM'])*100
print(df2)


df2['TIPO_GOBIERNO_NOMBRE']=df2.index
R=  (df2['MONTO_DEVENGADO'].sum() / df2['MONTO_PIM'].sum())*100
print('R= ', R)
R2=  (DFBD['MONTO_DEVENGADO'].sum() / DFBD['MONTO_PIM'].sum())*100
print('R2= ', R2)

print('len= ', len(df2['TIPO_GOBIERNO_NOMBRE'].unique().tolist()))
print('--->',[""]*len(df2['TIPO_GOBIERNO_NOMBRE'].unique().tolist()) )
#and ACTIVIDAD_ACCION_OBRA in (5006300,5006299

df2['Avance%'] = df2['Avance%'].round(2)

#ACTIVIDAD_ACCION_OBRA_NOMBRE

 
#print('Act df2 = ', (df2['Act'].unique().tolist()))


df3 = DFBD.groupby(['TIPO_GOBIERNO_NOMBRE', 'Act'])['MONTO_DEVENGADO', 'MONTO_PIM'].sum()
#print(df3)
df3['Avance%']= (df3['MONTO_DEVENGADO'] / df3['MONTO_PIM'])*100
df3['TIPO_GOBIERNO_NOMBRE']=df3.index

df3[['b1', 'b2']] = pd.DataFrame(df3['TIPO_GOBIERNO_NOMBRE'].tolist(), index=df3.index)

print(df3)




labels0  = df2['TIPO_GOBIERNO_NOMBRE'].unique().tolist()            +df3['b2'].tolist()
parents0 = [""]*len(df2['TIPO_GOBIERNO_NOMBRE'].unique().tolist())  +df3['b1'].tolist()
values0  = df2['Avance%'].unique().tolist()                         +df3['Avance%'].tolist()

print('labels=  ', labels0 )
print('parents= ', parents0)
print('values=  ', values0 )

print(df3[['b1', 'b2', 'Avance%']])


import itertools
ab = itertools.chain(['it'], ['was'], ['annoying'])
list(ab)



# Text field
def pag0():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Tablero de Seguimiento"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])


# Text field
def Hitos():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Avance de las medidas del Plan Con Punche Perú"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

# Text field
def detalleHitos():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Seguimiento a la implementación de las medidas"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

import plotly.graph_objects as go

# Iris bar figure
def drawFigure():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.bar(
                        df, x="sepal_width", y="sepal_length", color="species"
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'green',
                        paper_bgcolor= 'red',
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])

#fig.show()

# Iris bar figure
def draw1():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(go.Sunburst(
                        labels=df2['TIPO_GOBIERNO_NOMBRE'].unique().tolist(),
                        #parents=df2['Avance%'].unique().tolist(),
                        parents= [""]*len(df2['TIPO_GOBIERNO_NOMBRE'].unique().tolist()) ,
                        values=df2['Avance%'].unique().tolist(),
                    )).update_layout(margin = dict(t=0, l=0, r=0, b=0))
                ) 
            ])
        ),  
    ])
"""
def draw2_():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(go.Sunburst(
                        labels=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
                        parents=[""  , "Eve" , "Eve" , "Seth", "Seth", "Eve" , "Eve" , "Awan" , "Eve" ],
                        values= [10  ,   14  ,   12  ,  10   ,     2 ,   6   ,  6    ,    4   ,   4],
                    )).update_layout(margin = dict(t=0, l=0, r=0, b=0))
                ) 
            ])
        ),  
    ])
"""


#df = px.data.tips()
#fig = px.sunburst(px.data.tips(), path=['day', 'time', 'sex'], values='total_bill')
#fig.show()


print(px.data.tips())
def draw4():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.sunburst(px.data.tips(), path=['day', 'time'], values='total_bill').update_layout(margin = dict(t=0, l=0, r=0, b=0))
                ) 
            ])
        ),  
    ])

def draw5():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.sunburst(df3, path=['b1', 'b2'], values='Avance%').update_layout(margin = dict(t=0, l=0, r=0, b=0))
                ) 
            ])
        ),  
    ])

def draw2():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(go.Sunburst(
                        labels=labels0,
                        parents=parents0,
                        values= values0,
                    )).update_layout(margin = dict(t=0, l=0, r=0, b=0))
                ) 
            ])
        ),  
    ])
def draw3():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(go.Indicator(
    mode = "number+delta",
    value = R,
    number= { "suffix": "%" },

    title = {"text": "PP 0083<br><span style='font-size:0.8em;color:gray'>Avance</span><br><span style='font-size:0.8em;color:gray'>_</span>"},
    #delta = {'reference': 400, 'relative': True},
    domain = {'x': [0.6, 1], 'y': [0, 1]}))

                ) 
            ])
        ),  
    ])

def getcriterio():
    global gcols
    columnsCriterio1=[
                        #{'name': 'Intervención',    'id': 'Intervención',   'type': 'text',     'editable': True},
                        #{'name': 'N_',              'id': 'N_',             'type': 'text',     'editable': False},
                        #{'name': 'Indice',          'id': 'Indice',         'type': 'text',     'editable': False},
                        #{'name': 'Des_1',           'id': 'Des_1',          'type': 'text',     'editable': False},
                        
                        {'name': 'Actividad',       'id': 'Actividad',      'type': 'text',     'editable': False},
                        {'name': 'Tarea',       'id': 'Tarea',      'type': 'text',     'editable': False},

                        {'name': 'Subtarea',       'id': 'Subtarea',      'type': 'text',     'editable': False},

                        {'name': 'Inicio',          'id': 'Inicio',         'type': 'datetime', 'editable': False},
                        {'name': 'Fin',             'id': 'Fin',            'type': 'datetime', 'editable': False},
                        {'name': 'Alerta',          'id': 'Alerta',         'type': 'text',  'editable': False},
                        #{'name': 'Flag',          'id': 'Flag',         'type': 'text',  'editable': False},
                        #{'name': 'diff',          'id': 'diff',         'type': 'numeric',  'editable': False},


                        #{'name': 'PAvance',         'id': 'PAvance',        'type': 'numeric',     'editable': True,        'format':  {"specifier": "$,.2f", "locale": {"symbol": ["", "%"]}}},
                        {'name': '% de Avance',         'id': 'Pt_str0',        'type': 'numeric', 'editable': True, 'min':0,'max':100},
                        {'name': 'Responsable',     'id': 'Responsable',    'type': 'text',     'editable': False},
                        #{'name': 'C',     'id': 'C',    'type': 'text',     'editable': False},
                    ]
    columnsCriterio2=[
                        #{'name': 'Intervención',    'id': 'Intervención',   'type': 'text',     'editable': True},
                        #{'name': 'N_',              'id': 'N_',             'type': 'text',     'editable': False},
                        #{'name': 'Indice',          'id': 'Indice',         'type': 'text',     'editable': False},
                        #{'name': 'Des_1',           'id': 'Des_1',          'type': 'text',     'editable': False},
                        
                        #
                        {'name': 'Actividad',       'id': 'Actividad',      'type': 'text',     'editable': False},
                        {'name': 'Subtarea',       'id': 'Subtarea',      'type': 'text',     'editable': False},

                        {'name': 'Inicio',          'id': 'Inicio',         'type': 'datetime', 'editable': False},
                        {'name': 'Fin',             'id': 'Fin',            'type': 'datetime', 'editable': False},
                        #{'name': 'Alerta',          'id': 'Alerta',         'type': 'numeric',  'editable': False},
                        #{'name': 'PAvance',         'id': 'PAvance',        'type': 'numeric',     'editable': True},
                        #{'name': 'Responsable',     'id': 'Responsable',    'type': 'text',     'editable': True},
                        #{'name': 'C',     'id': 'C',    'type': 'text',     'editable': False},
                    ]
    if gcols==0:
        print("gcols 0")
        return columnsCriterio1
    if gcols==1:
        print("gcols 1")

        return columnsCriterio2
def drawF():
    return dbc.Card(
            children=[
                
        
                dbc.CardHeader("Tabla de Hitos"),
                dbc.CardBody(
                    [
                        
                    dash_table.DataTable(
                    id="datable_full",
                  
                    #DFMetadata = DFMetadata[DFMetadata.C.isin(state)]
                    data=DFMetadata.to_dict('records'),

                    #data= DFMetadata[DFMetadata.C.isin(state)].to_dict('records'),
                    
                    sort_action='native',
                    #style_cell={"whiteSpace": "pre-line"},
                    style_cell={"whiteSpace": "normal","height": "auto",'textAlign': 'left' },
                    #virtualization=True,
                    #style_table={'width':'{}'.format(str(int(viewport['width'])-20))+'px'},
                    style_header={
                                'backgroundColor': 'gray',
                                'fontWeight': 'bold',
                                'text-align': 'center'

                            },
                    #config={
                    #    'displayModeBar': False
                    #},
                    #¢Proyecto	Actividad	Tarea	FechaInicio	FechaFinProy	FechaFinReal	PAvance	Responsable	DiasReminder	Secuencia	Obs
                    
                    #columns=[
                    #    {'name': 'Proyecto',       'id': 'Proyecto',    'type': 'text', 'editable': False},
                    #    {'name': 'Actividad',      'id': 'Actividad',   'type': 'text'},
                    #    {'name': 'Tarea',          'id': 'Tarea',       'type': 'text'},
                    #    {'name': 'FechaInicio',    'id': 'FechaInicio', 'type': 'datetime'},
                    #    {'name': 'FechaFinProy',   'id': 'FechaFinProy','type': 'datetime'},
                    #    {'name': 'FechaFinReal',   'id': 'FechaFinReal','type': 'datetime'},
                    #    {'name': 'PAvance',        'id': 'PAvance',     'type': 'numeric'},
                    #    #{'name': 'Responsable',    'id': 'Responsable', 'type': 'text'},
                    #    #{'name': 'DiasReminder',   'id': 'DiasReminder','type': 'text'},
                    #    #{'name': 'Secuencia',      'id': 'Secuencia',   'type': 'text'},
                    #    {'name': 'Obs',            'id': 'Obs',         'type': 'text'},
                    #],
                    
                    columns=getcriterio(),


                    merge_duplicate_headers=True,

                    #merge_duplicate_rows=True,
                    filter_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    row_selectable="multi",
                    row_deletable=True,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 10,
                    editable=True,
                    style_as_list_view=True,
                    style_data_conditional=[
                        
#0090d1 celeste
#3d6aa7 azul
#018444 verde
#005b91 azul 2
                        {
                            'if': {
                                #'filter_query': '{c} contains "New"'
                                'filter_query': '{C} = "1"'
                            },
                            #'backgroundColor': '#0074D9',
                            'backgroundColor': '#3d6aa7',

                            'color': 'white',
                            'textDecoration': 'underline',
                            'textDecorationStyle': 'dotted',
                        },
                        {
                            'if': {
                                #'filter_query': '{c} contains "New"'
                                'filter_query': '{C} = "2"'
                            },
                            #'backgroundColor': '#19a645', verde
                            #'backgroundColor': '#018444',
                            'backgroundColor': '#9ad69e',
                            

                            'color': 'white',
                            'textDecoration': 'underline',
                            'textDecorationStyle': 'dotted',
                        },
                        #{
                        #    'if': {
                        #        'filter_query': '{PAvance} >= 0 && {PAvance} <= 39 && {Flag} = 1' ,
                        #        'column_id': 'PAvance'
                        #    },
                        #    'backgroundColor': 'darkred',
                        #     'color': 'dark'
                        #},
                        {
                            'if': {
                                'filter_query': '{PAvance} >= 90 && {PAvance} <= 100',
                                #'column_id': 'PAvance'
                                 'column_id': 'PAvance'
                                
                            },
                            #'backgroundColor': 'green',
                            #'backgroundColor': '#57f261',
                            'backgroundColor': '#red',


                            'color': 'dark'
                        },
 {
                            'if': {
                                'filter_query': '{Pt_str0} >= 90 && {Pt_str0} <= 100',
                                 'column_id': 'Pt_str0'
                                
                            },
                            'backgroundColor': 'red',


                            'color': 'dark'
                        },



                        
                        {
                            'if': {
                                'filter_query': '{PAvance} >= 40 && {PAvance} <= 89',
                                'column_id': 'PAvance'
                            },
                            'backgroundColor': '#ff7e00',
                            'color': 'dark'
                        },
                        {
                            'if': {
                                'state': 'active'  # 'active' | 'selected'
                            },
                        'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                        'border': '1px solid rgb(0, 116, 217)'
                        },


  
                    
                    {'if': {'column_id': 'Actividad'},   'textAlign': 'left'},     
                    {'if': {'column_id': 'Subtarea'},    'textAlign': 'left'},     
                    {'if': {'column_id': 'Inicio'},      'textAlign': 'center'},     
                    {'if': {'column_id': 'Fin'},         'textAlign': 'center'},     
                    {'if': {'column_id': 'Flag'},        'textAlign': 'center'},    
                    {'if': {'column_id': 'Pt_str0'},     'textAlign': 'right'},    
                    {'if': {'column_id': 'PAvance'},     'textAlign': 'right'},    
                    {'if': {'column_id': 'Responsable'}, 'textAlign': 'left'},    
                          
                         {
                            'if': {
                                'filter_query': '{PAvance} >= 90 && {PAvance} <= 100',
                                'column_id': 'Pt_str0'
                            },
                            'backgroundColor': '#57f261',
                            'color': 'dark'
                        },
                        {
                            'if': {
                                'filter_query': '{PAvance} >= 40 && {PAvance} <= 89',
                                'column_id': 'Pt_str0'
                            },
                            'backgroundColor': '#ff7e00',
                            'color': 'dark'
                        },
                       

                        {
                            'if': {
                                'filter_query': '{Hitos} = 1',
                                'column_id': ['Actividad', 'Tarea','Subtarea','Inicio', 'Fin','Alerta','Pt_str0','Responsable']
                           },
                            "fontWeight": "bold",
                        #'backgroundColor': '#ff7e00',
                            #'color': 'dark'
                        },


                        

                    ]
                )

                    ]
                )    
                ],color="#3d6aa7"
)

def drawF_mini():
    return dbc.Card(
            children=[
                dbc.CardHeader("Tabla de Hitos"),
                dbc.CardBody(
                    [
                        
                    dash_table.DataTable(
                    
                    id="datable_mini",
                    #DFMetadata = DFMetadata[DFMetadata.C.isin(state)]
                    #data=DFMetadata3[DFMetadata3['N_']==c()].to_dict('records'),
                    data=DFMetadata3.to_dict('records'),

                    #data= DFMetadata[DFMetadata.C.isin(state)].to_dict('records'),
                    
                    sort_action='native',
                    #style_cell={"whiteSpace": "pre-line"},
                    style_cell={"whiteSpace": "normal","height": "auto",'textAlign': 'right' },
                    #virtualization=True,
                    #style_table={'width':'{}'.format(str(int(viewport['width'])-20))+'px'},
                    style_header={
                                'backgroundColor': 'gray',
                                'fontWeight': 'bold',
                                'text-align': 'center'
                            },
                    #config={
                    #    'displayModeBar': False
                    #},
                    #¢Proyecto	Actividad	Tarea	FechaInicio	FechaFinProy	FechaFinReal	PAvance	Responsable	DiasReminder	Secuencia	Obs
                    
                    #columns=[
                    #    {'name': 'Proyecto',       'id': 'Proyecto',    'type': 'text', 'editable': False},
                    #    {'name': 'Actividad',      'id': 'Actividad',   'type': 'text'},
                    #    {'name': 'Tarea',          'id': 'Tarea',       'type': 'text'},
                    #    {'name': 'FechaInicio',    'id': 'FechaInicio', 'type': 'datetime'},
                    #    {'name': 'FechaFinProy',   'id': 'FechaFinProy','type': 'datetime'},
                    #    {'name': 'FechaFinReal',   'id': 'FechaFinReal','type': 'datetime'},
                    #    {'name': 'PAvance',        'id': 'PAvance',     'type': 'numeric'},
                    #    #{'name': 'Responsable',    'id': 'Responsable', 'type': 'text'},
                    #    #{'name': 'DiasReminder',   'id': 'DiasReminder','type': 'text'},
                    #    #{'name': 'Secuencia',      'id': 'Secuencia',   'type': 'text'},
                    #    {'name': 'Obs',            'id': 'Obs',         'type': 'text'},
                    #],
                    


                    columns=[
                        {'name': 'Intervención',    'id': 'Intervención',    'type': 'text',        'editable': False},
                        {'name': ' ',               'id': 'Estado',          'type': 'text',        'editable': False},
                        #{'name': '% de Avance',    'id': 'Pt_',            'type': 'numeric',      'editable': False},
                        {'name': '% de Avance',     'id': 'Pt_str',          'type': 'numeric',     'editable': False},
                        {'name': 'Alerta',          'id': 'Alerta',          'type': 'text',        'editable': False},
                        {'name': 'Comentario',      'id': 'Comentario',      'type': 'text',        'editable': True},
                        #{'name': 'PAvance',      'id': 'PAvance',      'type': 'text',        'editable': True},

                    ],


                    filter_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    row_selectable="multi",
                    row_deletable=True,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 10,
                    editable=True,
                    style_as_list_view=True,
                    
                    style_data_conditional=[

                

                    {'if': {'column_id': 'Intervención'},   'textAlign': 'left'},     
                    {'if': {'column_id': 'Estado'},         'textAlign': 'left'},     
                    {'if': {'column_id': 'Pt_str'},         'textAlign': 'right'},     
                    {'if': {'column_id': 'Alerta'},         'textAlign': 'left'},     
                    {'if': {'column_id': 'Comentario'},     'textAlign': 'left'},    

                        

                    {
                        'if': {
                            'filter_query': '{PAvance} >= 70 && {PAvance} <= 100',
                            'column_id': 'Pt_str'
                        },
                        'backgroundColor': '#57f261',
                        'color': 'dark'
                    },
                     {
                        'if': {
                            'filter_query': '{PAvance} >= 11 && {PAvance} <= 69',
                            'column_id': 'Pt_str'
                        },
                        'backgroundColor': '#f2c957',
                        'color': 'dark'
                    },
                    {
                        'if': {
                            'filter_query': '{PAvance} >= 0 && {PAvance} <= 10',
                            'column_id': 'Pt_str'
                        },
                        'backgroundColor': '#ed6145',
                        'color': 'dark'
                    },
                    ],
                )

                    ]
                )    
                ],color="#3d6aa7"
)

l=DFMetadata['N_'].unique().tolist()

def desplegable_medidas():
    
#         dcc.Dropdown(id='demo-dropdown',options=l, value=l[0] ),
#                     id="datable_mini",
#     l=DFMetadata['N_'].unique().tolist()
    return html.Div([
        dcc.Dropdown(id='demo-dropdown',
        options=l, value=l[0]
    ),
 ])


l=DFMetadata['N_'].unique().tolist()

def desplegable_medidas2():
    
#         dcc.Dropdown(id='demo-dropdown',options=l, value=l[0] ),
#                     id="datable_mini",
#     l=DFMetadata['N_'].unique().tolist()
    l=["Criterio 1","Criterio 2"]

    return html.Div([
        dcc.Dropdown(id='demo-dropdown2',
        options=l, value=l[0]
    ),
 ])

legal=DFMetadata['Dispositivo Legal'].unique().tolist()

def desplegable_legal():
    
#         dcc.Dropdown(id='demo-dropdown',options=l, value=l[0] ),
#                     id="datable_mini",
#     l=DFMetadata['N_'].unique().tolist()
    l=["Criterio 1","Criterio 2"]

    return html.Div([
        dcc.Dropdown(id='demo-dropdown_legal',
        options=legal, value=legal[0]
    ),
 ])

#################################################################
#################################################################


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "19rem",
    "padding": "2rem 1rem",
    "background-color": "#9cabb9",
}
#f8f9fa gris
#455260
#9cabb9
# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


import base64
image_filename = 'MEF_img_1.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())



test_png = 'conpunche3.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

#        #style={'height':'40%', 'width':'80%'}),

sidebar = html.Div(
    [
        html.H2("", className="display-4"),
        html.Img(src='data:image/png;base64,{}'.format(test_base64)), 
        html.Hr(),
        html.P(
            " ", className="lead"
        ),
        dbc.Nav(
            [
                #dbc.NavLink("Home", href="/", active="exact"),
               dbc.NavLink("Avance de las medidas", href="/page-1", active="exact"),
               dbc.NavLink("Seguimiento",           href="/page-2", active="exact"),
               dbc.NavLink("Actividades Diarias",   href="/page-3", active="exact"),
               dbc.NavLink("Consulta Amigable",     href="/page-4", active="exact"),

            ],
            vertical=True,
            pills=True
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        #return html.P("This is the content of the home page!")


        app.layout = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            pag0()
                        ], width=12),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        #dbc.Col([
                        #    desplegable_medidas() 
                        #], width=3),
                        #dbc.Col([
                        #    #drawFigure()
                        #    c()
                        #], width=2),
                    #    dbc.Col([
                    #        drawF() 
                    #    ], width=12),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        #dbc.Col([
                        #    c()
                        #], width=2),
                        #dbc.Col([
                        #    drawF_mini()
                        #], width=12),
                    ], align='center'),      
                ]), color = 'Dark'
            )
        ])
        #return html.P("This is the content of page 1. Yay!")
        return app.layout

    elif pathname == "/page-1":
        

        app.layout = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            Hitos()
                        ], width=12),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        #dbc.Col([
                        #    desplegable_medidas() 
                        #], width=3),
                        #dbc.Col([
                        #    #drawFigure()
                        #    c()
                        #], width=2),
                    #    dbc.Col([
                    #        drawF() 
                    #    ], width=12),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        #dbc.Col([
                        #    c()
                        #], width=2),
                        dbc.Col([
                            drawF_mini()
                        ], width=12),
                    ], align='center'),      
                ]), color = 'Dark'
            )
        ])
        #return html.P("This is the content of page 1. Yay!")
        return app.layout
    elif pathname == "/page-2":

        app.layout = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            detalleHitos()
                        ], width=12),
                        dbc.Col([
                            desplegable_medidas() 
                        ], width=3),
                        dbc.Col([
                            desplegable_legal() 
                        ], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        #dbc.Col([
                        #    drawFigure() 
                        #], width=3),
                        #dbc.Col([
                        #    #drawFigure()
                        #    c()
                        #], width=2),
                        dbc.Col([
                            drawF() 
                        ], width=12),
                    ], align='center'), 
                    #html.Br(),
                    #dbc.Row([
                        #dbc.Col([
                        #    c()
                        #], width=2),
                    #    dbc.Col([
                    #        drawF_mini()
                    #    ], width=12),
                    #]#, align='center'),      
                ]), color = 'Dark'
            )
        ])
        #return html.P("This is the content of page 1. Yay!")
        return app.layout
    elif pathname == "/page-3":

        app.layout = dbc.Container([
            dcc.Store(id="store"),
            html.H1("Alerta de actividades diarias"),
            html.Hr(),
            #dbc.Button(
            #    "Regenerate graphs",
            #    color="primary",
            #    id="button",
            #    className="mb-3",
            #),
 #'if': {'column_id': c},
 #                   'textAlign': 'left'
 #               } for c in ['Date', 'Region']
            
        #for c in ['Date', 'Region']

            dbc.Tabs(
                [
                 

                    dbc.Tab(label="Kathy", tab_id="tab_Kathy"),
                    dbc.Tab(label="Angel", tab_id="tab_Angel"),
                    dbc.Tab(label="Antonio", tab_id="tab_Antonio"),

                ],
                id="tabs",
                active_tab="tab_Antonio",
            ),
            html.Div(id="tab-content", className="p-4"),
        ])

        #return html.P("This is the content of page 1. Yay!")
        return app.layout
    elif pathname == "/page-4":

        app.layout = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            Hitos()
                        ], width=12),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                        #dbc.Col([
                        #    drawText()
                        #], width=3),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            draw4() 
                        ], width=6),
                        dbc.Col([
                            #drawFigure()
                            draw5()
                        ], width=6),
                    #    dbc.Col([
                    #        drawF() 
                    #    ], width=12),
                    ], align='center'), 
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            draw1()
                        ], width=6),
                        dbc.Col([
                            draw2()
                        ], width=6),
                    ], align='center'),  
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            draw1()
                        ], width=6),
                        dbc.Col([
                            draw3()
                        ], width=6),
                    ], align='center'),     
                ]), color = 'Dark'
            )
        ])
        #return html.P("This is the content of page 1. Yay!")
        return app.layout
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """

    #Alerta
    print('desde alertas!')
    print(    DFMetadata[DFMetadata['Alerta']!= ""])

    print(DFMetadata['Alerta'].unique())

    #if active_tab and data is not None:
    if active_tab :

        if active_tab == "tab_Kathy":



#for c in ['Date', 'Region']
            alerts = html.Div([
                #dbc.Alert("This is a primary alert", color="primary"),
                #dbc.Alert("This is a secondary alert", color="secondary"),
                #dbc.Alert("Actividad 1: concluida", color="success",
                #    dismissable=True),

                    dbc.Alert("Actividad 3: vencida y avance al 0% ",           color="danger",dismissable=True) ,
                dbc.Alert("Actividad 2: queda 1 día para su vencimiento.",  color="warning",dismissable=True),

            ]for c in ['Date', 'Region'])

            return alerts
        elif active_tab == "tab_Angel":
            alerts0 = html.Div([
                #dbc.Alert("This is a primary alert", color="primary"),
                #dbc.Alert("This is a secondary alert", color="secondary"),
                dbc.Alert("Actividad 1: concluida", color="success"),
                dbc.Alert("Actividad 2: qsueda 1 día para su vencimiento.", color="warning"),
                dbc.Alert("Actividad 3: vencida y avance al 0% ", color="danger"),

            ])
            portfolio_list=['1', '2']
            alerts1 = html.Div([
                
               [dbc.Alert("Actividad 1: concluida", color="success") for portfolio in (portfolio_list)] ])
            return alerts1
    return " --- "



#         dcc.Dropdown(id='demo-dropdown',options=l, value=l[0] ),
#                     id="datable_mini",
#     l=DFMetadata['N_'].unique().tolist()

#    data=DFMetadata3.to_dict('records'),


# ,Input("demo-dropdown2", "value")
# demo-dropdown_legal
@app.callback(
    Output("datable_full", "data"), 
    [Input("demo-dropdown", "value"),Input("demo-dropdown_legal", "value")]
)
def display_table(state,state2):
    global DFMetadata    
    print('state= ', state)  
    print('state2= ', state2)      
    gcols= 0
    
    #l=["Criterio 1","Criterio 2"]
    #if state2=="Criterio 1":
    #    gcols= 0
    #if state2=="Criterio 2":
    #    gcols= 1
        
    DFMetadata_= DFMetadata[DFMetadata['N_'].isin([state])]
    DFMetadata_= DFMetadata_[DFMetadata_['Dispositivo Legal'].isin([state2])]

    return DFMetadata_.to_dict("records")


#@app.callback(
#    Output("datable_full", "data"), 
#    Input("demo-dropdown2", "value")
#)
def display_table(state):
#    global DFMetadata    
    print('state2= ', state)      
#    #DFMetadata_= DFMetadata[DFMetadata['N_'].isin([state])]
#    #return DFMetadata_.to_dict("records")



#0090d1 celeste

#3d6aa7 azul

#018444 verde

#005b91 azul 2


if __name__ == "__main__":
    app.run_server(port=8888)



"""

añdir leyenda


nivel de gobierno 

productos y70 activifsafrd



alerta


aviso al cronograma 

alerrta a los hitos 




Cambiar de tipo de letra 

hitos en negrita 
"""
