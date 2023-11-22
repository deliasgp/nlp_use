# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:15:42 2022

@author: daniel
"""
#-------------------
#E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/
import sys
sys.path.append('E:/Mi unidad/dgavidia_minedu/BD USE/NLP/PILOTO/script/dgavidia')
#sys.path.append('H:/Mi unidad/dgavidia_minedu/BD USE/NLP/PILOTO/script/dgavidia')
#---------------
maindir_minedu     = 'C:/Users/dgavidia/OneDrive - Ministerio de Educación'
maindir_hogar      = 'C:/Users/dgavidia/OneDrive - Ministerio de Educación'

maindir            =  maindir_minedu
subdir      =  '/BD USE/NLP/TABLETAS/Input/'
output      =  '/BD USE/NLP/TABLETAS/Output'
#-------------------
#import datatable as dt
import pandas as pd
import numpy as np
import re
import psycopg2
#---------------------------------------------*
from os import walk
filenames = next(walk(maindir+subdir), (None, None, []))[2]  # [] if no file
r = re.compile(".*.csv")
filenames = list(filter(r.match, filenames)) # Read Note below
#---------------------------------------------*
db_params = {
    "dbname": "DGAVIDIA",
    "host": "localhost",
    "user": "postgres",
    "port": 5432,
    "password": "a07119420"
}

conn_dg = psycopg2.connect(**db_params)
query_test = """SELECT iddia, COUNT(*) 
               FROM public.sm_tablets_obs 
               GROUP BY iddia;"""
               
count_iddia          = pd.read_sql_query(query_test, conn_dg)
count_iddia
#---------------------------------------------*
fec_t1 = np.max(count_iddia['iddia'])
fec_t0 = np.min(count_iddia['iddia'])
#---------------------------------------------*
query_fec_t1 = """SELECT * FROM public.sm_tablets_obs WHERE iddia="""+ str(fec_t1)
#recepcion_dir  = maindir + subdir + 'recepcion_' + str(fec_t1) +'.csv'
recepcion = pd.read_sql_query(query_fec_t1, conn_dg)
#----------------------------------------------------------------------------*
#
#
#           RECEPCION
#
#
#----------------------------------------------------------------------------*
#obs_recepcion_dir1  = "C:\\Users\\dgavidia\\OneDrive - Ministerio de Educación\\BD_USE\\NLP\\TABLETAS\\Output\\BI\\20231018\\obs_recepcion_20231018.csv"
obs_recepcion_dir1 = 'D:/obs_recepcion_20231018.csv'
obs_recepcion_t0 = pd.read_csv(obs_recepcion_dir1, encoding='Latin-1')
obs_recepcion_t0.drop('Unnamed: 0', axis=1, inplace=True)
#obs_recepcion_dir  = maindir + output + '/clasificacion visual/'+fec_t0 + '/obs_recepcion_'+ fec_t0+'.csv'
#obs_recepcion_t0 = dt.fread(obs_recepcion_dir).to_pandas()
#str(obs_recepcion_t0['idinstitucioneducativa']).zfill(7)
obs_recepcion_t0['idinstitucioneducativa']=obs_recepcion_t0['idinstitucioneducativa'].astype(str)
obs_recepcion_t0['idinstitucioneducativa']=obs_recepcion_t0.idinstitucioneducativa.str.pad(7,side='left',fillchar='0')
obs_recepcion_t0['iddia'] = fec_t0

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
obs_recepcion_t0.to_sql('obs_recepcion', engine, if_exists='replace', index=False)
#----------------------------------------------------------------------------*
recep_data = recepcion[['CODIGO_MODULAR','OBSERVACION_RECEPCION']][recepcion.OBSERVACION_RECEPCION.str.len()>0].drop_duplicates()
#----------------------------------------------------------------------------*
for col in obs_recepcion_t0.columns:
    print(col)    

obs_recepcion_t1 = recep_data.merge(obs_recepcion_t0, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','OBSERVACION_RECEPCION']),
                                right_on=(['idinstitucioneducativa','OBSERVACION_RECEPCION']),
                                indicator = True)
  
print(obs_recepcion_t1['_merge'].value_counts())
print(obs_recepcion_t1[obs_recepcion_t1['idcatrecepcion'].isnull()].shape)

obs_recepcion_t1.drop('_merge', axis=1, inplace=True)
obs_recepcion_t1.drop('CODIGO_MODULAR', axis=1, inplace=True)
obs_recepcion_t1 = obs_recepcion_t1[obs_recepcion_t0.columns]     
obs_recepcion_t1['iddia'] = fec_t1
#----------------------------------------------------------------------------*
from sqlalchemy import create_engine
#engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
obs_recepcion_t1.to_sql('obs_recepcion', engine, if_exists='append', index=False)
#----------------------------------------------------------------------------*

query_test = """SELECT iddia, COUNT(*) 
               FROM public.obs_recepcion 
               GROUP BY iddia;"""
               
pd.read_sql_query(query_test, conn_dg)
#----------------------------------------------------------------------------*
#
#
#
#
#           ASIGNACION
#
#
#
#
#----------------------------------------------------------------------------*
obs_asigna_dir = 'D:/obs_asigna_20231018.csv'
obs_asigna_t0 = pd.read_csv(obs_asigna_dir, encoding='Latin-1')
obs_asigna_t0.drop('Unnamed: 0', axis=1, inplace=True)
obs_asigna_t0['idinstitucioneducativa']=obs_asigna_t0['idinstitucioneducativa'].astype(str)
obs_asigna_t0['idinstitucioneducativa']=obs_asigna_t0.idinstitucioneducativa.str.pad(7,side='left',fillchar='0')
obs_asigna_t0['iddia'] = fec_t0
#------------------------*
for col in obs_asigna_t0.columns:
    print(col)  
    
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
obs_asigna_t0.to_sql('obs_asigna', engine, if_exists='replace', index=False)
#------------------------*
#
#
#----------------------------------------------------------------------------*
asigna_data = recepcion[['CODIGO_MODULAR',
                         'SERIE_EQUIPO',
                         'OBSERVACIONES_A']][recepcion.OBSERVACIONES_A.str.len()>0].drop_duplicates()

asigna_data = asigna_data.rename(columns = {'CODIGO_MODULAR': 'idinstitucioneducativa',
                                              'SERIE_EQUIPO': 'serie_equipo',
                                              'OBSERVACIONES_A': 'observaciones'}, inplace = False)
#obs_asigna_t0.groupby(['idinstitucioneducativa','serie_equipo','observaciones']).ngroups
#----------------------------------------------------------------------------*
obs_asigna_t1 = asigna_data.merge(obs_asigna_t0[['idinstitucioneducativa','serie_equipo','observaciones']].drop_duplicates(), 
                                how = 'left', 
                                left_on=(['idinstitucioneducativa','serie_equipo','observaciones']), 
                                right_on=(['idinstitucioneducativa','serie_equipo','observaciones']),
                                indicator = True,validate = 'one_to_one').query("_merge == 'left_only'").drop(columns="_merge")

obs_asigna_t1['iddia'] = fec_t1    
for col in obs_asigna_t1.columns:
    print(col)
#----------------------------------------------------------------------------*
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
obs_asigna_t1.to_sql('obs_asigna', engine, if_exists='append', index=False)
#------------------------*
