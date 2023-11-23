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
query_test = """SELECT iddia, COUNT(*) FROM public.sm_tablets_obs GROUP BY iddia;"""
               
count_iddia          = pd.read_sql_query(query_test, conn_dg)
#---------------------------------------------*
fec_t1 = np.max(count_iddia['iddia'])
fec_t0 = np.min(count_iddia['iddia'])
# separate fec_t0 into year, month and day
fec_t0 = str(fec_t0)
fecha_corte_t0 = fec_t0[0:4] + '-' + fec_t0[4:6] + '-' + fec_t0[6:8]
#---------------------------------------------*
query_fec_t1 = """SELECT DISTINCT A."CODIGO_MODULAR",A."OBSERVACION_RECEPCION"
                  FROM public.sm_tablets_obs A
                  LEFT JOIN (SELECT * FROM public.obs_recepcion_bi WHERE "fecha_corte" = '"""+fecha_corte_t0+"""') B
                  ON A."CODIGO_MODULAR" = B."idinstitucioneducativa" AND A."OBSERVACION_RECEPCION" = B."OBSERVACION_RECEPCION"
                  WHERE B.idcatrecepcion IS NULL
                  AND A."OBSERVACION_RECEPCION" IS NOT NULL 
                  AND A."OBSERVACION_RECEPCION" != '' 
                  AND A.iddia="""+ str(fec_t1) + ";"
recepcion = pd.read_sql_query(query_fec_t1, conn_dg)
recepcion['iddia'] = fec_t1
print(recepcion.shape)

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
recepcion.to_sql('obs_recepcion', engine, if_exists='replace', index=False)
#----------------------------------------------------------------------------*
#           ASIGNACION
#----------------------------------------------------------------------------*
query_asigna_t1 = """SELECT DISTINCT A."CODIGO_MODULAR",A."SERIE_EQUIPO",A."OBSERVACIONES_A"
                  FROM public.sm_tablets_obs A
                  LEFT JOIN (SELECT * FROM public."obs_asigna_bi" 
                           WHERE "fecha_corte"= (SELECT MIN (DISTINCT fecha_corte) FROM public."obs_recepcion_bi")
                           ) B
                  ON A."CODIGO_MODULAR" = B."idinstitucioneducativa"AND
                     A."SERIE_EQUIPO" = B."serie_equipo" AND
                     A."OBSERVACIONES_A" = B."observaciones"
                  WHERE B.idcatasignacion IS NULL
                  AND LENGTH(A."OBSERVACIONES_A")>0 
                  AND A.iddia=(SELECT MAX (DISTINCT iddia) FROM public.sm_tablets_obs);"""
#----------------------------------------------------------------------------*
asigna_data = pd.read_sql_query(query_asigna_t1, conn_dg)
print(asigna_data.shape)
asigna_data['iddia'] = fec_t1    
#----------------------------------------------------------------------------*
asigna_data.to_sql('obs_asigna', engine, if_exists='replace', index=False)
#------------------------*
