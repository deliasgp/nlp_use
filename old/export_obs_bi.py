# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:50:15 2022

@author: DGAVIDIA
"""
#-------------------
import sys
sys.path.append('D:/dgavidia_minedu/dgavidia_minedu/BD USE/NLP/PILOTO/script/dgavidia')
#sys.path.append('H:/Mi unidad/dgavidia_minedu/BD USE/NLP/PILOTO/script/dgavidia')
#---------------
maindir_minedu     = 'E:/Mi unidad/dgavidia_minedu'
maindir_hogar      = 'E:/Mi unidad/dgavidia_minedu'
maindir            =  maindir_minedu
subdir      =  '/BD USE/NLP/TABLETAS/Input/'
output      =  '/BD USE/NLP/TABLETAS/Output/'
#-------------------
#import datatable as dt
import pandas as pd
import numpy as np
import re
#-----------------------------------------------------------------------------*
from os import walk
filenames = next(walk(maindir+subdir), (None, None, []))[2]  # [] if no file}
r = re.compile(".*.csv")
filenames = list(filter(r.match, filenames)) # Read Note below
#-----------------------------------------------------------------------------*
fec_t0 = "20230823"
fecha_corte = '2023-08-23'
#-----------------------------------------------------------------------------*
#CARGAMOS LOS DATOS DE SIAGIE MATERIALES
recepcion_dir  = maindir + subdir + 'recepcion_' + fec_t0+'.csv'
recepcion = pd.read_csv(recepcion_dir)
#-----------------------------------------------------------------------------*
#CARGAMOS EL CSV CON LAS OBSERVACIONES CLASIFICADAS.
obs_recepcion_dir  = maindir + output + '/clasificacion visual/'+fec_t0 + '/obs_recepcion_'+ fec_t0+'.xlsx'
obs_recepcion = pd.read_excel(obs_recepcion_dir)

for col in recepcion.columns:
    print(col)

recepcion.groupby(['FASE','TIPO_EQUIPO'])['FASE'].count()

recep_data = recepcion[['CODIGO_MODULAR','NRO_PECOSA','OBSERVACION_RECEPCION','FECHA_CREACION_R','FECHA_MODIFICACION_R','FASE']][recepcion.OBSERVACION_RECEPCION.str.len()>0].drop_duplicates()
#idinstitucioneducativa	nro_pecosa	OBSERVACION_RECEPCION	fecha_creacion	fecha_modificacion	idcatrecepcion	flg_bin	tipo_categoria	fecha_corte	fase
#----------------------------------------------------------------------------*
obs_recepcion['OBS_TEMP'] = obs_recepcion['OBSERVACION_RECEPCION'].str.upper()
recep_data['OBS_TEMP'] = recep_data['OBSERVACION_RECEPCION'].str.upper()
obs_recepcion = obs_recepcion.rename(columns = {'OBSERVACION_RECEPCION': 'OBSERVACION_RECEPCION_R'},inplace = False)
#----------------------------------------------------------------------------*
obs_recepcion_t1 = recep_data.merge(obs_recepcion, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','OBS_TEMP']),
                                right_on=(['CODIGO_MODULAR','OBS_TEMP']),
                                indicator = True)

print(obs_recepcion_t1['_merge'].value_counts())
print(obs_recepcion_t1[obs_recepcion_t1['flg_bin'].isnull()].shape)
#----------------------------------------------------------------------------*
for col in obs_recepcion_t1.columns:
    print(col)
#----------------------------------------------------------------------------*
obs_recepcion_t1 = obs_recepcion_t1[['CODIGO_MODULAR',
                                     'NRO_PECOSA',
                                     'OBSERVACION_RECEPCION','FECHA_CREACION_R','FECHA_MODIFICACION_R',
                                     'flg_cat','flg_bin','FASE']]
#----------------------------------------------------------------------------*
obs_recepcion_t1 = obs_recepcion_t1.rename(columns = {'CODIGO_MODULAR': 'idinstitucioneducativa',
                                                      'NRO_PECOSA': 'nro_pecosa',
                                                      'FECHA_CREACION_R':'fecha_creacion',
                                                      'FECHA_MODIFICACION_R':'fecha_modificacion',
                                                      'flg_cat': 'idcatrecepcion',
                                                      'FASE': 'fase'}, inplace = False)
obs_recepcion_t1['tipo_categoria'] = 1    
obs_recepcion_t1['fecha_corte'] = fecha_corte
#temp1 = obs_recepcion_t1[obs_recepcion_t1['idcatrecepcion'].isnull()==True]
#temp2 = obs_recepcion[obs_recepcion['CODIGO_MODULAR']==267849]
#----------------------------------------------------------------------------*
obs_recep_dir = maindir + output + '/BI/'+ fec_t0 + '/obs_recepcion_'+fec_t0+'.csv'
obs_recepcion_t1.to_csv(obs_recep_dir,  encoding = 'latin-1')
#----------------------------------------------------------------------------*
#
#
#
#
#
#----------------------------------------------------------------------------*
#obs_asigna_dir  = maindir + output + '/clasificacion visual/'+fec_t0+'/obs_asigna_'+fec_t0+'.csv'
#obs_asigna = dt.fread(obs_asigna_dir).to_pandas()
#------------------------*
obs_asigna_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_asigna_'+fec_t0+'.xlsx'
obs_asigna = pd.read_excel(obs_asigna_dir, sheet_name=0)
#------------------------*
for col in obs_asigna.columns:
    print(col)
#------------------------*
#
#
#----------------------------------------------------------------------------*
asigna_data = recepcion[['CODIGO_MODULAR',
                         'SERIE_EQUIPO',
                         'CODIGO',
                         'NRO_PECOSA',
                         'OBSERVACIONES_A',
                         'FECHA_CREACION_AT',
                         'FECHA_MODIFICACION_AT',
                         'FASE']][recepcion.OBSERVACIONES_A.str.len()>0].drop_duplicates()
#for col in recepcion.columns:
#    print(col)
#----------------------------------------------------------------------------*
obs_asigna['OBS_TEMP'] = obs_asigna['OBSERVACIONES_A'].str.upper()
asigna_data['OBS_TEMP'] = asigna_data['OBSERVACIONES_A'].str.upper()
obs_asigna = obs_asigna.rename(columns = {'OBSERVACIONES_A': 'OBSERVACIONES_A_R'},inplace = False)

obs_asigna_t1 = asigna_data.merge(obs_asigna, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP']), 
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP']),
                                indicator = True)

print(obs_asigna_t1['_merge'].value_counts())
print(obs_asigna_t1[obs_asigna_t1['flg_bin'].isnull()].shape)
#----------------------------------------------------------------------------*
#----------------------------------------------------------------------------*
obs_asigna_t1 = obs_asigna_t1[['CODIGO_MODULAR',                        
                        'NRO_PECOSA',
                        'CODIGO',
                        'SERIE_EQUIPO',
                        'OBSERVACIONES_A',
                        'FECHA_CREACION_AT',
                        'FECHA_MODIFICACION_AT',
                        'flg_cat',
                        'flg_bin',
                        'FASE']]
#----------------------------------------------------------------------------*
obs_asigna_t1['tipo_categoria'] = 1    
obs_asigna_t1['fecha_corte'] = fecha_corte
obs_asigna_t1['fecha_siagie'] = fecha_corte
#----------------------------------------------------------------------------*
obs_asigna_t1 = obs_asigna_t1.rename(columns = {'CODIGO_MODULAR': 'idinstitucioneducativa', 
                                              'NRO_PECOSA': 'nro_pecosa',
                                              'CODIGO': 'idcodigo',
                                              'SERIE_EQUIPO': 'serie_equipo',
                                              'OBSERVACIONES_A': 'observaciones',
                                              'FECHA_CREACION_AT':'fecha_creacion',
                                              'FECHA_MODIFICACION_AT':'fecha_modificacion',
                                              'flg_cat': 'idcatasignacion',
                                              'FASE': 'fase'}, inplace = False)
#----------------------------------------------------------------------------*#
#temp1 = obs_asigna_t1[obs_asigna_t1['idcatasignacion'].isnull()==True]
#temp2 = obs_recepcion[obs_recepcion['CODIGO_MODULAR']==267849]
#----------------------------------------------------------------------------*
obs_asigna_dir = maindir + output + '/BI/'+ fec_t0 + '/obs_asigna_'+fec_t0+'.csv'
obs_asigna_t1.to_csv(obs_asigna_dir,  encoding = 'latin-1')
#obs_asigna_dir = maindir + subdir + 'obs_asigna_'+fec_t0+'.xlsx'
#obs_asigna_t1.to_excel(obs_asigna_dir,  encoding = 'latin-1')
#-------------------------------------------------------------


#A = recepcion[recepcion['SERIE_EQUIPO']=='HGAKHL04']


#https://drive.google.com/drive/folders/1dOuCifcHpNP6hjeN-Omi_9hiDfu8F7hh




import math
n = 6
k = 2
math.comb(n, k)























