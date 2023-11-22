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
#---------------------------------------------*
from os import walk
filenames = next(walk(maindir+subdir), (None, None, []))[2]  # [] if no file
r = re.compile(".*.csv")
filenames = list(filter(r.match, filenames)) # Read Note below
#---------------------------------------------*
fec_t1 = "20230913"
fec_t0 = "20230823"
#---------------------------------------------*
#recepcion_20230628
recepcion_dir  = maindir + subdir + 'recepcion_' + fec_t1+'.csv'
recepcion = pd.read_csv(recepcion_dir)
#------------------------
#----------------------------------------------------------------------------*
#
#
#
#
#           RECEPCION
#
#
#
#
#----------------------------------------------------------------------------*
obs_recepcion_dir  = maindir + output + '/clasificacion visual/'+fec_t0 + '/obs_recepcion_'+ fec_t0+'.xlsx'
obs_recepcion_t0 = pd.read_excel(obs_recepcion_dir,sheet_name=0)
#obs_recepcion_dir  = maindir + output + '/clasificacion visual/'+fec_t0 + '/obs_recepcion_'+ fec_t0+'.csv'
#obs_recepcion_t0 = dt.fread(obs_recepcion_dir).to_pandas()
#----------------------------------------------------------------------------*
recep_data = recepcion[['CODIGO_MODULAR','OBSERVACION_RECEPCION']][recepcion.OBSERVACION_RECEPCION.str.len()>0].drop_duplicates()
#----------------------------------------------------------------------------*
for col in obs_recepcion_t0.columns:
    print(col)    

obs_recepcion_t1 = recep_data.merge(obs_recepcion_t0, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','OBSERVACION_RECEPCION']),
                                right_on=(['CODIGO_MODULAR','OBSERVACION_RECEPCION']),
                                indicator = True)

print(obs_recepcion_t1['_merge'].value_counts())
print(obs_recepcion_t1[obs_recepcion_t1['flg_bin'].isnull()].shape)
#----------------------------------------------------------------------------*
obs_recepcion_t1 = obs_recepcion_t1[['CODIGO_MODULAR','OBSERVACION_RECEPCION',
                                     'flg_cat','flg_bin','flg_motivo','flg_nombre']]
#----------------------------------------------------------------------------*
for col in obs_recepcion_t1.columns:
    print(col)
#----------------------------------------------------------------------------*
obs_recepcion_t1 = obs_recepcion_t1.sort_values('flg_cat')
obs_recepcion_dir_t1 = maindir + subdir + 'obs_recepcion_'+fec_t1+'.xlsx'
obs_recepcion_t1.to_excel(obs_recepcion_dir_t1,  encoding = 'latin-1')
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
obs_asigna_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_asigna_'+fec_t0+'.xlsx'
obs_asigna_t0 = pd.read_excel(obs_asigna_dir, sheet_name=0)
#obs_asigna_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_asigna_'+fec_t0+'.csv'
#obs_asigna_t0 = dt.fread(obs_asigna_dir).to_pandas()
#------------------------*
for col in obs_asigna_t0.columns:
    print(col)
#------------------------*
#
#
#----------------------------------------------------------------------------*
asigna_data = recepcion[['CODIGO_MODULAR',
                         'SERIE_EQUIPO',
                         'OBSERVACIONES_A']][recepcion.OBSERVACIONES_A.str.len()>0].drop_duplicates()
#for col in recepcion.columns:
#    print(col)
#----------------------------------------------------------------------------*
obs_asigna_t1 = asigna_data.merge(obs_asigna_t0, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A']), 
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A']),
                                indicator = True)

print(obs_asigna_t1['_merge'].value_counts())
print(obs_asigna_t1[obs_asigna_t1['flg_bin'].isnull()].shape)
#----------------------------------------------------------------------------*
obs_asigna_t1 = obs_asigna_t1[['CODIGO_MODULAR',   
                        'SERIE_EQUIPO',
                        'OBSERVACIONES_A',
                        'flg_cat',
                        'flg_bin',
                        'flg_motivo',
                        'flg_nombre']]
#----------------------------------------------------------------------------*
obs_asigna_t1 = obs_asigna_t1.sort_values('flg_cat')
#----------------------------------------------------------------------------*
obs_asigna_dir = maindir + subdir + 'obs_asigna_'+fec_t1+'.xlsx'
obs_asigna_t1.to_excel(obs_asigna_dir,  encoding = 'latin-1')
#----------------------------------------------------------------------------*
#
#
#           PERDIDA
#
#
#----------------------------------------------------------------------------*
obs_perdida_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_perdida_'+fec_t0+'.xlsx'
obs_perdida_t0 = pd.read_excel(obs_perdida_dir,sheet_name = 0)
#obs_perdida_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_perdida_'+fec_t0+'.csv'
#obs_perdida_t0 = dt.fread(obs_perdida_dir).to_pandas()
#------------------------*
for col in obs_perdida_t0.columns:
    print(col)
#------------------------*    
perdida_data = recepcion[['CODIGO_MODULAR',
                         'SERIE_EQUIPO',
                         'OBSERVACIONES_PERDIDA']][recepcion.OBSERVACIONES_PERDIDA.str.len()>0].drop_duplicates()
#------------------------*    
#----------------------------------------------------------------------------*
obs_perdida_t1 = perdida_data.merge(obs_perdida_t0, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_PERDIDA']),
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_PERDIDA']),
                                indicator = True)
#----------------------------------------------------------------------------*
print(obs_perdida_t1['_merge'].value_counts())
print(obs_perdida_t1[obs_perdida_t1['flg_bin'].isnull()].shape)
#----------------------------------------------------------------------------*
obs_perdida_t1 = obs_perdida_t1[['CODIGO_MODULAR',   
                        'SERIE_EQUIPO',
                        'OBSERVACIONES_PERDIDA',
                        'flg_cat',
                        'flg_bin',
                        'flg_motivo',
                        'flg_nombre']]
#----------------------------------------------------------------------------*
obs_perdida_t1 = obs_perdida_t1.sort_values('flg_cat')
obs_perdida_dir = maindir + subdir + 'obs_perdida_'+fec_t1+'.xlsx'
obs_perdida_t1.to_excel(obs_perdida_dir,  encoding = 'latin-1')
#----------------------------------------------------------------------------*
#----------------------------------------------------------------------------*
#
#
#           EQUIPO
#
#
#----------------------------------------------------------------------------*
obs_equipo_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_equipo_'+fec_t0+'.xlsx'
obs_equipo_t0 = pd.read_excel(obs_equipo_dir,sheet_name = 0)
#obs_equipo_dir  = maindir + output + '/clasificacion visual/'+ fec_t0 + '/obs_equipo_'+fec_t0+'.csv'
#obs_equipo_t0 = dt.fread(obs_equipo_dir).to_pandas()
#------------------------*
for col in obs_equipo_t0.columns:
    print(col)
#------------------------*    
equipo_data = recepcion[['CODIGO_MODULAR',
                         'SERIE_EQUIPO',
                         'OBSERVACION_EQUIPO']][recepcion.OBSERVACION_EQUIPO.str.len()>0].drop_duplicates()
#------------------------*    
#----------------------------------------------------------------------------*
obs_equipo_t1 = equipo_data.merge(obs_equipo_t0, 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACION_EQUIPO']),
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACION_EQUIPO']),
                                indicator = True)
#----------------------------------------------------------------------------*
print(obs_equipo_t1['_merge'].value_counts())
print(obs_equipo_t1[obs_equipo_t1['flg_bin'].isnull()].shape)
#----------------------------------------------------------------------------*
obs_equipo_t1 = obs_equipo_t1[['CODIGO_MODULAR',   
                        'SERIE_EQUIPO',
                        'OBSERVACION_EQUIPO',
                        'flg_cat',
                        'flg_bin',
                        'flg_motivo',
                        'flg_nombre']]
#----------------------------------------------------------------------------*
obs_equipo_t1 = obs_equipo_t1.sort_values('flg_cat')
obs_equipo_dir = maindir + subdir + 'obs_equipo_'+fec_t1+'.xlsx'
obs_equipo_t1.to_excel(obs_equipo_dir,  encoding = 'latin-1')
#----------------------------------------------------------------------------*
