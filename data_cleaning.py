# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
import pandas as pd
import numpy as np
#-----------------------------------------------------------------------------*
fec_t = "20230809"
obs_recep = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/observaciones/'+fec_t+'/obs_recepcion_'+fec_t+'.xlsx'
obs_asigna = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/observaciones/'+fec_t+'/obs_asigna_'+fec_t+'.xlsx'
datos_recep = pd.read_excel(obs_recep)
datos_asigna = pd.read_excel(obs_asigna)
#-----------------------------------------------------------------------------*
datos_recep.columns
#-----------------------------------------------------------------------------*
import re
#(datos['flg_bin']==1) & (datos['flg_cat']==1)
patron1 = r'(?i)(\bfalta\b|\bno\b||\bsin\b).*(valid|coinc|tiene|carg|regis|actua|recog|\bfirmar\b)'
patron2 = r'(?i)((\bdni\b).*(tramit|trámit))|((\bfalta\b).*(dni))'
patron3 =r'(?i)((\bse\b).*(entreg))|(recog)'

datos_asigna['target'] = np.where((datos_asigna['flg_cat']!=1) & 
                            (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)celu|telef', regex=True)==False) &
                            (datos_asigna['OBSERVACIONES_A'].str.contains(patron2, regex=True)==False) & 
                            (datos_asigna['OBSERVACIONES_A'].str.contains(patron1, regex=True)==False) & 
                            (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)dni|reniec|firm', regex=True)),
                           1, #Asignada beneficiarios o familiares
                           0)

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)interne|energ', regex=True)==False) &
                             (datos_asigna['OBSERVACIONES_A'].str.contains(patron3, regex=True)) & 
                             (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)((\bno\b).(\bse\b).*(entreg))|((\bno\b).*(recog|valid|coinc))|(no entrega)', regex=True)==False) & 
                             (datos_asigna['target']==0),
                           1, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['target']==0 )& (datos_asigna['flg_cat']==15) & (datos_asigna['flg_bin']==1) &
                           (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)regul|subsa', regex=True)==False),
                           1, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['target']==0 )& (datos_asigna['flg_cat']==6) & (datos_asigna['flg_bin']==1),
                           1, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
datos_asigna['target'] = np.where(
                           (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(\bdni\b).*(registr)', regex=True)),
                           1, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where(((datos_asigna['target']==1) & (datos_asigna['flg_bin']==0)) | 
                           ((datos_asigna['target']==0) & (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)dni|reniec|firm|siagie', regex=True))),
                           2, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])

datos_asigna['target'] = np.where(((datos_asigna['target']==1) & (datos_asigna['flg_bin']==0)) | 
                           ((datos_asigna['target']==0) & (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b).*(valid)', regex=True))),
                           2, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(rot|raja|negro|quiña|malogr|rob|extrav|p.rdi)|((\bno\b).*func)', regex=True)) | 
                           (datos_asigna['flg_cat']==5),
                           3, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(\bsin\b|falta|\bno\b).*((chic|chip|internet|cobert|conect|saldo|megas)|(plan|datos))', regex=True)) & ((datos_asigna['target']==0) | (datos_asigna['flg_cat']==97)),
                           4, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(ni|solicito).*((chip|internet|cobert|conect)|(plan|datos_asigna|megas))', regex=True)) & (datos_asigna['target']==0),
                           4, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(no|perder).*((chip|internet|cobert|conect|saldo)|(plan|datos))|((chip).*(bloq))', regex=True)) & (datos_asigna['target']==0),
                           4, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(chip|internet|cobert|conect|saldo|megas|señal)', regex=True))  & 
                           (datos_asigna['flg_bin']==0) &
                           (datos_asigna['target']==0),
                           4, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(no|perder|sin|falta).*(carga|carcas|funda|llave|aguja|ahuja|panel|solar|a.*tador|corrient)', regex=True))  & 
                           (datos_asigna['flg_bin']==0) &
                           (datos_asigna['target']==0),
                           5, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(adaptador|corriente|cargad|cable|usb)', regex=True))  & 
                           (datos_asigna['flg_bin']==0) &
                           (datos_asigna['target']==0),
                           5, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['flg_cat']==2) & (datos_asigna['target']==0),
                           1, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['flg_cat']==9) & (datos_asigna['target']==0),
                           6, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(("primero|segundo|tercero|cuarto|quinto|sexto").*("primero|segundo|tercero|cuarto|quinto|sexto"))', regex=True)),
                           6, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])

datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)((1ro|2do|3ro|4to|5to|6to).*(1ro|2do|3ro|4to|5to|6to))', regex=True)),
                           6, #Dificultades con la conectividad de las tabletas
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where(((datos_asigna['flg_cat']==1) & (datos_asigna['flg_bin']==0) & (datos_asigna['target']==0) ),
                           2, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where(((datos_asigna['flg_cat']==4) & (datos_asigna['flg_bin']==0) & (datos_asigna['target']==0) ),
                           3, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
#
#
#
#

datos_asigna['target']  = np.where((datos_asigna['flg_cat']==97) & 
                                   (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(sin efecto)', regex=True)==False) & 
                                   (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b|\bsin\b).*(dni|\bvalidado\b)', regex=True)) & (datos_asigna['flg_cat']==97),
                           2, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*

datos_asigna['target']  = np.where((datos_asigna['flg_cat']==97) & 
                                   (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)((\bdni\b).*(registr))|(dni valid)', regex=True)) & (datos_asigna['flg_cat']==97),
                           1, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target']  = np.where((datos_asigna['target']==2) & (datos_asigna['flg_cat']==97) & 
                                   (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(sin efecto)', regex=True)==False) & 
                                   (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b|\bsin\b).*(dni|\bvalidado\b)', regex=True)==False),
                           2, #Problemas con validar y registrar al beneficiario
                           datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['target']==2) & (datos_asigna['flg_cat']==97) & 
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b|\bsin\b).*(dni|\bvalidado\b)', regex=True)==False),
                                  1, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['target']==3) & (datos_asigna['flg_cat']==97) & 
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(apoderad|recog)', regex=True)==True),
                                  1, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])

#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['target']==3) & (datos_asigna['flg_cat']==97) & 
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)((ning|sin|no).*(probl|prob))|((se).*(entreg))|(buen estad)', regex=True)==True),
                                  0, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['target']==3) & (datos_asigna['flg_cat']==97) &
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(dni|firm|asign|recib|recep|nombre|alumno)', regex=True)==True),
                                  1, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['target']==3) & (datos_asigna['flg_cat']==97) & 
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)((\bno\b).*(funciona))|(malog|raja|perdi)', regex=True)==False),
                                  0, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
datos_asigna['target'] = np.where((datos_asigna['flg_cat']==97) & 
                                 (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)((otro).*(est|doc))|(otro nivel)|((se).*(entreg).*(otro))', regex=True)),
                                  6, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(correlativo)', regex=True)),
                                  0, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['flg_cat']==97) & 
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(cambio)', regex=True)),
                                  6, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['flg_cat']==97) &  #5
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(niña)', regex=True)==False) & 
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(entreg|tiene).*(otr).*(tablet|carga|chip)', regex=True)),
                                  5, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
datos_asigna['target'] = np.where((datos_asigna['flg_cat']==97) & (datos_asigna['target']==0) &
                                  (datos_asigna['OBSERVACIONES_A'].str.contains(r'(?i)(padre|apoderad|madre|tío|tía|tio|tia|abuel)')),
                                  1, #Asignada beneficiarios o familiares
                                  datos_asigna['target'])
#-----------------------------------------------------------------------------*
pd.crosstab(datos_asigna['flg_cat'],datos_asigna['target'])
temp_rev = datos_asigna[(datos_asigna.flg_cat==97) & (datos_asigna.flg_bin==0) & (datos_asigna.target==0)]
#0. Otros comentarios
#1. Asignada a beneficiarios o familiares
#2. Problemas con validar y registrar al beneficiario
#3. Problemas software, técnicos, averías, robos
#4. Problemas de conectividad
#5. Problemas con cargador y otros complementos
#6. Redistribución y cambio de beneficiario
#-----------------------------------------------------------------------------*
datos_asigna1 = datos_asigna[['OBSERVACIONES_A','flg_cat','flg_bin','target']]#.drop_duplicates()
datos_asigna1['flg_data'] = 1
datos_asigna1['target_ninguna_obs'] = np.where((datos_asigna1.flg_cat==97) & (datos_asigna1.flg_bin==1) & (datos_asigna1.target==0),1,0)
datos_recep1 = datos_recep[['OBSERVACION_RECEPCION','flg_cat','flg_bin']]#.drop_duplicates()
datos_recep1['flg_data'] = 0
datos_recep1['target_ninguna_obs'] = np.where((datos_recep1.flg_cat==97) & (datos_recep1.flg_bin==0),1,0)
#-----------------------------------------------------------------------------*
# Renaming columns (optional)
datos_asigna1 = datos_asigna1.rename(columns={'OBSERVACIONES_A': 'OBSERVACIONES'})
datos_recep1 = datos_recep1.rename(columns={'OBSERVACION_RECEPCION': 'OBSERVACIONES'})
#-----------------------------------------------------------------------------*
# Concatenate DataFrames vertically
res_fin = pd.concat([datos_recep1, datos_asigna1], ignore_index=True)
#-----------------------------------------------------------------------------*
res_fin.target_ninguna_obs.value_counts()
#-----------------------------------------------------------------------------*
dir_export = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/'+'obs_'+fec_t+'.xlsx'
res_fin.to_excel(dir_export)
#-----------------------------------------------------------------------------*
datos_asigna1 = datos_asigna[['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A','flg_cat','flg_bin','target']]#.drop_duplicates()
dir_export_asigna = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/'+'train_obs_asigna'+fec_t+'.xlsx'
datos_asigna1.to_excel(dir_export_asigna)
#-----------------------------------------------------------------------------*
#
#
#
#
# Assuming df is your DataFrame
conditions = [datos_recep['flg_cat'] == 1,
              datos_recep['flg_cat'] == 2,
              datos_recep['flg_cat'] == 3,
              datos_recep['flg_cat'] == 4]

choices = [1, 2, 3, 4]


datos_recep1 = datos_recep[['CODIGO_MODULAR','OBSERVACION_RECEPCION','flg_cat','flg_bin']]#.drop_duplicates()

datos_recep1['target'] = pd.Series(pd.Categorical(np.select(conditions, choices, default=0)))
#----------------------------------------------------------------------------*
datos_recep1['target'].value_counts(normalize=True)
datos_recep1['target'].value_counts()
#----------------------------------------------------------------------------*
#temp_rev= datos[(datos['OBSERVACION_RECEPCION'].str.contains("dev"))]
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)interne|datos|chip|conect|conex') & (datos_recep1['target']==0)) | (datos_recep1['flg_cat']==10),
                           3, #Dificultades con la conectividad de las tabletas
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)cargador|cargar|elect|bater|energ')),
                           4, #Dificultades con la carga de las tableta
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)dev')) | (datos_recep1['flg_cat']==16),
                           5, #Devoluciones
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['flg_cat']==9) | (datos_recep1['flg_cat']==15) | (datos_recep1['flg_cat']==11)  | (datos_recep1['flg_cat']==18),
                           6, #Problemas con la recepción del equipo
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['flg_cat']==5) | (datos_recep1['flg_cat']==6) | (datos_recep1['flg_cat']==7)  | (datos_recep1['flg_cat']==8),
                           7, #Problemas con los registros y sistemas 
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['target']!=2) & 
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(ning).*(incon|observ)', regex=True)==False) & 
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(\bno\b).*(se).*(recep)', regex=True)),
                           6,
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'] = np.where((datos_recep1['target']!=2) & 
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(ning).*(incon|observ)', regex=True)==False) & 
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(\bno\b).*(se).*(recep)', regex=True)) | 
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(capac|orientac)', regex=True)),
                           6,
                           datos_recep1['target'])

datos_recep1['target'] = np.where((datos_recep1['flg_cat']==96) & (datos_recep1['target']==0) &
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(aun|falta|perdi)', regex=True)),
                           6,
                           datos_recep1['target'])


datos_recep1['target'] = np.where((datos_recep1['flg_cat']==96)  & (datos_recep1['target']==0) &
                           (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(amenaz|no lo requi|(no.*utiliza)|abiert)', regex=True)),
                           6,
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['flg_cat'] = np.where((datos_recep1['flg_cat']==96) & (datos_recep1['flg_bin']==1) & (datos_recep1['target']==0) &
                            (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(dotaci|fecha|lleg|pecosa|instituci|recep|asigna)', regex=True)) & 
                            (datos_recep1['OBSERVACION_RECEPCION'].str.contains(r'(?i)(aun|falta|perdi)', regex=True)==False),
                           97,
                           datos_recep1['flg_cat'])

datos_recep1['target'] = np.where((datos_recep1['flg_cat']!=97)  & (datos_recep1['target']==0),
                           6,
                           datos_recep1['target'])
#----------------------------------------------------------------------------*
datos_recep1['target'].value_counts(normalize=True).sort_index()
datos_recep1['target'].value_counts().sort_index()
#------------------------------------------------------------------------------*
pd.crosstab(datos_recep1['flg_cat'],datos_recep1['target'])
#------------------------------------------------------------------------------*
dir_export_recep = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/'+'train_obs_recep'+fec_t+'.xlsx'
datos_recep1.to_excel(dir_export_recep)
#-----------------------------------------------------------------------------*






