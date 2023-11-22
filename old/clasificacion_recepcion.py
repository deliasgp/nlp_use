# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:26:37 2023

@author: DGAVIDIA
"""
#-----------------------------------------------------------------------------*
import pandas as pd
import os
import pickle 
#nltk.download('punkt')
import sys
sys.path.append('D:/repositorios_git/nlp_use/')
import normalizar_texto as nt
from normalizar_texto import palabras_repetidas
from joblib import load
#-----------------------------------------------------------------------------*
#CARGANDO DATOS
#-----------------------------------------------------------------------------*
def importar_modelos(git_dir, recep=True, asigna=True):
    """
    Esta función importa modelos y características de un directorio git específico.

    Parámetros:
    git_dir (str): Ruta del directorio donde se encuentran los modelos y características.
    recep (bool): Indicador para determinar si importar características y modelos de "recep".
    asigna (bool): Indicador para determinar si importar características y modelos de "asigna".

    Devuelve:
    list: Una lista de modelos y características importados.
    """
    try:
        # Cargando modelos y características comunes
        ninguna_obs_model = cargar_modelo(git_dir, 'ninguna_obs_os_svm_tf_idf.joblib')
        ninguna_obs_feature = cargar_caracteristica(git_dir, 'ninguna_obs_os_tf_idf.pickle')
        
        res_ninguna_obs = [ninguna_obs_model, ninguna_obs_feature]
        res_recep, res_asigna = None, None

        if recep:
            recep_model = cargar_modelo(git_dir, 'recepcion_os_tf_idf_svm.joblib')
            recep_feature = cargar_caracteristica(git_dir, 'feat_engine_recepcion_os_tf_idf.pickle')
            res_recep = [recep_model, recep_feature]
        
        if asigna:
            asigna_model = cargar_modelo(git_dir, 'aisgna_os_tf_idf_rf.joblib')
            asigna_feature = cargar_caracteristica(git_dir, 'feat_engine_asigna_os_tf_idf.pickle')
            #---*
            asigna_fam_model = cargar_modelo(git_dir, 'asigna_fam_os_tf_idf_rf.joblib')
            asigna_fam_feature = cargar_caracteristica(git_dir, 'feat_engine_asigna_fam_os_tf_idf.pickle')
            res_asigna = [asigna_fam_model,asigna_fam_feature,asigna_model, asigna_feature]

        return res_ninguna_obs, res_recep, res_asigna
    except Exception as e:
        print(f"Ocurrió un error al importar los modelos: {e}")
        return None


def cargar_modelo(directorio, nombre_archivo):
    """Carga un modelo desde un archivo."""
    ruta_archivo = os.path.join(directorio, nombre_archivo)
    with open(ruta_archivo, 'rb') as f:
        return load(f)


def cargar_caracteristica(directorio, nombre_archivo):
    """Carga una característica desde un archivo."""
    ruta_archivo = os.path.join(directorio, nombre_archivo)
    with open(ruta_archivo, 'rb') as f:
        return pickle.load(f)
    
#IMPORTANDO MODELOS
git_repo_dir = 'D:/repositorios_git/nlp_use/'    
modelos = importar_modelos(git_repo_dir,recep = True, asigna = True)
#-----------------------------------------------------------------------------*
ninguna_obs = modelos[0] #MODELO DE FEATURE ENGINEERING Y CLASIFICACION DE NINGUNTA OBSERVACION
recep = modelos[1] #MODELO DE FEATURE ENGINEERING Y CLASIFICACION DE RECEPCION
asigna = modelos[2] #MODELO DE FEATURE ENGINEERING Y CLASIFICACION DE ASIGNACION
#-----------------------------------------------------------------------------*
import nltk
stopword_list = nltk.corpus.stopwords.words('spanish')

stop_words_nombres = pd.read_csv('E:/Mi unidad/dgavidia_minedu/BD USE/NLP/NOMBRES.csv')
stop_words_apellidos = pd.read_csv('E:/Mi unidad/dgavidia_minedu/BD USE/NLP/APELLIDOS.csv')
stop_words_tablets = nt.stop_words_use(local_file=False,maindir='',label_benef=False) + stopword_list 
stop_words_tablets = stop_words_tablets + list(stop_words_nombres['word'])
stop_words_tablets = stop_words_tablets + list(stop_words_apellidos['word'])
#stop_words_tablets = stop_words_tablets + ['efrain','socrates','condori']
#-----------------------------------------------------------------------------*
eliminar_stop_words = ['no','si','solo','se','custodia','sin'] + ['tiene','estado']
eliminar_stop_words = eliminar_stop_words + ['estudiante','padre','madre','padres','madres'] + ['él','segun','san']
eliminar_stop_words = eliminar_stop_words + ["primero","segundo","tercero","cuarto","quinto","sexto","por","to","ro","do","er","grado","grados",
"estudiantes",'docentes',"alumnos","del","primaria","secundaria","multigrado","institución","institucion","ie","iiee",
"alumnas","alumno","alumna"]
for word in eliminar_stop_words:
    if word in stop_words_tablets:
        stop_words_tablets.remove(word)        
#-----------------------------------------------------------------------------*
#
#
#                               OBTENIENDO NUEVOS DATOS
#
#
#-----------------------------------------------------------------------------*
maindir_minedu     = 'E:/Mi unidad/dgavidia_minedu'
maindir_hogar      = 'E:/Mi unidad/dgavidia_minedu'
maindir            =  maindir_minedu
subdir      =  '/BD USE/NLP/TABLETAS/Input/'
output      =  '/BD USE/NLP/TABLETAS/Output/'
#---------------------------------------------*
from os import walk
import re
filenames = next(walk(maindir+subdir), (None, None, []))[2]  # [] if no file
r = re.compile(".*.xlsx")
filenames = list(filter(r.match, filenames)) # Read Note below
#---------------------------------------------*
fec_t1 = "20230913"
fec_t0 = "20230809"
#-----------------------------------------------------------------------------*
#recepcion_20230628
recepcion_dir  = maindir + subdir + 'recepcion_' + fec_t1+'.csv'

dir_recep_t0 = maindir_minedu + subdir+'train_obs_recep'+fec_t0+'.xlsx'
dir_asigna_t0 = maindir_minedu + subdir+'train_obs_asigna'+fec_t0+'.xlsx'
datos_recep_t0 = pd.read_excel(dir_recep_t0)
datos_asigna_t0 = pd.read_excel(dir_asigna_t0)

dir_recep_t1  = maindir + subdir + '/obs_recepcion_'+fec_t1+'.xlsx'
dir_asigna_t1  = maindir + subdir + '/obs_asigna_'+fec_t1+'.xlsx'
datos_recep_t1 = pd.read_excel(dir_recep_t1)
datos_asigna_t1 = pd.read_excel(dir_asigna_t1)
#-----------------------------------------------------------------------------*
def limpiar_texto(x,stopwords,autocorrecion=False):
    texto_limpio = []
    text_corpus = nt.normalizar_texto(x,
                                    contraction_expansion=True,
                                    accented_char_removal=True, 
                                    text_lower_case=True, 
                                    text_stemming=False, text_lemmatization=True, 
                                    special_char_removal=True, remove_digits=True,
                                    stopword_removal=True, special_cases = True,
                                    autocorrecion=False,
                                    stopwords = stopwords)    
    #-----------------------------------*
    for doc in text_corpus:
        word = palabras_repetidas(doc)
        texto_limpio.append(word)
    #-----------------------------------*
    return(texto_limpio)
#-----------------------------------------------------------------------------*
def clasificacion(x,modelo,feature): 
    res_feat = feature.transform(x)
    res_model = modelo.predict(res_feat)    
    return(res_model)
#------------------------------------------------------------------------------*
recep_new = datos_recep_t1.merge(datos_recep_t0[["CODIGO_MODULAR", "OBSERVACION_RECEPCION"]], 
                                       on=["CODIGO_MODULAR", "OBSERVACION_RECEPCION"], 
                                       how="outer", 
                                       indicator=True).query("_merge == 'left_only'").drop(columns="_merge")

recep_new = recep_new[['CODIGO_MODULAR', 'OBSERVACION_RECEPCION']]
#-----------------------------------------------------------------------------*
x = limpiar_texto(recep_new['OBSERVACION_RECEPCION'],stopwords = stop_words_tablets)
#-----------------------------------------------------------------------------*
ninguna_obs_cat = clasificacion(x, modelo = ninguna_obs[0], feature = ninguna_obs[1])
recep_cat = clasificacion(x,modelo = recep[0],feature = recep[1])
import numpy as np
ninguna_obs_cat = np.where(pd.Series(x).str.contains(r'(?i)(ninguno).(observacion)') | ninguna_obs_cat==1 ,1,ninguna_obs_cat)
recep_cat = np.where(ninguna_obs_cat==1,99,recep_cat)
recep_new['target'] = recep_cat
#-----------------------------------------------------------------------------*

#1 Faltan o necesitan más equipos
#2 Tabletas con defectos técnicos
#3 Tabletas sin plan de datos
#4 Dificultades con la carga de las tableta
#5 Comentarios sobre devolución de equipos a IE o UGEL
#6 Problemas con la recepción del equipo
#7 Problemas con los registros y sistemas 
def d_cat_recep(x):
    if x == 1:
        return '1. Faltan o necesitan más equipos'
    elif x == 2: 
        return '2. Tabletas con defectos técnicos'
    elif x == 3: 
        return '3. Problemas de conectividad'
    elif x == 4: 
        return '4. Dificultades con la carga de las tableta'
    elif x == 5: 
        return '5. Comentarios sobre devolución de equipos a IE o UGEL'
    elif x == 6: 
        return '6. Problemas con la recepción del equipo'
    elif x == 7: 
        return '7. Problemas con los registros y sistemas'
    elif x == 99: 
        return '99. Ninguna observación'
    
   
#recep_new['idcatrecepcion'] = recep_new['idcatrecepcion'].apply(d_cat_recep)
#-----------------------------------------------------------------------------*
#
#
#  ASIGNACION
asigna_new = datos_asigna_t1.merge(datos_asigna_t0[["CODIGO_MODULAR","SERIE_EQUIPO","OBSERVACIONES_A"]],
                                       on=["CODIGO_MODULAR","SERIE_EQUIPO" , "OBSERVACIONES_A"], 
                                       how="outer", 
                                       indicator=True).query("_merge == 'left_only'").drop(columns="_merge")

asigna_new = asigna_new[["CODIGO_MODULAR","SERIE_EQUIPO","OBSERVACIONES_A"]]
x = limpiar_texto(asigna_new['OBSERVACIONES_A'],stopwords = stop_words_tablets)
#-----------------------------------------------------------------------------*
# 1. asigna_fam_model, 2.asigna_fam_feature, 3.asigna_model, 4.asigna_feature
ninguna_obs_cat = clasificacion(x, modelo = ninguna_obs[0], feature = ninguna_obs[1])
asigna_fam_cat = clasificacion(x, modelo = asigna[0], feature = asigna[1])
asigna_cat = clasificacion(x, modelo = asigna[2], feature = asigna[3])

asigna_new['ninguna_obs_cat'] = ninguna_obs_cat
asigna_new['asigna_fam_cat'] = asigna_fam_cat
asigna_new['asigna_cat'] = asigna_cat

ninguna_obs_cat = np.where(pd.Series(x).str.contains(r'(?i)(en buen estado)') | ninguna_obs_cat==1 ,1,ninguna_obs_cat)
asigna_cat = np.where(ninguna_obs_cat==1,99,asigna_cat)
asigna_cat = np.where(asigna_fam_cat==1,1,asigna_cat)
asigna_new['ninguna_obs_cat']  = ninguna_obs_cat
asigna_new['target'] = asigna_cat
#0. Otros comentarios
#1. Asignada beneficiarios o familiares
#2. Problemas con validar y registrar al beneficiario
#3. Problemas software, técnicos, averías, robos
#4. Problemas de conectividad
#5. Problemas con cargador y otros complementos
#6. Redistribución y cambio de beneficiario
def d_cat_asigna(x):
    if x == 0:
        return '1. Otros comentarios'
    elif x == 1: 
        return '2. Asignación exitosa a beneficiarios'
    elif x == 2: 
        return '3. Problemas con validar y registrar al beneficiario'
    elif x == 3: 
        return '4. Problemas con el software, técnicos, averías y/o robos'
    elif x == 4: 
        return '5. Problemas de conectividad'
    elif x == 5: 
        return '6. Problemas con cargador y otros complementos'
    elif x == 6: 
        return '7. Redistribución y cambio de beneficiario'
    elif x == 99: 
        return '99. Ninguna observación'
#-----------------------------------------------------------------------------*    
#
#
#
#          CARGAMOS LOS DATOS DE SIAGIE MATERIALES
#
#
#
#-----------------------------------------------------------------------------*         
recepcion_dir  = maindir + subdir + 'recepcion_' + fec_t1+'.csv'
recepcion = pd.read_csv(recepcion_dir)




recepcion.groupby(['FASE','TIPO_EQUIPO'])['FASE'].count()

recep_data = recepcion[['CODIGO_MODULAR',
                        'NRO_PECOSA',
                        'OBSERVACION_RECEPCION',
                        'FECHA_CREACION_R',
                        'FECHA_MODIFICACION_R',
                        'FASE']][recepcion.OBSERVACION_RECEPCION.str.len()>0].drop_duplicates()
#idinstitucioneducativa	nro_pecosa	OBSERVACION_RECEPCION	fecha_creacion	fecha_modificacion	idcatrecepcion	flg_bin	tipo_categoria	fecha_corte	fase
#------------------------------------------------------------------------------*
datos_recep_t0['OBS_TEMP'] = datos_recep_t0['OBSERVACION_RECEPCION'].str.upper()
recep_new['OBS_TEMP'] = recep_new['OBSERVACION_RECEPCION'].str.upper()
recep_new = recep_new.rename(columns = {'target': 'target_new'},inplace = False)
recep_data['OBS_TEMP'] = recep_data['OBSERVACION_RECEPCION'].str.upper()
#-----------------------------------------------------------------------------*
obs_recepcion_t1 = recep_data.merge(datos_recep_t0[['CODIGO_MODULAR','OBS_TEMP','target']], 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','OBS_TEMP']),
                                right_on=(['CODIGO_MODULAR','OBS_TEMP']),
                                indicator = False)
#-----------------------------------------------------------------------------*
obs_recepcion_t1 = obs_recepcion_t1.merge(recep_new[['CODIGO_MODULAR','OBS_TEMP','target_new']], 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','OBS_TEMP']),
                                right_on=(['CODIGO_MODULAR','OBS_TEMP']),
                                indicator = True)
#-----------------------------------------------------------------------------*
obs_recepcion_t1['flg_cat'] = np.where(obs_recepcion_t1['target'].notnull(),
                                       obs_recepcion_t1['target'],
                                       obs_recepcion_t1['target_new'])

obs_recepcion_t1['tipo_categoria'] = np.where(obs_recepcion_t1['_merge']=='both',
                                       2,
                                       1)
#-----------------------------------------------------------------------------*
#-----------------------------------------------------------------------------*
print(obs_recepcion_t1['_merge'].value_counts())
#----------------------------------------------------------------------------*
obs_recepcion_t1 = obs_recepcion_t1[['CODIGO_MODULAR',
                                     'NRO_PECOSA',
                                     'OBSERVACION_RECEPCION','FECHA_CREACION_R','FECHA_MODIFICACION_R',
                                     'flg_cat','FASE','tipo_categoria']]
#----------------------------------------------------------------------------*
for col in obs_recepcion_t1.columns:
    print(col)
#----------------------------------------------------------------------------*    
obs_recepcion_t1 = obs_recepcion_t1.rename(columns = {'CODIGO_MODULAR': 'idinstitucioneducativa',
                                                      'NRO_PECOSA': 'nro_pecosa',
                                                      'FECHA_CREACION_R':'fecha_creacion',
                                                      'FECHA_MODIFICACION_R':'fecha_modificacion',
                                                      'flg_cat': 'idcatrecepcion',
                                                      'FASE': 'fase'}, inplace = False)    
fec_t1
fecha_corte = '2023-09-13'
obs_recepcion_t1['fecha_corte'] = fecha_corte
#temp1 = obs_recepcion_t1[obs_recepcion_t1['idcatrecepcion'].isnull()==True]
#temp2 = obs_recepcion[obs_recepcion['CODIGO_MODULAR']==267849]
#----------------------------------------------------------------------------*
obs_recep_dir = maindir + output + '/BI/'+ fec_t1 + '/obs_recepcion_'+fec_t0+'.csv'
obs_recepcion_t1.to_csv(obs_recep_dir,  encoding = 'latin-1')
#-----------------------------------------------------------------------------*
#
#
#
#                                   ASIGNACION
#
#
#
#-----------------------------------------------------------------------------*
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
datos_asigna_t0['OBS_TEMP'] = datos_asigna_t0['OBSERVACIONES_A'].str.upper()
asigna_new['OBS_TEMP'] = asigna_new['OBSERVACIONES_A'].str.upper()
asigna_new = asigna_new.rename(columns = {'target': 'target_new'},inplace = False)
asigna_data['OBS_TEMP'] = asigna_data['OBSERVACIONES_A'].str.upper()
#-----------------------------------------------------------------------------*
obs_asigna_t1 = asigna_data.merge(datos_asigna_t0[['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP','target']], 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP']),
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP']),
                                indicator = False)
#-----------------------------------------------------------------------------*
obs_asigna_t1 = obs_asigna_t1.merge(asigna_new[['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP','target_new']], 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP']),
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBS_TEMP']),
                                indicator = True)
#-----------------------------------------------------------------------------*
print(obs_asigna_t1['_merge'].value_counts())
#----------------------------------------------------------------------------*
obs_asigna_t1['flg_cat'] = np.where(obs_asigna_t1['target'].notnull(),
                                       obs_asigna_t1['target'],
                                       obs_asigna_t1['target_new'])

obs_asigna_t1['tipo_categoria'] = np.where(obs_asigna_t1['_merge']=='both',
                                       2,
                                       1)
#----------------------------------------------------------------------------*
obs_asigna_t1 = obs_asigna_t1[['CODIGO_MODULAR',                        
                        'NRO_PECOSA',
                        'CODIGO',
                        'SERIE_EQUIPO',
                        'OBSERVACIONES_A',
                        'FECHA_CREACION_AT',
                        'FECHA_MODIFICACION_AT',
                        'flg_cat',
                        'FASE','tipo_categoria']]
#----------------------------------------------------------------------------*
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
obs_asigna_dir = maindir + output + '/BI/'+ fec_t1+ '/obs_asigna_'+fec_t1+'.csv'
obs_asigna_t1.to_csv(obs_asigna_dir,  encoding = 'latin-1')    
#-----------------------------------------------------------------------------*    
    
asigna_new['d_cat_asignacion'] = asigna_new['target_new'].apply(d_cat_asigna)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    