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
import numpy as np
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
    list: Una lista de modelos y características importados.0
    """
    try:
        # Cargando modelos y características comunes
        ninguna_obs_model = cargar_modelo(git_dir, 'ninguna_obs_os_rf_tf_idf.joblib')
        ninguna_obs_feature = cargar_caracteristica(git_dir, 'feat_engine_ninguna_obs_os_tf_idf.pickle')
        
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
import psycopg2
db_params = {
    "dbname": "DGAVIDIA",
    "host": "localhost",
    "user": "postgres",
    "port": 5432,
    "password": "a07119420"
}
conn_dg = psycopg2.connect(**db_params)
#-----------------------------------------------------------------------------*
import nltk
stopword_list = nltk.corpus.stopwords.words('spanish')
#dir_nlp = 'C:/Users/dgavidia/OneDrive - Ministerio de Educación/BD_USE/NLP/'
#-----------------------------------------------------------------------------*
#stop_words_nombres = pd.read_csv(dir_nlp+'NOMBRES.csv')
#stop_words_apellidos = pd.read_csv(dir_nlp+'APELLIDOS.csv')

stop_words_nombres = pd.read_sql_query('SELECT * FROM public.stop_words_nombres', conn_dg)
stop_words_apellidos = pd.read_sql_query('SELECT * FROM public.stop_words_apellidos', conn_dg)
#-----------------------------------------------------------------------------*
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
conn_dg = psycopg2.connect(**db_params)
query_test = """SELECT iddia, COUNT(*) 
               FROM public.sm_tablets_obs 
               GROUP BY iddia;"""
               
count_iddia   = pd.read_sql_query(query_test, conn_dg)

#---------------------------------------------*
fec_t1 = np.max(count_iddia['iddia'])
fec_t0 = np.min(count_iddia['iddia'])
fec_t1 = str(fec_t1)
fec_t0 = str(fec_t0)
#-----------------------------------------------------------------------------*
dir_recep_t1  = """SELECT * FROM public.obs_recepcion"""
datos_recep_t1 = pd.read_sql_query(dir_recep_t1,conn_dg)
datos_recep_t1.columns
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
recep_new = datos_recep_t1
recep_new = recep_new[['CODIGO_MODULAR', 'OBSERVACION_RECEPCION']]
#-----------------------------------------------------------------------------*
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
#-----------------------------------------------------------------------------*
#1 Faltan o necesitan más equipos
#2 Tabletas con defectos técnicos
#3 Tabletas sin plan de datos
#4 Dificultades con la carga de las tableta
#5 Comentarios sobre devolución de equipos a IE o UGEL
#6 Problemas con la recepción del equipo
#7 Problemas con los registros y sistemas  
#-----------------------------------------------------------------------------*
# execute only if recep_new has data
if recep_new.shape[0] > 0:
    x = limpiar_texto(recep_new['OBSERVACION_RECEPCION'],stopwords = stop_words_tablets)
    #-----------------------------------------------------------------------------*
    ninguna_obs_cat = clasificacion(x, modelo = ninguna_obs[0], feature = ninguna_obs[1])
    recep_cat = clasificacion(x,modelo = recep[0],feature = recep[1])
    import numpy as np
    ninguna_obs_cat = np.where(pd.Series(x).str.contains(r'(?i)(ninguno).(observacion)') | ninguna_obs_cat==1 ,1,ninguna_obs_cat)
    recep_cat = np.where(ninguna_obs_cat==1,99,recep_cat)
    recep_new['target'] = recep_cat
    recep_new['d_catrecepcion'] = recep_new['target'].apply(d_cat_recep)

#print all characters of a dataframe
#-----------------------------------------------------------------------------*
#
#
#
#  ASIGNACION
#
#
#
#-----------------------------------------------------------------------------*
dir_asigna_t1  = """SELECT * FROM public.obs_asigna"""
datos_asigna_t1 = pd.read_sql_query(dir_asigna_t1,conn_dg)
asigna_new = datos_asigna_t1
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
asigna_new['target'] = np.where((asigna_new['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b).(entreg)')) & 
                                (asigna_new['target']==99),
                                0,
                                asigna_new['target']) 
#-----------------------------------------------------------------------------*
asigna_new['target'] = np.where((asigna_new['OBSERVACIONES_A'].str.contains(r'(?i)(hurto)')) & 
                                (asigna_new['target']!=3),
                                3,
                                asigna_new['target']) 
#-----------------------------------------------------------------------------*
asigna_new['d_asigna'] = asigna_new['target'].apply(d_cat_asigna)
print(asigna_new.shape) 

#temp_asigna = asigna_new[['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A','target','d_asigna']]
#temp_asigna.to_excel('D:/repositorios_git/nlp_use/temp_asigna.xlsx',index=False)
#-----------------------------------------------------------------------------*    
#
#
#
#          CARGAMOS LOS DATOS DE SIAGIE MATERIALES
#
#
#
#-----------------------------------------------------------------------------* 
# 
conn_dg = psycopg2.connect(**db_params)
query_test = """SELECT iddia, COUNT(*) FROM public.sm_tablets_obs GROUP BY iddia;"""
               
count_iddia          = pd.read_sql_query(query_test, conn_dg)
#---------------------------------------------*
fec_t1 = np.max(count_iddia['iddia'])
fec_t0 = np.min(count_iddia['iddia'])
# separate fec_t0 into year, month and day
fec_t0 = str(fec_t0)
fecha_corte_t0 = fec_t0[0:4] + '-' + fec_t0[4:6] + '-' + fec_t0[6:8]

query_fec_t1 = """SELECT DISTINCT A."CODIGO_MODULAR",
                         A."NRO_PECOSA",
                         A."OBSERVACION_RECEPCION",
                         A."FECHA_CREACION_R",
                         A."FECHA_MODIFICACION_R",
                         A."FASE",
                         B."tipo_categoria",
                         B."idcatrecepcion"
                FROM public."sm_tablets_obs" A
                LEFT JOIN (SELECT * FROM public."obs_recepcion_bi" 
                           WHERE "fecha_corte"= '"""+fecha_corte_t0+"""'
                           ) B
                ON (A."CODIGO_MODULAR" = B."idinstitucioneducativa" AND
                    A."OBSERVACION_RECEPCION" = B."OBSERVACION_RECEPCION")
                WHERE LENGTH(A."OBSERVACION_RECEPCION")>0 
                AND A."iddia"=(SELECT MAX (DISTINCT iddia) FROM public.sm_tablets_obs)
                ;"""

recep_data = pd.read_sql_query(query_fec_t1, conn_dg)
recep_data['idcatrecepcion'].value_counts()
recep_data['tipo_categoria'].isnull().sum()
recep_data['idcatrecepcion'].isnull().sum()
recep_data[recep_data['idcatrecepcion'].isnull()].shape
#------------------------------------------------------------------------------*
recep_new.columns
#-----------------------------------------------------------------------------*
obs_recepcion_temp = recep_data.merge(recep_new[['CODIGO_MODULAR','OBSERVACION_RECEPCION','target']], 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','OBSERVACION_RECEPCION']),
                                right_on=(['CODIGO_MODULAR','OBSERVACION_RECEPCION']),
                                indicator = True)
#-----------------------------------------------------------------------------*
#ACTUALIZANDO INFORMACION DE OBSERVACIONES DE RECEPCION
obs_recepcion_temp['tipo_categoria'] = np.where(obs_recepcion_temp['tipo_categoria'].notnull(),
                                                obs_recepcion_temp['tipo_categoria'],
                                                2)

obs_recepcion_temp['idcatrecepcion'] = np.where(obs_recepcion_temp['idcatrecepcion'].notnull(),
                                                obs_recepcion_temp['idcatrecepcion'],
                                                obs_recepcion_temp['target'])

temo_obs_recepcion_temp = obs_recepcion_temp[['CODIGO_MODULAR','OBSERVACION_RECEPCION','target','tipo_categoria','idcatrecepcion','target']]
temo_obs_recepcion_temp.to_excel('D:/temo_obs_recepcion_temp.xlsx',index=False)
#-----------------------------------------------------------------------------*
#collumns with nulls in obs_recepcion_temp
for col in obs_recepcion_temp.columns:
    print(col)
#----------------------------------------------------------------------------*    
obs_recepcion_temp = obs_recepcion_temp.rename(columns = {'CODIGO_MODULAR': 'idinstitucioneducativa',
                                                      'NRO_PECOSA': 'nro_pecosa',
                                                      'FECHA_CREACION_R':'fecha_creacion',
                                                      'FASE':'fase',
                                                      'FECHA_MODIFICACION_R':'fecha_modificacion'}, inplace = False)    
fec_t1
# SEPARATE FEC_T1 EN AÑO, MES Y DIA
fec_t1 = str(fec_t1)
fecha_corte = fec_t1[0:4]+'-'+fec_t1[4:6]+'-'+fec_t1[6:8]

obs_recepcion_temp['fecha_corte'] = fecha_corte


obs_recepcion_temp = obs_recepcion_temp[['idinstitucioneducativa',
                                    'nro_pecosa',
                                    'OBSERVACION_RECEPCION',
                                    'fecha_creacion',
                                    'fecha_modificacion',
                                    'idcatrecepcion',
                                    'tipo_categoria',
                                    'fase',
                                    'fecha_corte']]

for col in obs_recepcion_temp.columns:
   print(col)
   print(obs_recepcion_temp[col].isnull().sum())
#----------------------------------------------------------------------------*
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
obs_recepcion_temp.to_sql('obs_recepcion_bi', engine, if_exists='append', index=False)
#-----------------------------------------------------------------------------*
#
#
#
#                                   ASIGNACION
#
#
#
#-----------------------------------------------------------------------------*
query_fec_t1 = """SELECT DISTINCT A."CODIGO_MODULAR",
                         A."SERIE_EQUIPO",
                         A."CODIGO",
                         A."NRO_PECOSA",
                         A."OBSERVACIONES_A",
                         A."FECHA_CREACION_AT",
                         A."FECHA_MODIFICACION_AT",
                         A."FASE",
                         B."idcatasignacion",
                         B."tipo_categoria"
                FROM public."sm_tablets_obs" A
                LEFT JOIN (SELECT * FROM public."obs_asigna_bi" 
                           WHERE "fecha_corte"= '"""+fecha_corte_t0+"""'
                           ) B
                ON (A."CODIGO_MODULAR" = B."idinstitucioneducativa" AND
                    A."SERIE_EQUIPO" = B."serie_equipo" AND
                    A."OBSERVACIONES_A" = B."observaciones")
                WHERE LENGTH(A."OBSERVACIONES_A")>0 
                AND A."iddia"=(SELECT MAX (DISTINCT iddia) FROM public.sm_tablets_obs)
                ;"""

asigna_data = pd.read_sql_query(query_fec_t1, conn_dg)

asigna_data.columns
asigna_new.columns
#-----------------------------------------------------------------------------*
obs_asigna_t1 = asigna_data.merge(asigna_new[['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A','target']], 
                                how = 'left', 
                                left_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A']),
                                right_on=(['CODIGO_MODULAR','SERIE_EQUIPO','OBSERVACIONES_A']),
                                indicator = True)
#-----------------------------------------------------------------------------*
obs_asigna_t1.columns
print(obs_asigna_t1['_merge'].value_counts())

obs_asigna_t1['tipo_categoria'].value_counts()
obs_asigna_t1.shape
obs_asigna_t1['CODIGO_MODULAR'].isnull().sum()
#----------------------------------------------------------------------------*
obs_asigna_t1['idcatasignacion'] = np.where(obs_asigna_t1['idcatasignacion'].notnull(),
                                            obs_asigna_t1['idcatasignacion'],
                                            obs_asigna_t1['target'])

obs_asigna_t1['tipo_categoria'] = np.where(obs_asigna_t1['_merge']=='both',
                                       2,
                                       obs_asigna_t1['tipo_categoria'])    
#----------------------------------------------------------------------------*
obs_asigna_t1.columns

obs_asigna_t1 = obs_asigna_t1[['CODIGO_MODULAR',                        
                        'NRO_PECOSA',
                        'CODIGO',
                        'SERIE_EQUIPO',
                        'OBSERVACIONES_A',
                        'FECHA_CREACION_AT',
                        'FECHA_MODIFICACION_AT',
                        'idcatasignacion',
                        'fase','tipo_categoria']]
#----------------------------------------------------------------------------*
#fecha_corte = fecha_corte
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

obs_asigna_t1 = obs_asigna_t1.drop_duplicates()

obs_asigna_t1.columns
#  re order columns
obs_asigna_t1 = obs_asigna_t1[['idinstitucioneducativa',
                        'nro_pecosa',
                        'idcodigo',
                        'serie_equipo',
                        'observaciones',
                        'fecha_creacion',
                        'fecha_modificacion',
                        'idcatasignacion',
                        'fase',
                        'tipo_categoria',
                        'fecha_corte',
                        'fecha_siagie']]

for col in obs_asigna_t1.columns:
   print(col)
   print(obs_asigna_t1[col].isnull().sum())
#----------------------------------------------------------------------------*
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:a07119420@localhost:5432/DGAVIDIA')
obs_asigna_t1.to_sql('obs_asigna_bi', engine, if_exists='append', index=False)
#-----------------------------------------------------------------------------*        
asigna_bi = pd.read_sql_query('SELECT * FROM public.obs_asigna_bi', conn_dg)    
recep_bi = pd.read_sql_query('SELECT * FROM public.obs_recepcion_bi', conn_dg)


    
    
    
    
    
    