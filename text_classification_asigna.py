# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:01:03 2023

@author: DGAVIDIA
"""
# -*- coding: utf-8 -*-
#en cmd prompt
#pip install -U pip setuptools wheel
#pip install -U spacy
#python -m spacy download es_core_news_sm
#pip install pyspellchecker

import pandas as pd
import numpy as np
import gensim
import nltk
#nltk.download('punkt')
import sys

from joblib import dump
import pickle 


sys.path.append('D:/repositorios_git/nlp_use/')
import normalizar_texto as nt
#-----------------------------------------------------------------------------*
fec_t = "20230809"
minedu_dir  = 'C:/Users/dgavidia/OneDrive - Ministerio de Educación'
obs_dir = minedu_dir + '/BD_USE/NLP/TABLETAS/Input/observaciones/' +fec_t+'/obs_asigna_'+fec_t+'.xlsx'
datos = pd.read_excel(obs_dir)
#-----------------------------------------------------------------------------*
stopword_list = nltk.corpus.stopwords.words('spanish')

stop_words_nombres = pd.read_csv(minedu_dir + '/BD_USE/NLP/NOMBRES.csv')
stop_words_apellidos = pd.read_csv(minedu_dir + '/BD_USE/NLP/APELLIDOS.csv')
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
        
#import re
#(datos['flg_bin']==1) & (datos['flg_cat']==1)
patron1 = r'(?i)(\bfalta\b|\bno\b||\bsin\b).*(valid|coinc|tiene|carg|regis|actua|recog|\bfirmar\b)'
patron2 = r'(?i)((\bdni\b).*(tramit|trámit))|((\bfalta\b).*(dni))'
patron3 =r'(?i)((\bse\b).*(entreg))|(recog)'

datos['target'] = np.where((datos['flg_cat']!=1) & 
                            (datos['OBSERVACIONES_A'].str.contains(r'(?i)celu|telef', regex=True)==False) &
                            (datos['OBSERVACIONES_A'].str.contains(patron2, regex=True)==False) & 
                            (datos['OBSERVACIONES_A'].str.contains(patron1, regex=True)==False) & 
                            (datos['OBSERVACIONES_A'].str.contains(r'(?i)dni|reniec|firm', regex=True)),
                           1, #Asignada beneficiarios o familiares
                           0)

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)interne|energ', regex=True)==False) &
                             (datos['OBSERVACIONES_A'].str.contains(patron3, regex=True)) & 
                             (datos['OBSERVACIONES_A'].str.contains(r'(?i)((\bno\b).(\bse\b).*(entreg))|((\bno\b).*(recog|valid|coinc))|(no entrega)', regex=True)==False) & 
                             (datos['target']==0),
                           1, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['target']==0 )& (datos['flg_cat']==15) & (datos['flg_bin']==1) &
                           (datos['OBSERVACIONES_A'].str.contains(r'(?i)regul|subsa', regex=True)==False),
                           1, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['target']==0 )& (datos['flg_cat']==6) & (datos['flg_bin']==1),
                           1, #Dificultades con la conectividad de las tabletas
                           datos['target'])
datos['target'] = np.where(
                           (datos['OBSERVACIONES_A'].str.contains(r'(?i)(\bdni\b).*(registr)', regex=True)),
                           1, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where(((datos['target']==1) & (datos['flg_bin']==0)) | 
                           ((datos['target']==0) & (datos['OBSERVACIONES_A'].str.contains(r'(?i)dni|reniec|firm|siagie', regex=True))),
                           2, #Problemas con validar y registrar al beneficiario
                           datos['target'])

datos['target'] = np.where(((datos['target']==1) & (datos['flg_bin']==0)) | 
                           ((datos['target']==0) & (datos['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b).*(valid)', regex=True))),
                           2, #Problemas con validar y registrar al beneficiario
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(rot|raja|negro|quiña|malogr|rob|extrav|p.rdi)|((\bno\b).*func)', regex=True)) | 
                           (datos['flg_cat']==5),
                           3, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(\bsin\b|falta|\bno\b).*((chic|chip|internet|cobert|conect|saldo|megas)|(plan|datos))', regex=True)) & ((datos['target']==0) | (datos['flg_cat']==97)),
                           4, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(ni|solicito).*((chip|internet|cobert|conect)|(plan|datos|megas))', regex=True)) & (datos['target']==0),
                           4, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(no|perder).*((chip|internet|cobert|conect|saldo)|(plan|datos))|((chip).*(bloq))', regex=True)) & (datos['target']==0),
                           4, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(chip|internet|cobert|conect|saldo|megas|señal)', regex=True))  & 
                           (datos['flg_bin']==0) &
                           (datos['target']==0),
                           4, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(no|perder|sin|falta).*(carga|carcas|funda|llave|aguja|ahuja|panel|solar|a.*tador|corrient)', regex=True))  & 
                           (datos['flg_bin']==0) &
                           (datos['target']==0),
                           5, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(adaptador|corriente|cargad|cable|usb)', regex=True))  & 
                           (datos['flg_bin']==0) &
                           (datos['target']==0),
                           5, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==2) & (datos['target']==0),
                           1, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==9) & (datos['target']==0),
                           6, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(("primero|segundo|tercero|cuarto|quinto|sexto").*("primero|segundo|tercero|cuarto|quinto|sexto"))', regex=True)),
                           6, #Dificultades con la conectividad de las tabletas
                           datos['target'])

datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)((1ro|2do|3ro|4to|5to|6to).*(1ro|2do|3ro|4to|5to|6to))', regex=True)),
                           6, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where(((datos['flg_cat']==1) & (datos['flg_bin']==0) & (datos['target']==0) ),
                           2, #Problemas con validar y registrar al beneficiario
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where(((datos['flg_cat']==4) & (datos['flg_bin']==0) & (datos['target']==0) ),
                           3, #Problemas con validar y registrar al beneficiario
                           datos['target'])
#-----------------------------------------------------------------------------*
#
#
#
#

datos['target']  = np.where((datos['flg_cat']==97) & 
                                   (datos['OBSERVACIONES_A'].str.contains(r'(?i)(sin efecto)', regex=True)==False) & 
                                   (datos['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b|\bsin\b).*(dni|\bvalidado\b)', regex=True)) & (datos['flg_cat']==97),
                           2, #Problemas con validar y registrar al beneficiario
                           datos['target'])
#-----------------------------------------------------------------------------*

datos['target']  = np.where((datos['flg_cat']==97) & 
                                   (datos['OBSERVACIONES_A'].str.contains(r'(?i)((\bdni\b).*(registr))|(dni valid)', regex=True)) & (datos['flg_cat']==97),
                           1, #Problemas con validar y registrar al beneficiario
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target']  = np.where((datos['target']==2) & (datos['flg_cat']==97) & 
                                   (datos['OBSERVACIONES_A'].str.contains(r'(?i)(sin efecto)', regex=True)==False) & 
                                   (datos['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b|\bsin\b).*(dni|\bvalidado\b)', regex=True)==False),
                           2, #Problemas con validar y registrar al beneficiario
                           datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']==2) & (datos['flg_cat']==97) & 
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(\bno\b|\bsin\b).*(dni|\bvalidado\b)', regex=True)==False),
                                  1, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']==3) & (datos['flg_cat']==97) & 
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(apoderad|recog)', regex=True)==True),
                                  1, #Asignada beneficiarios o familiares
                                  datos['target'])

#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']==3) & (datos['flg_cat']==97) & 
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)((ning|sin|no).*(probl|prob))|((se).*(entreg))|(buen estad)', regex=True)==True),
                                  0, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']==3) & (datos['flg_cat']==97) &
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(dni|firm|asign|recib|recep|nombre|alumno)', regex=True)==True),
                                  1, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']==3) & (datos['flg_cat']==97) & 
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)((\bno\b).*(funciona))|(malog|raja|perdi)', regex=True)==False),
                                  0, #Asignada beneficiarios o familiares
                                  datos['target'])
datos['target'] = np.where((datos['flg_cat']==97) & 
                                 (datos['OBSERVACIONES_A'].str.contains(r'(?i)((otro).*(est|doc))|(otro nivel)|((se).*(entreg).*(otro))', regex=True)),
                                  6, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['OBSERVACIONES_A'].str.contains(r'(?i)(correlativo)', regex=True)),
                                  0, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==97) & 
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(cambio)', regex=True)),
                                  6, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==97) &  #5
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(niña)', regex=True)==False) & 
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(entreg|tiene).*(otr).*(tablet|carga|chip)', regex=True)),
                                  5, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==97) & (datos['target']==0) &
                                  (datos['OBSERVACIONES_A'].str.contains(r'(?i)(padre|apoderad|madre|tío|tía|tio|tia|abuel)')),
                                  1, #Asignada beneficiarios o familiares
                                  datos['target'])
#-----------------------------------------------------------------------------*
#0. Otros comentarios
#1. Asignada beneficiarios o familiares
#2. Problemas con validar y registrar al beneficiario
#3. Problemas software, técnicos, averías, robos
#4. Problemas de conectividad
#5. Problemas con cargador y otros complementos
#6. Redistribución y cambio de beneficiario

datos['target'].value_counts()

pd.crosstab(datos['flg_cat'],datos['target'])        
datos = datos[((datos['flg_cat']==97) | (datos['target']==1))==False]

pd.crosstab(datos['flg_cat'],datos['target'])
print(datos['target'].value_counts())    
#-----------------------------------------------------------------------------*
text_corpus = nt.normalizar_texto(datos['OBSERVACIONES_A'],
                                    contraction_expansion=True,
                                    accented_char_removal=True, 
                                    text_lower_case=True, 
                                    text_stemming=False, text_lemmatization=True, 
                                    special_char_removal=True, remove_digits=True,
                                    stopword_removal=True, special_cases = True,
                                    autocorrecion=False,
                                    stopwords = stop_words_tablets)
#-----------------------------------------------------------------------------*
#Eliminando palabras repetidas
from normalizar_texto import palabras_repetidas
texto_limpio = []
for doc in text_corpus:
    word = palabras_repetidas(doc)
    texto_limpio.append(word)
#-----------------------------------------------------------------------------*
datos['obs'] = texto_limpio
datos['obs'] = datos.obs.replace('', 'NA')
datos['obs'] = datos['obs'].str.replace("él", "se")
#-----------------------------------------------------------------------------*
print(datos['target'].value_counts(normalize=True))
#-----------------------------------------------------------------------------*
datos['target'].value_counts(normalize=True).sort_index()
datos['target'].value_counts().sort_index()
#----------------------------------------------------------------------------*
result = datos['target'].value_counts(normalize=True).reset_index()
result.columns = ['Value', 'Proportion']
result['Count'] = result['Value'].map(datos['target'].value_counts())

print(result)
#print(np.mean(datos['target']))
#%%2 - Modelos de ingienería de características
#Objetivo: Transformar los textos en matrices de datos para los modelos
#Construyendo datos de prueba y datos de entrenamiento
from sklearn.model_selection import train_test_split
train_corpus, test_corpus, train_label_nums, test_label_nums = train_test_split(
    datos['obs'], #np.array(datos['obs'])
    datos['target'], #np.array(datos['target'])
    test_size=1/10, random_state=42)

#os_train_corpus
df_temp = train_corpus.copy()
df_temp = pd.DataFrame(df_temp)
df_temp['target'] = train_label_nums
#--------*
print(df_temp['target'].value_counts().sort_index())
datos['target'].value_counts(normalize=True).sort_index() #print(np.mean(df_temp['target']))
#-------------------------------------------------------------------*
print(result)
count_class_0, count_class_2, count_class_3, count_class_4, count_class_5, count_class_6 = df_temp.target.value_counts().sort_index()

df_class_0 = df_temp[df_temp['target'] == 0]
df_class_1 = df_temp[df_temp['target'] == 1]
df_class_2 = df_temp[df_temp['target'] == 2]
df_class_3 = df_temp[df_temp['target'] == 3]
df_class_4 = df_temp[df_temp['target'] == 4]
df_class_5 = df_temp[df_temp['target'] == 5]
df_class_6 = df_temp[df_temp['target'] == 6]

df_class_0_sample = df_class_0.sample(count_class_3-count_class_0,replace=True)
df_class_2_sample = df_class_2.sample(count_class_3-count_class_2,replace=True)
df_class_3_sample = df_class_3
df_class_4_sample = df_class_4.sample(count_class_3-count_class_4,replace=True)
df_class_5_sample = df_class_5.sample(count_class_3-count_class_5,replace=True)
df_class_6_sample = df_class_6.sample(count_class_3-count_class_6,replace=True)


df_train_over = pd.concat([df_class_0_sample, df_class_2_sample,
                           df_class_4_sample,df_class_5_sample,
                           df_class_6_sample], axis=0)

df_train_over = pd.concat([df_train_over, 
                           df_class_0,
                           df_class_1,
                           df_class_2,                           
                           df_class_3,
                           df_class_4,
                           df_class_5,
                           df_class_6], axis=0)



print('Random over-sampling:')
print(df_train_over.target.value_counts())

#df_train_over.is_fraud.value_counts().plot(kind='bar', title='Is Fraud');

over_train_corpus = df_train_over['obs']
over_train_label_nums = df_train_over['target']

#%%%2.1 - Bag of Words (term frequency) model:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(train_corpus)
cv_test_features = cv.transform(test_corpus)
#%%%2.2 - Bag of N-Grams model:
bv = CountVectorizer(binary=True, ngram_range=(2,2))
bv_train_features = bv.fit_transform(train_corpus)
bv_test_features = bv.transform(test_corpus)

#%%%2.3 - TF-IDF model: 
#term frequency-inverse document frequency.
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
tv_train_features = tv.fit_transform(train_corpus)
tv_test_features = tv.transform(test_corpus)

#%%%2.4 - Word2Vec model:
tokenized_train = [nt.tokenizer.tokenize(text)
                   for text in train_corpus]
tokenized_test = [nt.tokenizer.tokenize(text)
                   for text in test_corpus]
#-----------------------------------------------------------------------------*
# average_word_vectors: 
#   calcula el vector promedio de una lista de palabras 
#   dadas utilizando el modelo Word2Vec y el vocabulario. 
#   El vector promedio se calcula sumando los vectores de todas las palabras presentes en el vocabulario y 
#   dividiéndolo por el número total de palabras en el vocabulario. 
#
# document_vectorizer:
#   Finalmente, se aplica esta función a cada documento (fila) en el corpus utilizando una comprensión de lista, 
#   y los resultados se almacenan en una lista llamada features.
#-----------------------------------------------------------------------------*
def average_word_vectors(words, model, vocabulary, num_features):
    # Crear un vector de características inicializado con ceros
    feature_vector = np.zeros((num_features,), dtype="float64")  
    nwords = 0.  # Contador de palabras válidas
    for word in words:  # Iterar sobre las palabras del documento
        # Verificar si la palabra está en el vocabulario del modelo Word2Vec
        if word in vocabulary:  
            # Incrementar el contador de palabras válidas
            nwords = nwords + 1. 
            # Sumar el vector de la palabra actual (modelada por w2vec) al vector de características
            feature_vector = np.add(feature_vector, model.wv[word])  
    if nwords > 0:  # Verificar si hay palabras válidas en el documento
        # Calcular el promedio dividiendo el vector de características por el número de palabras válidas
        feature_vector = np.divide(feature_vector, nwords)  
    return feature_vector


def document_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)  # Crear un conjunto de palabras del modelo Word2Vec
    features = np.zeros((len(corpus), num_features), dtype="float64")  # Crear una matriz de ceros para almacenar los vectores de características de todos los documentos
    for i, tokenized_sentence in enumerate(corpus):  # Iterar sobre los documentos tokenizados en el corpus
        feature_vector = average_word_vectors(
            tokenized_sentence,
            model, 
            vocabulary, 
            num_features
        )  # Calcular el vector de características promedio para el documento actual
        features[i] = feature_vector  # Almacenar el vector de características en la fila correspondiente de la matriz
        
    return features  # Devolver la matriz de vectores de característica
#-----------------------------------------------------------------------------*
# build word2vec model
#-----------------------------------------------------------------------------*
w2v_num_features = 1000
w2v_model = gensim.models.Word2Vec(tokenized_train, 
                                   vector_size=w2v_num_features, 
                                   window=100,
                                   min_count=2, 
                                   sample=1e-3, 
                                   sg=1, epochs =5, workers=10)
#-----------------------------------------------------------------------------*
a1 = ['recepcion', 'tableta', 'educativo']
a2 = set(w2v_model.wv.index_to_key)
#-----------------------------------------------------------------------------*
average_word_vectors(words = a1,
                     model = w2v_model,
                     vocabulary= a2,
                     num_features = w2v_num_features)


avg_wv_train_features = document_vectorizer(corpus = tokenized_train, 
                                            model = w2v_model,
                                            num_features = w2v_num_features)

avg_wv_test_features = document_vectorizer(corpus = tokenized_test, 
                                           model = w2v_model,
                                           num_features = w2v_num_features)    

print('Word2Vec model:> Train features shape:', 
      avg_wv_train_features.shape,
      ' Test features shape:', avg_wv_test_features.shape)
#%%%2.5 - GloVe model:   
    
#%% 3-Modelos de clasificacióm - Feat. Eng. Bag of Words 
#%%% 3.1-Multinomial Naïve Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

mnb = MultinomialNB(alpha=1)
mnb.fit(cv_train_features, train_label_nums)
mnb_bow_cv_scores = cross_val_score(mnb, cv_train_features, train_label_nums, cv=5)
mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)
print('CV Accuracy (5-fold):', mnb_bow_cv_scores)
print('Mean CV Accuracy:', mnb_bow_cv_mean_score)
mnb_bow_test_score = mnb.score(cv_test_features, test_label_nums)
print('Test Accuracy:', mnb_bow_test_score)

#%%% 3.2-Logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(cv_train_features, train_label_nums)
lr_bow_cv_scores = cross_val_score(lr, cv_train_features, train_label_nums, cv=5)
lr_bow_cv_mean_score = np.mean(lr_bow_cv_scores)
print('CV Accuracy (5-fold):', lr_bow_cv_scores)
print('Mean CV Accuracy:', lr_bow_cv_mean_score)
lr_bow_test_score = lr.score(cv_test_features, test_label_nums)
print('Test Accuracy:', lr_bow_test_score)
#%%% 3.3-Support vector machines
from sklearn.svm import LinearSVC
svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(cv_train_features, train_label_nums)
svm_bow_cv_scores = cross_val_score(svm, cv_train_features, train_label_nums, cv=5)
svm_bow_cv_mean_score = np.mean(svm_bow_cv_scores)
print('CV Accuracy (5-fold):', svm_bow_cv_scores)
print('Mean CV Accuracy:', svm_bow_cv_mean_score)
svm_bow_test_score = svm.score(cv_test_features, test_label_nums)
print('Test Accuracy:', svm_bow_test_score)

#%%% 3.4-Random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(cv_train_features, train_label_nums)
rfc_bow_cv_scores = cross_val_score(rfc, cv_train_features, train_label_nums, cv=5)
rfc_bow_cv_mean_score = np.mean(rfc_bow_cv_scores)
print('CV Accuracy (5-fold):', rfc_bow_cv_scores)
print('Mean CV Accuracy:', rfc_bow_cv_mean_score)
rfc_bow_test_score = rfc.score(cv_test_features, test_label_nums)
print('Test Accuracy:', rfc_bow_test_score)
#%%% 3.5-SGDC (Stochastic Gradient Descent)
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import SGDClassifier

svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
svm_sgd.fit(cv_train_features, train_label_nums)
svmsgd_bow_cv_scores = cross_val_score(svm_sgd, cv_train_features, train_label_nums, cv=5)
svmsgd_bow_cv_mean_score = np.mean(svmsgd_bow_cv_scores)
print('CV Accuracy (5-fold):', svmsgd_bow_cv_scores)
print('Mean CV Accuracy:', svmsgd_bow_cv_mean_score)
svmsgd_bow_test_score = svm_sgd.score(cv_test_features, test_label_nums)
print('Test Accuracy:', svmsgd_bow_test_score)





#%%% 3.6-Gradient boosting machine

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(cv_train_features, train_label_nums)
gbc_bow_cv_scores = cross_val_score(gbc, cv_train_features, train_label_nums, cv=5)
gbc_bow_cv_mean_score = np.mean(gbc_bow_cv_scores)
print('CV Accuracy (5-fold):', gbc_bow_cv_scores)
print('Mean CV Accuracy:', gbc_bow_cv_mean_score)
gbc_bow_test_score = gbc.score(cv_test_features, test_label_nums)
print('Test Accuracy:', gbc_bow_test_score)



#%% 4-Modelos de clasificacióm - Feat. Eng.  Bag of N-grams 
#%%% 4.1-Multinomial Naïve Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(bv_train_features, train_label_nums)
mnb_bng_bv_scores = cross_val_score(mnb, bv_train_features, train_label_nums, cv=5)
mnb_bng_bv_mean_score = np.mean(mnb_bng_bv_scores)
print('CV Accuracy (5-fold):', mnb_bng_bv_scores)
print('Mean CV Accuracy:', mnb_bng_bv_mean_score)
mnb_bng_test_score = mnb.score(bv_test_features, test_label_nums)
print('Test Accuracy:', mnb_bng_test_score)
#%%% 4.2-Logistic regression
lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(bv_train_features, train_label_nums)
lr_bng_bv_scores = cross_val_score(lr, bv_train_features, train_label_nums, cv=5)
lr_bng_bv_mean_score = np.mean(lr_bng_bv_scores)
print('CV Accuracy (5-fold):', lr_bng_bv_scores)
print('Mean CV Accuracy:', lr_bng_bv_mean_score)
lr_bng_test_score = lr.score(bv_test_features, test_label_nums)
print('Test Accuracy:', lr_bng_test_score)
#%%% 4.3-Support vector machines
svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(bv_train_features, train_label_nums)
svm_bng_bv_scores = cross_val_score(svm, bv_train_features, train_label_nums, cv=5)
svm_bng_bv_mean_score = np.mean(svm_bng_bv_scores)
print('CV Accuracy (5-fold):', svm_bng_bv_scores)
print('Mean CV Accuracy:', svm_bng_bv_mean_score)
svm_bng_test_score = svm.score(bv_test_features, test_label_nums)
print('Test Accuracy:', svm_bng_test_score)

#%%% 4.4-Random forest
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(bv_train_features, train_label_nums)
rfc_bng_bv_scores = cross_val_score(rfc, bv_train_features, train_label_nums, cv=5)
rfc_bng_bv_mean_score = np.mean(rfc_bng_bv_scores)
print('CV Accuracy (5-fold):', rfc_bng_bv_scores)
print('Mean CV Accuracy:', rfc_bng_bv_mean_score)
rfc_bng_test_score = rfc.score(bv_test_features, test_label_nums)
print('Test Accuracy:', rfc_bng_test_score)

#%%% 4.5-SGDC (Stochastic Gradient Descent)
svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
svm_sgd.fit(bv_train_features, train_label_nums)
svmsgd_bng_bv_scores = cross_val_score(svm_sgd, bv_train_features, train_label_nums, cv=5)
svmsgd_bng_bv_mean_score = np.mean(svmsgd_bng_bv_scores)
print('CV Accuracy (5-fold):', svmsgd_bng_bv_scores)
print('Mean CV Accuracy:', svmsgd_bng_bv_mean_score)
svmsgd_bng_test_score = svm_sgd.score(bv_test_features, test_label_nums)
print('Test Accuracy:', svmsgd_bng_test_score)
#%%% 4.6-Gradient boosting machine
gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(bv_train_features, train_label_nums)
gbc_bng_bv_scores = cross_val_score(gbc, bv_train_features, train_label_nums, cv=5)
gbc_bng_bv_mean_score = np.mean(gbc_bng_bv_scores)
print('CV Accuracy (5-fold):', gbc_bng_bv_scores)
print('Mean CV Accuracy:', gbc_bng_bv_mean_score)
gbc_bng_test_score = gbc.score(bv_test_features, test_label_nums)
print('Test Accuracy:', gbc_bng_test_score)



#%% 5-Modelos de clasificacióm - Feat. Eng. TF-IDF 
#%%% 5.1-Multinomial Naïve Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(tv_train_features, train_label_nums)
mnb_tfidf_cv_scores = cross_val_score(mnb, tv_train_features, train_label_nums, cv=5)
mnb_tfidf_cv_mean_score = np.mean(mnb_tfidf_cv_scores)
print('CV Accuracy (5-fold):', mnb_tfidf_cv_scores)
print('Mean CV Accuracy:', mnb_tfidf_cv_mean_score)
mnb_tfidf_test_score = mnb.score(tv_test_features, test_label_nums)
print('Test Accuracy:', mnb_tfidf_test_score)
#%%% 5.2-Logistic regression

lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(tv_train_features, train_label_nums)
lr_tfidf_cv_scores = cross_val_score(lr, tv_train_features, train_label_nums, cv=5)
lr_tfidf_cv_mean_score = np.mean(lr_tfidf_cv_scores)
print('CV Accuracy (5-fold):', lr_tfidf_cv_scores)
print('Mean CV Accuracy:', lr_tfidf_cv_mean_score)
lr_tfidf_test_score = lr.score(tv_test_features, test_label_nums)
print('Test Accuracy:', lr_tfidf_test_score)

#%%% 5.3-Support vector machines

svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(tv_train_features, train_label_nums)
svm_tfidf_cv_scores = cross_val_score(svm, tv_train_features, train_label_nums, cv=5)
svm_tfidf_cv_mean_score = np.mean(svm_tfidf_cv_scores)
print('CV Accuracy (5-fold):', svm_tfidf_cv_scores)
print('Mean CV Accuracy:', svm_tfidf_cv_mean_score)
svm_tfidf_test_score = svm.score(tv_test_features, test_label_nums)
print('Test Accuracy:', svm_tfidf_test_score)

#%%% 5.4-Random forest
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(tv_train_features, train_label_nums)
rfc_tfidf_cv_scores = cross_val_score(rfc, tv_train_features, train_label_nums, cv=5)
rfc_tfidf_cv_mean_score = np.mean(rfc_tfidf_cv_scores)
print('CV Accuracy (5-fold):', rfc_tfidf_cv_scores)
print('Mean CV Accuracy:', rfc_tfidf_cv_mean_score)
rfc_tfidf_test_score = rfc.score(tv_test_features, test_label_nums)
print('Test Accuracy:', rfc_tfidf_test_score)
#%%% 5.5-SGDC (Stochastic Gradient Descent)
svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
svm_sgd.fit(tv_train_features, train_label_nums)
svmsgd_tfidf_cv_scores = cross_val_score(svm_sgd, tv_train_features, train_label_nums, cv=5)
svmsgd_tfidf_cv_mean_score = np.mean(svmsgd_tfidf_cv_scores)
print('CV Accuracy (5-fold):', svmsgd_tfidf_cv_scores)
print('Mean CV Accuracy:', svmsgd_tfidf_cv_mean_score)
svmsgd_tfidf_test_score = svm_sgd.score(tv_test_features, test_label_nums)
print('Test Accuracy:', svmsgd_tfidf_test_score)

#%%% 5.6-Gradient boosting machine

gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(tv_train_features, train_label_nums)
gbc_tfidf_cv_scores = cross_val_score(gbc, tv_train_features, train_label_nums, cv=5)
gbc_tfidf_cv_mean_score = np.mean(gbc_tfidf_cv_scores)
print('CV Accuracy (5-fold):', gbc_tfidf_cv_scores)
print('Mean CV Accuracy:', gbc_tfidf_cv_mean_score)
gbc_tfidf_test_score = gbc.score(tv_test_features, test_label_nums)
print('Test Accuracy:', gbc_tfidf_test_score)

#%% A-Comparación
cuadro =  pd.DataFrame([['Naive Bayes', 
               mnb_bow_cv_mean_score, mnb_bow_test_score, 
               mnb_bng_bv_mean_score, mnb_bng_test_score, 
               mnb_tfidf_cv_mean_score, mnb_tfidf_test_score],
              ['Logistic Regression', 
               lr_bow_cv_mean_score, lr_bow_test_score, 
               lr_bng_bv_mean_score, lr_bng_test_score, 
               lr_tfidf_cv_mean_score, lr_tfidf_test_score],
              ['Linear SVM', 
               svm_bow_cv_mean_score, svm_bow_test_score, 
               svm_bng_bv_mean_score, svm_bng_test_score, 
               svm_tfidf_cv_mean_score, svm_tfidf_test_score],
              ['Linear SVM (SGD)', 
               svmsgd_bow_cv_mean_score, svmsgd_bow_test_score, 
               svmsgd_bng_bv_mean_score, svmsgd_bng_test_score, 
               svmsgd_tfidf_cv_mean_score, svmsgd_tfidf_test_score],
              ['Random Forest', 
               rfc_bow_cv_mean_score, rfc_bow_test_score, 
               rfc_bng_bv_mean_score, rfc_bng_test_score, 
               rfc_tfidf_cv_mean_score, rfc_tfidf_test_score],
              ['Gradient Boosted Machines', 
               gbc_bow_cv_mean_score, gbc_bow_test_score, 
               gbc_bng_bv_mean_score, gbc_bng_test_score, 
               gbc_tfidf_cv_mean_score, gbc_tfidf_test_score]],
             columns=['Model', 
                      'CV Score (BOW)', 'Test Score (BOW)', 
                      'CV Score (B-NGRAM)', 'Test Score (B-NGRAM)', 
                      'CV Score (TF-IDF)', 'Test Score (TF-IDF)'],
             ).T

#%% 6-Modelos de clasificacióm - Feat. Eng. Word2Vec model 
print("Pendiente Word2Vec - GloVe")
#%%% 6.1-%Logistic regression
lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(avg_wv_train_features, train_label_nums)
lr_w2v_cv_scores = cross_val_score(lr, avg_wv_train_features, train_label_nums, cv=5)
lr_w2v_cv_mean_score = np.mean(lr_w2v_cv_scores)
print('CV Accuracy (5-fold):', lr_w2v_cv_scores)
print('Mean CV Accuracy:', lr_w2v_cv_mean_score)
lr_w2v_test_score = lr.score(avg_wv_test_features, test_label_nums)
print('Test Accuracy:', lr_w2v_test_score)
#%%% 6.2-Support vector machines
svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
svm.fit(avg_wv_train_features, train_label_nums)
svm_w2v_cv_scores = cross_val_score(svm, avg_wv_train_features, train_label_nums, cv=5)
svm_w2v_cv_mean_score = np.mean(svm_w2v_cv_scores)
print('CV Accuracy (5-fold):', svm_w2v_cv_scores)
print('Mean CV Accuracy:', svm_w2v_cv_mean_score)
svm_w2v_test_score = svm.score(avg_wv_test_features, test_label_nums)
print('Test Accuracy:', svm_w2v_test_score)

#%%% 6.3-Gradient boosting machine
#%%% 7-Modelos de clasificacióm - Feat. Eng. GloVe model 
print("Pendiente Clasificación - GloVe")
#%% B.OVER SAMPLING
print("Over Sampling")
gir_repo_dir = 'D:/repositorios_git/nlp_use/'
#%%% B-2.1 - Bag of Words (term frequency) model:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(over_train_corpus)
cv_test_features = cv.transform(test_corpus)

dir_os_bow = gir_repo_dir+'/feat_engine_asigna_os_bow.pickle'
with open(dir_os_bow, 'wb') as f:
    pickle.dump(cv, f)
#%%% B-2.2 - Bag of N-Grams model:
bv = CountVectorizer(binary=True, ngram_range=(2,2))
bv_train_features = bv.fit_transform(over_train_corpus)
bv_test_features = bv.transform(test_corpus)

#%%% B-2.3 - TF-IDF model: 
#term frequency-inverse document frequency.
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
tv_train_features = tv.fit_transform(over_train_corpus)

dir_os_tf_idf = gir_repo_dir+'/feat_engine_asigna_os_tf_idf.pickle'
with open(dir_os_tf_idf, 'wb') as f:
    pickle.dump(cv, f)
#gir_repo_dir = 'D:/repositorios_git/nlp_use/'
#dir_os_tfidf = gir_repo_dir+'/feat_engine_2_os_tf_idf.pickle'
#with open(dir_os_tfidf, 'wb') as f:
#    pickle.dump(tv, f)    
#tv_test_features = tv.transform(test_corpus)
#%%% B-2.4 - Word2Vec model:
tokenized_train = [nt.tokenizer.tokenize(text)
                   for text in over_train_corpus]
tokenized_test = [nt.tokenizer.tokenize(text)
                   for text in test_corpus]
#-----------------------------------------------------------------------------*
#-----------------------------------------------------------------------------*
# build word2vec model
#-----------------------------------------------------------------------------*
w2v_num_features = 1000
w2v_model = gensim.models.Word2Vec(tokenized_train, 
                                   vector_size=w2v_num_features, 
                                   window=100,
                                   min_count=2, 
                                   sample=1e-3, 
                                   sg=1, epochs =5, workers=10)
#-----------------------------------------------------------------------------*
a1 = ['recepcion', 'tableta', 'educativo']
a2 = set(w2v_model.wv.index_to_key)
#-----------------------------------------------------------------------------*
average_word_vectors(words = a1,
                     model = w2v_model,
                     vocabulary= a2,
                     num_features = w2v_num_features)


avg_wv_train_features = document_vectorizer(corpus = tokenized_train, 
                                            model = w2v_model,
                                            num_features = w2v_num_features)

avg_wv_test_features = document_vectorizer(corpus = tokenized_test, 
                                           model = w2v_model,
                                           num_features = w2v_num_features)    

print('Word2Vec model:> Train features shape:', 
      avg_wv_train_features.shape,
      ' Test features shape:', avg_wv_test_features.shape)
#%%% B-2.5 - GloVe model:  
    
#%% B-3-Modelos de clasificacióm - Feat. Eng. Bag of Words 
#%%% 3.1-Multinomial Naïve Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

mnb = MultinomialNB(alpha=1)
mnb.fit(cv_train_features, over_train_label_nums)
mnb_bow_cv_scores = cross_val_score(mnb, cv_train_features, over_train_label_nums, cv=5)
mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)
print('CV Accuracy (5-fold):', mnb_bow_cv_scores)
print('Mean CV Accuracy:', mnb_bow_cv_mean_score)
mnb_bow_test_score = mnb.score(cv_test_features, test_label_nums)
print('Test Accuracy:', mnb_bow_test_score)

#%%% 3.2-Logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(cv_train_features, over_train_label_nums)
lr_bow_cv_scores = cross_val_score(lr, cv_train_features, over_train_label_nums, cv=5)
lr_bow_cv_mean_score = np.mean(lr_bow_cv_scores)
print('CV Accuracy (5-fold):', lr_bow_cv_scores)
print('Mean CV Accuracy:', lr_bow_cv_mean_score)
lr_bow_test_score = lr.score(cv_test_features, test_label_nums)
print('Test Accuracy:', lr_bow_test_score)
#%%% 3.3-Support vector machines
from sklearn.svm import LinearSVC
svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(cv_train_features, over_train_label_nums)
svm_bow_cv_scores = cross_val_score(svm, cv_train_features, over_train_label_nums, cv=5)
svm_bow_cv_mean_score = np.mean(svm_bow_cv_scores)
print('CV Accuracy (5-fold):', svm_bow_cv_scores)
print('Mean CV Accuracy:', svm_bow_cv_mean_score)
svm_bow_test_score = svm.score(cv_test_features, test_label_nums)
print('Test Accuracy:', svm_bow_test_score)

gir_repo_dir = 'D:/repositorios_git/nlp_use/'
dir_os_svm_bow = gir_repo_dir+'/asigna_os_svm_bow.joblib'
dump(svm, dir_os_svm_bow)
#%%% 3.4-Random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(cv_train_features, over_train_label_nums)
rfc_bow_cv_scores = cross_val_score(rfc, cv_train_features, over_train_label_nums, cv=5)
rfc_bow_cv_mean_score = np.mean(rfc_bow_cv_scores)
print('CV Accuracy (5-fold):', rfc_bow_cv_scores)
print('Mean CV Accuracy:', rfc_bow_cv_mean_score)
rfc_bow_test_score = rfc.score(cv_test_features, test_label_nums)
print('Test Accuracy:', rfc_bow_test_score)

#gir_repo_dir = 'D:/repositorios_git/nlp_use/'
#dir_os_bow_rf = gir_repo_dir+'/modelo_entrenado_bow_rf.joblib'
#dump(rfc, dir_os_bow_rf)
#%%% 3.5-SGDC (Stochastic Gradient Descent)
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import SGDClassifier

svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
svm_sgd.fit(cv_train_features, over_train_label_nums)
svmsgd_bow_cv_scores = cross_val_score(svm_sgd, cv_train_features, over_train_label_nums, cv=5)
svmsgd_bow_cv_mean_score = np.mean(svmsgd_bow_cv_scores)
print('CV Accuracy (5-fold):', svmsgd_bow_cv_scores)
print('Mean CV Accuracy:', svmsgd_bow_cv_mean_score)
svmsgd_bow_test_score = svm_sgd.score(cv_test_features, test_label_nums)
print('Test Accuracy:', svmsgd_bow_test_score)

#%%% 3.6-Gradient boosting machine

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(cv_train_features, over_train_label_nums)
gbc_bow_cv_scores = cross_val_score(gbc, cv_train_features, over_train_label_nums, cv=5)
gbc_bow_cv_mean_score = np.mean(gbc_bow_cv_scores)
print('CV Accuracy (5-fold):', gbc_bow_cv_scores)
print('Mean CV Accuracy:', gbc_bow_cv_mean_score)
gbc_bow_test_score = gbc.score(cv_test_features, test_label_nums)
print('Test Accuracy:', gbc_bow_test_score)


#%% B-4-Modelos de clasificacióm - Feat. Eng.  Bag of N-grams 
#%%% 4.1-Multinomial Naïve Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(bv_train_features, over_train_label_nums)
mnb_bng_bv_scores = cross_val_score(mnb, bv_train_features, over_train_label_nums, cv=5)
mnb_bng_bv_mean_score = np.mean(mnb_bng_bv_scores)
print('CV Accuracy (5-fold):', mnb_bng_bv_scores)
print('Mean CV Accuracy:', mnb_bng_bv_mean_score)
mnb_bng_test_score = mnb.score(bv_test_features, test_label_nums)
print('Test Accuracy:', mnb_bng_test_score)
#%%% 4.2-Logistic regression
lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(bv_train_features, over_train_label_nums)
lr_bng_bv_scores = cross_val_score(lr, bv_train_features, over_train_label_nums, cv=5)
lr_bng_bv_mean_score = np.mean(lr_bng_bv_scores)
print('CV Accuracy (5-fold):', lr_bng_bv_scores)
print('Mean CV Accuracy:', lr_bng_bv_mean_score)
lr_bng_test_score = lr.score(bv_test_features, test_label_nums)
print('Test Accuracy:', lr_bng_test_score)
#%%% 4.3-Support vector machines
svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(bv_train_features, over_train_label_nums)
svm_bng_bv_scores = cross_val_score(svm, bv_train_features, over_train_label_nums, cv=5)
svm_bng_bv_mean_score = np.mean(svm_bng_bv_scores)
print('CV Accuracy (5-fold):', svm_bng_bv_scores)
print('Mean CV Accuracy:', svm_bng_bv_mean_score)
svm_bng_test_score = svm.score(bv_test_features, test_label_nums)
print('Test Accuracy:', svm_bng_test_score)

#%%% 4.4-Random forest
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(bv_train_features, over_train_label_nums)
rfc_bng_bv_scores = cross_val_score(rfc, bv_train_features, over_train_label_nums, cv=5)
rfc_bng_bv_mean_score = np.mean(rfc_bng_bv_scores)
print('CV Accuracy (5-fold):', rfc_bng_bv_scores)
print('Mean CV Accuracy:', rfc_bng_bv_mean_score)
rfc_bng_test_score = rfc.score(bv_test_features, test_label_nums)
print('Test Accuracy:', rfc_bng_test_score)

#%%% 4.5-SGDC (Stochastic Gradient Descent)
svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
svm_sgd.fit(bv_train_features, over_train_label_nums)
svmsgd_bng_bv_scores = cross_val_score(svm_sgd, bv_train_features, over_train_label_nums, cv=5)
svmsgd_bng_bv_mean_score = np.mean(svmsgd_bng_bv_scores)
print('CV Accuracy (5-fold):', svmsgd_bng_bv_scores)
print('Mean CV Accuracy:', svmsgd_bng_bv_mean_score)
svmsgd_bng_test_score = svm_sgd.score(bv_test_features, test_label_nums)
print('Test Accuracy:', svmsgd_bng_test_score)
#%%% 4.6-Gradient boosting machine
gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(bv_train_features, over_train_label_nums)
gbc_bng_bv_scores = cross_val_score(gbc, bv_train_features, over_train_label_nums, cv=5)
gbc_bng_bv_mean_score = np.mean(gbc_bng_bv_scores)
print('CV Accuracy (5-fold):', gbc_bng_bv_scores)
print('Mean CV Accuracy:', gbc_bng_bv_mean_score)
gbc_bng_test_score = gbc.score(bv_test_features, test_label_nums)
print('Test Accuracy:', gbc_bng_test_score)



#%% B-5-Modelos de clasificacióm - Feat. Eng. TF-IDF 
#%%% 5.1-Multinomial Naïve Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(tv_train_features, over_train_label_nums)
mnb_tfidf_cv_scores = cross_val_score(mnb, tv_train_features, over_train_label_nums, cv=5)
mnb_tfidf_cv_mean_score = np.mean(mnb_tfidf_cv_scores)
print('CV Accuracy (5-fold):', mnb_tfidf_cv_scores)
print('Mean CV Accuracy:', mnb_tfidf_cv_mean_score)
mnb_tfidf_test_score = mnb.score(tv_test_features, test_label_nums)
print('Test Accuracy:', mnb_tfidf_test_score)
#%%% 5.2-Logistic regression

lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(tv_train_features, over_train_label_nums)
lr_tfidf_cv_scores = cross_val_score(lr, tv_train_features, over_train_label_nums, cv=5)
lr_tfidf_cv_mean_score = np.mean(lr_tfidf_cv_scores)
print('CV Accuracy (5-fold):', lr_tfidf_cv_scores)
print('Mean CV Accuracy:', lr_tfidf_cv_mean_score)
lr_tfidf_test_score = lr.score(tv_test_features, test_label_nums)
print('Test Accuracy:', lr_tfidf_test_score)

#%%% 5.3-Support vector machines


svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(tv_train_features, over_train_label_nums)
svm_tfidf_cv_scores = cross_val_score(svm, tv_train_features, over_train_label_nums, cv=5)
svm_tfidf_cv_mean_score = np.mean(svm_tfidf_cv_scores)
print('CV Accuracy (5-fold):', svm_tfidf_cv_scores)
print('Mean CV Accuracy:', svm_tfidf_cv_mean_score)
svm_tfidf_test_score = svm.score(tv_test_features, test_label_nums)
print('Test Accuracy:', svm_tfidf_test_score)

#gir_repo_dir = 'D:/repositorios_git/nlp_use/'
#dir_os_tfidf_svm = gir_repo_dir+'/modelo_entrenado_2_os_tf_idf_svm.joblib'
#dump(svm, dir_os_tfidf_svm)

#%%% 5.4-Random forest
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(tv_train_features, over_train_label_nums)
rfc_tfidf_cv_scores = cross_val_score(rfc, tv_train_features, over_train_label_nums, cv=5)
rfc_tfidf_cv_mean_score = np.mean(rfc_tfidf_cv_scores)
print('CV Accuracy (5-fold):', rfc_tfidf_cv_scores)
print('Mean CV Accuracy:', rfc_tfidf_cv_mean_score)
rfc_tfidf_test_score = rfc.score(tv_test_features, test_label_nums)
print('Test Accuracy:', rfc_tfidf_test_score)

dir_os_tfidf_rf = gir_repo_dir+'/aisgna_os_tf_idf_rf.joblib'
dump(rfc, dir_os_tfidf_rf)
#%%% 5.5-SGDC (Stochastic Gradient Descent)
svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, random_state=42)
svm_sgd.fit(tv_train_features, over_train_label_nums)
svmsgd_tfidf_cv_scores = cross_val_score(svm_sgd, tv_train_features, over_train_label_nums, cv=5)
svmsgd_tfidf_cv_mean_score = np.mean(svmsgd_tfidf_cv_scores)
print('CV Accuracy (5-fold):', svmsgd_tfidf_cv_scores)
print('Mean CV Accuracy:', svmsgd_tfidf_cv_mean_score)
svmsgd_tfidf_test_score = svm_sgd.score(tv_test_features, test_label_nums)
print('Test Accuracy:', svmsgd_tfidf_test_score)

#%%% 5.6-Gradient boosting machine

gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(tv_train_features, over_train_label_nums)
gbc_tfidf_cv_scores = cross_val_score(gbc, tv_train_features, over_train_label_nums, cv=5)
gbc_tfidf_cv_mean_score = np.mean(gbc_tfidf_cv_scores)
print('CV Accuracy (5-fold):', gbc_tfidf_cv_scores)
print('Mean CV Accuracy:', gbc_tfidf_cv_mean_score)
gbc_tfidf_test_score = gbc.score(tv_test_features, test_label_nums)
print('Test Accuracy:', gbc_tfidf_test_score)

#%% B-Comparación
cuadro_over =  pd.DataFrame([['Naive Bayes', 
               mnb_bow_cv_mean_score, mnb_bow_test_score, 
               mnb_bng_bv_mean_score, mnb_bng_test_score, 
               mnb_tfidf_cv_mean_score, mnb_tfidf_test_score],
              ['Logistic Regression', 
               lr_bow_cv_mean_score, lr_bow_test_score, 
               lr_bng_bv_mean_score, lr_bng_test_score, 
               lr_tfidf_cv_mean_score, lr_tfidf_test_score],
              ['Linear SVM', 
               svm_bow_cv_mean_score, svm_bow_test_score, 
               svm_bng_bv_mean_score, svm_bng_test_score, 
               svm_tfidf_cv_mean_score, svm_tfidf_test_score],
              ['Linear SVM (SGD)', 
               svmsgd_bow_cv_mean_score, svmsgd_bow_test_score, 
               svmsgd_bng_bv_mean_score, svmsgd_bng_test_score, 
               svmsgd_tfidf_cv_mean_score, svmsgd_tfidf_test_score],
              ['Random Forest', 
               rfc_bow_cv_mean_score, rfc_bow_test_score, 
               rfc_bng_bv_mean_score, rfc_bng_test_score, 
               rfc_tfidf_cv_mean_score, rfc_tfidf_test_score],
              ['Gradient Boosted Machines', 
               gbc_bow_cv_mean_score, gbc_bow_test_score, 
               gbc_bng_bv_mean_score, gbc_bng_test_score, 
               gbc_tfidf_cv_mean_score, gbc_tfidf_test_score]],
             columns=['Model', 
                      'CV Score (BOW)', 'Test Score (BOW)', 
                      'CV Score (B-NGRAM)', 'Test Score (B-NGRAM)', 
                      'CV Score (TF-IDF)', 'Test Score (TF-IDF)'],
             ).T



1+1
