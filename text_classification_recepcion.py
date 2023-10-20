# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:25:55 2023

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
obs_dir = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/observaciones/'+fec_t+'/obs_recepcion_'+fec_t+'.xlsx'
datos = pd.read_excel(obs_dir)
#-----------------------------------------------------------------------------*
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
text_corpus = nt.normalizar_texto(datos['OBSERVACION_RECEPCION'],
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
#datos['target'] = np.where(datos.flg_cat==97,1,0)

datos['flg_cat'].value_counts()

# Assuming df is your DataFrame
conditions = [datos['flg_cat'] == 1,
              datos['flg_cat'] == 2,
              datos['flg_cat'] == 3,
              datos['flg_cat'] == 4]

choices = [1, 2, 3, 4]

temp_rev = datos[datos['flg_cat']==16]

datos['target'] = pd.Series(pd.Categorical(np.select(conditions, choices, default=0)))
#----------------------------------------------------------------------------*
datos['target'].value_counts(normalize=True)
datos['target'].value_counts()
#----------------------------------------------------------------------------*
#temp_rev= datos[(datos['OBSERVACION_RECEPCION'].str.contains("dev"))]
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['obs'].str.contains("interne|datos|chip|conect|conex") & (datos['target']==0)) | (datos['flg_cat']==10),
                           3, #Dificultades con la conectividad de las tabletas
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['obs'].str.contains("cargador|cargar|elect|bater|energ")),
                           4, #Dificultades con la carga de las tableta
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['obs'].str.contains("dev")) | (datos['flg_cat']==16),
                           5, #Devoluciones
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==9) | (datos['flg_cat']==15) | (datos['flg_cat']==11)  | (datos['flg_cat']==18),
                           6, #Problemas con la recepción del equipo
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['flg_cat']==5) | (datos['flg_cat']==6) | (datos['flg_cat']==7)  | (datos['flg_cat']==8),
                           7, #Problemas con los registros y sistemas 
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']!=2) & 
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(ning).*(incon|observ)', regex=True)==False) & 
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(\bno\b).*(se).*(recep)', regex=True)),
                           6,
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'] = np.where((datos['target']!=2) & 
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(ning).*(incon|observ)', regex=True)==False) & 
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(\bno\b).*(se).*(recep)', regex=True)) | 
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(capac|orientac)', regex=True)),
                           6,
                           datos['target'])

datos['target'] = np.where((datos['flg_cat']==96) & (datos['target']==0) &
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(aun|falta|perdi)', regex=True)),
                           6,
                           datos['target'])


datos['target'] = np.where((datos['flg_cat']==96)  & (datos['target']==0) &
                           (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(amenaz|no lo requi|(no.*utiliza)|abiert)', regex=True)),
                           6,
                           datos['target'])
#----------------------------------------------------------------------------*
datos['flg_cat'] = np.where((datos['flg_cat']==96) & (datos['flg_bin']==1) & (datos['target']==0) &
                            (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(dotaci|fecha|lleg|pecosa|instituci|recep|asigna)', regex=True)) & 
                            (datos['OBSERVACION_RECEPCION'].str.contains(r'(?i)(aun|falta|perdi)', regex=True)==False),
                           97,
                           datos['flg_cat'])

datos['target'] = np.where((datos['flg_cat']!=97)  & (datos['target']==0),
                           6,
                           datos['target'])
#----------------------------------------------------------------------------*
datos['target'].value_counts(normalize=True).sort_index()
datos['target'].value_counts().sort_index()
pd.crosstab(datos['flg_cat'],datos['target'])
#----------------------------------------------------------------------------*
result = datos['target'].value_counts(normalize=True).reset_index()
result.columns = ['Value', 'Proportion']
result['Count'] = result['Value'].map(datos['target'].value_counts())
print(result)


#1 Faltan o necesitan más equipos
#2 Tabletas con defectos técnicos
#3 Tabletas sin plan de datos
#4 Dificultades con la carga de las tableta
#5 Comentarios sobre devolución de equipos a IE o UGEL
#6 Problemas con la recepción del equipo
#7 Problemas con los registros y sistemas 
datos = datos[(datos['flg_cat']!=97) & (datos['target']!=0)]   
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
datos['target'].value_counts().sort_index()
datos['target'].value_counts(normalize=True).sort_index() #print(np.mean(df_temp['target']))
#--------*
print(result)
count_class_1,  count_class_2, count_class_3, count_class_4, count_class_5, count_class_6, count_class_7= df_temp.target.value_counts().sort_index()


df_class_1 = df_temp[df_temp['target'] == 1]
df_class_2 = df_temp[df_temp['target'] == 2]
df_class_3 = df_temp[df_temp['target'] == 3]
df_class_4 = df_temp[df_temp['target'] == 4]
df_class_5 = df_temp[df_temp['target'] == 5]
df_class_6 = df_temp[df_temp['target'] == 6]
df_class_7 = df_temp[df_temp['target'] == 7]

df_class_1_sample = df_class_1
df_class_2_sample = df_class_2.sample(count_class_1-count_class_2,replace=True)
df_class_3_sample = df_class_3.sample(count_class_1-count_class_3,replace=True)
df_class_4_sample = df_class_4.sample(count_class_1-count_class_4,replace=True)
df_class_5_sample = df_class_5.sample(count_class_1-count_class_5,replace=True)
df_class_6_sample = df_class_6.sample(count_class_1-count_class_6,replace=True)
df_class_7_sample = df_class_7.sample(count_class_1-count_class_7,replace=True)

df_train_over = pd.concat([df_class_2_sample,
                           df_class_3_sample,df_class_4_sample,df_class_5_sample,
                           df_class_6_sample,df_class_7_sample], axis=0)

df_train_over = pd.concat([df_train_over,                            
                           df_class_1,
                           df_class_2,                           
                           df_class_3,
                           df_class_4,
                           df_class_5,
                           df_class_6,
                           df_class_7], axis=0)



print('Random under-sampling:')
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
#%%% B-2.1 - Bag of Words (term frequency) model:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(over_train_corpus)
cv_test_features = cv.transform(test_corpus)

gir_repo_dir = 'D:/repositorios_git/nlp_use/'
dir_os_bow = gir_repo_dir+'/recep_os_bow.pickle'
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

gir_repo_dir = 'D:/repositorios_git/nlp_use/'
dir_os_tfidf = gir_repo_dir+'/recep_os_tf_idf.pickle'
with open(dir_os_tfidf, 'wb') as f:
    pickle.dump(tv, f)
    
tv_test_features = tv.transform(test_corpus)
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
dir_os_svm_rf = gir_repo_dir+'/recepcion_os_svm.joblib'
dump(svm, dir_os_svm_rf)
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

gir_repo_dir = 'D:/repositorios_git/nlp_use/'
dir_os_bow_rf = gir_repo_dir+'/modelo_entrenado_bow_rf.joblib'
dump(rfc, dir_os_bow_rf)
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

gir_repo_dir = 'D:/repositorios_git/nlp_use/'
dir_os_tfidf_svm = gir_repo_dir+'/recepcion_os_tf_idf_svm.joblib'
dump(svm, dir_os_tfidf_svm)
#%%% 5.4-Random forest
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(tv_train_features, over_train_label_nums)
rfc_tfidf_cv_scores = cross_val_score(rfc, tv_train_features, over_train_label_nums, cv=5)
rfc_tfidf_cv_mean_score = np.mean(rfc_tfidf_cv_scores)
print('CV Accuracy (5-fold):', rfc_tfidf_cv_scores)
print('Mean CV Accuracy:', rfc_tfidf_cv_mean_score)
rfc_tfidf_test_score = rfc.score(tv_test_features, test_label_nums)
print('Test Accuracy:', rfc_tfidf_test_score)


#gir_repo_dir = 'D:/repositorios_git/nlp_use/'
#dir_os_tfidf_rf = gir_repo_dir+'/modelo_entrenado_2_os_tf_idf_rf.joblib'
#dump(rfc, dir_os_tfidf_rf)
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
