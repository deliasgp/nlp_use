# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:08:27 2021

@author: daniel
"""
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import spacy
import re
import unicodedata
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('spanish')
#-----------------------------------------------------------------------------*
nlp = spacy.load('es_core_news_sm')
#----------------------------------------------------------------------------*
from spellchecker import SpellChecker
#----------------------------------------------------------------------------*
def corregir_comentarios(comments):
    # Inicializar el corrector ortográfico
    spell = SpellChecker(language='es')

    # Corregir el comentario
    words = comments.split()
    corrected_words = []
    for word in words:
        # Si la palabra no está en el diccionario, corregir la ortografía
        if not spell.correction(word) == word and spell.correction(word) is not None:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    corrected_comment = ' '.join(corrected_words)

    # Devolver el comentario corregido
    return corrected_comment
#----------------------------------------------------------------------------*
def lemmatize_text(text):
    #La lematización es un proceso lingüístico que consiste en, 
    #dada una forma flexionada (es decir, en plural, en femenino, conjugada, etc), hallar el lema correspondiente. 
    #El lema es la forma que por convenio se acepta como representante de todas las formas flexionadas de una misma palabra
    text = nlp(text)    
    text = ' '.join([word.lemma_ for word in text])
    return text
#lemmatize_text(texto)
#----------------------------------------------------------------------------*
def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
#simple_porter_stemming(texto)
#----------------------------------------------------------------------------*
def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens
#correct_tokens = remove_repeated_characters(nltk.word_tokenize(a))
#' '.join(correct_tokens)
#----------------------------------------------------------------------------*
CONTRACTION_MAP = {
"a el": "al",
"de el": "del"
}
#----------------------------------------------------------------------------*
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
#----------------------------------------------------------------------------*
#def remove_accented_chars(text):
#    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#    return text
#-----------------------------------------------------------------------------*
def remove_accented_chars(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = unicodedata.normalize('NFC', text)
    return text
#----------------------------------------------------------------------------*
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]'
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(pattern, '', text)
    return text
#----------------------------------------------------------------------------*
def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#----------------------------------------------------------------------------*
def expandir_grados(text):
    text= re.sub("1ro","primero",text)
    text= re.sub("2do","segundo",text)
    text= re.sub("3ro","tercero",text)
    text= re.sub("4to","cuarto",text)
    text= re.sub("5to","quinto",text)
    text= re.sub("6to","primero",text)
    return text
#----------------------------------------------------------------------------*  
def remove_special_cases(text):
    text = re.sub('\S*\d+\S*', '', text) # CUALQUIER PALABRA QUE CONTENGA NÚMEROS
    text = re.sub('\s+', ' ', text) #MULTIPLES ESPACIOS EN BLANCO A UNO
    text = re.sub('seri[^. ]*\W', '', text) #Palabra serie
    text = re.sub('serí[^. ]*\W', '', text) #Palabra serie  
    text = re.sub('[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]|[0-9]', '', text) #NUMEROS Y PUNTUACION
    text = re.sub('fat\w+|falt\w+', 'falta', text) #variantes de la palabra falta
    text = re.sub('dificu\w+', 'dificultad', text) #variantes de la palabra falta
    text = re.sub('operativ\w+', 'operativas', text) #variantes de la palabra falta
    text = re.sub('\w+tabl\w+|tabl\w+', 'tableta', text) #variantes de la tablet
    text = re.sub('\w+decep\w+|decep\w+', 'recibió', text) #variantes de la         
    text = re.sub('hg[\w]*', '', text) #Palabra serie  
    text = re.sub('aoc*', '', text) #Palabra serie  
    text = re.sub('ugel', 'ugel ', text) #MULTIPLES ESPACIOS EN BLANCO A UNO
    text = re.sub('\s+', ' ', text) #MULTIPLES ESPACIOS EN BLANCO A UNO
    return text
#-----------------------------------------------------------------------------
def palabras_repetidas(text):
    text = text.split()
    text = ' '.join(sorted(set(text), key=text.index))
    return text    
#-----------------------------------------------------------------------------
def stop_words_use(local_file,maindir,label_benef=True,sw_cen_pob=True):
    #-----------------------*
    if local_file==True:
        archivo_dir = maindir 
    else:
        archivo_dir = "https://raw.githubusercontent.com/deliasgp/nlp_use/main/ubigeo_padron_web.csv" #maindir + 'ubigeo_padron_web.csv'
    #-----------------------*
    ubigeo_use = pd.read_csv(archivo_dir, encoding="latin-1")
    dias_mes = ["enero","febrero","marzo","abril","mayo","junio",
                "julio","agosto","setiembre","octubre","noviembre","diciembre",
                "lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
    
    numeros = ["uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve","diez",
               "once","doce","trece","catorce","quince","dieciséis","diecisiete","dieciocho","diecinueve",
               "veinte","veintiuno","veintidos","veintitres","veinticuatro","veinticinco","veintiseis","veintisiete","veintiocho","veintinueve",
               "treinta","cuarenta","cincuenta","sesenta","setenta","ochenta","noventa",
               "cien","doscientos","trescientos","cuatrocientos","quinientos","seiscientos","setecientos","ochocientos","novecientos","mil"]
    

    if label_benef==True:
        sw_1 = ["un","una",
                "el","la","los","las","para","de","en","y","que","vista",
                "a","e","i","o","u","niño","docente","niña","niñas","niños","me","nos",
                "primero","segundo","tercero","cuarto","quinto","sexto","por","to","ro","do","er","grado","grados",
                "estudiantes",'docentes',"alumnos","del","primaria","secundaria","multigrado","institución","institucion","ie","iiee",
                "alumnas","alumno","alumna","dotacion","dotación"] 
    else:
        sw_1 = ["a","e","i","o","u"]      
        
    d_dpto = np.unique(ubigeo_use['REGION']).tolist()
    for i in range(len(d_dpto)):
        d_dpto[i] = d_dpto[i].lower()
    #----------------------------*  
    d_prov = np.unique(ubigeo_use['D_PROV']).tolist()
    
    for i in range(len(d_prov)):
        d_prov[i] = d_prov[i].lower()
    #----------------------------*    
    d_dist = np.unique(ubigeo_use['D_DIST']).tolist()
    for i in range(len(d_dist)):
        d_dist[i] = d_dist[i].lower()
    #----------------------------*
    if sw_cen_pob==True:
        cen_pob = np.unique(ubigeo_use['CEN_POB']).tolist()
        for i in range(len(cen_pob)):
                     cen_pob[i] = cen_pob[i].lower()
                     #----------------------------*
    else:
        cen_pob = ["NA"]
    lista_stop_word = sw_1 + d_dpto + d_prov + d_dist + cen_pob + numeros + dias_mes
    return lista_stop_word
#----------------------------------------------------------------------------- 
def normalizar_texto(corpus, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_stemming=False, text_lemmatization=True, 
                     special_char_removal=True, remove_digits=True,
                     stopword_removal=True, special_cases = True,
                     autocorrecion = True,exp_grados = True,
                     stopwords=stopword_list):
    #largo = len(corpus)
    normalized_corpus = []
    # normalize each document in the corpus
    for i, doc in enumerate(corpus):        
        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))        
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        if exp_grados:
            doc = expandir_grados(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        if special_cases:
            doc = remove_special_cases(doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords=stopwords)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # autocorrecion
        if autocorrecion:
                doc=corregir_comentarios(doc)                       
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)        
        # stem text
        if text_stemming and not text_lemmatization:
        	doc = simple_porter_stemming(doc)       
        doc = re.sub(' +', ' ', doc)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        #print(i)    
        normalized_corpus.append(doc)
    #-------------------------------------------------------------------------*    
    return normalized_corpus
