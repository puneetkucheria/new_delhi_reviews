# import libraries
# import sqlite3
import sqlite3

#import pandas
import pandas as pd

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

exclude = string.punctuation                    # initiate punctuation removal list
stopwords_list = stopwords.words('english')     # initiate stopwords
stemmer = PorterStemmer()                       # initiate stemmer
lemmatizer = WordNetLemmatizer()                # initiate lemmatizer

# get data from the database
def get_data(table):
    # create sqlite connection
    con = sqlite3.connect('/Users/puneetkucheria/projects/data_science_course/capstone projects/Database.db')
    # print(pd.read_sql_query("SELECT * FROM sqlite_master", con))
    df = pd.read_sql_query("SELECT * FROM " + table + "", con) # New_Delhi_Reviews
    print("data imported")
    return df
# Remove punctuation 
def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))

# Remove stopwords
def remove_stopwords(text):
    data = [word for word in text.split() if word not in stopwords_list]
    return " ".join(data)

# apply Stemming
def apply_stemming(text):
    data = [stemmer.stem(word) for word in text.split()]
    return " ".join(data)

# apply lemma
def apply_lemma(text):
    data = [lemmatizer.lemmatize(word, pos='v') for word in text.split()] # to improve further you can use word specific pos
    return " ".join(data)
    # ['eat', 'mango'] - 'eat mango'

# Final data cleaning full
def text_data_cleaning(df):
    print("starting data cleaning")
    df['review_original'] = df['review_full'].copy()
    df['review_full'] = df['review_full'].str.lower()
    print('converted to lowercase')
    df['review_full'] = df['review_full'].apply(remove_punc)  
    print('removed punctuations')
    df['review_full'] = df['review_full'].apply(remove_stopwords)
    print('removed stopwords')
    df['review_full'] = df['review_full'].apply(apply_stemming)
    print('applied stemming')
    df['review_full'] = df['review_full'].apply(apply_lemma)
    print('applied lemmatization')
    return df

