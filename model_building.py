
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from rake_nltk import Rake

tfidf = TfidfVectorizer(min_df=0.01,max_df=0.1)
r = Rake()

def tfidf_features_fit(df):
    tfidf_matrix = tfidf.fit_transform(df['review_full'])
    return tfidf,pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf.get_feature_names_out())

# tfidf transform
def tfidf_features_transform(df):
    tfidf_matrix = tfidf.transform(df['review_full'])
    return tfidf,pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf.get_feature_names_out())

# implement RAKE
def rake_features(df):
    r.extract_keywords_from_text(df['review_full'])
    return r.get_ranked_phrases_with_scores()