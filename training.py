from data_processing_features import get_data, text_data_cleaning
from model_building import tfidf_features_fit
import pandas as pd
from sklearn.cluster import KMeans
from rake_nltk import Rake

# get data
df = get_data('New_Delhi_Reviews')

# clean data 
df = text_data_cleaning(df)

tfidf, tfidf_matrix = tfidf_features_fit(df)

df_clustering = pd.merge(df['rating_review'],tfidf_matrix,left_index=True,right_index=True, how='inner')

kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(df_clustering)

df_clustering['cluster_labels'] = kmeans.labels_

merged_df = pd.merge(df, df_clustering['cluster_labels'], left_index=True, right_index=True, how='inner')





print(df_clustering.head())







