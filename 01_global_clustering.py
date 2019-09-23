'''
this script makes clusters over the intensity data.
Both spatially and temporally.
First get optimal K with the silhuouette method
Then cluster with that K.
Blocks with zero intensity over all days are removed from spatial clustering.
During the clustering, a few "outliers" are found. They are removed from the dataset before clustering
These "outliers" will be analyzed separately. 
'''

import numpy as np
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.metrics import consensus_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# make connection
try:
    conn = psycopg2.connect("dbname='c211' user='s6039677' host='gip.itc.utwente.nl' port='5434' password='_s6039677_'")
    cur = conn.cursor()
except:
    print('something went wrong, cannot connect to database')
    quit()

query = "select * " \
        "from s6039677.intensity as intensity " \
        "where intensity.block not in (select block from s6039677.block_exclude) " \
        "order by intensity.block "

df = pd.read_sql(query, conn)
print(len(df))
df = df[(df['block'] != 453)]
col_list= list(df)
col_list.remove('block')
df['sum_'] = df[col_list].sum(axis=1, skipna=True)
# print(df.sum_.max(), df.sum_.min())
print(len(df[(df['sum_']==0)]))

# remove rows with ALL zeros - blocks that NEVER had observations
df_no_all_zero = df[(df['sum_'] != 0)]
# print(len(df_0))
print(len(df_no_all_zero))

'''
# looking for an optimal K with the silhuouette method
df.drop(columns=['block', 'sum_'], inplace=True)
df_no_all_zero.drop(columns=['block', 'sum_'], inplace=True)

# temporal
df_t = df.T
df_no_all_zero_t = df_no_all_zero.T
range_n_clusters = [2, 3, 4]
for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(df_t)
    centers = clusterer.cluster_centers_

    score = silhouette_score (df_t, preds, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


# spatial
range_n_clusters = [2, 3, 4]
print(df_no_all_zero.head())
for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(df_no_all_zero)
    centers = clusterer.cluster_centers_

    score = silhouette_score (df_no_all_zero, preds, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
'''

''' result: 
temporal: 
For n_clusters = 2, silhouette score is 0.5467139278724319)
For n_clusters = 3, silhouette score is 0.0736959907015903)
For n_clusters = 4, silhouette score is 0.06287528298070764))

spatial: 
For n_clusters = 2, silhouette score is 0.9086495825405276)
For n_clusters = 3, silhouette score is 0.814294919403279)
For n_clusters = 4, silhouette score is 0.814013631579899)
'''

# temporal cluster
# dropped day 34 as outlier
df.drop(columns=['block', 'sum_', 'day_34'], inplace=True)
df_t = df.T
clusterer = KMeans(n_clusters=2)
preds = clusterer.fit_predict(df_t)
print(preds)
np.savetxt('day_label.csv', [preds], delimiter=',', fmt='%d')
'''cluster label without day 34!
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 0 0]
 '''

# quit()

# spatial cluster
clusterer = KMeans(n_clusters=2)
spatial_df = df[['block']]
df_no_all_zero['cluster_labels'] = clusterer.fit_predict(df_no_all_zero.iloc[:, 1:-1])
spatial_df = spatial_df.join(df_no_all_zero.set_index('block'), on='block')
spatial_df.fillna(2, inplace=True)
spatial_df = spatial_df[['block', 'cluster_labels']]
print(len(spatial_df[(spatial_df['cluster_labels']==0)]), len(spatial_df[(spatial_df['cluster_labels']==1)]), len(spatial_df[(spatial_df['cluster_labels']==2)]), len(spatial_df[(spatial_df['cluster_labels']==3)]))
spatial_df.to_csv('spatialcluster_without_zero_3_clusters_0917.csv')
