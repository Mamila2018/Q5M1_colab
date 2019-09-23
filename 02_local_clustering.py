'''
this script does spatial clustering within the temporal clusters
the number of clusters remain the same as the script "global_clustering"
'''

import psycopg2
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.cluster import KMeans


# get day labels
arr = genfromtxt('day_label.csv', delimiter=',')
day_label = list(map(int, arr.tolist()))
day_id = list(range(1, 52, 1))
day_id.remove(34)
day_dict = dict(zip(day_id, day_label))
def reverse_label(base, searchvalue):
    result = list()
    for key, value in base.items():
        if value == searchvalue:
            result.append(key)
    return result

# make connection
try:
    conn = psycopg2.connect("dbname='c211' user='s6039677' host='gip.itc.utwente.nl' port='5434' password='_s6039677_'")
    cur = conn.cursor()
except:
    print('something went wrong, cannot connect to database')
    quit()

query = "select block, %s " \
        "from s6039677.intensity as intensity " \
        "where intensity.block not in (select block from s6039677.block_exclude) " \
        "order by intensity.block " %(','.join(map((lambda x: 'day_'+str(x)), reverse_label(day_dict, 1))))

df = pd.read_sql(query, conn)
df = df[(df['block'] != 453)]  # drop day 453
# print(df)
col_list= list(df)
col_list.remove('block')
df['sum_'] = df[col_list].sum(axis=1, skipna=True)
print(len(df[(df['sum_']==0)]))

# remove rows with ALL zeros - blocks that NEVER had observations
df_no_all_zero = df[(df['sum_'] != 0)]
# print(len(df_0))
print(len(df_no_all_zero))
# print(df_no_all_zero.iloc[:, 1:-1])
# print(df_no_all_zero)

clusterer = KMeans(n_clusters=2)
spatial_df = df[['block']]
df_no_all_zero['cluster_labels'] = clusterer.fit_predict(df_no_all_zero.iloc[:, 1:-1])
spatial_df = spatial_df.join(df_no_all_zero.set_index('block'), on='block')
spatial_df.fillna(2, inplace=True)
spatial_df = spatial_df[['block', 'cluster_labels']]
print(len(spatial_df[(spatial_df['cluster_labels']==0)]), len(spatial_df[(spatial_df['cluster_labels']==1)]), len(spatial_df[(spatial_df['cluster_labels']==2)]), len(spatial_df[(spatial_df['cluster_labels']==3)]))
spatial_df.to_csv('spatialcluster_2_zero_as_class_temporal_label_1.csv')


# do the same thing with day label 0
'''
try:
    conn = psycopg2.connect("dbname='c211' user='s6039677' host='gip.itc.utwente.nl' port='5434' password='_s6039677_'")
    cur = conn.cursor()
except:
    print('something went wrong, cannot connect to database')
    quit()

query = "select block, %s " \
        "from s6039677.intensity as intensity " \
        "where intensity.block not in (select block from s6039677.block_exclude) " \
        "order by intensity.block " %(','.join(map((lambda x: 'day_'+str(x)), reverse_label(day_dict, 0))))

df = pd.read_sql(query, conn)
df = df[(df['block'] != 453)]  # drop day 453
# print(df)
col_list= list(df)
col_list.remove('block')
df['sum_'] = df[col_list].sum(axis=1, skipna=True)
print(len(df[(df['sum_']==0)]))
# remove rows with ALL zeros - blocks that NEVER had observations
df_no_all_zero = df[(df['sum_'] != 0)]
# print(len(df_0))
print(len(df_no_all_zero))
clusterer = KMeans(n_clusters=2)
spatial_df = df[['block']]
df_no_all_zero['cluster_labels'] = clusterer.fit_predict(df_no_all_zero.iloc[:, 1:-1])
spatial_df = spatial_df.join(df_no_all_zero.set_index('block'), on='block')
spatial_df.fillna(2, inplace=True)
spatial_df = spatial_df[['block', 'cluster_labels']]
print(len(spatial_df[(spatial_df['cluster_labels']==0)]), len(spatial_df[(spatial_df['cluster_labels']==1)]), len(spatial_df[(spatial_df['cluster_labels']==2)]), len(spatial_df[(spatial_df['cluster_labels']==3)]))
spatial_df.to_csv('spatialcluster_2_zero_as_class_temporal_label_0.csv')
'''
