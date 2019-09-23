import pandas as pd
import psycopg2
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# make connection
try:
    conn = psycopg2.connect("dbname='c211' user='s6039677' host='gip.itc.utwente.nl' port='5434' password='_s6039677_'")
    cur = conn.cursor()
except:
    print('something went wrong, cannot connect to database')
    quit()

query_1 = "select observer, count(observer)" \
          "from s6039677.observation " \
          "group by observer"

df_1 = pd.read_sql(query_1, conn)
description_1 = stats.describe(df_1['count'])
print(description_1)
qqplot = stats.probplot(df_1['count'], plot=plt)
plt.show()
plt.hist(df_1['count'], bins=10, color='gray')
plt.show()
# conclusion: super skewed distribution

# log - log plots
query_2 = "with observers as" \
          "(select observer, count(observer) as n_observation " \
          "from s6039677.observation " \
          "group by observer) " \
          "select count(observer) as n_observers, n_observation " \
          "from observers " \
          "group by n_observation " \
          "order by n_observation "

df_2 = pd.read_sql(query_2, conn)
df_2['log_n_observation'] = np.log10(df_2['n_observation'])
df_2['log_n_observers'] = np.log10(df_2['n_observers'])
plt.scatter(df_2['log_n_observers'], df_2['log_n_observation'])
plt.show()


# '''something about the temporal distribution'''
query_3 = "select count(id), obsdate " \
         "from s6039677.observation " \
         "group by obsdate " \
         "order by obsdate"

df_3 = pd.read_sql(query_3, conn)
df_3['obsdate'] = pd.to_datetime(df_3['obsdate'])
# print(df_3.dtypes)
ax = plt.subplot(111)
ax.bar(df_3['obsdate'], df_3['count'], width=np.timedelta64(1, 'D'))
ax.xaxis_date()
plt.show()
# conclusion: higher spikes in weekend days -> more "meaningful" data on weekend days

# a spatial example to plot the data
query_4 = "select blc.block, count(obs.observer), blc.geom as geom " \
          "from s6039677.observation as obs, s6039677.block as blc " \
          "where obs.obsdate = '2017-05-21' and obs.block = blc.block " \
          "group by blc.block, geom"
df = gpd.GeoDataFrame.from_postgis(query_4, conn, geom_col='geom')
df.crs = {'init':'epsg:28992'}
# df = df.to_crs(epsg=4326)
# print(df.head())
df.plot(column='count')
plt.show()
df['log_n_observer'] = np.log10(df['count'])
plt.hist(df['log_n_observer'])
plt.show()
plt.hist(df['count'])
plt.show()
# conclusion: most block have very few observers

conn.close()