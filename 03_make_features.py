

'''
this script get the "Features" for all blocks, in the entire time frame (all 51 weekend days except day 34)
same thing can be done by changing line 56, 
for temporal cluster 0 and temporal cluster 1
don't forget to change the output file name
'''
import psycopg2
import numpy as np
import pandas as pd

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

query = "select * " \
        "from s6039677.intensity as intensity " \
        "where intensity.block not in (select block from s6039677.block_exclude) " \
        "order by intensity.block "

df = pd.read_sql(query, conn)

# make dayid to mdate dict
day_query = "select mdate, dayid " \
        "from s6039677.weekends"
day_df = pd.read_sql(day_query, conn)
mday_dict = dict(zip(day_df.dayid, day_df.mdate))
# print(mday_dict[1])

# query average temperature etc.
query_0 = "select t.block, avg(temper), stddev_samp(temper), avg(precip), stddev_samp(precip) " \
               "from s6039677.temperature as t, s6039677.precipitation as r " \
               "where (t.dtime in (select mdate from s6039677.weekends where dayid in (%s))) " \
               "    and (t.block in (select block from s6039677.block_uniq_new)) " \
          "         and (t.block not in (select block from s6039677.block_exclude))" \
          "         and (t.block = r.block)" \
          "         and (t.dtime = r.dtime)" \
               "group by t.block " %((', '.join(map(str,day_id))))  #(', '.join(map(str, reverse_label(day_dict, 1))))

print(query_0)
df_0 = pd.read_sql(query_0, conn)
print(df_0.head())

# query the other stable attributes
query_demo = "SELECT * " \
             "FROM s6039677.block_demo_new " \
             "WHERE block not in (select block from s6039677.block_exclude) " \
             "order by block"
demo = pd.read_sql(query_demo, conn)
demo.drop(columns=['id', 'geom', 'join_count', 'target_fid'], inplace=True)
print(demo.head())
final = df_0.join(demo.set_index('block'), on='block')

query_access = "select distinct block, total " \
               "from s6039677.road_new " \
               "order by block"
access = pd.read_sql(query_access, conn)
final = final.join(access.set_index('block'), on='block')

query_landuse = "select distinct block, major_landuse, reserve_percent " \
                "from s6039677.block_uniq_new " \
                "order by block"
landuse = pd.read_sql(query_landuse, conn)
final = final.join(landuse.set_index('block'), on='block')

print(final)
print(final.columns)
# final.drop(columns=['Unnamed: 0'], inplace=True)
final.to_csv('features_all_0919.csv')

conn.close()

