"""
this script uses a RF classifier.
The cluster label of the days are the targets.
The weather/climate attributes of the days are the Xs.
"""

import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

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
df = df[(df['block'] != 453)]  # drop block 453
df.drop(columns=['day_34'], inplace=True)  # drop day 34

# get day labels
arr = np.genfromtxt('day_label.csv', delimiter=',')
day_label = list(map(int, arr.tolist()))
day_id = list(range(1, 52, 1))
day_id.remove(34)
day_dict = dict(zip(day_id, day_label))

df.drop(columns=['block'], inplace=True)
daily_avg_int = pd.DataFrame(df.sum())  # (day_i, average intensity)
index_list = daily_avg_int.index.str.split('_').tolist()
daily_avg_int['dayid'] = day_id
daily_avg_int['day_label'] = daily_avg_int['dayid'].map(day_dict)
# print(daily_avg_int)

query_t = "select w.dayid, avg(temper) as avg_t, stddev_samp(temper) as std_t " \
          "from s6039677.temperature as t, s6039677.weekends as w " \
          "where t.dtime = w.mdate and w.dayid in (%s) " \
          "group by w.dayid" % (','.join(map(str, day_id)))

query_r = "select w.dayid, avg(precip) as avg_r, stddev_samp(precip) as std_r " \
          "from s6039677.precipitation as r, s6039677.weekends as w " \
          "where r.dtime = w.mdate and w.dayid in (%s) " \
          "group by w.dayid" % (','.join(map(str, day_id)))

temper = pd.read_sql(query_t, conn)
temper.set_index('dayid')
rain = pd.read_sql(query_r, conn)
weater = temper.join(rain.set_index('dayid'), on='dayid')
# print(weater)

day_final = daily_avg_int.join(weater.set_index('dayid'), on='dayid')
# day_final['label'] = day_final['dayid']
# print(day_final)
# day_final.to_csv('days.csv')


# regression result is ridiculous
# cls?

y = day_final['day_label']
X = day_final.iloc[:, 3:]
labels = X.columns
print(labels)

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf.fit(train_features, train_labels)

std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for feature in zip(labels, clf.feature_importances_):
    print(feature)

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# confusion matrix
predicted = clf.predict(test_features)
matrix = confusion_matrix(test_labels, predicted)
print(matrix)

''' temporal cluster
('avg_t', 0.4800231803342435)
('std_t', 0.14554696852818994)
('avg_r', 0.24227828301127657)
('std_r', 0.13215156812629)

matrix: 
[[6 2]
 [0 5]]

accuracy: 11/13
'''