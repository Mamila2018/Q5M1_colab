'''
this script runs a random forest classifier
the label (from the spatial clustering in the previous step) is the target
the features are made from the script "make_features"
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv('spatialcluster_without_zero_3_clusters_0917.csv')
df.cluster_labels = df.cluster_labels.astype('int32')
df.drop(columns=['Unnamed: 0'], inplace=True)

# get the some new features, landuse into category and do some percentages
features = pd.read_csv('features_all_0919.csv')
features.drop(columns=['Unnamed: 0'], inplace=True)
# features.drop(columns=['longit', 'latit', 'cluster'], inplace=True)  # for not 0919
features.major_landuse = features.major_landuse.astype('category')
features['landuse'] = features['major_landuse'].cat.codes
features.drop(columns=['major_landuse'], inplace=True)
features['percent_nl'] = features['n_dutch_2']/features['aantal_inw']
features['percent_west'] = features['n_west_2']/features['aantal_inw']
features.drop(columns=['n_dutch_2', 'n_west_2'], inplace=True)
features.dropna(inplace=True)

# doubt
features['percent_1'] = features['aantal_i_1']/features['aantal_inw']
features['percent_2'] = features['aantal_i_2']/features['aantal_inw']
features['percent_3'] = features['aantal_i_3']/features['aantal_inw']
features['percent_4'] = features['aantal_i_4']/features['aantal_inw']
features['percent_5'] = features['aantal_i_5']/features['aantal_inw']
features.drop(columns=['aantal_i_1', 'aantal_i_2', 'aantal_i_3', 'aantal_i_4', 'aantal_i_5'], inplace=True)

features = features.join(df.set_index('block'), on='block')
features.drop(columns=['block'], inplace=True)
features.dropna(inplace=True)


print(features.columns)

X = features.iloc[:, :-1]
y = features[['cluster_labels']]
# print(X.columns)

labels = X.columns

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

clf = RandomForestClassifier(n_estimators=5000, random_state=0, n_jobs=-1)
clf.fit(train_features, train_labels)


std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for feature in zip(labels, clf.feature_importances_):
    print(feature)

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
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


# result on aggregated population
'''feature importance after dropped some population variables, 1000 estimator
('avg', 0.1173008357753718)
('stddev_samp', 0.11943687488759909)
('avg.1', 0.11532296267135053)
('stddev_samp.1', 0.10792705271351376)
('aantal_inw', 0.11727764927676346)
('stedelijkh', 0.009954735189534322)
('total', 0.12648868821371428)
('reserve_percent', 0.04068384160431885)
('landuse', 0.018258739928565387)
('percent_nl', 0.11318613810361752)
('percent_west', 0.11416248163565101)

matrix
[[6111    0  425]
 [  44    1    0]
 [1271    0  696]]
accuracy: 79.6
'''


'''with age percentages, 5000 estimator
('avg', 0.07556878697207889)
('stddev_samp', 0.07708839634202386)
('avg.1', 0.0761630347713674)
('stddev_samp.1', 0.06918579731309388)
('aantal_inw', 0.07643460046304114)
('stedelijkh', 0.0071733615537954215)
('total', 0.09153377746804521)
('reserve_percent', 0.03589650204232111)
('landuse', 0.01459672540176017)
('percent_nl', 0.07493958161882901)
('percent_west', 0.07526020484353001)
('percent_1', 0.06360856404590548)
('percent_2', 0.06454469985596434)
('percent_3', 0.06605927947008046)
('percent_4', 0.06602560561104474)
('percent_5', 0.06592108222711891)

matrix
[[6141    0  395]
 [  43    2    0]
 [1283    0  684]]
accuracy 80

# 10000 estimator similar score and only slightly worse accuracy
'''




# result on block population not aggregated
'''
('avg', 0.1185183569863746)
('stddev_samp', 0.1169490226218834)
('avg.1', 0.11019133687991986)
('stddev_samp.1', 0.10242022301815859)
('aantal_inw', 0.06412265234093706)
('stedelijkh', 0.0130213178740376)
('total', 0.13118642426586186)
('reserve_percent', 0.035063946429038005)
('landuse', 0.01386187357668364)
('percent_nl', 0.05903727848753882)
('percent_west', 0.025199801678956024)
('percent_1', 0.03951096351170991)
('percent_2', 0.03719072985449814)
('percent_3', 0.03932696990574016)
('percent_4', 0.05242507768913195)
('percent_5', 0.041974024879530496)

matrix
[[4887    0  280]
 [  23    0    1]
 [1183    0  429]]
 
accuracy: 78
'''

