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
features.major_landuse = features.major_landuse.astype('category')
features['landuse'] = features['major_landuse'].cat.codes
features.drop(columns=['major_landuse'], inplace=True)
features['percent_nl'] = features['n_dutch_2']/features['aantal_inw']
features['percent_west'] = features['n_west_2']/features['aantal_inw']
features.drop(columns=['n_dutch_2', 'n_west_2'], inplace=True)
features.dropna(inplace=True)

features = features.join(df.set_index('block'), on='block')
features.drop(columns=['block'], inplace=True)
features.dropna(inplace=True)

# doubt
features.drop(columns=['aantal_i_1', 'aantal_i_2', 'aantal_i_3', 'aantal_i_4', 'aantal_i_5'], inplace=True)
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

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
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


# result
'''feature importance after dropped some population variables
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

