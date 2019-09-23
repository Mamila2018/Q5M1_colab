"""
this script does the same thing as the "global_spatial_classify", the only difference
is that it does classification WITHIN the temporal clusters.

!!!! IMPORTANT !!!!
change the input for df and features accordingly in LINE 18 AND 32
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv('spatialcluster_2_zero_as_class_temporal_label_1.csv')
df.cluster_labels = df.cluster_labels.astype('int32')
df.drop(columns=['Unnamed: 0'], inplace=True)

features = pd.read_csv('features_1_0919.csv')
features.drop(columns=['Unnamed: 0'], inplace=True)
# features.drop(columns=['Unnamed: 0', 'longit', 'latit'], inplace=True)
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

print(len(features[(features['cluster_labels']==0)]), len(features[(features['cluster_labels']==1)]), len(df[(df['cluster_labels']==2)]))

X = features.iloc[:, :-1]
y = features[['cluster_labels']]

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


'''temporal label 1 (the earlier part of the half year)
('avg', 0.08157282976783872)
('stddev_samp', 0.08088890486552329)
('avg.1', 0.06874508655809879)
('stddev_samp.1', 0.06821911723162924)
('aantal_inw', 0.08443472203924422)
('stedelijkh', 0.00701577236656282)
('total', 0.08496876560746157)
('reserve_percent', 0.03995358123358736)
('landuse', 0.014823042062963962)
('percent_nl', 0.07148679868372347)
('percent_west', 0.0732098427434012)
('percent_1', 0.061875427674116114)
('percent_2', 0.06495478971172275)
('percent_3', 0.06884048737077071)
('percent_4', 0.06461721700139257)
('percent_5', 0.06439361508196324)

matrix:
[[4000    0 1030]
 [  13    0    0]
 [1314    0 2191]]
 
accuracu: 72
'''

'''for temporal label 0 (later part of the half year
('avg', 0.07687549136394761)
('stddev_samp', 0.07250453328004658)
('avg.1', 0.07051421408047023)
('stddev_samp.1', 0.06828552877208309)
('aantal_inw', 0.08097773100706748)
('stedelijkh', 0.006365290883905281)
('total', 0.08796482375237728)
('reserve_percent', 0.04502212305686992)
('landuse', 0.019285767083778917)
('percent_nl', 0.07522729892521254)
('percent_west', 0.07249007439741073)
('percent_1', 0.06353584253646591)
('percent_2', 0.06402325821269796)
('percent_3', 0.06589677042089409)
('percent_4', 0.06678329221295018)
('percent_5', 0.06424796001382228)

matrix:
[[4741    0  801]
 [  57    1    2]
 [1417    0 1529]]

accuracy 73
'''