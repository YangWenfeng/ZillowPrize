# Version 1: XGBoost without outlier.
# Public score: 0.0645843
# MeanEncoder regionidcity, regionidneighborhood, regionidzip
# Training result: [220] train-mae:0.0508244 test-mae:0.0526386
# Public score: 0.0645385
# try1: Add to MeanEncoder rawcensustractandblock, censustractandblock
# Training result: [200] train-mae:0.0508592 test-mae:0.0526378
# Public score: 0.0646074
# try2: Add to propertyzoningdesc
# Training result: [210] train-mae:0.0508218 test-mae:0.0526126
# Public score: 0.0645737
# try3: Add to propertycountylandusecode
# Training result: [210] train-mae:0.0508666 test-mae:0.0526252
# Public score:0.0645598
# try4: Add to propertyzoningdesc, propertycountylandusecode
# Training result: [230] train-mae:0.0506804 test-mae:0.052604
# Public score: 0.0645467
# Drop rawcensustractandblock, censustractandblock
# Training result: [230] train-mae:0.0507928 test-mae:0.0526542
# Public score: 0.0645721

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from mean_encoder import MeanEncoder

OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4
FOLDS = 5


print('Reading training data, properties and test data.')
train = pd.read_csv("../../data/train_2016_v2.csv")
properties = pd.read_csv('../../data/properties_2016.csv')
test = pd.read_csv('../../data/sample_submission.csv')

print('Encoding missing data.')
for column in properties.columns:
    properties[column] = properties[column].fillna(-1)
    if properties[column].dtype == 'object':
        label_encoder = LabelEncoder()
        list_value = list(properties[column].values)
        label_encoder.fit(list_value)
        properties[column] = label_encoder.transform(list_value)

print ('Mean encode data.')
mean_encoder = MeanEncoder(
    categorical_features=['regionidcity', 'regionidneighborhood', 'regionidzip'],
    target_type='regression'
)

print('Combining training data with properties.')
train_with_properties = train.merge(properties, how='left', on='parcelid')
print('Original training data with properties shape: {}'
      .format(train_with_properties.shape))

print('Dropping out outliers.')
train_with_properties = train_with_properties[
    train_with_properties.logerror > OUTLIER_LOWER_BOUND]
train_with_properties = train_with_properties[
    train_with_properties.logerror < OUTLIER_UPPER_BOUND]
print('New training data with properties without outliers shape: {}'
      .format(train_with_properties.shape))

print('Creating training and test data for xgboost.')
x_train = train_with_properties.drop(
    ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
     'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = train_with_properties['logerror']
y_train_mean = np.mean(y_train)

x_train = mean_encoder.fit_transform(x_train, y_train)
x_train = x_train.drop(mean_encoder.categorical_features, axis=1)

print('Training the model with cross validation.')
d_train = xgb.DMatrix(x_train, y_train)

# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_train_mean,
    'silent': 1
}

# cross validation.
cv_result = xgb.cv(
    xgb_params, d_train, nfold=FOLDS, num_boost_round=350,
    early_stopping_rounds=50, verbose_eval=10, show_stdv=False)
num_boost_rounds = int(round(len(cv_result) * np.sqrt(FOLDS/(FOLDS-1))))
model = xgb.train(
    dict(xgb_params, silent=1), d_train, num_boost_round=num_boost_rounds)

print('Building test set.')
test['parcelid'] = test['ParcelId']
df_test = test.merge(properties, how='left', on='parcelid')
df_test = mean_encoder.transform(df_test)
d_test = xgb.DMatrix(df_test[x_train.columns])


print('Predicting on test data.')
p_test = model.predict(d_test)
test = pd.read_csv('../../data/sample_submission.csv')
for column in test.columns[test.columns != 'ParcelId']:
    test[column] = p_test

print('Writing to csv.')
test.to_csv('../../data/xgboost_without_outlier.csv', index=False, float_format='%.4f')

print('Congratulation!!!')
