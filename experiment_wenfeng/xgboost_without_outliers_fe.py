# Version 1: XGBoost without outlier.
# Training result: [220] train-mae:0.0509354 test-mae:0.0526506
# Public score: 0.0645843
# LabelCountEncoder, Local CV: 0.05263960, Public score: 0.0645880
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from feature_utils import LabelCountEncoder
from feature_utils import OutlierEncoder

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

fe_columns = ['yearbuilt', 'taxamount']
properties.replace(to_replace=fe_columns, value=np.nan, inplace=True)

properties['yearbuilt'] = 2016 - properties['yearbuilt']

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

print('Building test set.')
test['parcelid'] = test['ParcelId']
df_test = test.merge(properties, how='left', on='parcelid')

def feature_scaler(x_train, df_test):
    #
    columns = ['taxamount', 'yearbuilt']
    for col in columns:
        scaler = RobustScaler()
        x_train[col] = scaler.fit_transform(x_train[[col]])[:, 0]
        df_test[col] = scaler.transform(df_test[[col]])[:, 0]

    return x_train, df_test

# x_train, df_test = feature_scaler(x_train, df_test)

y_train_mean = y_train.mean()

print('Training the model with cross validation.')
d_train = xgb.DMatrix(x_train, y_train.values)

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

print('Cross validation result, test-mae-mean = %.8f' % cv_result['test-mae-mean'].values[-1])
print('Use num_boost_rounds = %d' % num_boost_rounds)

def feature_outlier_explore(x_train, y_train, xgb_params):
    result = []
    for col in fe_columns:
        for method in ['iqr', 'spe']:
            for replace in ['mean', 'median', 'boundary', 'nan']:
                print 'col = %s, method = %s, replace = %s' % (col, method, replace)

                x_train_new = x_train.copy()
                outlier_encoder = OutlierEncoder(method=method, replace=replace)
                outlier_encoder.fit(x_train_new[col])
                x_train_new[col] = outlier_encoder.transform(x_train_new[col])

                d_train = xgb.DMatrix(x_train_new, y_train.values)
                # cross validation.
                cv_result = xgb.cv(
                    xgb_params, d_train, nfold=FOLDS, num_boost_round=350,
                    early_stopping_rounds=50, verbose_eval=10, show_stdv=False)

                print '%s,%s,%s,%.6f' % (col, method, replace, cv_result['test-mae-mean'].values[-1])
                result.append([col, method, replace, cv_result['test-mae-mean'].values[-1]])

    print 'Feature Engineering with OutlierEncoder.'
    print '\n'.join(','.join([str(e) for e in one]) for one in result)


def feature_scaler_explore(x_train, y_train, xgb_params):
    """
    # default localCV: 0.05264420
    # RobustScaler only with taxamount, LocalCV: 0.05264420, PB: 0.0645843
    # RobustScaler with taxamount & yearbuilt, LocalCV: 0.05263960, PB: 0.0645880
    """

    columns = ['taxamount', 'yearbuilt']
    result = []
    for col in columns:
        scaler = RobustScaler()
        x_train_new = x_train.copy()
        x_train_new[col] = scaler.fit_transform(x_train_new[[col]])[:, 0]

        d_train = xgb.DMatrix(x_train_new, y_train.values)
        # cross validation.
        cv_result = xgb.cv(
            xgb_params, d_train, nfold=FOLDS, num_boost_round=350,
            early_stopping_rounds=50, verbose_eval=10, show_stdv=False)

        print('Cross validation result, test-mae-mean = %.8f' % cv_result['test-mae-mean'].values[-1])
        result.append([col, cv_result['test-mae-mean'].values[-1]])

    print '\n'.join(','.join([str(e) for e in one]) for one in result)

# feature_scaler_explore(x_train, y_train, xgb_params)
feature_outlier_explore(x_train, y_train, xgb_params)
import sys
sys.exit(0)

model = xgb.train(
    dict(xgb_params, silent=1), d_train, num_boost_round=num_boost_rounds)

d_test = xgb.DMatrix(df_test[x_train.columns])

print('Predicting on test data.')
p_test = model.predict(d_test)
test = pd.read_csv('../../data/sample_submission.csv')
for column in test.columns[test.columns != 'ParcelId']:
    test[column] = p_test

print('Writing to csv.')
test.to_csv('../../data/xgboost_without_outlier.csv', index=False, float_format='%.4f')

print('Congratulation!!!')
