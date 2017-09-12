# Version 1: XGBoost without outlier.
# Training result: [220] train-mae:0.0509354 test-mae:0.0526506
# Public score: 0.0645843
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import cPickle
import time

OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4
FOLDS = 5
PICKLE_FILE = '../../data/xgboost_without_outliers_fe.p'

def save_data():
    start_time = time.time()
    print('Reading training data, properties and test data.')
    train = pd.read_csv("../../data/train_2016_v2.csv")
    properties = pd.read_csv('../../data/properties_2016.csv')
    test = pd.read_csv('../../data/sample_submission.csv')
    properties = properties.isnull()

    print('Encoding missing data.')
    for column in properties.columns:
        properties[column] = properties[column].fillna(-1)
        if properties[column].dtype == 'object':
            label_encoder = LabelEncoder()
            list_value = list(properties[column].values)
            label_encoder.fit(list_value)
            properties[column] = label_encoder.transform(list_value)

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

    cPickle.dump((x_train, y_train, df_test, ), open(PICKLE_FILE, 'wb'))

    print("--- %s seconds ---" % (time.time() - start_time))

def load_data():
    start_time = time.time()

    with open(PICKLE_FILE, 'rb') as fp:
      x_train, y_train, df_test = cPickle.load(fp)

    print("--- %s seconds ---" % (time.time() - start_time))

    return x_train, y_train, df_test

def run():
    x_train, y_train, df_test = load_data()
    y_train_mean = np.mean(y_train)

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

    print('Cross validation result, test-mae-mean = %.8f' % cv_result[' test-mae-mean'][-1])
    print('Use num_boost_rounds = %d' % num_boost_rounds)
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


if __name__ == "__main__":
    # save_data()
    run()
