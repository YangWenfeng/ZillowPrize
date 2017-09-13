# Version 1: XGBoost without outlier.
# Training result: [220] train-mae:0.0509354 test-mae:0.0526506
# Public score: 0.0645843
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import time

OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4
FOLDS = 5
HDF_FILE = '../../data/xgboost_without_outliers_fe.h5'

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

    store = pd.HDFStore(HDF_FILE, 'w', complib=str('zlib'), complevel=5)
    store.put('x_train', x_train, data_columns=x_train.columns)
    store.put('y_train', y_train)
    store.put('df_test', df_test, data_columns=df_test.columns)
    store.close()

    print("--- %s seconds ---" % (time.time() - start_time))

def load_data():
    start_time = time.time()

    x_train = pd.read_hdf(HDF_FILE, 'x_train')
    y_train = pd.read_hdf(HDF_FILE, 'y_train')
    df_test = pd.read_hdf(HDF_FILE, 'df_test')

    print("--- %s seconds ---" % (time.time() - start_time))

    return x_train, y_train.values, df_test

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