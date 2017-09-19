# Version 1: XGBoost without outlier.
# Training result: [220] train-mae:0.0509354 test-mae:0.0526506
# Public score: 0.0645843
# FE OutlierEncoder taxamount/yearbuilt 0.0646074
# FE Merge without MeanEncoder, PB


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from feature_utils import LabelCountEncoder, FeatureInteraction
from feature_utils import OutlierEncoder
from mean_encoder import MeanEncoder

OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4
FOLDS = 5

debug = True

def get_xgb_params(y_train_mean):
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
    return xgb_params

def reset_nan(df, columns):
    for col in columns:
        df[col].replace(to_replace=-1, value=np.nan, inplace=True)
    return df

def get_train_test_data():
    print('Reading training data, properties and test data.')
    train = pd.read_csv("../../data{}/train_2016_v2.csv".format('_debug' if debug else ''))
    properties = pd.read_csv('../../data{}/properties_2016.csv'.format('_debug' if debug else ''))
    test = pd.read_csv('../../data{}/sample_submission.csv'.format('_debug' if debug else ''))

    print('Encoding missing data.')
    for column in properties.columns:
        properties[column].fillna(-1, inplace=True)
        if properties[column].dtype == 'object':
            label_encoder = LabelEncoder()
            list_value = list(properties[column].values)
            label_encoder.fit(list_value)
            properties[column] = label_encoder.transform(list_value)

    # yearbuilt
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

    x_train.fillna(-1, inplace=True)
    df_test.fillna(-1, inplace=True)

    return x_train, y_train, df_test

def xgboost_cross_validation(x_train, y_train):
    xgb_params = get_xgb_params(y_train.mean())

    print('Training the model with cross validation.')
    d_train = xgb.DMatrix(x_train, y_train.values)

    # cross validation.
    num_boost_round = 350 if not debug else 10
    cv_result = xgb.cv(
        xgb_params, d_train, nfold=FOLDS, num_boost_round=num_boost_round,
        early_stopping_rounds=50, verbose_eval=10, show_stdv=False)
    num_boost_rounds = int(round(len(cv_result) * np.sqrt(FOLDS/(FOLDS-1))))

    best_score = cv_result['test-mae-mean'].values[-1]
    print('Cross validation best score, test-mae-mean = %.8f' % best_score)
    print('Use num_boost_rounds = %d' % num_boost_rounds)
    return best_score, num_boost_rounds

def train_and_predict(x_train, y_train, df_test, num_boost_rounds, output_suffix=''):
    print('Training the model and predict.')
    y_train_mean = y_train.mean()

    # xgboost params
    xgb_params = get_xgb_params(y_train.mean())

    d_train = xgb.DMatrix(x_train, y_train.values)

    model = xgb.train(
        dict(xgb_params, silent=1), d_train, num_boost_round=num_boost_rounds)

    d_test = xgb.DMatrix(df_test[x_train.columns])

    print('Predicting on test data.')
    p_test = model.predict(d_test)
    test = pd.read_csv('../../data{}/sample_submission.csv'.format('_debug' if debug else ''))
    for column in test.columns[test.columns != 'ParcelId']:
        test[column] = p_test

    output_file = '../../data/xgboost_without_outlier{}.csv'.format(output_suffix)
    print('Writing to csv [%s].' % output_file)
    test.to_csv(output_file, index=False, float_format='%.4f')

def feature_scaler(x_train, df_test):
    print 'Feature Scaler.'
    columns = ['taxamount', 'yearbuilt']
    for col in columns:
        scaler = RobustScaler()
        x_train[col] = scaler.fit_transform(x_train[[col]])[:, 0]
        df_test[col] = scaler.transform(df_test[[col]])[:, 0]

    return x_train, df_test

def feature_outlier(x_train, df_test):
    fe_columns = ['taxamount', 'yearbuilt']

    # reset nan
    x_train = reset_nan(x_train, fe_columns)
    df_test = reset_nan(df_test, fe_columns)

    print 'Feature Outlier.'
    outlier_encoder = OutlierEncoder(method='iqr', replace='median')
    col = 'yearbuilt'
    outlier_encoder.fit(x_train[col])
    x_train[col] = outlier_encoder.transform(x_train[col])
    df_test[col] = outlier_encoder.transform(df_test[col])

    outlier_encoder = OutlierEncoder(method='spe', replace='mean')
    col = 'taxamount'
    outlier_encoder.fit(x_train[col])
    x_train[col] = outlier_encoder.transform(x_train[col])
    df_test[col] = outlier_encoder.transform(df_test[col])

    # x_train.fillna(-1, inplace=True)
    # df_test.fillna(-1, inplace=True)

    return x_train, df_test

def explore_feature_outlier():
    print "Explore feature outlier."
    x_train, y_train, df_test = get_train_test_data()

    fe_columns = ['taxamount', 'yearbuilt']

    # reset nan
    x_train = reset_nan(x_train, fe_columns)

    result = []
    for col in fe_columns:
        for method in ['iqr', 'spe']:
            for replace in ['mean', 'median', 'boundary', 'nan']:
                if replace == 'boundary' and method != 'iqr':
                    continue
                print 'col = %s, method = %s, replace = %s' % (col, method, replace)

                x_train_new = x_train.copy()
                outlier_encoder = OutlierEncoder(method=method, replace=replace)
                outlier_encoder.fit(x_train_new[col])
                x_train_new[col] = outlier_encoder.transform(x_train_new[col])

                best_score, _ = xgboost_cross_validation(x_train_new, y_train)
                result.append([col, method, replace, best_score])

    print 'Feature Engineering with OutlierEncoder.'
    print '\n'.join(','.join([str(e) for e in one]) for one in result)


def explore_feature_scaler():
    """
    # default localCV: 0.05264420
    # RobustScaler only with taxamount, LocalCV: 0.05264420, PB: 0.0645843
    # RobustScaler with taxamount & yearbuilt, LocalCV: 0.05263960, PB: 0.0645880
    """
    print "Explore feature scaler."
    x_train, y_train, df_test = get_train_test_data()
    fe_columns = ['taxamount', 'yearbuilt']

    result = []
    for col in fe_columns:
        scaler = RobustScaler()
        x_train_new = x_train.copy()
        x_train_new[col] = scaler.fit_transform(x_train_new[[col]])[:, 0]

        best_score, _ = xgboost_cross_validation(x_train_new, y_train)
        result.append([col, best_score])

    print '\n'.join(','.join([str(e) for e in one]) for one in result)

def explore_feature_interaction():
    x_train, y_train, df_test = get_train_test_data()

    print 'do_mean_encoder'
    x_train, y_train, df_test = FeatureInteraction().do_mean_encoder(x_train, y_train, df_test)

    result = []

    print 'add_feature_missing_count'
    x_train_new, _ = FeatureInteraction().add_feature_missing_count(x_train.copy(), df_test.copy())
    best_score, _ = xgboost_cross_validation(x_train_new, y_train)
    result.append(['add_feature_missing_count', best_score])
    print 'result', ','.join(str(e) for e in result[-1])

    print 'add_regionidzip_centroid_distance'
    x_train_new, _ = FeatureInteraction().add_regionidzip_centroid_distance(x_train.copy(), df_test.copy())
    best_score, _ = xgboost_cross_validation(x_train_new, y_train)
    result.append(['regionidzip_centroid_distance', best_score])
    print 'result', ','.join(str(e) for e in result[-1])

    print 'add_feature_division'
    for col1 in ['structuretaxvaluedollarcnt', 'taxamount', 'landtaxvaluedollarcnt']:
        for col2 in ['calculatedfinishedsquarefeet', 'lotsizesquarefeet']:
            x_train_new, _ = FeatureInteraction().add_feature_division(x_train.copy(), df_test.copy(), col1, col2)
            best_score, _ = xgboost_cross_validation(x_train_new, y_train)
            result.append(['add_feature_division_%s_div_%s' % (col1, col2), best_score])
            print 'result', ','.join(str(e) for e in result[-1])

    print 'add_feature_mod'
    for col in ['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']:
        for num in [10, 100, 1000, 10000]:
            x_train_new, _ = FeatureInteraction().add_feature_mod(x_train.copy(), df_test.copy(), col, num)
            best_score, _ = xgboost_cross_validation(x_train_new, y_train)
            result.append(['add_feature_mod_%s_mod_%d' % (col, num), best_score])
            print 'result', ','.join(str(e) for e in result[-1])

    for col in ['lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12']:
        for num in [10, 20, 50, 100]:
            x_train_new, _ = FeatureInteraction().add_feature_mod(x_train.copy(), df_test.copy(), col, num)
            best_score, _ = xgboost_cross_validation(x_train_new, y_train)
            result.append(['add_feature_mod_%s_mod_%d' % (col, num), best_score])
            print 'result', ','.join(str(e) for e in result[-1])

    print 'add_feature_around'
    for decimals in [0, 1, 2, 3, 4, 5, 6, 7]:
        x_train_new, tmp = FeatureInteraction().add_feature_around(
            x_train.copy(), df_test.copy(), 'latitude', decimals)
        x_train_new, _ = FeatureInteraction().add_feature_around(
            x_train_new, tmp, 'longitude', decimals)
        best_score, _ = xgboost_cross_validation(x_train_new, y_train)
        result.append(['add_feature_around_%s_%d' % ('latitudelongitude', decimals), best_score])
        print 'result', ','.join(str(e) for e in result[-1])

    print 'add_feature_log'
    for col in ['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount',
                'lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12']:
        x_train_new, _ = FeatureInteraction().add_feature_log(x_train.copy(), df_test.copy(), col)
        best_score, _ = xgboost_cross_validation(x_train_new, y_train)
        result.append(['add_feature_log_%s' % col, best_score])
        print 'result', ','.join(str(e) for e in result[-1])

    print 'add_feature_exp'
    for col in ['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount',
                'lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12']:
        x_train_new, _ = FeatureInteraction().add_feature_exp(x_train.copy(), df_test.copy(), col)
        best_score, _ = xgboost_cross_validation(x_train_new, y_train)
        result.append(['add_feature_exp_%s' % col, best_score])
        print 'result', ','.join(str(e) for e in result[-1])

    print 'LabelCountEncoder'
    for col in ['airconditioningtypeid', 'architecturalstyletypeid',
                'buildingclasstypeid', 'buildingqualitytypeid', 'decktypeid',
                'heatingorsystemtypeid',
                'propertylandusetypeid', 'storytypeid', 'typeconstructiontypeid',
                'regionidcity', 'regionidcounty', 'regionidneighborhood',
                'regionidzip', 'rawcensustractandblock', 'censustractandblock',
                'fips', 'pooltypeid2', 'pooltypeid7',
                'fips', 'pooltypeid2', 'pooltypeid7',
                'pooltypeid10']:
        lce = LabelCountEncoder()
        x_train_new = x_train.copy()
        # df_test_new = df_test.copy()
        lce.fit(x_train_new[col])
        x_train_new[col] = lce.transform(x_train_new[col])
        # df_test[col] = lce.transform(df_test[col])
        best_score, _ = xgboost_cross_validation(x_train_new, y_train)
        result.append(['LabelCountEncoder_%s' % col, best_score])
        print 'result', ','.join(str(e) for e in result[-1])

    print '\n'.join(','.join([str(e) for e in one]) for one in result)

def explore_feature_geo():
    """
    Local CV: 0.05264520
    """
    x_train, y_train, df_test = get_train_test_data()
    x_train, df_test = FeatureInteraction.add_regionidzip_centroid_distance(x_train, df_test)

    result = []
    best_score, _ = xgboost_cross_validation(x_train, y_train)
    result.append(['regionidzip_centroid_distance', best_score])

    print '\n'.join(','.join([str(e) for e in one]) for one in result)

def run_fe_merge():
    x_train, y_train, df_test = get_train_test_data()

    print 'do_mean_encoder'
    x_train, y_train, df_test = FeatureInteraction().do_mean_encoder(x_train, y_train, df_test)

    # LabelCountEncoder
    # for col in ['propertylandusetypeid', 'censustractandblock', 'buildingqualitytypeid',
    #             'rawcensustractandblock']:
    #     lce = LabelCountEncoder()
    #     lce.fit(x_train[col])
    #     x_train[col] = lce.transform(x_train[col])
    #     df_test[col] = lce.transform(df_test[col])

    # add_feature_division
    # [230]	train-mae:0.0507164	test-mae:0.0526378
    # test-mae-mean = 0.05263020
    # x_train, df_test = FeatureInteraction(True).add_feature_division(
    #     x_train, df_test, 'landtaxvaluedollarcnt', 'lotsizesquarefeet')
    # x_train, df_test = FeatureInteraction(True).add_feature_division(
    #     x_train, df_test, 'taxamount', 'calculatedfinishedsquarefeet')

    best_score, num_boost_rounds = xgboost_cross_validation(
        x_train, y_train
    )

    train_and_predict(x_train, y_train, df_test, num_boost_rounds, output_suffix='_fe_merge')

def run_feature_outlier():
    x_train, y_train, df_test = get_train_test_data()

    x_train, df_test = feature_outlier(x_train, df_test)

    best_score, num_boost_rounds = xgboost_cross_validation(
        x_train, y_train
    )

    train_and_predict(x_train, y_train, df_test, num_boost_rounds, output_suffix='_fe_outlier')

def run():
    x_train, y_train, df_test = get_train_test_data()

    best_score, num_boost_rounds = xgboost_cross_validation(
        x_train, y_train
    )

    # feature_scaler_explore(x_train, y_train, xgb_params)
    # feature_outlier_explore(x_train, y_train, xgb_params)
    # import sys
    # sys.exit(0)

    # x_train, df_test = feature_scaler(x_train, df_test)
    # x_train, df_test = feature_outlier(x_train, df_test)

    train_and_predict(x_train, y_train, df_test, num_boost_rounds, output_suffix='')

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and 'nodebug' == sys.argv[-1]:
        debug = False

    func_name = 'run'
    if len(sys.argv) >= 2:
        func_name = sys.argv[1]

    # explore_feature_scaler, explore_feature_outlier
    # func_name = 'explore_feature_interaction'

    print 'Call %s(), debug is %s' % (func_name, debug)
    eval(func_name)()
    print('Congratulation!!!')
