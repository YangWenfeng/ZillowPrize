import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost_baseline import XGBoostModel
import common_utils as cu

FEATURE_IMPORTANCE_FILE = '../../data/feature_importance.csv'

def gen_feature_importance(importance_type='gain'):
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X, y)

    # train model.
    xgbm = XGBoostModel()
    xgbm.train(X_train, y_train, X_holdout, y_holdout)

    # feature importance
    tmp = xgbm.base_model.get_score(importance_type=importance_type)
    columns, importances = [], []
    for c, i in tmp.items():
        columns.append(c)
        importances.append(i)

    importance_df = pd.DataFrame({'column_name': columns, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=True)

    importance_df = importance_df.reset_index(drop=True)

    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print 'Generate feature importance success.'

def get_feature_importance_df():
    importance_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

    return importance_df

def get_feature_missing_df(X):
    missing_df = X.isnull().sum(axis=0).reset_index()

    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_rate'] = missing_df['missing_count'] / float(X.shape[0])
    missing_df = missing_df.sort_values(by='missing_count', ascending=False)
    missing_df = missing_df.reset_index(drop=True)

    # pd.options.display.max_rows = 65
    # print missing_df

    return missing_df

def get_category_features():
    category_features = ['airconditioningtypeid', 'architecturalstyletypeid',
                         'buildingclasstypeid', 'buildingqualitytypeid', 'decktypeid',
                         'heatingorsystemtypeid', 'propertycountylandusecode',
                         'propertylandusetypeid', 'storytypeid', 'typeconstructiontypeid',
                         'regionidcity', 'regionidcounty', 'regionidneighborhood',
                         'regionidzip', 'rawcensustractandblock', 'censustractandblock',
                         'propertyzoningdesc']

    return category_features

def get_bool_features():
    bool_features = ['hashottuborspa', 'taxdelinquencyflag', 'fireplaceflag']

    return bool_features

def encode_category_bool_features(df, encode_non_object):
    new_df = df.copy()
    category_features = get_category_features()
    bool_features = get_bool_features()

    print 'Encode category & bool features: [%s], [%s]' % (','.join(category_features),
                                                           ','.join(bool_features))
    for column in new_df.columns:
        if column in category_features or column in bool_features:
            missing = new_df[column].isnull()
            new_df[column].fillna(-1, inplace=True)
            label_encoder = LabelEncoder()
            list_value = list(new_df[column].values)
            label_encoder.fit(list_value)
            new_df[column] = label_encoder.transform(list_value)
            if not encode_non_object:
                new_df[column][missing] = np.nan

    return new_df

def get_features_by_missing_rate(missing_df, missing_rate):
    drop_columns = missing_df[missing_df['missing_rate'] >= missing_rate]['column_name'].values

    return list(drop_columns)

def get_zero_variance_features():
    # X, _ = cu.get_train_data(encode_non_object=False)
    # X.fillna(X.median(), inplace=True)  # IMPORTANT
    #
    # from sklearn.feature_selection import VarianceThreshold
    # selector = VarianceThreshold()
    #
    # selector.fit(X)
    # zero_variance_columns = [col for i, col in enumerate(X.columns) if selector.variances_[i] == 0]

    zero_variance_columns = ['buildingclasstypeid', 'decktypeid', 'poolcnt', 'pooltypeid10',
                             'pooltypeid2', 'pooltypeid7', 'storytypeid', 'assessmentyear']

    return zero_variance_columns

if __name__ == '__main__':

    X, y = cu.get_train_data(encode_non_object=False)

    # feature importance
    # gen_feature_importance()
    # print get_feature_importance_df()

    # missing rate
    missing_df = get_feature_missing_df(X)
    print missing_df
    print get_features_by_missing_rate(missing_df, 0.90)

    # Removing features with low variance
    # Remove feature assessmentyear, cause it's variance equal 0
    # print X['assessmentyear'].value_counts(dropna=False)
    # print get_zero_variance_features()
