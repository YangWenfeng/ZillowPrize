import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import common_utils as cu

class LabelCountEncoder:
    """
    LabelCount Encoding
    Author: Wenfeng Yang
    Inspired by: https://www.slideshare.net/HJvanVeen/feature-engineering-72376750
    """
    def __init__(self):
        self._data = None

    def fit(self, values):
        series = pd.Series(values)
        vc = series.value_counts()
        vc = vc.sort_values(ascending=True, inplace=False)

        self._data = dict()
        for i, v in enumerate(vc.index):
            self._data[v] = i

    def transform(self, values):
        return np.array([self._data.get(v, np.nan) for v in values])


class OutlierEncoder:
    """
    Outlier Encoding
    Author: Wenfeng Yang
    Replace feature outliers with np.nan or other values like mean, median etc
    """

    def __init__(self, method='iqr', replace='mean'):
        assert method in ['iqr', 'spe']
        assert replace in ['nan', 'mean', 'median']

        self._method = method
        self._replace = replace
        self._llimit = None
        self._ulimit = None
        self._replace_value = np.nan

    @staticmethod
    def get_series_percentile(series, lpercentile=0.5, upercentile=99.5):
        new_series = series[series.notnull()]

        llimit = np.percentile(new_series.values, lpercentile)
        ulimit = np.percentile(new_series.values, upercentile)

        return llimit, ulimit

    @staticmethod
    def get_series_q1q3(series):
        return OutlierEncoder.get_series_percentile(series, 25, 75)

    def fit(self, series):
        if self._method == 'iqr':
            q1, q3 = self.get_series_q1q3(series)
            iqr = q3 - q1
            self._llimit = q1 - 1.5 * iqr
            self._ulimit = q3 + 1.5 * iqr
        elif self._method == 'spe':
            self._llimit, self._ulimit = self.get_series_percentile(series)

        if self._replace == 'mean':
            self._replace_value = series.mean()
        elif self._replace == 'median':
            self._replace_value = series.median()
        elif self._replace_value == 'nan':
            self._replace_value = np.nan

    def transform(self, series):
        new_series = series.copy()

        new_series.loc[new_series > self._ulimit] = self._replace_value
        new_series.loc[new_series < self._llimit] = self._replace_value

        return new_series

def get_feature_importance_df(importance_type='gain'):
    from xgboost_baseline import XGBoostModel

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
                         'propertyzoningdesc', 'fips', 'pooltypeid2', 'pooltypeid7',
                         'pooltypeid10']

    return category_features

def get_bool_features():
    bool_features = ['hashottuborspa', 'taxdelinquencyflag', 'fireplaceflag']

    return bool_features

def get_year_features():
    return ['yearbuilt', 'assessmentyear', 'taxdelinquencyyear']

def get_latitude_longitude_features():
    return ['latitude', 'longitude']

def get_all_properties_features():
    s = 'airconditioningtypeid,architecturalstyletypeid,basementsqft,bathroomcnt,' \
        'bedroomcnt,buildingqualitytypeid,buildingclasstypeid,calculatedbathnbr,' \
        'decktypeid,threequarterbathnbr,finishedfloor1squarefeet,calculatedfinishedsquarefeet,' \
        'finishedsquarefeet6,finishedsquarefeet12,finishedsquarefeet13,finishedsquarefeet15,' \
        'finishedsquarefeet50,fips,fireplacecnt,fireplaceflag,fullbathcnt,garagecarcnt,' \
        'garagetotalsqft,hashottuborspa,heatingorsystemtypeid,latitude,longitude,' \
        'lotsizesquarefeet,numberofstories,parcelid,poolcnt,poolsizesum,pooltypeid10,' \
        'pooltypeid2,pooltypeid7,propertycountylandusecode,propertylandusetypeid,' \
        'propertyzoningdesc,rawcensustractandblock,censustractandblock,regionidcounty,' \
        'regionidcity,regionidzip,regionidneighborhood,roomcnt,storytypeid,typeconstructiontypeid,' \
        'unitcnt,yardbuildingsqft17,yardbuildingsqft26,yearbuilt,taxvaluedollarcnt,' \
        'structuretaxvaluedollarcnt,landtaxvaluedollarcnt,taxamount,assessmentyear,' \
        'taxdelinquencyflag,taxdelinquencyyear'
    return s.split(',')

def get_scale_features():
    return list(set(get_all_properties_features())
                - set(get_category_features())
                - set(get_bool_features())
                - set(['parcelid'])
                )

def data_preprocessing(df, encode_non_object, standard_scaler_flag=False):
    print 'Data preprocessing.'
    new_df = df.copy()

    print 'Data preprocessing with year features'
    year_features = get_year_features()
    for col in year_features:
        new_df[col] = 2016 - new_df[col]

    category_features = get_category_features()
    bool_features = get_bool_features()

    print 'Encode category & bool features: [%s], [%s]' % (','.join(category_features),
                                                           ','.join(bool_features))

    latitude_longitude_features = get_latitude_longitude_features()
    for column in new_df.columns:
        if column in ['parcelid']:
            print 'Data preprocessing skip parcelid.'
            continue

        if column in category_features or column in bool_features:
            missing = new_df[column].isnull()
            new_df[column].fillna(0, inplace=True)
            label_encoder = LabelEncoder()
            list_value = list(new_df[column].values)
            label_encoder.fit(list_value)
            new_df[column] = label_encoder.transform(list_value)
            if not encode_non_object:
                new_df[column][missing] = np.nan
        elif encode_non_object:
            if standard_scaler_flag:
                v_mean, v_std = new_df[column].mean(), new_df[column].std()
                new_df[column] = (new_df[column] - v_mean) / v_std

            if column in latitude_longitude_features:
                new_df[column].fillna(new_df[column].median(), inplace=True)
            else:
                new_df[column].fillna(0, inplace=True)

    return new_df

def fillna_zero(df):
    new_df = df.copy()

    columns = ['hashottuborspa', 'airconditioningtypeid', 'poolcnt', 'fireplacecnt',
               'decktypeid', 'regionidcity', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10']
    for col in columns:
        new_df[col].fillna(0, inplace=True)

    return new_df

def get_features_by_missing_rate(missing_df, missing_rate):
    drop_columns = missing_df[missing_df['missing_rate'] >= missing_rate]['column_name'].values

    return list(drop_columns)

def gen_zero_variance_features():
    X, _ = cu.get_train_data(encode_non_object=False)

    X.fillna(X.median(), inplace=True)  # IMPORTANT

    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()

    selector.fit(X)
    zero_variance_columns = [col for i, col in enumerate(X.columns) if selector.variances_[i] == 0]

    return zero_variance_columns


if __name__ == '__main__':

    X, y = cu.get_train_data(encode_non_object=False)

    X = fillna_zero(X)
    print X.shape

    # feature importance
    print 'Generate feature importance.'
    print get_feature_importance_df()

    # missing rate
    print 'Missing rate.'
    missing_df = get_feature_missing_df(X)
    print missing_df

    print 'Missing rate >= 0.90'
    print get_features_by_missing_rate(missing_df, 0.90)

    # Removing features with low variance
    # Remove feature assessmentyear, cause it's variance equal 0
    # print X['assessmentyear'].value_counts(dropna=False)
    # print get_zero_variance_features()

    #
    print gen_zero_variance_features()
