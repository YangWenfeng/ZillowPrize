# Common utils shared by different models.
from sklearn.preprocessing import LabelEncoder
import data_utils as du
import numpy as np

FOLD_NUM = 5
TRAIN_SPLIT_LINE = 80172
OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4

RESULT_FILE = "../../data/result.csv"


def encode_data(df, encode_non_object):
    print('Encoding missing data.')
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(-1, inplace=True)
            label_encoder = LabelEncoder()
            list_value = list(df[column].values)
            label_encoder.fit(list_value)
            df[column] = label_encoder.transform(list_value)
        elif encode_non_object:
            df[column].fillna(-1, inplace=True)
    return df


def get_train_data(encode_non_object):
    print('Getting train data.')
    train = du.get_completed_train_data(encode_non_object)
    train = train[train.logerror > OUTLIER_LOWER_BOUND]
    train = train[train.logerror < OUTLIER_UPPER_BOUND]
    X = train.drop(['parcelid', 'logerror', 'transactiondate',
                    'propertyzoningdesc', 'propertycountylandusecode',
                    'fireplacecnt', 'fireplaceflag'], axis=1)
    y = train['logerror'].values
    return X, y


def get_test_data(encode_non_object):
    print('Getting test data.')
    return du.get_completed_test_data(encode_non_object)


def get_one_kfold(length, split_index):
    train_idx = range(TRAIN_SPLIT_LINE)
    holdout_idx = []
    for i in xrange(TRAIN_SPLIT_LINE, length):
        if i % FOLD_NUM == split_index:
            holdout_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, holdout_idx


def get_full_kfold(length):
    full_list = []
    for i in xrange(FOLD_NUM):
        full_list.append(np.array(get_one_kfold(length, i)))
    return full_list


def get_cv(X, y):
    print('Creating train and holdout data.')
    train_idx, holdout_idx = get_one_kfold(len(X), 0)
    x_train = X.iloc[train_idx]
    y_train = y[train_idx]
    x_holdout = X.iloc[holdout_idx]
    y_holdout = y[holdout_idx]
    return x_train, y_train, x_holdout, y_holdout


def write_result(y_pred):
    print('Writing to csv.')
    result_df = du.get_test_data()
    for column in result_df.columns[result_df.columns != 'ParcelId']:
        result_df[column] = y_pred
    result_df.to_csv(RESULT_FILE, index=False, float_format='%.4f')
    print('Congratulation!')
