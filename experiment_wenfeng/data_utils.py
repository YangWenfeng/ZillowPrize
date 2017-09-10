# Generate completed training and test data, which is time-consuming to
# generate every time.
import feature_utils as fu
import pandas as pd

TRAIN_DATA_FILE = "../../data/train_2016_v2.csv"
PROPERTIES_FILE = "../../data/properties_2016.csv"
TEST_DATA_FILE = "../../data/sample_submission.csv"

COMPLETED_TRAIN_DATA_FILE = "../../data/completed_train.csv"
COMPLETED_TRAIN2_DATA_FILE = "../../data/completed_train2.csv"
COMPLETED_TEST_DATA_FILE = "../../data/completed_test.csv"
COMPLETED_TEST2_DATA_FILE = "../../data/completed_test2.csv"

STANDARD_SCALER_TRAIN_DATA_FILE = "../../data/standard_scaler_train.csv"
STANDARD_SCALER_TEST_DATA_FILE = "../../data/standard_scaler_test.csv"

_train_df, _test_df, _properties_df = None, None, None

def get_completed_train_data(encode_non_object, standard_scaler_flag=False):
    if encode_non_object:
        file_name = COMPLETED_TRAIN2_DATA_FILE
    else:
        if standard_scaler_flag:
            file_name = STANDARD_SCALER_TRAIN_DATA_FILE
        else:
            file_name = COMPLETED_TRAIN_DATA_FILE
    return pd.read_csv(file_name)


def get_completed_test_data(encode_non_object, standard_scaler_flag=False):
    if encode_non_object:
        file_name = COMPLETED_TEST2_DATA_FILE
    else:
        if standard_scaler_flag:
            file_name = STANDARD_SCALER_TEST_DATA_FILE
        else:
            file_name = COMPLETED_TEST_DATA_FILE
    return pd.read_csv(file_name)


def get_test_data():
    return pd.read_csv(TEST_DATA_FILE)


def generate_train_data(encode_non_object, standard_scaler_flag=False):
    print('Generating train data.')
    train_df = _train_df.copy()
    properties_df = \
        fu.data_preprocessing(_properties_df.copy(), encode_non_object, standard_scaler_flag)
    train_properties_df = \
        train_df.merge(properties_df, how='left', on='parcelid')
    if encode_non_object:
        file_name = COMPLETED_TRAIN2_DATA_FILE
    else:
        file_name = COMPLETED_TRAIN_DATA_FILE
    train_properties_df.to_csv(file_name, index=False, float_format='%f')


def generate_test_data(encode_non_object, standard_scaler_flag=False):
    print('Generating test data.')
    test_df = _test_df.copy()
    test_df['parcelid'] = test_df['ParcelId']
    properties_df =\
        fu.data_preprocessing(_properties_df.copy(), encode_non_object, standard_scaler_flag)
    test_properties_df = test_df.merge(properties_df, how='left', on='parcelid')
    test_properties_df = test_properties_df.drop(
        ['ParcelId', '201610', '201611', '201612', '201710', '201711',
         '201712'], axis=1)
    if encode_non_object:
        file_name = COMPLETED_TEST2_DATA_FILE
    else:
        file_name = COMPLETED_TEST_DATA_FILE
    test_properties_df.to_csv(file_name, index=False, float_format='%f')


if __name__ == '__main__':
    print 'Read origin train data.'
    _train_df = pd.read_csv(TRAIN_DATA_FILE)

    print 'Read origin test data.'
    _test_df = pd.read_csv(TEST_DATA_FILE)

    print 'Read origin properties data.'
    _properties_df = pd.read_csv(PROPERTIES_FILE)

    generate_train_data(False)
    generate_test_data(False)
    generate_train_data(True)
    generate_test_data(True)
    generate_train_data(True, standard_scaler_flag=True)
    generate_test_data(True, standard_scaler_flag=True)
