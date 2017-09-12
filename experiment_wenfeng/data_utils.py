# Generate completed training and test data, which is time-consuming to
# generate every time.
import common_utils as cu
import pandas as pd

TRAIN_DATA_FILE = "../../data/train_2016_v2.csv"
PROPERTIES_FILE = "../../data/properties_2016.csv"
TEST_DATA_FILE = "../../data/sample_submission.csv"

COMPLETED_TRAIN_DATA_FILE = "../../data/completed_train.csv"
COMPLETED_TRAIN2_DATA_FILE = "../../data/completed_train2.csv"
COMPLETED_TEST_DATA_FILE = "../../data/completed_test.csv"
COMPLETED_TEST2_DATA_FILE = "../../data/completed_test2.csv"


def get_completed_train_data(encode_non_object):
    if encode_non_object:
        file_name = COMPLETED_TRAIN2_DATA_FILE
    else:
        file_name = COMPLETED_TRAIN_DATA_FILE
    return pd.read_csv(file_name)


def get_completed_test_data(encode_non_object):
    if encode_non_object:
        file_name = COMPLETED_TEST2_DATA_FILE
    else:
        file_name = COMPLETED_TEST_DATA_FILE
    return pd.read_csv(file_name)


def get_test_data():
    return pd.read_csv(TEST_DATA_FILE)


def generate_train_data(encode_non_object):
    print('Generating train data.')
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    properties_df = \
        cu.encode_data(pd.read_csv(PROPERTIES_FILE), encode_non_object)
    train_properties_df = \
        train_df.merge(properties_df, how='left', on='parcelid')
    if encode_non_object:
        file_name = COMPLETED_TRAIN2_DATA_FILE
    else:
        file_name = COMPLETED_TRAIN_DATA_FILE
    train_properties_df.to_csv(file_name, index=False, float_format='%f')


def generate_test_data(encode_non_object):
    print('Generating test data.')
    test_df = pd.read_csv(TEST_DATA_FILE)
    test_df['parcelid'] = test_df['ParcelId']
    properties_df =\
        cu.encode_data(pd.read_csv(PROPERTIES_FILE), encode_non_object)
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
    generate_train_data(False)
    generate_train_data(True)
    generate_test_data(False)
    generate_test_data(True)
