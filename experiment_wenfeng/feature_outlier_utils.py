import numpy as np
import pandas as pd
from collections import OrderedDict

__author__ = 'yangwenfeng'


def get_feature_outlier_index(pd, feature, method):
    """
    :param pd:
    :param feature:
    :param method:
    :return: index list of outliers
    """
    methods = ['inter_quartile_range', 'small_probability_event']
    assert method in methods

    ret = None
    values = pd[feature].values

    if method == methods[0]:
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        ret = [i for i, v in enumerate(values) if v < q1-1.5*iqr or v > q3+1.5*iqr]

    elif method == methods[1]:
        ulimit = np.percentile(values, 99.5)
        llimit = np.percentile(values, 0.05)
        ret = [i for i, v in enumerate(values) if v > ulimit or v < llimit]

    return ret

def replace_feature_outlier_value(df, feature, indexs, value):
    new_df = df.copy()
    new_df[feature][indexs] = value
    return new_df

def replace_feature_outlier_iqr_boundary(df, feature):
    new_df = df.copy()

    q1 = np.percentile(new_df[feature].values, 25)
    q3 = np.percentile(new_df[feature].values, 75)
    iqr = q3 - q1
    llimit = q1 - 1.5 * iqr
    ulimit = q3 + 1.5 * iqr

    new_df[feature].loc[new_df[feature] > ulimit] = ulimit
    new_df[feature].loc[new_df[feature] < llimit] = llimit

    return new_df

def replace_feature_outlier_boundary(df, feature, lpercentile=0.5, upercentile=99.5):
    new_df = df.copy()

    llimit = np.percentile(new_df[feature].values, lpercentile)
    ulimit = np.percentile(new_df[feature].values, upercentile)

    new_df[feature].loc[new_df[feature] > ulimit] = ulimit
    new_df[feature].loc[new_df[feature] < llimit] = llimit

    return new_df

def generate_df(df, feature):
    ret = OrderedDict()

    outlier_indexs_iqr = get_feature_outlier_index(df, feature, 'inter_quartile_range')
    outlier_indexs_spe = get_feature_outlier_index(df, feature, 'small_probability_event')

    ret['inter_quartile_range_replace_nan'] = \
        replace_feature_outlier_value(df, feature, outlier_indexs_iqr, np.NAN)

    ret['inter_quartile_range_replace_median'] = \
        replace_feature_outlier_value(df, feature, outlier_indexs_iqr, df[feature].median())

    ret['inter_quartile_range_replace_boundary'] = \
        replace_feature_outlier_iqr_boundary(df, feature)

    ret['small_probability_event_replace_nan'] = \
        replace_feature_outlier_value(df, feature, outlier_indexs_spe, np.NAN)

    ret['small_probability_event_replace_median'] = \
        replace_feature_outlier_value(df, feature, outlier_indexs_spe, df[feature].median())

    ret['small_probability_event_replace_boundary'] = \
        replace_feature_outlier_boundary(df, feature)

    return ret


if __name__ == '__main__':
    numbers = np.array([1, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 22])
    numbers_df = pd.DataFrame({'numbers': numbers})

    outlier_indexs_iqr = get_feature_outlier_index(numbers_df, 'numbers', 'inter_quartile_range')
    outlier_indexs_spe = get_feature_outlier_index(numbers_df, 'numbers', 'small_probability_event')
    print 'outlier_indexs_iqr', outlier_indexs_iqr
    print 'outlier_indexs_spe', outlier_indexs_spe
    #
    # new_numbers_df = replace_feature_outlier_value(numbers_df, 'numbers', outlier_indexs_spe, np.NaN)
    # print numbers_df, new_numbers_df

    for name, df in generate_df(numbers_df, 'numbers').items():
        print name
        print df
