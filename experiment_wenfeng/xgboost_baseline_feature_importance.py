# XGBoost baseline for feature engineering.
#
# Training result: [192] train-mae:0.051412 holdout-mae:0.051941
# Public score: 0.0646266
# Drop missing rate >= 0.9950 features:
# [buildingclasstypeid,finishedsquarefeet13,basementsqft,
# storytypeid,yardbuildingsqft26,architecturalstyletypeid,
# typeconstructiontypeid,finishedsquarefeet6]
# Training result: [178] train-mae:0.051431 holdout-mae:0.052045
# Public score: 0.0646217 improve 0.68% = (0.0646266 - 0.0646217)/(0.0646266 - 0.0639094)
# Drop missing rate >= 0.9500 features:
# [buildingclasstypeid,finishedsquarefeet13,basementsqft,
# storytypeid,yardbuildingsqft26,architecturalstyletypeid,
# typeconstructiontypeid,finishedsquarefeet6,decktypeid,
# poolsizesum,pooltypeid10,pooltypeid2,taxdelinquencyyear,
# yardbuildingsqft17,finishedsquarefeet15]
# Training result: [181] train-mae:0.051411 holdout-mae:0.051993
# Public score: 0.0646936
# Drop zero variance features:
# [buildingclasstypeid,decktypeid,poolcnt,pooltypeid10,
# pooltypeid2,pooltypeid7,storytypeid,assessmentyear]
# Training result: [178] train-mae:0.051562 holdout-mae:0.052138
# Public score: 0.0647126

import common_utils as cu
import xgboost as xgb


class XGBoostModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout, y_holdout):
        print('Training the model.')
        params = {
            'eta': 0.033,
            'max_depth': 6,
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'silent': 1
        }
        xgboost_X_train = xgb.DMatrix(X_train, label=y_train)
        xgboost_X_holdout = xgb.DMatrix(X_holdout, label=y_holdout)
        watchlist = [(xgboost_X_train, 'train'), (xgboost_X_holdout, 'holdout')]
        self.base_model = xgb.train(
            params, xgboost_X_train, 10000, watchlist,
            early_stopping_rounds=100, verbose_eval=10)

    def predict(self, predict_df):
        return self.base_model.predict(xgb.DMatrix(predict_df))

def run_missing_rate():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # feature utils
    from feature_utils import get_feature_missing_df, get_features_by_missing_rate
    missing_df = get_feature_missing_df(X)

    result = []
    for missing_rate in [0.95, 0.995]:
        print 'Missing Rate is %.4f' % missing_rate

        columns = get_features_by_missing_rate(missing_df, missing_rate)

        newX = X.drop(columns, axis=1)

        # get CV from train data.
        X_train, y_train, X_holdout, y_holdout = cu.get_cv(newX, y)

        # train model.
        xgbm = XGBoostModel()
        xgbm.train(X_train, y_train, X_holdout, y_holdout)

        result.append([missing_rate, xgbm.base_model.best_score])

    print '\n'.join(','.join(str(o) for o in one) for one in result)

def run_zero_variance():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # feature utils
    from feature_utils import get_zero_variance_features
    columns = get_zero_variance_features()

    print 'Drop zero variance features [%s]' % ','.join(columns)
    X = X.drop(columns, axis=1)

    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X, y)

    # train model.
    xgbm = XGBoostModel()
    xgbm.train(X_train, y_train, X_holdout, y_holdout)

    # read test data.
    T = cu.get_test_data(encode_non_object=False)

    # predict result.
    print('Predicting.')
    y_pred = xgbm.predict(T[X_train.columns])

    # write result.
    cu.write_result(y_pred)

def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # feature utils
    from feature_utils import get_feature_missing_df, get_features_by_missing_rate
    missing_df = get_feature_missing_df(X)
    missing_rate = 0.995
    columns = get_features_by_missing_rate(missing_df, missing_rate)
    print "Drop missing rate >= %.4f features: [%s]" % (missing_rate, ','.join(columns))
    X = X.drop(columns, axis=1)

    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X, y)

    # train model.
    xgbm = XGBoostModel()
    xgbm.train(X_train, y_train, X_holdout, y_holdout)

    # read test data.
    T = cu.get_test_data(encode_non_object=False)

    # predict result.
    print('Predicting.')
    y_pred = xgbm.predict(T[X_train.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
    # run_missing_rate()
    # run_zero_variance()
