# XGBoost baseline for feature engineering.
# baseline
# Training result: [192] train-mae:0.051412 holdout-mae:0.051941
# Public score: 0.0646266
import common_utils as cu
import xgboost as xgb
from feature_outlier_utils import *


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

def run_feature_outlier():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # transform feature 'yearbuilt'
    X['yearbuilt'] = 2016 - X['yearbuilt']

    result = []
    for feature in ['taxamount', 'yearbuilt']:
        for name, newSeries in generate_feature_replace_outlier(X[feature]).items():
            print 'Try to deal with feature[%s] outlier by [%s].' % (feature, name)

            # get CV from train data.
            newX = X.copy()
            newX[feature] = newSeries
            X_train, y_train, X_holdout, y_holdout = cu.get_cv(newX, y)

            # train model.
            xgbm = XGBoostModel()
            xgbm.train(X_train, y_train, X_holdout, y_holdout)

            result.append([feature, name, xgbm.base_model.best_score])

    print '\n'.join(','.join(str(o) for o in one) for one in result)

def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    print('Transform, replace feature outliers.')
    X['yearbuilt'] = 2016 - X['yearbuilt']
    X['yearbuilt'] = replace_feature_outlier_iqr_boundary(X['yearbuilt'])
    X['taxamount'] = replace_feature_outlier_iqr_boundary(X['taxamount'])

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
    # run_feature_outlier()
    # Feature outlier result
    # yearbuilt,iqr_boundary,0.051907
    # taxamount,iqr_boundary,0.051929
