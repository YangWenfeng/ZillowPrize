# XGBoost baseline for feature engineering.
# baseline
# Training result: [192] train-mae:0.051412 holdout-mae:0.051941
# Public score: 0.0646266
# baseline norm: y -= y_mean; y /= y_std before train, y_pred *= y_std; y_pred += y_mean
# Train result: [89] train-mae:0.623345 holdout-mae:0.621901
# Public score: 0.0645886

import common_utils as cu
import xgboost as xgb
import numpy as np


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


def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)
    y_mean, y_std = y.mean(), y.std()
    y -= y_mean; y /= y_std

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
    y_pred *= y_std; y_pred += y_mean

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
