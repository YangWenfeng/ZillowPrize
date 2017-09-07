# Ensemble generation to be stacked by LinearRegression in the end.
# There are two types of models:
# 1. Tree model, such as XGBoost and LightGBM.
# 2. Linear model, such as Linear Regression and Bayesian Ridge.
#
# Attention: KFold is fixed now for developing. It will be changed to random
# mode before the final submission.
#
# XGBoost          : 0.0646224
# LightGBM         : 0.0648655
# Linear Regression: 0.0648547
# Ridge            : 0.0648670
# Lasso            : 0.0649758
# Elastic Net      : 0.0649684
# Lasso LARS       : 0.0652184
# Bayesian Ridge   : 0.0648601
# Final            : 0.0648263
from xgboost_baseline import XGBoostModel
from lightgbm_baseline import LightGBMModel
from linear_regression_baseline import LinearRegressionModel
from ridge_baseline import RidgeModel
from lasso_baseline import LassoModel
from elastic_net_baseline import ElasticNetModel
from lasso_lars_baseline import LassoLarsModel
from bayesian_ridge_baseline import BayesianRidgeModel
import common_utils as cu
import numpy as np


class Ensemble(object):
    def __init__(self, stacker, base_models):
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        # get folds index for train and holdout.
        folds = cu.get_full_kfold(len(y))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        # run each model.
        for i, base_model in enumerate(self.base_models):
            print("Fitting for base model %d: %s" % (i, self.base_models))
            S_test_i = np.zeros((T.shape[0], len(folds)))

            # run each fold.
            for j, (train_idx, test_idx) in enumerate(folds):
                print("Fitting for fold %d" % j)
                X_train = X.iloc[train_idx]
                y_train = y[train_idx]
                X_holdout = X.iloc[test_idx]
                y_holdout = y[test_idx]
                base_model.train(X_train, y_train, X_holdout, y_holdout)
                y_pred = base_model.predict(X_holdout)
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = base_model.predict(T)

            # get mean of all folds.
            S_test[:, i] = S_test_i.mean(1)

        # second layer to fit the result from the first layer cross validation.
        self.stacker.train(S_train, y, None, None)
        y_pred = self.stacker.predict(S_test)
        return y_pred


def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)

    # create base models.
    base_models = [
        XGBoostModel(),
        LightGBMModel(),
        LinearRegressionModel(),
        RidgeModel(),
        LassoModel(),
        ElasticNetModel(),
        LassoLarsModel(),
        BayesianRidgeModel(),
    ]

    # setup ensemble parameters.
    ensemble = Ensemble(
        stacker=LinearRegressionModel(),
        base_models=base_models
    )

    # ensemble result.
    print('Ensembling result.')
    y_pred = ensemble.fit_predict(X, y, T[X.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == '__main__':
    run()
