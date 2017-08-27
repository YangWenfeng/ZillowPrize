# LightGBM baseline for feature engineering.
#
# Training result: [1355] train's l1: 0.0517065 holdout's l1: 0.0525932
# Public score: 0.0646075
import common_utils as cu
import lightgbm as lgbm


class LightGBMModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout, y_holdout):
        print('Training the model.')
        params = {
            'max_bin': 10,
            'learning_rate': 0.0021,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'sub_feature': 0.5,
            'bagging_fraction': 0.85,
            'bagging_freq': 40,
            'num_leaves': 512,
            'min_data': 500,
            'min_hessian': 0.05,
            'verbose': 0
        }
        lgbm_X_train = lgbm.Dataset(X_train, label=y_train)
        lgbm_X_holdout = lgbm.Dataset(X_holdout, label=y_holdout)
        self.base_model = lgbm.train(
            params, lgbm_X_train, num_boost_round=10000,
            valid_sets=[lgbm_X_train, lgbm_X_holdout],
            valid_names=['train', 'holdout'], early_stopping_rounds=100,
            verbose_eval=10)

    def predict(self, predict_df):
        return self.base_model.predict(predict_df)


def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X, y)

    # train model.
    lgbmm = LightGBMModel()
    lgbmm.train(X_train, y_train, X_holdout, y_holdout)

    # read test data.
    T = cu.get_test_data(encode_non_object=False)

    # predict result.
    print('Predicting.')
    y_pred = lgbmm.predict(T[X_train.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
