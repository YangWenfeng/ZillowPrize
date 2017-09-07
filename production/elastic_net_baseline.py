# Elastic Net baseline for feature engineering.
#
# Public score: 0.0650432
from sklearn.linear_model import ElasticNet
import common_utils as cu


class ElasticNetModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout=None, y_holdout=None):
        print('Training model.')
        self.base_model = ElasticNet(alpha=0.1, l1_ratio=0.7)
        self.base_model.fit(X_train, y_train)

    def predict(self, predict_df):
        return self.base_model.predict(predict_df)


def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # train model.
    lrm = ElasticNetModel()
    lrm.train(X, y)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)

    # predict result.
    print('Predicting.')
    y_pred = lrm.predict(T[X.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
