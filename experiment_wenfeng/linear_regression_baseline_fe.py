# Linear regression baseline for feature engineering.
#
# Public score: 0.0649163
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import common_utils as cu


class LinearRegressionModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout, y_holdout):
        print('Training model.')
        self.base_model = LinearRegression()
        self.base_model.fit(X_train, y_train)

        y_pred = self.predict(X_train)
        mae = mean_absolute_error(y_true=y_train, y_pred=y_pred)
        print 'Training result: %.6f' % mae

    def predict(self, predict_df):
        return self.base_model.predict(predict_df)

def run_fe():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # feature utils
    from feature_utils import get_category_features, get_bool_features
    category_bool_columns = []
    category_bool_columns.extend(get_category_features())
    category_bool_columns.extend(get_bool_features())
    print 'Drop category & bool columns: %s' % ','.join(category_bool_columns)
    X = X.drop(category_bool_columns, axis=1)

    # from sklearn.preprocessing import StandardScaler
    print 'Standard Scaler.'
    for col in X.columns:
        if col in category_bool_columns:
            continue
        col_mean, col_std = X[col].mean(), X[col].std()
        X[col] = (X[col] - col_mean) / col_std

    # train model.
    lrm = LinearRegressionModel()
    lrm.train(X, y, None, None)

def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # train model.
    lrm = LinearRegressionModel()
    lrm.train(X, y, None, None)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)

    # predict result.
    print('Predicting.')
    y_pred = lrm.predict(T[X.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()  # 0.052993
    # run_fe()  # 0.052991
