# MLPRegressor baseline for feature engineering.
#
# Public score: 57305.2981687 (MLP is sensitive to feature scaling)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import common_utils as cu


class MLPRegressorModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout=None, y_holdout=None):
        print('Training model.')
        params = {
            'hidden_layer_sizes': (X_train.shape[1]+1,) * 4
        }
        self.base_model = MLPRegressor(params)
        self.base_model.fit(X_train, y_train)

        y_pred = self.predict(X_train)
        mae = mean_absolute_error(y_true=y_train, y_pred=y_pred)
        print 'Training result: %.6f' % mae

    def predict(self, predict_df):
        return self.base_model.predict(predict_df)

def run_grid():
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

    feature_cnt = X.columns.shape[0]

    print 'Grid Search.'
    parameters = {
        'hidden_layer_sizes': [(feature_cnt+1,) * n for n in [1, 2, 3, 4, 5, 6]],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200, 400, 600],
        'early_stopping': [False, True]
    }
    grid = GridSearchCV(MLPRegressor(), parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')
    grid.fit(X, y)

    print 'best_score_', grid.best_score_
    print 'best_params_', grid.best_params_

def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # train model.
    lrm = MLPRegressorModel()
    lrm.train(X, y)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)

    # predict result.
    print('Predicting.')
    y_pred = lrm.predict(T[X.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    # run()
    run_grid()
