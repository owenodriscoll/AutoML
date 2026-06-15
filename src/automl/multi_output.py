import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin, clone

class MultiOutputStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking regressor for multi-output regression.
    sklearn's StackingRegressor calls column_or_1d(y) unconditionally and
    therefore cannot handle 2-D targets. This class replicates stacking
    manually via cross_val_predict, keeping all targets together.
    """

    def __init__(self, estimators, final_estimator, cv):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        y = np.asarray(y)

        # --- build meta-feature matrix from OOF predictions of every base estimator
        meta_features = []
        self.fitted_estimators_ = []

        for name, estimator in self.estimators:
            # cross_val_predict handles multi-output estimators fine
            oof_preds = cross_val_predict(estimator, X, y, cv=self.cv)  # (n, n_targets)
            meta_features.append(oof_preds)

            # also fit each base estimator on the full training set
            fitted_est = clone(estimator).fit(X, y)
            self.fitted_estimators_.append((name, fitted_est))

        # meta_X shape: (n_samples, n_estimators * n_targets)
        meta_X = np.hstack(meta_features)

        # RidgeCV supports multi-output natively — no MultiOutputRegressor wrapper needed
        self.final_estimator_ = clone(self.final_estimator).fit(meta_X, y)

        return self

    def predict(self, X):
        meta_features = [est.predict(X) for _, est in self.fitted_estimators_]
        meta_X = np.hstack(meta_features)
        return self.final_estimator_.predict(meta_X)