import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X.numpy()


class PitchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=127, output_distribution="normal"):

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

        self.sc1 = QuantileTransformer(n_quantiles=n_quantiles,
                                       output_distribution=output_distribution)

        self.sc2 = MinMaxScaler()

    def mtof(self, m):
        return 440 * 2**((m - 69) / 12)

    def ftom(self, f):
        return 12 * np.log2(f / 440) + 69

    def fit(self, X, y=None):
        X = self.ftom(X)
        X = self.sc1.fit_transform(X)
        self.sc2.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.ftom(X)
        X = self.sc1.transform(X)
        out = self.sc2.transform(X)
        return out

    def inverse_transform(self, X, y=None):
        X = self.sc2.inverse_transform(X)
        X = self.sc1.inverse_transform(X)
        out = self.mtof(X)
        return out


class LoudnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=100, output_distribution="normal"):

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

        self.sc1 = QuantileTransformer(n_quantiles=n_quantiles,
                                       output_distribution=output_distribution)

        self.sc2 = MinMaxScaler()

    def fit(self, X, y=None):
        X = self.sc1.fit_transform(X)
        self.sc2.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.sc1.transform(X)
        out = self.sc2.transform(X)
        return out

    def inverse_transform(self, X, y=None):
        X = self.sc2.inverse_transform(X)
        out = self.sc1.inverse_transform(X)
        return out
