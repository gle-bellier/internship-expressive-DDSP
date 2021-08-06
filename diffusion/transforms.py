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

        # self.sc1 = MinMaxScaler(feature_range=(-1, 1))

    def mtof(self, m):
        return 440 * 2**((m - 69) / 12)

    def ftom(self, f):
        m = 12 * np.log2(f / 440) + 69
        m = m.clip(0, 128)
        return m

    def fit(self, X, y=None):
        X = self.ftom(X)
        self.sc1.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.ftom(X)
        out = self.sc1.transform(X)
        out /= 10
        return out

    def inverse_transform(self, X, y=None):
        X *= 10
        X = self.sc1.inverse_transform(X)
        out = self.mtof(X)
        return out


class LoudnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=100, output_distribution="normal"):

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

        self.sc1 = QuantileTransformer(n_quantiles=n_quantiles,
                                       output_distribution=output_distribution)

    def fit(self, X, y=None):
        X = self.sc1.fit(X)
        return self

    def transform(self, X, y=None):
        out = self.sc1.transform(X)
        out /= 10
        return out

    def inverse_transform(self, X, y=None):
        X = X * 10
        out = self.sc1.inverse_transform(X)
        return out
