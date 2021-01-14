import pandas as pd
import statsmodels.api as sm
import numpy as np


class FMacBeth:

    def __init__(self, r: pd.DataFrame, x: pd.DataFrame, rf: pd.DataFrame):

        self.r = r  # R_T: The return panel. T x I for I assets and T periods
        self.rf = rf                     # R_f: Risk free rate. T x 1
        self.x = x                       # E.g. FF7 Matrix

        self.summary_table = None
        self.B = None
        self.lambda_T = None
        self.lambda_MB = None
        self.R_squared = None
        self.t_stat = None
        self.lambda_r2 = None
        self.beta_r2 = None

    def fit(self):
        # First pass
        self.B, self.beta_r2 = self._get_betas_for_assets()
        # Second pass
        self.lambda_T, self.lambda_r2 = self._get_lambda_per_t()

        T = len(self.lambda_T)
        self.lambda_MB = np.mean(self.lambda_T)

        self.t_stat = self.lambda_MB / (np.std(self.lambda_T) / np.sqrt(T))

    def _get_betas_for_assets(self):
        B_I_T = []
        r_sq = []
        for i in self.r:
            excess_return = self.r[i] - self.rf

            X_ = sm.add_constant(self.x)  # Add alpha as alpha = (1 1 1 ... 1 1 1)

            res = sm.GLS(excess_return, X_).fit()
            B_i_T = res.params.values[1:]
            B_I_T.append(B_i_T)
            r_sq.append(res.rsquared_adj)
        return pd.DataFrame(B_I_T, index=self.r.columns), pd.DataFrame(r_sq, index=self.r.columns)

    def _get_lambda_per_t(self):
        lambda_T_f = []
        r_sq = []
        for t in range(self.r.shape[0]):
            excess_return = self.r.iloc[t,:] - self.rf.iloc[t]

            X_ = sm.add_constant(self.B)         # Add alpha as alpha = (1 1 1 ... 1 1 1)
            res = sm.GLS(excess_return, X_).fit()
            lambda_t_f = res.params.values[1:]
            lambda_T_f.append(lambda_t_f)
            r_sq.append(res.rsquared_adj)

        return pd.DataFrame(lambda_T_f, index=self.r.index), pd.DataFrame(r_sq, index=self.r.index)

