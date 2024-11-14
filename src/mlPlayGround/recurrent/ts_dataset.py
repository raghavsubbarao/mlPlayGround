import numpy as np
import pandas as pd
import tensorflow as tf
tfk = tf.keras

from core.strategy.trendfactor import cfm_trend_factor

class tsDataset:
    """
    Wrapper around a time series dataset
    """
    def __init__(self, input_ts, covar_ts=None, targt_ts=None,
                 input_len=12, output_len=1, shift=1, steps=1,
                 batch_size=32, val_frac=0.0):

        # TODO: check dimensionality and any other ordering info between tgt and covars
        # TODO: how do we handle multiple time series of different lengths?

        self.input_ts = input_ts  # t x n
        self.incov_ts = covar_ts  # assumed to be t x n x c or t x n
        self.targt_ts = targt_ts  # assumed to be ?

        self.input_len = input_len
        self.output_len = output_len
        self.shift = shift
        self.steps = steps

        self.total_len = self.input_len + (self.shift + self.output_len - 1) + (self.steps - 1)

        self.batch_size = batch_size
        self.val_frac = val_frac

        # do we need to adapt these? here jsut setting for densest possible
        self.stride = 1
        self.rate = 1

        # generate features from the raw target ts and covars
        self.ftr_ts = None
        self.tgt_ts = None
        self.cov_ts = None
        self.gen_features()

        self.ts_names = self.ftr_ts.columns.unique(0)

    def __repr__(self):
        # TODO: add more detail here
        return '\n'.join([f'Total window size: {self.total_len}',
                          f'Input length of {self.input_len}',
                          f'Output length of {self.output_len}',
                          f'shifted by {self.shift}',
                          f'for {self.steps} steps.'])

    def gen_features(self):
        """
        Generate features from the input series over here. This should be overwritten
        in any subclass if required. By default, the method combines the target and
        covar ts based on the input length
        """
        # TODO: we assume the covars are lined up with targets while darts allows shifting

        # generate features. Here the feature is the time series itself
        self.ftr_ts = pd.DataFrame(data=self.input_ts.values,
                                   index=self.input_ts.index,
                                   columns=pd.MultiIndex.from_tuples([(c, 'ts') for c in self.input_ts.columns]))

        # handle covariates. default is to just copy them over
        if self.incov_ts is None:
            self.cov_ts = None
        elif len(self.incov_ts.columns.levels) == 1:
            # need to add a dummy level
            self.ftr_ts = pd.DataFrame(data=self.incov_ts.values,
                                       index=self.incov_ts.index,
                                       columns=pd.MultiIndex.from_tuples([(c, 'ts') for c in self.incov_ts.columns]))
        else:
            self.cov_ts = self.incov_ts

        # handle targets
        # TODO: handle actual targets. Ow the target will be the time series itself
        if self.targt_ts is None:
            self.tgt_ts = self.input_ts.shift(-self.shift)
        else:
            raise Exception('How do we handle this???')

        # TODO: check date alignment among the three dfs

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self):
        assert(self.steps == 1)  # only implemented for steps==1
        # TODO: how do we handle covars?

        n = self.ts_names[0]

        def single_ds(a):
            assert a in self.ts_names

            # use window to create a dataset of datasets
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=self.ftr_ts[n].values,
                                                                      targets=self.tgt_ts[n].values,
                                                                      sequence_length=self.input_len,
                                                                      batch_size=self.batch_size,
                                                                      sequence_stride=self.stride,
                                                                      sampling_rate=self.rate,
                                                                      shuffle=True)

        for n in self.ts_names[1:]:
            ds.concatenate(tf.keras.preprocessing.timeseries_dataset_from_array(data=self.ftr_ts[n].values,
                                                                                targets=self.tgt_ts[n].values,
                                                                                sequence_length=self.input_len,
                                                                                batch_size=self.batch_size,
                                                                                sequence_stride=self.stride,
                                                                                sampling_rate=self.rate,
                                                                                shuffle=True))

        return ds

    @property
    def train(self):
        return self.make_dataset(self.input_ts)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


class deepTrendDataset(tsDataset):

    def __init__(self, px_ts, input_len=20, steps=5, batch_size=32, val_frac=0.0,
                 lags=[1, 21, 63, 126, 252], tf_lags=[66, 250], tau=10):

        self.lags = lags
        self.tf_lags = tf_lags
        self.tau = tau

        self.inp_slice = slice(0, input_len + steps - 1)  # assuming output_len==1
        self.tgt_slice = slice(-steps, None)  # assuming output_len==1

        super(deepTrendDataset, self).__init__(input_ts=px_ts, covar_ts=None, targt_ts=None,
                                               input_len=input_len, output_len=1, shift=1, steps=steps,
                                               batch_size=batch_size, val_frac=val_frac)

    @staticmethod
    def rolling_vol(df_ret, tau=10, vol_type='ewma'):
        if vol_type.lower() == 'roll':
            return np.sqrt((df_ret ** 2).rolling(tau).mean())
        elif vol_type.lower() == 'ewma':
            return np.sqrt((df_ret ** 2).ewm(adjust=False, span=tau).mean())
        elif vol_type.lower() == 'ecfm' or vol_type.lower() == 'ewma_cfm':
            f = partial(ewmacfm, lambda_=1 - 2 / (tau + 1))
            return np.sqrt((df_ret ** 2).rolling(tau).apply(f, raw=True))
        else:
            raise Exception('Unknown rolling vol type')

    def gen_features(self):

        self.input_ts.sort_index(axis=1, level=0, inplace=True)  # sort by dates

        features = ['Ret_{:04d}'.format(lag) for lag in self.lags] + \
                   ['TrendFactor_{:04d}'.format(lag) for lag in self.tf_lags]
        assets = list(self.input_ts.columns)

        # Returns
        rdf = self.input_ts.diff(1).dropna(axis=0, how='any')  # difference return

        # vol
        sigma = self.rolling_vol(rdf, self.tau, vol_type='ewma').shift(1)

        # feature vectors
        fv = lambda a: pd.concat([100 * (self.input_ts[a].diff(l) / (sigma[a] * np.sqrt(l))) for l in self.lags] +
                                 [cfm_trend_factor(self.input_ts[a], tau=l) for l in self.tf_lags],  # trend factors
                                 axis=1, join='inner')
        cols = pd.MultiIndex.from_product([assets, features], names=['Asset', 'Feature'])
        self.ftr_ts = pd.DataFrame(data=np.hstack([fv(a) for a in assets]),
                                   columns=cols, index=self.input_ts.index).dropna()

    def split_window(self, features):
        inputs = features[:, self.inp_slice, :]
        targts = features[:, self.tgt_slice, :self.output_len]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_len + self.steps - 1, None])
        targts.set_shape([None, self.steps, self.output_len])
        return inputs, targts

    def make_dataset(self):

        def single_ds(a):
            assert(a in self.ts_names)
            ds = tfk.preprocessing.timeseries_dataset_from_array(data=self.ftr_ts[a].values, targets=None,
                                                                 sequence_length=self.total_len,
                                                                 batch_size=self.batch_size, shuffle=True,
                                                                 sequence_stride=self.stride, sampling_rate=self.rate)
            ds = ds.map(self.split_window)
            return ds

        ds = single_ds(self.ts_names[0])  # start with the first asset

        for n in self.ts_names[1:]:
            ds = ds.concatenate(single_ds(n))

        return ds


# def rolling_vol(df_ret, tau=10, vol_type='ewma'):
#     if vol_type.lower() == 'roll':
#         return np.sqrt((df_ret ** 2).rolling(tau).mean())
#     elif vol_type.lower() == 'ewma':
#         return np.sqrt((df_ret ** 2).ewm(adjust=False, span=tau).mean())
#     elif vol_type.lower() == 'ecfm' or vol_type.lower() == 'ewma_cfm':
#         f = partial(ewmacfm, lambda_=1 - 2 / (tau + 1))
#         return np.sqrt((df_ret ** 2).rolling(tau).apply(f, raw=True))
#     else:
#         raise Exception('Unknown rolling vol type')
#
#
# def build_fv(px_df, tau=10):
#     from core.strategy.trendfactor import cfm_trend_factor
#     px_df.sort_index(axis=1, level=0, inplace=True)  # sort by dates
#     lags = [1, 21, 63, 126, 252]
#     tf_lags = [66, 250]
#
#     features = ['Ret_{:04d}'.format(lag) for lag in lags] + \
#                ['TrendFactor_{:04d}'.format(lag) for lag in tf_lags] +\
#                ['Return']
#     assets = list(px_df.columns)
#
#     # Returns
#     rdf = px_df.diff(1).dropna(axis=0, how='any')  # difference return
#
#     # vol
#     sigma = rolling_vol(rdf, tau, vol_type='ewma').shift(1)
#
#     # generate feature vectors and outputs for training
#     fv = lambda a: pd.concat([100 * (px_df[a].diff(periods=lag) / (sigma[a] * np.sqrt(lag))) for lag in lags] +
#                              [cfm_trend_factor(px_df[a], tau=lag) for lag in tf_lags] +   # trend factor
#                              [(px_df[a].diff(periods=1) / sigma[a]).shift(-1)], axis=1, join='inner')   # fwd return
#
#     cols = pd.MultiIndex.from_product([assets, features], names=['Asset', 'Feature'])
#     fvdf = pd.DataFrame(data=np.hstack([fv(a) for a in assets]), columns=cols, index=px_df.index).dropna()
#
#     return fvdf


if __name__ == "__main__":
    # import yfinance as yf
    # import datetime as dt
    #
    # today = dt.datetime.today()
    # offset = 0
    # if today.weekday() > 4:
    #     offset = today.weekday() - 4  # max(1, (today.weekday() + 6) % 7 - 3) #get last business day
    # timed = dt.timedelta(offset)
    # today_business = today - timed
    # print("d1 =", today_business)
    #
    # tickers = ['IXC', 'MXI', 'EXI', 'RXI', 'KXI', 'IXJ', 'IXG', 'IXN', 'IXP', 'JXI', 'IGF']
    # start = '2004-01-01'
    # end = today_business.strftime("%Y-%m-%d")
    #
    # data_df = yf.download(tickers, start, end)

    data_df = pd.read_csv('../../../data/gics1.csv').set_index('Date')
    data_df.index = data_df.index.map(lambda x: pd.Timestamp(x))

    dtds = deepTrendDataset(data_df)
    ds = dtds.make_dataset()