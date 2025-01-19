import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
# from tensorflow.keras.losses import binary_crossentropy as bce
# from tensorflow.keras import Sequential
import tensorflow_addons as tfa

#################################
#          TS Generator         #
#################################
# window generator class for data winnowing - splitting into input, output pairs
class WindowGenerator:
    def __init__(self, input_width, label_width, shift=None, batch_size=32):

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        if shift is None:
            self.shift = input_width
        else:
            self.shift = shift
        self.batch_size = batch_size

        self.total_window_size = self.shift + self.label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([f'Total window size: {self.total_window_size}',
                          f'Input indices: {self.input_indices}',
                          f'Label indices: {self.label_indices}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data,
                                                                  targets=None,
                                                                  sequence_length=self.total_window_size,
                                                                  sequence_stride=1,
                                                                  shuffle=True,
                                                                  batch_size=self.batch_size)

        ds = ds.map(self.split_window)
        return ds

    def __call__(self, ds):
        return self.make_dataset(ds)

#################################
#               TCN             #
#################################
class temporalBlockTf(layers.Layer):
    """
    For implementation of TCNs as described in "An Empirical Evaluation
    of Generic Convolutional and Recurrent Networks for Sequence Modeling"
    by Bai, Koleter and Koltun, 2018.

    While TCNs can be taken to mean 1d convolution with dilations, here
    they describe a "Temporal Block" that is stacked to build a TCN.
    """
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate,
                 dropout=0.2, trainable=True, name=None,
                 dtype=None, activity_regularizer=None, **kwargs):

        super(temporalBlockTf, self).__init__(trainable=trainable, dtype=dtype,
                                              activity_regularizer=activity_regularizer,
                                              name=name, **kwargs)

        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None

        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(filters=self.n_outputs, kernel_size=self.kernel_size, strides=self.strides,
                                   padding='causal', dilation_rate=self.dilation_rate, name="conv1")
        self.conv1 = tfa.layers.WeightNormalization(self.conv1)

        self.conv2 = layers.Conv1D(filters=self.n_outputs, kernel_size=self.kernel_size, strides=self.strides,
                                   padding='causal', dilation_rate=self.dilation_rate, name="conv2")
        self.conv2 = tfa.layers.WeightNormalization(self.conv2)

        # noise_shape ensures the same dropout is applied across batch and time steps
        self.dropout1 = layers.Dropout(self.dropout, noise_shape=(1, 1, self.n_outputs))
        self.dropout2 = layers.Dropout(self.dropout, noise_shape=(1, 1, self.n_outputs))

        if self.n_outputs != input_shape[-1]:
            # we will need a linear layer for the skip connection
            self.conv3 = layers.Conv1D(filters=self.n_outputs, kernel_size=1, name='conv3')

    def call(self, inputs, training=True, **kwargs):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)

        if self.conv3 is not None:
            res = self.conv3(inputs)
        else:
            res = inputs

        return x + res  # skip connection!

class temporalConvolutionNetwork(layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2, dilation_rate=2):
        super(temporalConvolutionNetwork, self).__init__()

        self.blocks = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = dilation_rate ** i
            self.blocks.append(temporalBlockTf(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                                               dropout=dropout, name=f"tblock_{i}"))

    def call(self, inputs, training=True, **kwargs):
        outputs = inputs
        for layer in self.blocks:
            outputs = layer(outputs, training=training)
        return outputs


#################################
#             N-BEATS           #
#################################
class GenericBlock(keras.layers.Layer):
    """
    Generic block definition as described in the paper.
    We can't have explanation from this kind of block because g coefficients are learnt.

    Parameters
    ----------
    horizon     : (int) Future interval to predict
    lookback    : (int) Past interval given
    n_neurons   : (int) Number of neurons in fully connected layers
    n_quantiles : (int) Number of quantiles in `QuantileLossError'
    dropout_rate: (int) Dropout rate
    """

    def __init__(self, horizon, lookback, n_neurons, n_quantiles, dropout_rate, **kwargs):
        super(GenericBlock, self).__init__(**kwargs)

        # 4 dense layers for the fully connected block
        self._fc_stack = [keras.layers.Dense(n_neurons, activation='relu',
                                             kernel_initializer="glorot_uniform") for _ in range(4)]
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._fc_backcast = self.add_weight(shape=(n_neurons, n_neurons), trainable=True,
                                            initializer="glorot_uniform", name='fc_backcast')
        self._fc_forecast = self.add_weight(shape=(n_quantiles, n_neurons, n_neurons), trainable=True,
                                            initializer="glorot_uniform", name='fx_forecast')
        self._lin_backcast = keras.layers.Dense(lookback, kernel_initializer="glorot_uniform")
        self._lin_forecast = keras.layers.Dense(horizon, kernel_initializer="glorot_uniform")

    def call(self, inputs, training=True, **kwargs):
        x = inputs  # batch_size x lookback
        for dense_layer in self._fc_stack:
            x = dense_layer(x)  # batch_size x nb_neurons
            x = self._dropout(x, training=training)

        theta_forecast = x @ self._fc_forecast  # n_quantiles x batch_size x n_neurons
        theta_backcast = x @ self._fc_backcast  # batch_size x n_neurons

        y_forecast = self._lin_forecast(theta_forecast)  # n_quantiles x batch_size x horizon
        y_backcast = self._lin_backcast(theta_backcast)  # batch_size x lookback

        return y_forecast, y_backcast


class TrendBlock(layers.Layer):
    """
    Trend block definition. Output layers are constrained to polynomial functions
    of small degree. Therefore it is possible to get explanation from this block.

    Parameters
    ----------
    horizon     : (int) Future interval to predict
    lookback    : (int) Past interval given
    n_neurons   : (int) Number of neurons in Fully connected layers
    n_quantiles : (int) Number of quantiles in `QuantileLossError'
    dropout_rate: (int) Dropout rate
    p_degree    : (int) Degree of the polynomial
    """

    def __init__(self, horizon, lookback, n_neurons, n_quantiles, dropout_rate, p_degree, **kwargs):
        super(TrendBlock, self).__init__(**kwargs)

        # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self._p_degree = tf.reshape(tf.range(p_degree + 1, dtype='float32'), shape=(-1, 1))
        self._horizon = tf.cast(horizon, dtype='float32')
        self._lookback = tf.cast(lookback, dtype='float32')
        self._n_neurons = n_neurons
        self._n_quantiles = n_quantiles

        self._fc_stack = [tf.keras.layers.Dense(n_neurons, activation='relu',
                                                kernel_initializer="glorot_uniform") for _ in range(4)]
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._fc_backcast = self.add_weight(shape=(n_neurons, p_degree + 1), trainable=True,
                                            initializer="glorot_uniform", name='fc_backcast')
        self._fc_forecast = self.add_weight(shape=(n_quantiles, n_neurons, p_degree + 1), trainable=True,
                                            initializer="glorot_uniform", name='fc_forecast')

        self._forecast_basis = (tf.range(self._horizon) / self._horizon) ** self._p_degree
        self._backcast_basis = (tf.range(self._lookback) / self._lookback) ** self._p_degree

    def call(self, inputs, training=True, **kwargs):
        x = inputs  # batch_size x lookback
        for dense_layer in self._fc_stack:
            x = dense_layer(x)  # batch_size x n_neurons
            x = self._dropout(x, training=training)

        theta_forecast = x @ self._fc_forecast  # n_quantiles x batch_size x p_degree
        theta_backcast = x @ self._fc_backcast  # batch_size x p_degree

        y_forecast = theta_forecast @ self._forecast_basis  # shape: n_quantiles x batch_size x horizon
        y_backcast = theta_backcast @ self._backcast_basis  # shape: batch_size x lookback

        return y_forecast, y_backcast


class SeasonalBlock(layers.Layer):
    """
    Seasonal block definition. Output layers are set to be the appropriate fourier series.
    Each expansion coefficient becomes a coefficient of the fourier series.

    Parameters
    ----------
    horizon     : (int) Future interval to predict
    lookback    : (int) Past interval given
    n_neurons   : (int) Number of neurons in Fully connected layers
    n_quantiles : (int) Number of quantiles in `QuantileLossError'
    dropout_rate: (int) Dropout rate
    fore_periods: (int) number of seasonal periods in the forecast interval. For eg. if we are dealing with annual
                        seasonality in a 1y forecast then periods=1
    back_periods: (int) number of seasonal periods in the backcast interval. For eg. if we are dealing with annual
                        seasonality in a 1y backcast then periods=1
    forecast_order: (int) Number of Fourier series to use. Higher values signifies complex fourier series
    backcast_order: (int) Number of Fourier series to use. Higher values signifies complex fourier series
    """

    def __init__(self, horizon, lookback, n_neurons, n_quantiles, dropout_rate,
                 fore_periods, back_periods, forecast_order, backcast_order, **kwargs):
        super(SeasonalBlock, self).__init__(**kwargs)

        self._horizon = horizon
        self._lookback = lookback

        # Broadcast horizons on multiple periods
        self._fore_periods = tf.cast(tf.reshape(fore_periods, (1, -1)), 'float32')
        self._back_periods = tf.cast(tf.reshape(back_periods, (1, -1)), 'float32')
        self._forecast_order = tf.reshape(tf.range(forecast_order, dtype='float32'), shape=(-1, 1))
        self._backcast_order = tf.reshape(tf.range(backcast_order, dtype='float32'), shape=(-1, 1))

        # Workout the number of neurons needed to compute seasonality coefficients
        horizon_neurons = tf.reduce_sum(2 * forecast_order)
        lookback_neurons = tf.reduce_sum(2 * backcast_order)

        self._fc_stack = [tf.keras.layers.Dense(n_neurons, activation='relu',
                                                kernel_initializer="glorot_uniform") for _ in range(4)]
        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        self._fc_backcast = self.add_weight(shape=(n_neurons, lookback_neurons), trainable=True,
                                            initializer="glorot_uniform", name='fc_backcast')
        self._fc_forecast = self.add_weight(shape=(n_quantiles, n_neurons, horizon_neurons), trainable=True,
                                            initializer="glorot_uniform", name='fc_forecast')

        # Workout cos and sin seasonal basis
        time_horizon = tf.range(self._horizon, dtype='float32') / self._fore_periods
        horizon_seasonality = 2 * np.pi * self._forecast_order * time_horizon
        self._forecast_basis = tf.concat((tf.cos(horizon_seasonality), tf.sin(horizon_seasonality)), axis=0)

        time_lookback = tf.range(self._lookback, dtype='float32') / self._back_periods
        lookback_seasonality = 2 * np.pi * self._backcast_order * time_lookback
        self._backcast_basis = tf.concat((tf.cos(lookback_seasonality), tf.sin(lookback_seasonality)), axis=0)

    def call(self, inputs, training=True, **kwargs):
        x = inputs  # batch_size x lookback
        for dense in self._fc_stack:
            x = dense(x)  # batch_size x n_neurons
            x = self._dropout(x, training=training)

        theta_forecast = x @ self._fc_forecast  # n_quantiles x batch_size x (2 * forecast_order)
        theta_backcast = x @ self._fc_backcast  # batch_size x (2 * backcast_order)

        y_forecast = theta_forecast @ self._forecast_basis  # n_quantiles x batch_size x horizon
        y_backcast = theta_backcast @ self._backcast_basis  # batch_size x lookback

        return y_forecast, y_backcast


class Stack(tf.keras.layers.Layer):
    """
    A stack is a series of blocks where each block produces two outputs, the forecast and the backcast.
    The residual between the input and the backcast at each block is passed as input to the next block.
    The block forecasts are summed to get the stack forecast.

    Parameters
    ----------
    blocks: list of trend, seasonal or generic blocks
    """

    def __init__(self, blocks, **kwargs):
        super(Stack, self).__init__(**kwargs)
        self._blocks = blocks

    def call(self, inputs, training=True, **kwargs):
        stack_forecast = 0.
        stack_residual = inputs
        for block in self._blocks:
            block_forecast, block_backcast = block(stack_residual,
                                                   training=training,
                                                   **kwargs)  # n_quantiles x batch_size x horizon, batch_size x lookback
            stack_residual = tf.subtract(stack_residual, block_backcast)  # residual
            stack_forecast = tf.add(stack_forecast, block_forecast)  # n_quantiles x batch_size x horizon
        return stack_forecast, stack_residual


class nBEATS(tf.keras.Model):
    """
    N-BEATS is a univariate model which can be interpretable or generic. It's strong advantage is its
    internal structure which allows us to extract the trend and the seasonality of a time series,
    available from the attributes `seasonality` and `trend`.

    N-BEATS: Neural basis expansion analysis for interpretable time series horizoning,
    Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados and Yoshua Bengio

    Parameters
    ----------
    stacks: list of `Stack` layers for the nbeats model. It can be full of `Trend`, `Seasonal` or `Genereic` blocks
    """

    def __init__(self, stacks, **kwargs):
        super(nBEATS, self).__init__(**kwargs)
        self._stacks = stacks
        self._stack_forecasts = None

    def call(self, inputs, training=True, **kwargs):
        # self._stack_forecasts = tf.TensorArray(tf.float32, size=len(self._stacks))  # trend/seasonality during inference
        model_forecast = 0.
        residual = inputs
        for idx, stack in enumerate(self._stacks):
            stack_forecast, residual = stack(residual, training=training, **kwargs)
            # self._stack_forecasts.write(idx, stack_forecast)
            model_forecast = tf.add(model_forecast, stack_forecast)
        return model_forecast

    def trend(self, inputs, **kwargs):
        stack_forecast, _ = self._stacks[0](inputs, training=False, **kwargs)
        return stack_forecast

    def seasonal(self, inputs, **kwargs):
        _, residual = self._stacks[0](inputs, training=False, **kwargs)
        stack_forecast, _ = self._stacks[1](residual, training=False, **kwargs)
        return stack_forecast

#################################
#           Transformer         #
#################################
def get_angles(pos, i, d_model, base=10000):
    angle_rates = 1 / np.power(base, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model, base=10000):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model, base)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention from "Attention is all you need"
    tensorflow_addons also implements this as a layer but does not allow a bias in the
    linear transformation of the query, key and value on input. This implementation is
    adapted from https://www.tensorflow.org/text/tutorials/transformer but the arguments
    have been redefined to match the tfa implementation of multi-head attention.
    """
    def __init__(self, head_size, num_heads, output_size=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size  # = num_heads x embed_dim. Therefore d_model has to be divisible by num_heads
        self.d_model = head_size * num_heads
        self.output_size = output_size

        self.wq = None
        self.wk = None
        self.wv = None
        self.dense = None

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        output_size = self.output_size if self.output_size is not None else num_value_features

        self.wq = layers.Dense(self.d_model, input_shape=(num_query_features,))
        self.wk = layers.Dense(self.d_model, input_shape=(num_key_features,))
        self.wv = layers.Dense(self.d_model, input_shape=(num_value_features,))
        self.dense = layers.Dense(output_size)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, head_size)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def dot_product_attention(q, k, v, mask=None):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions which equals the embedding dim
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
            q   : query ... x seq_len_q x embed_depth
            k   : key ... x seq_len_k x embed_depth
            v   : value ... x seq_len_v x depth_v
            mask: Float tensor with shape broadcastable
                  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            dpa, attn_wts
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        attn_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor
        if mask is not None:
            attn_logits += (mask * -1e9)  # sets mask values to -inf => softmax=0

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attn_wts = tf.nn.softmax(attn_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        dpa = tf.matmul(attn_wts, v)  # (..., seq_len_q, depth_v)
        return dpa, attn_wts

    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # batch_size x seq_len x d_model
        k = self.wk(k)  # batch_size x seq_len x d_model
        v = self.wv(v)  # batch_size x seq_len x d_model

        q = self.split_heads(q, batch_size)  # batch_size x num_heads x seq_len_q x head_size
        k = self.split_heads(k, batch_size)  # batch_size x num_heads x seq_len_k x head_size
        v = self.split_heads(v, batch_size)  # batch_size x num_heads x seq_len_v x head_size

        # attn_mat is batch_size x num_heads x seq_len_q x head_size
        # attn_wts is batch_size x num_heads x seq_len_q x seq_len_k
        attn_mat, attn_wts = self.dot_product_attention(q, k, v, mask)

        attn_mat = tf.transpose(attn_mat, perm=[0, 2, 1, 3])  # batch_size x seq_len_q x num_heads x head_size
        attn_mat = tf.reshape(attn_mat, (batch_size, -1, self.d_model))  # batch_size x seq_len_q x d_model
        attn_mat = self.dense(attn_mat)  # (batch_size, seq_len_q, output_size)

        return attn_mat, attn_wts


class EncoderBlock(layers.Layer):
    def __init__(self, head_size, num_heads, dff, rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(head_size, num_heads)
        self.fc1 = layers.Dense(dff, activation='relu')
        self.fc2 = None  # layers.Dense(head_size * num_heads)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape):
        self.fc2 = layers.Dense(input_shape[-1])

    def call(self, input, training=True, mask=None, **kwargs):
        attn, _ = self.mha([input, input, input], mask=mask)  # (batch_size, input_seq_len, d_model)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(input + attn)  # (batch_size, input_seq_len, d_model)

        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.fc1(out1)
        ffn_output = self.fc2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderBlock(layers.Layer):
    def __init__(self, head_size, num_heads, dff, rate=0.1):
        super(DecoderBlock, self).__init__()

        self.head_size = head_size
        self.num_heads = num_heads

        self.mha1 = MultiHeadAttention(head_size, num_heads)
        self.mha2 = None  # MultiHeadAttention(head_size, num_heads)

        # self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.fc1 = layers.Dense(dff, activation='relu')
        self.fc2 = None  # layers.Dense(head_size * num_heads)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def build(self, input_shape):
        self.mha2 = MultiHeadAttention(self.head_size, self.num_heads, output_size=input_shape[0][-1])
        self.fc2 = layers.Dense(input_shape[0][-1])

    def call(self, inputs, training=True, look_ahead_mask=None, padding_mask=None, **kwargs):
        x, enc_output = inputs  # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_wts1 = self.mha1([x, x, x], mask=look_ahead_mask)  # (batch_size, tgt_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_wts2 = self.mha2([out1, enc_output, enc_output], mask=padding_mask)  # (batch_size, tgt_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, tgt_seq_len, d_model)

        ffn_output = self.fc1(out2)
        ffn_output = self.fc2(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, tgt_seq_len, d_model)

        return out3, attn_wts1, attn_wts2


class Encoder(layers.Layer):
    def __init__(self, embed_dim,
                 num_layers, head_size, num_heads, dff,
                 input_vocab_size, maximum_position_encoding,
                 rate=0.1, base=10000):
        super(Encoder, self).__init__()

        self.head_size = head_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = layers.Embedding(input_vocab_size, embed_dim)
        self.base = base
        self.pos_encoding = positional_encoding(maximum_position_encoding, embed_dim, base)

        self.enc_layers = [EncoderBlock(head_size, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training=True, mask=None, **kwargs):
        seq_len = tf.shape(inputs)[-1]

        # adding embedding and position encoding.
        x = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(layers.Layer):
    def __init__(self, embed_dim,
                 num_layers, head_size, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, rate=0.1, base=10000):
        super(Decoder, self).__init__()

        self.head_size = head_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = layers.Embedding(target_vocab_size, embed_dim)
        self.base = base
        self.pos_encoding = positional_encoding(maximum_position_encoding, embed_dim, base)

        self.dec_layers = [DecoderBlock(head_size, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training=True, look_ahead_mask=None, padding_mask=None, **kwargs):
        x, enc_output = inputs
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i]([x, enc_output],
                                                   training=training,
                                                   look_ahead_mask=look_ahead_mask,
                                                   padding_mask=padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self, inp_embed_dim, out_embed_dim, num_layers, head_size, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, base=10000):
        super().__init__()
        self.encoder = Encoder(inp_embed_dim, num_layers, head_size, num_heads,
                               dff, input_vocab_size, pe_input, rate, base)
        self.decoder = Decoder(out_embed_dim, num_layers, head_size, num_heads,
                               dff, target_vocab_size, pe_target, rate, base)
        self.final_layer = layers.Dense(target_vocab_size, activation='softmax')

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="acc")

    @staticmethod
    def create_padding_mask(seq):
        """
        Mask for self-attention layers. mask==1 => the value is ignored
        Here the mask is used to ignore the padding zeros
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(tar):
        """
        Mask for encoder-decoder attention. mask==1 => the value is ignored
        Here the mask is used to prevent looking ahead in the decoder.
        """
        size = tf.shape(tar)[1]
        # upper-triangular part of the matrix with zeros along diagonal
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tar)
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def call(self, inputs, training=True):
        inp, tgt = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tgt)
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        dec_output, attn_wts = self.decoder((tgt, enc_output),  # (batch_size, tar_seq_len, d_model)
                                            training,
                                            look_ahead_mask,
                                            dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

    @staticmethod
    def masked_loss(y_true, y_pred):
        mask = tf.logical_not(tf.math.equal(y_true, 0))
        loss_ = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss_ = tf.where(mask, loss_, tf.zeros_like(loss_))  # set masked values to 0
        mask = tf.cast(mask, dtype=loss_.dtype)
        return tf.reduce_sum(loss_, axis=-1) / tf.reduce_sum(mask, axis=-1)

    @staticmethod
    def masked_acc(y_true, y_pred):
        mask = tf.logical_not(tf.math.equal(y_true, 0))
        acc_ = tf.cast(tf.equal(y_true, tf.cast(tf.argmax(y_pred, axis=2), dtype=y_true.dtype)),
                       dtype=tf.float32)
        acc_ = tf.where(mask, acc_, tf.zeros_like(acc_))  # set masked values to 0
        mask = tf.cast(mask, dtype=acc_.dtype)
        return tf.reduce_sum(acc_, axis=-1) / tf.reduce_sum(mask, axis=-1)

    def train_step(self, data):
        inp, tgt = data
        tgt_inp = tgt[:, :-1]
        tgt_real = tgt[:, 1:]
        with tf.GradientTape() as tape:
            preds = self.call((inp, tgt_inp), training=True)
            loss = tf.reduce_mean(self.masked_loss(tgt_real, preds))
            acc = tf.reduce_mean(self.masked_acc(tgt_real, preds))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.updateState(loss)
        self.acc_tracker.updateState(acc)

        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}


class Time2Vec(layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

        self.wb = None
        self.bb = None
        self.wa = None
        self.ba = None

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1],), initializer='uniform', trainable=True)

        # periodic
        self.wa = self.add_weight(name='wa', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp)  # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] * (self.k + 1)


class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = tfa.layers.MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x

class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [
            AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in
            range(num_layers)]

    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2]))  # flat vector of features out


if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras import Sequential

    # # test the temporal block works with a numpy array
    # x = np.random.normal(size=(32,  # batch_size
    #                            100,  # times
    #                            1))  # channels
    # tblock = TemporalBlock(8, 2, 1, 1)
    # output = tblock(x, training=True)

    # # Sequential MNIST with TCNs
    # # params
    # epochs = 50
    # batch_size = 64
    #
    # # data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train / 255
    # x_test = x_test / 255
    #
    # # targets are loaded as labels, NOT as one hot vectors
    # # therefore, we use categorical_crossentropy as the loss
    # dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape((-1, 28 * 28, 1)).astype('float32'),
    #                                               y_train.reshape((-1, 1)).astype('float32')))
    # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    # print(dataset.element_spec)
    #
    # # custom loss for sequential MNIST
    # # get the cross entropy only on the final outputs for each sequence
    # def seq_mnist_loss(y_true, y_pred):
    #     return tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred[:, -1, :]))
    #
    # # model
    # model = Sequential([TCN(num_channels=[20] * 6, kernel_size=8, dropout=0.1),
    #                     layers.Dense(10, activation='softmax', name='softmax')])
    #
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #               loss=seq_mnist_loss)  # 'sparse_categorical_crossentropy')
    # model.fit(dataset, epochs=epochs)
    #
    # # class TCNMonitor(keras.callbacks.Callback):
    # #     def __init__(self, num_imgs=(4, 4), y_true):
    # #         self.num_imgs = num_imgs
    # #         self.tot_imgs = np.cumprod(num_imgs)[-1]
    # #         self.y_true = y_true
    # #         self.idxs = np.random.randint(0, y_true.shape[0], self.tot_figs)
    # #
    # #     def on_epoch_end(self, epoch, logs=None):
    # #
    # #         fig = plt.figure(figsize=self.num_imgs)
    # #         for i in range(self.tot_imgs):
    # #             plt.subplot(self.num_imgs[0], self.num_imgs[1], i + 1)
    # #             barlist = plt.bar(np.arange(10), y_pred[k, -1, :].numpy(), color='b')
    # #             barlist[y_true[k]].set_color('r')
    # #
    # #         plt.savefig("../../../results/mnist_gan_%04d.png" % (epoch))
    # #         plt.close()
    #
    # y_pred = model(x_test[:1000, :, :].reshape((-1, 28*28, 1)).astype('float32'))
    #
    # def plot_chart(y_true, y_pred, n_figs):
    #     import matplotlib.pyplot as plt
    #     tot_figs = np.cumprod(n_figs)[-1]
    #
    #     idxs = np.random.randint(0, y_pred.shape[0], tot_figs)
    #     fig = plt.figure(figsize=n_figs)
    #     for i, k in enumerate(idxs):
    #         # print(k)
    #         plt.subplot(n_figs[0], n_figs[1], i + 1)
    #         barlist = plt.bar(np.arange(10), y_pred[k, -1, :].numpy(), color='b')
    #         barlist[y_true[k]].set_color('r')
    #
    #     plt.savefig("./results/tcn_seq_mnist.png")
    #     plt.show()
    #
    # plot_chart(y_test[:1000], y_pred, (4, 4))

    # ## Transformer ###
    # # check dot=product attention
    # temp_q = tf.constant([[0, 0, 10],
    #                       [0, 10, 0],
    #                       [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    # temp_k = tf.constant([[10, 0, 0],
    #                       [0, 10, 0],
    #                       [0, 0, 10],
    #                       [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    # temp_v = tf.constant([[1, 0],
    #                       [10, 0],
    #                       [100, 5],
    #                       [1000, 6]], dtype=tf.float32)  # (4, 2)
    #
    # temp_out, temp_attn = MultiHeadAttention.dot_product_attention(temp_q, temp_k, temp_v, None)
    # print('Attention weights are:')
    # print(temp_attn)
    # print('Output is:')
    # print(temp_out)
    #
    # # check multi-head attention
    # temp_mha = MultiHeadAttention(head_size=64, num_heads=8)
    # y = tf.random.uniform((1, 60, 100))  # (batch_size, encoder_sequence, d_model)
    # out, attn = temp_mha([y, y, y], None)
    # print(out.shape, attn.shape)
    #
    # # check encoder block
    # sample_encoder_layer = EncoderBlock(64, 8, 2048)
    # sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 100)), training=False, mask=None)
    # print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    #
    # # check decoder block
    # sample_decoder_layer = DecoderBlock(64, 8, 2048)
    # sample_decoder_layer_output, _, _ = sample_decoder_layer([tf.random.uniform((64, 50, 150)),
    #                                                           sample_encoder_layer_output],
    #                                                          training=False,
    #                                                          look_ahead_mask=None,
    #                                                          padding_mask=None)
    # print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
    #
    # # check encoder
    # sample_encoder = Encoder(embed_dim=1000, num_layers=2, head_size=64, num_heads=8, dff=2048,
    #                          input_vocab_size=200, maximum_position_encoding=10000)
    # temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    # sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
    # print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    #
    # # check decoder
    # sample_decoder = Decoder(embed_dim=2000, num_layers=2, head_size=64, num_heads=8, dff=2048,
    #                          target_vocab_size=8000, maximum_position_encoding=5000)
    # temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
    # output, attn = sample_decoder((temp_input, sample_encoder_output),
    #                               training=False, look_ahead_mask=None, padding_mask=None)
    # print(output.shape, attn['decoder_layer2_block2'].shape)
    #
    # # check transformer
    # sample_transformer = Transformer(inp_embed_dim=1000, out_embed_dim=2000,
    #                                  num_layers=2, head_size=64, num_heads=8, dff=2048,
    #                                  input_vocab_size=8500, target_vocab_size=8000,
    #                                  pe_input=10000, pe_target=6000)
    # temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    # temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    #
    # fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
    #
    # print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    num_samples = 10000  # Number of samples to train on.
    base_dim = 10000
    data_path = "../../../../data/fra.txt"  # Path to the data txt file on disk.

    # Vectorize the data.
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    inp_lines = [line.split("\t")[0] for line in lines[: min(num_samples, len(lines) - 1)]]
    tgt_lines = ["\t" + line.split("\t")[1] + "\n" for line in lines[: min(num_samples, len(lines) - 1)]]
    inp_chars = sorted(list(set(''.join(inp_lines))))
    tgt_chars = sorted(list(set(''.join(tgt_lines))))

    n_enc_tokens = len(inp_chars)
    n_dec_tokens = len(tgt_chars)
    max_enc_len = max([len(txt) for txt in inp_lines])
    max_dec_len = max([len(txt) for txt in tgt_lines])

    inp_token_dict = dict([(char, i + 1) for i, char in enumerate(inp_chars)])
    tgt_token_dict = dict([(char, i + 1) for i, char in enumerate(tgt_chars)])

    # Reverse-lookup token index to decode sequences
    inp_token_dict_rev = dict((i, char) for char, i in inp_token_dict.items())
    tgt_token_dict_rev = dict((i, char) for char, i in tgt_token_dict.items())

    enc_inp_data = np.array([[inp_token_dict[l] for l in line] + [0] * (max_enc_len - len(line)) for line in inp_lines])
    dec_inp_data = np.array([[tgt_token_dict[l] for l in line] + [0] * (max_dec_len - len(line)) for line in tgt_lines])

    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((enc_inp_data[:num_samples, :],
                                                  dec_inp_data[:num_samples, :]))
    dataset = dataset.shuffle(buffer_size=1024).batch(128)
    # print(dataset.element_spec)
    # el = dataset.take(5)

    # learning schedule
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=1000):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            config = {'d_model': self.d_model,
                      'warmup_steps': self.warmup_steps}
            return config

    # Hyper parameter search
    import keras_tuner as kt

    def model_builder(hp):
        embed_dim = hp.Int('embed_dim', min_value=4, max_value=32, step=4)
        n_layers = hp.Int('n_layers', min_value=2, max_value=5, step=1)
        head_size = hp.Int('head_size', min_value=8, max_value=40, step=4)
        n_heads = hp.Int('n_heads', min_value=8, max_value=40, step=4)
        dff = hp.Int('dff', min_value=256, max_value=512, step=32)

        model = Transformer(inp_embed_dim=embed_dim, out_embed_dim=embed_dim,
                            num_layers=n_layers, head_size=head_size, num_heads=n_heads, dff=dff,
                            input_vocab_size=n_enc_tokens + 1, target_vocab_size=n_dec_tokens + 1,
                            pe_input=max_enc_len, pe_target=max_dec_len, rate=0.1, base=base_dim)

        learning_rate = CustomSchedule(head_size * n_heads)  # num_heads * head_size
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        return model

    tuner = kt.Hyperband(model_builder, objective='loss', max_epochs=10,
                         factor=3, directory='../../../../tuning', project_name=f's2s_char_trans_{base_dim}')

    tuner.search(dataset, epochs=20)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f'Embed dim : {best_hps.get("embed_dim")}')
    print(f'Num layers : {best_hps.get("n_layers")}')
    print(f'Head size : {best_hps.get("head_size")}')
    print(f'Num heads : {best_hps.get("n_heads")}')
    print(f'FeedFwd dim : {best_hps.get("dff")}')

    # these parameters were chosen by the keras tuner run above
    model = Transformer(inp_embed_dim=best_hps.get("embed_dim"), out_embed_dim=best_hps.get("embed_dim"),
                        num_layers=best_hps.get("n_layers"), head_size=best_hps.get("head_size"),
                        num_heads=best_hps.get("n_heads"), dff=best_hps.get("dff"),
                        input_vocab_size=n_enc_tokens + 1, target_vocab_size=n_dec_tokens + 1,
                        pe_input=max_enc_len, pe_target=max_dec_len, rate=0.1, base=base_dim)

    preds = model([enc_inp_data[:10, :], dec_inp_data[:10, :-1]])
    print(model.count_params())
    # Transformer.masked_loss(dec_inp_data[:10, 1:], preds)
    # Transformer.masked_acc(dec_inp_data[:10, 1:], preds)

    learning_rate = CustomSchedule(best_hps.get("head_size") * best_hps.get("n_heads"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    model.fit(dataset, epochs=500)

    model.save_weights(f"../../../../models/s2s_char_trans_{base_dim}")

