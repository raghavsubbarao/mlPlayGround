import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

# TODO: Add type specs

#################################
#               RBM             #
#################################
# class transposeLayer(layers.Dense):
#     """
#     Transpose layer for Restricted Boltzmann Machines
#     """
#     def __init__(self, denseLayer,
#                  activation=None, use_bias=True,
#                  bias_initializer='zeros',
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  bias_constraint=None,
#                  **kwargs):
#         """
#         :param denseLayer:
#         :param activation:
#         :param use_bias:
#         :param bias_initializer:
#         :param bias_regularizer:
#         :param activity_regularizer:
#         :param bias_constraint:
#         :param kwargs:
#         """
#         self.__denseLayer = denseLayer
#
#         super(transposeLayer, self).__init__(denseLayer._batch_input_shape[-1],
#                                              activation=activation,
#                                              use_bias=use_bias,
#                                              bias_initializer=bias_initializer,
#                                              bias_regularizer=bias_regularizer,
#                                              activity_regularizer=activity_regularizer,
#                                              bias_constraint=bias_constraint,
#                                              input_shape=(denseLayer.units,),
#                                              **kwargs)
#
#     @property
#     def denseLayer(self):
#         return self.__denseLayer
#
#     def build(self, input_shape):
#         dtype = dtypes.as_dtype(self.dtype or K.floatx())
#         if not (dtype.is_floating or dtype.is_complex):
#             raise TypeError('Unable to build `Dense` layer with non-floating point '
#                             'dtype %s' % (dtype,))
#
#         input_shape = tensor_shape.TensorShape(input_shape)
#         last_dim = tensor_shape.dimension_value(input_shape[-1])
#         if last_dim is None:
#             raise ValueError('The last dimension of the inputs to `Dense` '
#                              'should be defined. Found `None`.')
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
#
#         # self.kernel = tf.transpose(self.__denseLayer.kernel)
#
#         if self.use_bias:
#             self.bias = self.add_weight('bias',
#                                         shape=[self.units,],
#                                         initializer=self.bias_initializer,
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint,
#                                         dtype=self.dtype,
#                                         trainable=True)
#         else:
#             self.bias = None
#
#         self.built = True
#
#     def call(self, inputs):
#         z = tf.matmul(inputs, tf.transpose(self.denseLayer.weights[0]))
#         if self.use_bias:
#             z = tf.add(z, self.bias)
#         return z

class RBM(keras.Model):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()

        # self.wt = layers.Dense(units=n_hid, input_shape=(n_vis,), use_bias=True,
        #                        kernel_initializer="glorot_uniform", bias_initializer="zeros")
        # self.hidden = Sequential([self.wt])
        #
        # self.w = transposeLayer(self.wt, use_bias=True, bias_initializer='zeros')
        # self.visible = Sequential([self.w])
        k = np.sqrt(6.0 / (n_vis + n_hid))
        self.W = tf.Variable(tf.random.uniform((n_vis, n_hid), minval=-k, maxval=k, dtype=tf.float32))
        self.vb = tf.Variable(tf.zeros((n_vis,), dtype=tf.float32))
        self.hb = tf.Variable(tf.zeros((n_hid,), dtype=tf.float32))

        self.optimizer = None

        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.pos_loss_tracker = keras.metrics.Mean(name="pos_loss")
        self.neg_loss_tracker = keras.metrics.Mean(name="neg_loss")

    def compile(self, optimizer):
        super(RBM, self).compile()
        self.optimizer = optimizer

    def p_hidden(self, v):
        """
        Computes hidden state from the input.
        :param v: tensor of shape (batch_size, n_visible)
        :return: tensor of shape (batch_size, n_hidden)
        """
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.hb)

    def p_visible(self, h):
        """
        Computes visible state from hidden state.
        :param h: tensor of shape (batch_size, n_hidden)
        :return: tensor of shape (batch_size, n_visible)
        """
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vb)

    def reconstruct(self, x):
        """
        Computes visible state from the input. Reconstructs data.
        :param x: tensor of shape (batch_size, n_visible)
        :return: tensor of shape (batch_size, n_visible)
        """
        def sample_bernoulli(p):
            return tf.nn.relu(tf.sign(p - tf.random.uniform(tf.shape(p))))

        return self.p_visible(sample_bernoulli(self.p_hidden(x)))

    def free_energy(self, v):
        f1 = tf.reduce_sum(tf.math.log(1. + tf.math.exp(tf.matmul(v, self.W) + self.hb)), axis=1)
        f2 = tf.reduce_sum(tf.multiply(v, self.vb), axis=1)
        # tf.reduce_sum(tf.multiply(v, self.visible(tf.zeros_like(hidden_p))), axis=1)
        return - f1 - f2

    def energy(self, h, v):
        l1 = tf.reduce_sum(tf.multiply(h, tf.matmul(v, self.W) + self.hb), axis=1)
        l2 = tf.reduce_sum(tf.multiply(v, self.vb), axis=1)
        return l1 + l2

    def gibbs_sample(self, v, k=1):

        def sample_bernoulli(p):
            """
            p is the probability of 1
            """
            return tf.nn.relu(tf.sign(p - tf.random.uniform(tf.shape(p))))

        hidden_p = self.p_hidden(v)

        for i in range(k):
            visible_p = self.p_visible(sample_bernoulli(hidden_p))
            hidden_p = self.p_hidden(visible_p)

        return hidden_p, visible_p

    def call(self, inputs, training=None, mask=None):
        return self.p_hidden(inputs)

    def train_step(self, data):

        _, visible_z = self.gibbs_sample(data, 1)

        with tf.GradientTape() as tape:
            pos_loss = tf.reduce_mean(self.free_energy(data))
            neg_loss = tf.reduce_mean(self.free_energy(visible_z))
            total_loss = pos_loss - neg_loss
            recon_loss = tf.reduce_mean(tf.square(data - visible_z))
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.recon_loss_tracker.updateState(recon_loss)
        self.total_loss_tracker.updateState(total_loss)
        self.pos_loss_tracker.updateState(pos_loss)
        self.neg_loss_tracker.updateState(neg_loss)

        return {"recon_loss": self.recon_loss_tracker.result(),
                "total_loss": self.total_loss_tracker.result(),
                "pos_loss": self.pos_loss_tracker.result(),
                "neg_loss": self.neg_loss_tracker.result()}

#################################
#               VAE             #
#################################
class normalSamplingTf(layers.Layer):
    """
    Sampling layer for the re-parametrization trick. Given the normal
    distribution parameters (z_mean, z_log_var) samples from the pdf

    Has no trainable parameters so need to define the __init__
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class variationalAutoencoderTf(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        # initialization
        super(variationalAutoencoderTf, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = normalSamplingTf()

        # beta = 1.0 for Kingma, Welling but can be adjusted
        # for beta-VAEs. beta > 1 allows disentangling generative factors
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker]

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=None, mask=None):
        return self.encode(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sampler([z_mean, z_log_var])
            reconstruction = self.decode(z)

            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction),
                                                      axis=(1, 2)))
            kl_loss = tf.reduce_mean(tf.reduce_sum((tf.exp(z_log_var) + tf.square(z_mean) - z_log_var) / 2.,
                                                   axis=1))
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.updateState(total_loss)
        self.recon_loss_tracker.updateState(recon_loss)
        self.kl_loss_tracker.updateState(kl_loss)

        return {"total_loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


# class CVAE(VAE):
#     """Convolutional variational autoencoder for MNIST"""
#
#     def __init__(self, latent_dim):
#
#         self.latent_dim = latent_dim
#         encoder = self.encoder_network()
#         decoder = self.decoder_network()
#
#         super(CVAE, self).__init__(encoder, decoder)
#
#     def encoder_network(self):
#         model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
#                                      tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
#                                                             activation='relu', padding='same'),
#                                      tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
#                                                             activation='relu', padding='same'),
#                                      tf.keras.layers.Flatten(),
#                                      # output mean and log variance (for numerical stability)
#                                      tf.keras.layers.Dense(self.latent_dim + self.latent_dim),  # No activation
#                                      ], name='encoder')
#         return model
#
#     def decoder_network(self):
#         model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
#                                      tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
#                                      tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
#                                      tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
#                                                                      padding='same', activation='relu'),
#                                      tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
#                                                                      padding='same', activation='relu'),
#                                      tf.keras.layers.Conv2DTranspose(filters=1,  kernel_size=3, strides=1,
#                                                                      padding='same', activation='sigmoid'),
#                                      ], name="decoder")
#         return model

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(initial_value=w_init(shape=(self.embedding_dim,
                                                                  self.num_embeddings),
                                                           dtype="float32"),
                                      trainable=True, name="embeddings_vqvae")

    def call(self, x):
        # Calculate the input shape of the inputs then
        # flatten inputs keeping `embedding_dim` intact
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # # Calculate vector quantization loss and add that to the layer.
        # # You can learn more about adding losses to different layers here:
        # # https://keras.io/guides/making_new_layers_and_models_via_subclassing/
        # code_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)  # codebook loss
        # comm_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)  # commitment loss
        # self.add_loss(comm_loss + code_loss)
        #
        # # Straight-through estimator
        # quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-norm between inputs and codes
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                     + tf.reduce_sum(self.embeddings ** 2, axis=0)
                     - 2 * similarity)

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class VQVAE(keras.Model):
    def __init__(self, encoder, decoder,
                 latent_dim, num_embeddings,
                 beta=0.25, data_var=1.0, **kwargs):
        # initialization
        super(VQVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # hidden states
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # parameters
        self.data_var = data_var
        self.beta = beta  # This parameter is best kept between [0.1, 2.0] as per the paper.

        self.vq = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.code_loss_tracker = keras.metrics.Mean(name="codebook_loss")
        self.comm_loss_tracker = keras.metrics.Mean(name="commitment_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.code_loss_tracker,
                self.comm_loss_tracker,
                self.vq_loss_tracker]

    def encode(self, x):
        return self.vq(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=None, mask=None):
        return self.encode(inputs)

    def reconstruct(self, x):
        return self.decoder(self.encode(x))

    def train_step(self, x):
        with tf.GradientTape() as tape:
            enc_out = self.encoder(x)
            qua_lat = self.vq(enc_out)

            # Calculate vector quantization losses
            # Note: this has to be done before applying
            # the straight-through estimator below!
            code_loss = tf.reduce_mean((qua_lat - tf.stop_gradient(enc_out)) ** 2)  # codebook loss
            comm_loss = tf.reduce_mean((tf.stop_gradient(qua_lat) - enc_out) ** 2)  # commitment loss

            # Straight-through estimator
            qua_lat = enc_out + tf.stop_gradient(qua_lat - enc_out)

            # reconstruction
            recons = self.decode(qua_lat)

            # reconstruction loss
            recon_loss = tf.reduce_mean((x - recons) ** 2) / self.data_var

            # total loss
            total_loss = recon_loss + code_loss + self.beta * comm_loss

        # backpropagation
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking
        self.total_loss_tracker.updateState(total_loss)
        self.recon_loss_tracker.updateState(recon_loss)
        self.code_loss_tracker.updateState(code_loss)
        self.comm_loss_tracker.updateState(comm_loss)
        self.vq_loss_tracker.updateState(code_loss + self.beta * comm_loss)  # (tf.reduce_sum(self.vq.losses))

        # Log results.
        return {"total_loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "code_loss": self.code_loss_tracker.result(),
                "comm_loss": self.comm_loss_tracker.result(),
                "vq_loss": self.vq_loss_tracker.result()}


#################################
#               GAN             #
#################################
class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None
        self.d_loss_metric = None
        self.g_loss_metric = None

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        # Sample random points in the latent space
        batch_size = tf.shape(data)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # we need to gradient tapes as we have two gradients: generator and discriminator
        # By default, the resources held by the gradient tape are released as soon as the
        # gradient method is called. TODO: Can we use a persistent gradient tape?
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)  # on real images we expect 1
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)  # in fake images we want 0
            d_loss = real_loss + fake_loss

        gen_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        # # Decode them to fake images
        # generated_images = self.generator(random_latent_vectors, training=True)
        #
        # # Combine them with real images
        # combined_images = tf.concat([generated_images, data], axis=0)
        #
        # # Assemble labels discriminating real from fake images
        # labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        #
        # # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        #
        # # Train the discriminator
        # with tf.GradientTape() as tape:
        #     predictions = self.discriminator(combined_images)
        #     d_loss = self.loss_fn(labels, predictions)
        # grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        # self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        #
        # # Sample random points in the latent space
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        #
        # # Assemble labels that say "all real images"
        # misleading_labels = tf.zeros((batch_size, 1))
        #
        # # Train the generator (note that we should *not* update the weights
        # # of the discriminator)!
        # with tf.GradientTape() as tape:
        #     predictions = self.discriminator(self.generator(random_latent_vectors))
        #     g_loss = self.loss_fn(misleading_labels, predictions)
        # grads = tape.gradient(g_loss, self.generator.trainable_weights)
        # self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.updateState(d_loss)
        self.g_loss_metric.updateState(g_loss)
        return {"d_loss": self.d_loss_metric.result(),
                "g_loss": self.g_loss_metric.result()}


if __name__ == "__main__":
    import numpy as np

    # data
    (x_train, _), (x_test, _) = keras.datasets.mnist.loadData()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_data = np.concatenate([x_train, x_test], axis=0)

    # training details
    epochs = 50
    batch_size = 128

    # # RBM
    # hid_dim = 64
    # dataset = tf.data.Dataset.from_tensor_slices(x_data.reshape((-1, 28 * 28)).astype('float32'))
    # dataset = dataset.shuffle(buffer_size=x_data.shape[0], reshuffle_each_iteration=True).batch(batch_size)
    #
    # rbm = RBM(n_vis=28 * 28, n_hid=hid_dim)
    # rbm.compile(keras.optimizers.Adam(learning_rate=0.0001))
    # rbm.fit(dataset, epochs=epochs)

    # # VAE
    # dataset = tf.data.Dataset.from_tensor_slices(x_data.reshape((-1, 28, 28, 1)).astype('float32'))
    # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    # latent_dim = 2
    # beta = 1.0
    # encoder = tf.keras.Sequential([layers.InputLayer(input_shape=(28, 28, 1)),
    #                                layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
    #                                              activation='relu', padding='same'),
    #                                layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
    #                                              activation='relu', padding='same'),
    #                                layers.Flatten(),
    #                                # output mean and log variance (for numerical stability)
    #                                layers.Dense(latent_dim + latent_dim),  # No activation
    #                                ], name='encoder')
    #
    # decoder = tf.keras.Sequential([layers.InputLayer(input_shape=(latent_dim,)),
    #                                layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
    #                                layers.Reshape(target_shape=(7, 7, 32)),
    #                                layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
    #                                                padding='same', activation='relu'),
    #                                layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
    #                                                       padding='same', activation='relu'),
    #                                layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1,
    #                                                       padding='same', activation='sigmoid'),
    #                                ], name="decoder")
    # vae = VAE(encoder, decoder, beta=beta)
    # vae.compile(optimizer=keras.optimizers.Adam())
    # vae.fit(dataset, epochs=epochs)

    # VQ-VAE
    dataset = tf.data.Dataset.from_tensor_slices(x_data.reshape((-1, 28, 28, 1)).astype('float32'))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    latent_dim = 16
    num_embeddings = 128
    beta = 0.25
    data_variance = np.var(x_train)

    encoder = tf.keras.Sequential([layers.InputLayer(input_shape=(28, 28, 1)),
                                   layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                                 activation='relu', padding='same'),
                                   layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                                                 activation='relu', padding='same'),
                                   layers.Conv2D(latent_dim, 1, padding="same")
                                   ], name='encoder')

    decoder = tf.keras.Sequential([layers.InputLayer(input_shape=encoder.output.shape[1:]),
                                   layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                                                          padding='same', activation='relu'),
                                   layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                                                          padding='same', activation='relu'),
                                   layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1,
                                                          padding='same'),
                                   ], name="decoder")

    vqvae = VQVAE(encoder, decoder, latent_dim, num_embeddings, beta, data_variance)
    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae.fit(dataset, epochs=epochs)

    # # GAN
    # dataset = tf.data.Dataset.from_tensor_slices(x_data.reshape((-1, 28, 28, 1)).astype('float32'))
    # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    # epochs = 50
    # latent_dim = 100
    #
    # discriminator = Sequential([layers.InputLayer(input_shape=(28, 28, 1)),
    #                             layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
    #                             layers.LeakyReLU(),
    #                             layers.Dropout(0.3),
    #                             layers.Conv2D(128, kernel_size=5, strides=2, padding='same'),
    #                             layers.LeakyReLU(),
    #                             layers.Dropout(0.3),
    #                             layers.Flatten(),
    #                             layers.Dense(1)],
    #                            name='discriminator')
    # discriminator.summary()
    #
    # generator = keras.Sequential([keras.Input(shape=(latent_dim,)),
    #                               layers.Dense(7 * 7 * 256, use_bias=False),
    #                               layers.BatchNormalization(),
    #                               layers.LeakyReLU(alpha=0.3),
    #                               layers.Reshape((7, 7, 256)),
    #                               layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding="same",
    #                                                      use_bias=False),
    #                               layers.BatchNormalization(),
    #                               layers.LeakyReLU(),
    #                               layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same",
    #                                                      use_bias=False),
    #                               layers.BatchNormalization(),
    #                               layers.LeakyReLU(),
    #                               layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same",
    #                                                      use_bias=False,
    #                                                      activation="tanh")],
    #                              name="generator")
    # generator.summary()
    #
    # gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    # gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #             g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #             loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
    #
    # class GANMonitor(keras.callbacks.Callback):
    #     def __init__(self, num_imgs=3, latent_dim=128):
    #         self.num_imgs = num_imgs
    #         self.latent_dim = latent_dim
    #         self.tot_imgs = np.cumprod(num_imgs)[-1]
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         random_latent_vectors = tf.random.normal(shape=(self.tot_imgs, self.latent_dim))
    #         generated_images = self.model.generator(random_latent_vectors, training=False)
    #         generated_images = (generated_images * 127.5 + 127.5)  # 255
    #         # generated_images.numpy()
    #
    #         fig = plt.figure(figsize=self.num_imgs)
    #         for i in range(self.tot_imgs):
    #             # img = keras.preprocessing.image.array_to_img(generated_images[i])
    #             # img.save("../../../results/generated_img_%03d_%d.png" % (epoch, i))
    #             plt.subplot(self.num_imgs[0], self.num_imgs[1], i + 1)
    #             plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    #             plt.axis('off')
    #
    #         plt.savefig("../../../results/mnist_gan_%04d.png" % (epoch))
    #         plt.close()
    # gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(num_imgs=(4, 4), latent_dim=latent_dim)])

