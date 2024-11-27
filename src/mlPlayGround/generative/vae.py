import abc
from typing import Optional, Iterable, Union, overload, Tuple

import torch

from mlPlayGround.model import baseTorchModel, baseLossTracker


#################################
#               VAE             #
#################################
class variationalAutoencoder(baseTorchModel):
    """
    Variational Autoencoder implementation. Defaults to \beta=1
    but can also act as a beta VAE
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, beta: float = 1.0, **kwargs):
        # initialization
        super(variationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # beta = 1.0 for the original [Kingma,Welling 2013] version but can
        # be adjusted for beta-VAEs.
        # beta > 1 allows for disentangling of generative factors [Higgins 2017]
        self.beta = beta

        self.totalLossTracker = baseLossTracker(name="total_loss")
        self.reconLossTracker = baseLossTracker(name="recon_loss")
        self.klLossTracker = baseLossTracker(name="kl_loss")

    @property
    def metrics(self):
        return [self.totalLossTracker,
                self.reconLossTracker,
                self.klLossTracker]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logVar = torch.split(self.encoder(x), split_size_or_sections=2, dim=1)
        return mean, logVar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zMean, zLogVar = self.encode(inputs)  # get the mean and variance
        z = torch.normal(torch.zeros(zMean.shape), torch.ones(zMean.shape)).to(self.device)  # sample normal RV
        recon = self.decoder(zMean + torch.exp(zLogVar / 2.) * z)  # Reparameterization trick!
        return recon, zMean, zLogVar

    def trainStep(self, data, optimizer):
        # implement the abstract method for a single training update
        X = data.to(self.device)
        recon, zMean, zLogVar = self.forward(X)

        reconLoss = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy(recon, X, reduction='none'), dim=[1, 2, 3]))
        klLoss = torch.mean(torch.sum((torch.exp(zLogVar) + torch.square(zMean) - zLogVar) / 2., dim=1))
        totalLoss = reconLoss + self.beta * klLoss

        # Backpropagation
        totalLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.totalLossTracker.updateState(totalLoss)
        self.reconLossTracker.updateState(reconLoss)
        self.klLossTracker.updateState(klLoss)

        return {"totalLoss": self.totalLossTracker.result(),
                "reconLoss": self.reconLossTracker.result(),
                "klLoss": self.klLossTracker.result()}

    def validStep(self, data):
        # implement the abstract method for a single training update
        X = data.to(self.device)
        recon, zMean, zLogVar = self.forward(X)

        reconLoss = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy(recon, X, reduction='none'), dim=[1, 2, 3]))
        klLoss = torch.mean(torch.sum((torch.exp(zLogVar) + torch.square(zMean) - zLogVar) / 2., dim=1))
        totalLoss = reconLoss + self.beta * klLoss
        return totalLoss


#################################
#              VQ-VAE           #
#################################
class vaeVectorQuantizer(torch.nn.Module):
    def __init__(self, nEmbeddings, dEmbeddings, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.dEmbedding = dEmbeddings
        self.nEmbeddings = nEmbeddings

        # Initialize the embeddings which we will quantize.
        # w_init = tf.random_uniform_initializer()
        self.embeddings = torch.nn.Parameter(torch.zeros(self.dEmbedding, self.nEmbeddings), requires_grad=True)
        #
        # tf.Variable(initial_value=w_init(shape=(self.dEmbedding,
        #                                                       self.nEmbeddings),
        #                                                dtype="float32"),
        #                           trainable=True, name="embeddings_vqvae")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the input shape of the inputs then
        # flatten inputs keeping `embedding_dim` intact
        input_shape = tf.shape(x)
        flattened = torch.flatten(x, 0, -1)

        # Quantization.
        codeIndices = self.codeIndices(flattened)
        encodings = torch.nn.functional.one_hot(codeIndices, self.nEmbeddings)
        quantized = encodings * torch.transpose(self.embeddings, 0, 1)
        quantized = torch.reshape(quantized, input_shape)

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

    def codeIndices(self, y: torch.Tensor) -> torch.Tensor:
        # Calculate L2-norm between inputs and codes
        similarity = y * self.embeddings
        distances = torch.norm(y, p='fro', dim=1, keepdims=True)
        distances += torch.norm(self.embeddings, p='fro', dim=1, keepdims=True)
        distances -= 2 * similarity

        # Derive the indices for minimum distances.
        codes = tf.argmin(distances, axis=1)
        return codes


class vqVariationalAutoencoder(baseTorchModel):
    def __init__(self, encoder, decoder,
                 nDims, nEmbeddings,
                 beta=0.25, data_var=1.0, **kwargs):
        # initialization
        super(vqVariationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # hidden states
        self.nDims = nDims
        self.nEmbeddings = nEmbeddings

        # parameters
        self.data_var = data_var
        self.beta = beta  # This parameter is best kept between [0.1, 2.0] as per the paper.

        self.vq = VectorQuantizer(nEmbeddings, nDims)

        self.totalLossTracker = baseLossTracker(name="totalLoss")
        self.reconLossTracker = baseLossTracker(name="reconLoss")
        self.codingLossTracker = baseLossTracker(name="codingLoss")
        self.commitLossTracker = baseLossTracker(name="commitmentLoss")
        self.vqLossTracker = baseLossTracker(name="vqLoss")

    @property
    def metrics(self):
        return [self.totalLossTracker,
                self.reconLossTracker,
                self.codingLossTracker,
                self.commitLossTracker,
                self.vqLossTracker]

    def encode(self, x):
        return self.vq(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.encode(x)

    def reconstruct(self, x):
        return self.decoder(self.encode(x))

    def trainStep(self, data, optimizer):
        X = data.to(self.device)
        z_e = self.encoder(x)
        z_q = self.vq(z_e)

        # Calculate vector quantization losses
        codeLoss = torch.mean(torch.flatten(z_q - z_e.detach(), 1) ** 2, dim=-1)  # codebook loss
        commLoss = torch.mean(torch.flatten(z_q.detach() - z_e, 1) ** 2, dim=-1)  # commitment loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # reconstruction
        recon = self.decode(z_q)

        # reconstruction loss
        reconLoss = torch.mean(torch.flatten(X - recon, 1) ** 2, dim=-1)

        # total loss
        totalLoss = reconLoss / self.data_var + codeLoss + self.beta * commLoss

        # Backpropagation
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        # Loss tracking
        self.totalLossTracker.updateState(totalLoss)
        self.reconLossTracker.updateState(reconLoss)
        self.codingLossTracker.updateState(codeLoss)
        self.commitLossTracker.updateState(commLoss)
        self.vqLossTracker.updateState(codeLoss + self.beta * commLoss)

        # Log results.
        return {"totalLoss": self.totalLossTracker.result(),
                "reconLoss": self.reconLossTracker.result(),
                "codeLoss": self.codingLossTracker.result(),
                "commLoss": self.commitLossTracker.result(),
                "vqLoss": self.vqLossTracker.result()}

    def validStep(self, data):
        X = data.to(self.device)
        z_e = self.encoder(x)
        z_q = self.vq(z_e)

        # Calculate vector quantization losses
        codeLoss = torch.mean(torch.flatten(z_q - z_e.detach(), 1) ** 2, dim=-1)  # codebook loss
        commLoss = torch.mean(torch.flatten(z_q.detach() - z_e, 1) ** 2, dim=-1)  # commitment loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # reconstruction
        recon = self.decode(z_q)

        # reconstruction loss
        reconLoss = torch.mean(torch.flatten(X - recon, 1) ** 2, dim=-1)

        # total loss
        totalLoss = reconLoss / self.data_var + codeLoss + self.beta * commLoss

        return totalLoss
