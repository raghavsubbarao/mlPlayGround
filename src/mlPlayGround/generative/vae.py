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
