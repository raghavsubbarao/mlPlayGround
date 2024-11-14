import abc
from typing import Optional, Iterable, Union, overload

import numpy as np
import torch

from core.ml.model import baseTorchModel, baseLossTracker


#################################
#               VAE             #
#################################
class variationalAutoencoder(baseTorchModel):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        # initialization
        super(variationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # beta = 1.0 for Kingma, Welling but can be adjusted
        # for beta-VAEs. beta > 1 allows disentangling generative factors
        self.beta = beta

        self.total_loss_tracker = baseLossTracker(name="total_loss")
        self.recon_loss_tracker = baseLossTracker(name="recon_loss")
        self.kl_loss_tracker = baseLossTracker(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker]

    def encode(self, x):
        mean, logvar = torch.split(self.encoder(x), split_size_or_sections=2, dim=1)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, inputs):
        z_mean, z_logvar = self.encode(inputs)
        z = torch.normal(torch.zeros(z_mean.shape), torch.ones(z_mean.shape)).to(self.device)
        recon = self.decoder(z_mean + torch.exp(z_logvar / 2.) * z)
        return recon, z_mean, z_logvar

    def train_step(self, data, optimizer):
        X = data.to(self.device)

        recon, z_mean, z_log_var = self.forward(X)

        recon_loss = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy(recon, X, reduction='none'), dim=[1, 2, 3]))
        kl_loss = torch.mean(torch.sum((torch.exp(z_log_var) + torch.square(z_mean) - z_log_var) / 2., dim=1))
        total_loss = recon_loss + self.beta * kl_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {"total_loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


#################################
#            Real NVP           #
#################################
class normFlowModule(torch.nn.Module):
    """
    For real NVP (and other normalizing flows!) each layer also
    contributes to the loss being optimized. In TF this is handled
    through the tf_probability.bijector class. In torch however,
    there is no equivalent.
    This class aims to do something similar. To keep all the
    functionality offered by nn.Module we derive from it, but require
    that all the necessary functions be defined in subclasses to make
    sure that an concrete class can ultimately handle the required
    computations!
    """

    @abc.abstractmethod
    def inverseLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class normFlowSequential(torch.nn.Sequential, normFlowModule):

    @overload
    def __init__(self, *args: normFlowModule) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, normFlowModule]") -> None:
        ...

    def __init__(self, *args) -> None:
        super(normFlowSequential, self).__init__(*args)

    def inverseLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        ll = 0
        for module in self:
            ll = ll + module.forwardLogDetJacobian(y)
            y = module(y)
        return ll

    def forward(self, y: torch.Tensor):
        ll = []
        for module in self:
            y, fldj = module(y)
            ll.append(fldj)

        ll = torch.sum(torch.stack(ll, -1), 1)
        return y, ll


class nvpBatchNorm2d(torch.nn.modules.BatchNorm2d, normFlowModule):
    """
    Implementation of batch norm for real NVP. We use the batch norm in two
    places (i) the scale and shift sub networks - this is trivial and torch's
    implementation would be fine. (ii) in the coupling layers - here we need
    to account for the batch norm in the loss function.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None, dtype=None) -> None:
        super(nvpBatchNorm2d, self).__init__(num_features, eps, momentum, affine,
                                             track_running_stats, device, dtype)

    def inverseLogDetJacobian(self, y: torch.Tensor) -> torch.Tensor:
        lp = torch.log(self.weight) if self.affine else 0
        fldj = lp - 0.5 * torch.log(self.running_var + self.eps)
        return torch.sum(torch.flatten(fldj.view([1, self.num_features, 1, 1]).expand_as(y), 1), 1)

    def forward(self, input: torch.Tensor):
        return super(nvpBatchNorm2d, self).forward(input), self.inverseLogDetJacobian(input)


class realNVPCouplingLayer(normFlowModule):
    def __init__(self,
                 scaleModule: torch.nn.Module,
                 biasModule: torch.nn.Module,
                 dims: Iterable[int],
                 mask: str,
                 flip: Union[int, bool],
                 **kwargs):
        super(realNVPCouplingLayer, self).__init__(**kwargs)

        self.s = scaleModule
        self.t = biasModule

        self.sScale = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)
        self.tBias = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)
        self.tScale = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)

        if mask == 'check':
            mask = self.checkerBoardMask(dims)
        elif mask == 'channel':
            mask = self.channelMask(dims)
        else:
            raise Exception(f'Unknown masking type: {mask}')
        if flip:
            mask = 1 - mask
        self.register_buffer(name='mask', tensor=mask)

    @staticmethod
    def checkerBoardMask(dims):
        return torch.Tensor(1 - np.indices(dims).sum(axis=0) % 2)

    @staticmethod
    def channelMask(dims):
        assert len(dims) == 3
        assert dims[0] % 2 == 0
        mask = torch.cat([torch.zeros((dims[0] // 2, dims[1], dims[2])),
                          torch.ones((dims[0] // 2, dims[1], dims[2]))], dim=0)
        assert mask.shape == dims
        return mask

    def inverseLogDetJacobian(self, y: torch.Tensor) -> torch.Tensor:
        s = self.sScale * torch.tanh(self.s(self.mask * y))
        return torch.sum(torch.flatten((1 - self.mask) * s, 1), 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        t = self.tScale * self.t(self.mask * y) + self.tBias
        s = self.sScale * torch.tanh(self.s(self.mask * y))
        out = self.mask * y + (1 - self.mask) * (y * torch.exp(s) + t)
        ll = torch.sum(torch.flatten((1 - self.mask) * s, 1), 1)
        return out, ll


class realNonVolumePreserving(baseTorchModel):
    def __init__(self, baseBlock, dims, hidden=64, nScales=1, nFinal=4, **kwargs):
        # initialization
        super(realNonVolumePreserving, self).__init__(**kwargs)

        self.checkList = torch.nn.ModuleList()
        self.channelList = torch.nn.ModuleList()
        self.register_buffer('pi', torch.tensor(np.pi))

        for i in range(nScales):
            # each scale of the hierarchy consists of:
            # 1. three coupling layers with alternating checkerboard masks
            # 2. A squeeze operations s x s x c --> s/2 x s/2 x 4c
            # 3. three coupling layers with channel masks
            self.checkList.append(normFlowSequential(realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                          baseBlock(dims[0], hidden),
                                                                          dims, 'check', 0),
                                                     nvpBatchNorm2d(dims[0], affine=False),
                                                     realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                          baseBlock(dims[0], hidden),
                                                                          dims, 'check', 1),
                                                     nvpBatchNorm2d(dims[0], affine=False),
                                                     realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                          baseBlock(dims[0], hidden),
                                                                          dims, 'check', 0),
                                                     nvpBatchNorm2d(dims[0], affine=False)))

            dims = (4 * dims[0], dims[1] // 2, dims[2] // 2)

            self.channelList.append(normFlowSequential(realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                            baseBlock(dims[0], hidden),
                                                                            dims, 'channel', 1),
                                                       nvpBatchNorm2d(dims[0], affine=False),
                                                       realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                            baseBlock(dims[0], hidden),
                                                                            dims, 'channel', 0),
                                                       nvpBatchNorm2d(dims[0], affine=False),
                                                       realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                            baseBlock(dims[0], hidden),
                                                                            dims, 'channel', 1),
                                                       nvpBatchNorm2d(dims[0], affine=False)))

            dims = (dims[0] // 2, dims[1], dims[2])
            hidden = hidden * 2

        self.final = normFlowSequential(*[l for i in range(nFinal) for l in [realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                                                  baseBlock(dims[0], hidden),
                                                                                                  dims, 'check', i % 2 == 0),
                                                                             nvpBatchNorm2d(dims[0], affine=False)]])

        self.total_loss_tracker = baseLossTracker(name="total_loss")
        self.log_loss_tracker = baseLossTracker(name="log_loss")
        self.ll_loss_tracker = baseLossTracker(name="ll_loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def forward(self, y):
        yr = []
        ll = []

        for check, channel in zip(self.checkList, self.channelList):
            y, fldj = check(y)  # three couplings with checkerboard masking
            ll.append(fldj)

            y = torch.nn.functional.pixel_unshuffle(y, 2)  # squeeze

            y, fldj = channel(y)  # three couplings with channel masking
            ll.append(fldj)

            yr.append(torch.flatten(y[:, y.shape[1] // 2:, :, :], 1))
            y = y[:, :y.shape[1] // 2, :, :]

        y, fldj = self.final(y)
        ll.append(fldj)
        yr.append(torch.flatten(y, 1))

        return torch.flatten(torch.cat(yr, 1), 1), torch.sum(torch.stack(ll, -1), 1)

    def train_step(self, data, optimizer):
        X = data.to(self.device)

        y, ll = self.forward(X)

        log_loss = torch.sum((torch.log(2 * self.pi) + y * y) / 2.)
        ll_loss = -torch.sum(ll)
        total_loss = log_loss + ll_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.total_loss_tracker.update_state(total_loss)
        self.log_loss_tracker.update_state(log_loss)
        self.ll_loss_tracker.update_state(ll_loss)

        return {"total_loss": self.total_loss_tracker.result(),
                "log_loss": self.log_loss_tracker.result(),
                "ll_loss": self.ll_loss_tracker.result()}


if __name__ == "__main__":
    m = nvpBatchNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)
    output = m.inverseLogDetJacobian(input)

    shape = (1, 28, 28)
    planes = 64
    for i in range(6):
        print(i, shape)
        if i % 6 == 2:
            shape = (4 * shape[0], shape[1] // 2, shape[2] // 2)
