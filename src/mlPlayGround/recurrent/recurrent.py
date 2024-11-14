import abc
import torch

from core.ml.model import baseTorchModel, baseLossTracker


#################################
#               TCN             #
#################################
class chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class temporalBlock(torch.nn.Module):
    """
    For implementation of TCNs as described in "An Empirical Evaluation
    of Generic Convolutional and Recurrent Networks for Sequence Modeling"
    by Bai, Koleter and Koltun, 2018.

    While TCNs can be taken to mean 1d convolution with dilations, here
    they describe a "Temporal Block" that is stacked to build a TCN.
    """

    def __init__(self, nin, nout, kernelSize, strides=1, dilation=1, dropout=0.2):
        # Torch assumes input to a conv layer has shape batch x channels x steps
        # When padding <> 0 it pads both the beginning and the end. Therefore, for
        # causal padding need to remove the values at the end with the chomp layer

        super(temporalBlock, self).__init__()

        self.n_in = nin
        self.n_out = nout
        self.kernelSize = kernelSize
        self.strides = strides
        self.dilation = dilation
        self.dropout = dropout
        self.padding = (kernelSize - 1) * dilation

        self.conv1 = torch.nn.Conv1d(in_channels=nin, out_channels=nout, kernel_size=kernelSize,
                                     stride=strides, padding=self.padding, dilation=dilation,
                                     bias=True, padding_mode='zeros')
        self.conv1 = torch.nn.utils.weight_norm(self.conv1)
        self.drop1 = torch.nn.Dropout(p=dropout)

        self.conv2 = torch.nn.Conv1d(in_channels=nout, out_channels=nout, kernel_size=kernelSize,
                                     stride=strides, padding=self.padding, dilation=dilation,
                                     bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.utils.weight_norm(self.conv2)
        self.drop2 = torch.nn.Dropout(p=dropout)

        if self.padding == 0:
            self.net = torch.nn.Sequential(self.conv1,
                                           torch.nn.ReLU(),
                                           self.drop1,
                                           self.conv2,
                                           torch.nn.ReLU(),
                                           self.drop2)
        else:
            self.net = torch.nn.Sequential(self.conv1,
                                           chomp1d(self.padding),
                                           torch.nn.ReLU(),
                                           self.drop1,
                                           self.conv2,
                                           chomp1d(self.padding),
                                           torch.nn.ReLU(),
                                           self.drop2)

        if nin != nout:
            # we will need a linear layer for the skip connection
            self.conv3 = torch.nn.Conv1d(in_channels=nin, out_channels=nout,
                                         kernel_size=1, bias=True)
        else:
            self.conv3 = None

    def forward(self, inputs):
        x = self.net(inputs)
        res = inputs if self.conv3 is None else self.conv3(inputs)
        return x + res  # skip connection!


class temporalConvolutionNetwork(torch.nn.Module):
    def __init__(self, nChannels, kernelSize=2, dilation=2, dropout=0.2):
        super(temporalConvolutionNetwork, self).__init__()

        self.blocks = []
        for i, (nin, nout) in enumerate(zip(nChannels[:-1], nChannels[1:])):
            dilation_size = dilation ** i
            self.blocks.append(temporalBlock(nin, nout, kernelSize, strides=1,
                                             dilation=dilation_size, dropout=dropout))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.blocks:
            outputs = layer(outputs)
        return outputs


class temporalBlockQuantGan(temporalBlock):
    """
    Implements the temporal block as used in "Quant GANs:
    Deep Generation of Financial Time Series" by Wiese,Knobloch
    Korn and Kretschner 2019.

    They apply a relu on the skip connection output. So we only
    need to redefine the forward.
    """

    def __init__(self, nin, nout, kernelSize, strides=1, dilation=1, dropout=0.2):
        super(temporalBlockQuantGan, self).__init__(nin, nout, kernelSize,
                                                    strides, dilation, dropout)

        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        x = self.net(inputs)
        res = inputs if self.conv3 is None else self.conv3(inputs)
        return x, self.relu(x + res)  # skip connection!
