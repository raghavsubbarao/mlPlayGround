import abc
import torch
# import numpy as np

class baseTorchModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(baseTorchModel, self).__init__(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    def fit(self, dataset, optimizer, epochs):
        for t in range(epochs):
            train_loop(dataset, optimizer)
            test_loop(test_dl,)

    def train_loop(self, dataLoader, optimizer, epochs):

        size = len(dataLoader.dataset)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            # Set the model to training mode - important
            # for batch norm and dropout.
            self.train()

            for batch, data in enumerate(dataLoader):
                metrics = self.train_step(data, optimizer)

                if (batch + 1) % 100 == 0:
                    print(' '.join([f'{l}: {metrics[l]:>7f}' for l in metrics]) +
                          f'[{(batch + 1) * dataLoader.batch_size:>5d}/{size:>5d}]')

    @abc.abstractmethod
    def train_step(self, data, optimizer):
        pass


class baseLossTracker:
    def __init__(self, name):
        self.__name = name
        self.__losses = []

    def clear(self):
        self.__losses = []

    def update_state(self, loss):
        self.__losses.append(loss)

    def result(self):
        return self.__losses[-1]


class reshape(torch.nn.Module):
    def __init__(self, shape):
        super(reshape, self).__init__()
        self.__shape = shape

    def forward(self, inputs: torch.tensor):
        return inputs.view(-1, *self.__shape)
