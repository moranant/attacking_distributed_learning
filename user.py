import functools
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import data_sets

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def flatten_params(params):
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x,y:x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


class User:
    def __init__(self, user_id, batch_size, is_malicious, users_count, momentum, data_set=data_sets.MNIST):
        self.is_malicious = is_malicious
        self.user_id = user_id
        self.criterion = nn.NLLLoss()
        self.learning_rate = None
        self.grads = None
        self.data_set = data_set
        self.momentum = momentum
        if data_set == data_sets.MNIST:
            self.net = data_sets.MnistNet()
        elif data_set == data_sets.CIFAR10:
            self.net = data_sets.Cifar10Net()
        self.original_params = None
        dataset = self.net.dataset(True)
        sampler = None
        if users_count > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)

        self.train_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size, shuffle=sampler is None)
        self.train_iterator = iter(cycle(self.train_loader))

    def train(self, data, target):
        if self.data_set == data_sets.MNIST:
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28 * 28)
        else:
            b, c, h, w = data.size()
            data = data.view(b, c, h, w)
        self.optimizer.zero_grad()

        net_out = self.net(data)
        loss = self.criterion(net_out, target)
        loss.backward()
        #self.optimizer.step() # not stepping because reporting the gradients and the server is performing the step

    # user gets initial weights and learn the new gradients based on its data
    def step(self, current_params, learning_rate):
        if self.user_id == 0 and self.is_malicious:
            self.original_params = current_params.copy()
            self.learning_rate = learning_rate
        row_into_parameters(current_params, self.net.parameters())
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=self.momentum, weight_decay=5e-4)

        data, target = next(self.train_iterator)
        self.train(data, target)
        self.grads = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.net.parameters()])
