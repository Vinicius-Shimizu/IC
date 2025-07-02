import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from argparse import ArgumentParser
from tqdm import tqdm

from sppl.compilers.ast_to_spe import Id
from sppl.compilers.ast_to_spe import IfElse
from sppl.compilers.ast_to_spe import Sample
from sppl.compilers.ast_to_spe import IdArray
from sppl.compilers.ast_to_spe import For
from sppl.compilers.ast_to_spe import Sequence
from sppl.compilers.ast_to_spe import Transform
from sppl.distributions import bernoulli
from sppl.distributions import randint
from sppl.math_util import allclose

mnist_img_transform = torchvision.transforms.Compose([ # composes several transforms together
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])

# Dataset for MNIST Sum 2 Task
class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.mnist_dataset = torchvision.datasets.MNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.mnist_dataset)/2)
    
    def __getitem__(self, idx):
        # get 2 data points
        (a_img, a_digit) = self.mnist_dataset[self.index_map[idx*2]]
        (b_img, b_digit) = self.mnist_dataset[self.index_map[idx*2 + 1]]
        
        return (a_img, b_img, a_digit + b_digit)

    @staticmethod
    def collate_fn(batch):
        a_imgs = torch.stack([item[0] for item in batch])
        b_imgs = torch.stack([item[1] for item in batch])
        digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
        return ((a_imgs, b_imgs), digits)


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        MNISTSum2Dataset(
            data_dir,
            train=True,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTSum2Dataset(
            data_dir,
            train=False,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) # applies a 2D convolution over the input signal
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024,1024) # applies a linear transformation to the incoming data
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2) # applies a 2D pooling over the input signal
        x = F.max_pool2d(self.conv2(x), 2) 
        x = x.view(-1, 1024) # copy of x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.5, training=self.training) # randomly zeroes some elements of x with prob p
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def spplModel(p_digit1, p_digit2):
    n = len(p_digit1)
    digit1_0 = IdArray("digit1_0", n) 
    digit1_1 = IdArray("digit1_1", n)
    digit1_2 = IdArray("digit1_2", n)
    digit1_3 = IdArray("digit1_3", n)
    digit1_4 = IdArray("digit1_4", n)
    digit1_5 = IdArray("digit1_5", n)
    digit1_6 = IdArray("digit1_6", n)
    digit1_7 = IdArray("digit1_7", n)
    digit1_8 = IdArray("digit1_8", n)
    digit1_9 = IdArray("digit1_9", n)
    digit2_0 = IdArray("digit2_0", n) 
    digit2_1 = IdArray("digit2_1", n)
    digit2_2 = IdArray("digit2_2", n)
    digit2_3 = IdArray("digit2_3", n)
    digit2_4 = IdArray("digit2_4", n)
    digit2_5 = IdArray("digit2_5", n)
    digit2_6 = IdArray("digit2_6", n)
    digit2_7 = IdArray("digit2_7", n)
    digit2_8 = IdArray("digit2_8", n)
    digit2_9 = IdArray("digit2_9", n)
    
    program = Sequence(
        For(0, n, lambda i:
            Sequence(
                Sample(digit1_0[i], bernoulli(p=p_digit1[i][0])),
                Sample(digit1_1[i], bernoulli(p=p_digit1[i][1])),
                Sample(digit1_2[i], bernoulli(p=p_digit1[i][2])),
                Sample(digit1_3[i], bernoulli(p=p_digit1[i][3])),
                Sample(digit1_4[i], bernoulli(p=p_digit1[i][4])),
                Sample(digit1_5[i], bernoulli(p=p_digit1[i][5])),
                Sample(digit1_6[i], bernoulli(p=p_digit1[i][6])),
                Sample(digit1_7[i], bernoulli(p=p_digit1[i][7])),
                Sample(digit1_8[i], bernoulli(p=p_digit1[i][8])),
                Sample(digit1_9[i], bernoulli(p=p_digit1[i][9])),
                Sample(digit2_0[i], bernoulli(p=p_digit2[i][0])),
                Sample(digit2_1[i], bernoulli(p=p_digit2[i][1])),
                Sample(digit2_2[i], bernoulli(p=p_digit2[i][2])),
                Sample(digit2_3[i], bernoulli(p=p_digit2[i][3])),
                Sample(digit2_4[i], bernoulli(p=p_digit2[i][4])),
                Sample(digit2_5[i], bernoulli(p=p_digit2[i][5])),
                Sample(digit2_6[i], bernoulli(p=p_digit2[i][6])),
                Sample(digit2_7[i], bernoulli(p=p_digit2[i][7])),
                Sample(digit2_8[i], bernoulli(p=p_digit2[i][8])),
                Sample(digit2_9[i], bernoulli(p=p_digit2[i][9]))
            ),
        ),
    )
    model = program.interpret()
    probs = []
    for i in range(n):
        sum0 = (digit1_0[i] << {1} & digit2_0[i] << {1})            
        sum1 = ((digit1_0[i] << {1} & digit2_1[i] << {1}) | (digit1_1[i] << {1} & digit2_0[i] << {1}))
        sum2 = ((digit1_0[i] << {1} & digit2_2[i] << {1}) | (digit1_2[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_1[i] << {1}))
        sum3 = ((digit1_0[i] << {1} & digit2_3[i] << {1}) | (digit1_3[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_2[i] << {1}) | (digit1_2[i] << {1} & digit2_1[i] << {1}))    
        sum4 = ((digit1_0[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_3[i] << {1}) | (digit1_3[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_2[i] << {1}))
        sum5 = ((digit1_0[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_3[i] << {1}) | (digit1_3[i] << {1} & digit2_2[i] << {1}))
        sum6 = ((digit1_0[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_3[i] << {1}))
        sum7 = ((digit1_0[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_3[i] << {1}))
        sum8 = ((digit1_0[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_4[i] << {1}))
        sum9 = ((digit1_0[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_4[i] << {1}))
        sum10 = ((digit1_1[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_5[i] << {1}))
        sum11 = ((digit1_2[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_5[i] << {1}))
        sum12 = ((digit1_3[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_7[i] << {1}) | (digit1_6[i] << {1} & digit2_6[i] << {1}))
        sum13 = ((digit1_4[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_5[i] << {1}) | (digit1_6[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_6[i] << {1}))
        sum14 = ((digit1_5[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_5[i] << {1}) | (digit1_6[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_6[i] << {1}) | (digit1_7[i] << {1} & digit2_7[i] << {1}))
        sum15 = ((digit1_6[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_6[i] << {1}) | (digit1_7[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_7[i] << {1}))
        sum16 = ((digit1_7[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_7[i] << {1}) | (digit1_8[i] << {1} & digit2_8[i] << {1}))
        sum17 = ((digit1_8[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_8[i] << {1}))
        sum18 = (digit1_9[i] << {1} & digit2_9[i] << {1})

        probs.append([model.prob(sum0), model.prob(sum1), model.prob(sum2), model.prob(sum3), model.prob(sum4),
                    model.prob(sum5), model.prob(sum6), model.prob(sum7), model.prob(sum8), model.prob(sum9),
                    model.prob(sum10), model.prob(sum11), model.prob(sum12), model.prob(sum13), model.prob(sum14),
                    model.prob(sum15), model.prob(sum16), model.prob(sum17), model.prob(sum18)
                    ])
    return torch.tensor(probs, requires_grad=True)

class MNISTSum2Net(nn.Module):
    def __init__(self, provenance, k):
        super(MNISTSum2Net, self).__init__()

        self.mnist_net = MNISTNet()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        # Recognize the digits
        self.dig1dists = self.mnist_net(a_imgs) #tensor 64x 10
        self.dig2dists = self.mnist_net(b_imgs) #tensor 64x 10

        # def run_model(p_digit1, p_digit2):
            # n = len(p_digit1)
            # digit1_0 = IdArray("digit1_0", n) 
            # digit1_1 = IdArray("digit1_1", n)
            # digit1_2 = IdArray("digit1_2", n)
            # digit1_3 = IdArray("digit1_3", n)
            # digit1_4 = IdArray("digit1_4", n)
            # digit1_5 = IdArray("digit1_5", n)
            # digit1_6 = IdArray("digit1_6", n)
            # digit1_7 = IdArray("digit1_7", n)
            # digit1_8 = IdArray("digit1_8", n)
            # digit1_9 = IdArray("digit1_9", n)
            # digit2_0 = IdArray("digit2_0", n) 
            # digit2_1 = IdArray("digit2_1", n)
            # digit2_2 = IdArray("digit2_2", n)
            # digit2_3 = IdArray("digit2_3", n)
            # digit2_4 = IdArray("digit2_4", n)
            # digit2_5 = IdArray("digit2_5", n)
            # digit2_6 = IdArray("digit2_6", n)
            # digit2_7 = IdArray("digit2_7", n)
            # digit2_8 = IdArray("digit2_8", n)
            # digit2_9 = IdArray("digit2_9", n)
            
            # program = Sequence(
            #     For(0, n, lambda i:
            #         Sequence(
            #             Sample(digit1_0[i], bernoulli(p=p_digit1[i][0])),
            #             Sample(digit1_1[i], bernoulli(p=p_digit1[i][1])),
            #             Sample(digit1_2[i], bernoulli(p=p_digit1[i][2])),
            #             Sample(digit1_3[i], bernoulli(p=p_digit1[i][3])),
            #             Sample(digit1_4[i], bernoulli(p=p_digit1[i][4])),
            #             Sample(digit1_5[i], bernoulli(p=p_digit1[i][5])),
            #             Sample(digit1_6[i], bernoulli(p=p_digit1[i][6])),
            #             Sample(digit1_7[i], bernoulli(p=p_digit1[i][7])),
            #             Sample(digit1_8[i], bernoulli(p=p_digit1[i][8])),
            #             Sample(digit1_9[i], bernoulli(p=p_digit1[i][9])),
            #             Sample(digit2_0[i], bernoulli(p=p_digit2[i][0])),
            #             Sample(digit2_1[i], bernoulli(p=p_digit2[i][1])),
            #             Sample(digit2_2[i], bernoulli(p=p_digit2[i][2])),
            #             Sample(digit2_3[i], bernoulli(p=p_digit2[i][3])),
            #             Sample(digit2_4[i], bernoulli(p=p_digit2[i][4])),
            #             Sample(digit2_5[i], bernoulli(p=p_digit2[i][5])),
            #             Sample(digit2_6[i], bernoulli(p=p_digit2[i][6])),
            #             Sample(digit2_7[i], bernoulli(p=p_digit2[i][7])),
            #             Sample(digit2_8[i], bernoulli(p=p_digit2[i][8])),
            #             Sample(digit2_9[i], bernoulli(p=p_digit2[i][9]))
            #         ),
            #     ),
            # )
            # model = program.interpret()
            # probs = []
            # for i in range(n):
            #     sum0 = (digit1_0[i] << {1} & digit2_0[i] << {1})            
            #     sum1 = ((digit1_0[i] << {1} & digit2_1[i] << {1}) | (digit1_1[i] << {1} & digit2_0[i] << {1}))
            #     sum2 = ((digit1_0[i] << {1} & digit2_2[i] << {1}) | (digit1_2[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_1[i] << {1}))
            #     sum3 = ((digit1_0[i] << {1} & digit2_3[i] << {1}) | (digit1_3[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_2[i] << {1}) | (digit1_2[i] << {1} & digit2_1[i] << {1}))    
            #     sum4 = ((digit1_0[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_3[i] << {1}) | (digit1_3[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_2[i] << {1}))
            #     sum5 = ((digit1_0[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_3[i] << {1}) | (digit1_3[i] << {1} & digit2_2[i] << {1}))
            #     sum6 = ((digit1_0[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_3[i] << {1}))
            #     sum7 = ((digit1_0[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_4[i] << {1}) | (digit1_4[i] << {1} & digit2_3[i] << {1}))
            #     sum8 = ((digit1_0[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_4[i] << {1}))
            #     sum9 = ((digit1_0[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_0[i] << {1}) | (digit1_1[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_5[i] << {1}) | (digit1_5[i] << {1} & digit2_4[i] << {1}))
            #     sum10 = ((digit1_1[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_1[i] << {1}) | (digit1_2[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_5[i] << {1}))
            #     sum11 = ((digit1_2[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_2[i] << {1}) | (digit1_3[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_6[i] << {1}) | (digit1_6[i] << {1} & digit2_5[i] << {1}))
            #     sum12 = ((digit1_3[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_3[i] << {1}) | (digit1_4[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_7[i] << {1}) | (digit1_6[i] << {1} & digit2_6[i] << {1}))
            #     sum13 = ((digit1_4[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_4[i] << {1}) | (digit1_5[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_5[i] << {1}) | (digit1_6[i] << {1} & digit2_7[i] << {1}) | (digit1_7[i] << {1} & digit2_6[i] << {1}))
            #     sum14 = ((digit1_5[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_5[i] << {1}) | (digit1_6[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_6[i] << {1}) | (digit1_7[i] << {1} & digit2_7[i] << {1}))
            #     sum15 = ((digit1_6[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_6[i] << {1}) | (digit1_7[i] << {1} & digit2_8[i] << {1}) | (digit1_8[i] << {1} & digit2_7[i] << {1}))
            #     sum16 = ((digit1_7[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_7[i] << {1}) | (digit1_8[i] << {1} & digit2_8[i] << {1}))
            #     sum17 = ((digit1_8[i] << {1} & digit2_9[i] << {1}) | (digit1_9[i] << {1} & digit2_8[i] << {1}))
            #     sum18 = (digit1_9[i] << {1} & digit2_9[i] << {1})

            #     probs.append([model.prob(sum0), model.prob(sum1), model.prob(sum2), model.prob(sum3), model.prob(sum4),
            #                 model.prob(sum5), model.prob(sum6), model.prob(sum7), model.prob(sum8), model.prob(sum9),
            #                 model.prob(sum10), model.prob(sum11), model.prob(sum12), model.prob(sum13), model.prob(sum14),
            #                 model.prob(sum15), model.prob(sum16), model.prob(sum17), model.prob(sum18)
            #                 ])
            # return probs
        return (self.dig1dists, self.dig2dists)
        probs = run_model(self.dig1dists, self.dig2dists)
        return torch.tensor(probs, requires_grad=True)
    

def bce_loss(output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    return F.nll_loss(output, ground_truth)

class Trainer():
    def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):
        self.network = MNISTSum2Net(provenance, k)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader=train_loader
        self.test_loader=test_loader
        if loss == "nll":
            self.loss=nll_loss
        elif loss == "bce":
            self.loss=bce_loss
        else:
            raise Exception(f"Unknown loss function '{loss}'")
    
    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for (data, target) in iter:
            self.optimizer.zero_grad()
            output = self.network(data)
            sppl_output = spplModel(output[0], output[1])
            loss = self.loss(sppl_output, target)
            loss.backward()
            self.optimizer.step()
            iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    
    def test(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for(data, target) in iter:
                output = self.network(data)
                sppl_output = spplModel(output[0], output[1])
                test_loss += self.loss(sppl_output, target).item()
                pred = sppl_output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                
                perc = 100. * correct / num_items
                iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")


    def train(self, n_epochs):
        self.test(0)
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.test(epoch)

def main():
    n_epochs = 2
    batch_size_train = 128
    batch_size_test = 128
    learning_rate = 0.001
    loss_fn = "bce"
    k = 3
    provenance = "difftopkproofs"
    torch.manual_seed(1024)
    random.seed(1024)

    data_dir = os.path.abspath(os.path.join(os.getcwd(), "data"))
    train_loader, test_loader = mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test)
    trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)
    trainer.train(n_epochs)

main()
