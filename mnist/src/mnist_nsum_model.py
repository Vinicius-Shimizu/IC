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

import scallopy

mnist_img_transform = torchvision.transforms.Compose([ # composes several transforms together
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])

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


class MNISTSum2Net(nn.Module):
    def __init__(self, provenance, k):
        super(MNISTSum2Net, self).__init__()

        self.mnist_net = MNISTNet()

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance="difftopkproofs") # differenciable version of "topkproofs"
        self.scl_ctx.add_relation("dig1", (int,), input_mapping=[(i,) for i in range(10)]) # input mapping is a 10-tensor that contains the prob distribution for each digit
        self.scl_ctx.add_relation("dig2", (int,), input_mapping=[(i,) for i in range(10)])
        self.scl_ctx.add_rule("sum2(a + b) :- dig1(a), dig2(b)")

        # Encodes the output we want to get from Scallop, that is, when we get the probability for a tuple (x,),
        # we want to put it into the x+1-th position
        self.sum2 = self.scl_ctx.forward_function("sum2", output_mapping=[(i,) for i in range(19)])

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        # Recognize the digits
        a_distrs = self.mnist_net(a_imgs)
        b_distrs = self.mnist_net(b_imgs)

        return self.sum2(dig1=a_distrs, dig2=b_distrs)
    

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
            loss = self.loss(output, target)
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
                test_loss += self.loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                
                perc = 100. * correct / num_items
                iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")


    def train(self, n_epochs):
        self.test(0)
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.test(epoch)


if __name__ == "__main__":
    parser = ArgumentParser("mnist_sum2")
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--loss-fn", type=str, default="bce")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    loss_fn = args.loss_fn
    k = args.top_k
    provenance = args.provenance
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
    train_loader, test_loader = mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test)
    trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)

    if not args.test:
        trainer.train(n_epochs)
        # torch.save({
        #     'model_state_dict': trainer.network.state_dict(),
        #     'optimizer_state_dict': trainer.optimizer.state_dict(),
        #     'epoch': n_epochs,
        # }, '2Sum_model.pth')
        scripted_model = torch.jit.script(trainer.network)
        scripted_model.save("2Sum_model_scripted.pt")
    else:
        checkpoint = torch.load('2Sum_model.pth')
        trainer.network.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.test(0)