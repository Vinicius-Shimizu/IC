#python
import torch
import torchvision
import matplotlib.pyplot as plt

# Digit classification network definition.
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(1, 32, 3, padding=1),
      torch.nn.MaxPool2d(2),
      torch.nn.ReLU(True),
      torch.nn.Conv2d(32, 64, 3, padding=1),
      torch.nn.MaxPool2d(2),
      torch.nn.ReLU(True)
    )
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(64 * 7 * 7, 1024),
      torch.nn.ReLU(),
      torch.nn.Linear(1024, 10),
      torch.nn.Softmax(1)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.view(-1, 64 * 7 * 7)
    x = self.classifier(x)
    return x

# Return an instance of Net.
def digit_net(): return Net()

# Retrieve the MNIST data.
def mnist_data():
  train = torchvision.datasets.MNIST(root = "/tmp", train = True, download = True)
  test  = torchvision.datasets.MNIST(root = "/tmp", train = False, download = True)
  return train.data.float().reshape(len(train), 1, 28, 28)/255., train.targets, \
         test.data.float().reshape(len(test), 1, 28, 28)/255., test.targets

# Normalization function to center pixel values around mu with standard deviation sigma.
def normalize(X_R, Y_R, X_T, Y_T, mu, sigma): return (X_R-mu)/sigma, Y_R, (X_T-mu)/sigma, Y_T

train_X, train_Y, test_X, test_Y = normalize(*mnist_data(), 0.1307, 0.3081)

def pick_slice(data, which):
  h = len(data)//4
  if which == 0: return slice(0, h)
  elif which == 1: return slice(h, 2*h)
  elif which == 2: return slice(2*h, 3*h)
  return slice(3*h, 4*h)

# MNIST images for the train set.
def mnist_images_train(which): return train_X[pick_slice(train_X, which)]

# MNIST images for the test set.
def mnist_images_test(which): return test_X[pick_slice(test_X, which)]

# Observed atoms for training.
def mnist_labels_train():
  h = len(train_Y)//4
  labels = torch.concatenate((train_Y[:h].reshape(-1, 1),
                              train_Y[h:2*h].reshape(-1, 1), 
                              train_Y[2*h:3*h].reshape(-1, 1), 
                              train_Y[3*h:4*h].reshape(-1, 1)), axis=1)
  T = []
  for i in range(h): T.append(labels[i][0] + labels[i][1] + labels[i][2] + labels[i][3])
  T = torch.tensor(T)
  
  D = [[f"sum({a.item()+b.item() + c.item() + d.item()})"] for a, b, c, d in labels]
  return D
#end.

% Data of the first number.
input(0) ~ test(@mnist_images_test(0)), train(@mnist_images_train(0)).
input(1) ~ test(@mnist_images_test(1)), train(@mnist_images_train(1)).
% Data of the second number.
input(2) ~ test(@mnist_images_test(2)), train(@mnist_images_train(2)).
input(3) ~ test(@mnist_images_test(3)), train(@mnist_images_train(3)).

% Neural annotated disjunction over each number from 0 to 9; use Adam as optimizer
% and a learning rate of 0.001.
?::digit(X, {0..9}) as @digit_net with optim = "Adam", lr = 0.001 :- input(X).
% The sum.
sum(Z) :- digit(0, A), digit(1, B), digit(2, C), digit(3, D), Z = A+B+C+D.

% Learn the parameters of the program from the "sum(X)" atoms.
#learn @mnist_labels_train, lr = 0.001, niters = 10, alg = "lagrange", batch = 64.
#semantics maxent.
#query sum(X).
% Ask for the probability of all groundings of sum(X).