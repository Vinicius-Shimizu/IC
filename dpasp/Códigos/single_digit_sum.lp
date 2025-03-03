#python
import torch
import torchvision

# Digit classification network definition.
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(1, 6, 5),
      torch.nn.MaxPool2d(2, 2),
      torch.nn.ReLU(True),
      torch.nn.Conv2d(6, 16, 5),
      torch.nn.MaxPool2d(2, 2),
      torch.nn.ReLU(True)
    )
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(16 * 4 * 4, 120),
      torch.nn.ReLU(),
      torch.nn.Linear(120, 84),
      torch.nn.ReLU(),
      torch.nn.Linear(84, 10),
      torch.nn.Softmax(1)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.view(-1, 16 * 4 * 4)
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
def normalize(X_R, Y_R, X_T, Y_T, mu, sigma):
  return (X_R-mu)/sigma, Y_R, (X_T-mu)/sigma, Y_T

train_X, train_Y, test_X, test_Y = normalize(*mnist_data(), 0.1307, 0.3081)
# Whether to pick the first or second half of the dataset.
def pick_slice(data, which):
  h = len(data)//2
  return slice(h, len(data)) if which else slice(0, h)
# MNIST images for the train set.
def mnist_images_train(which): return train_X[pick_slice(train_X, which)]
# MNIST images for the test set.
def mnist_images_test(which): return test_X[pick_slice(test_X, which)]
# Observed atoms for training.
def mnist_labels_train():
  # We join the two halves (top and bottom) of MNIST and join them together to get
  # two digits side by side. The labels are atoms encoding the sum of the two digits.
  labels = torch.concatenate((train_Y[:(h := len(train_Y)//2)].reshape(-1, 1),
                              train_Y[h:].reshape(-1, 1)), axis=1)
  return [[f"sum({x.item() + y.item()})"] for x, y in labels]
#end.

% Data of the first digit.
input(0) ~ test(@mnist_images_test(0)), train(@mnist_images_train(0)).
% Data of the second digit.
input(1) ~ test(@mnist_images_test(1)), train(@mnist_images_train(1)).

% Neural annotated disjunction over each digit from 0 to 9; use Adam as optimizer
% and a learning rate of 0.001.
?::digit(X, {0..9}) as @digit_net with optim = "Adam", lr = 0.001 :- input(X).
% The sum.
sum(Z) :- digit(0, X), digit(1, Y), Z = X+Y.

% Learn the parameters of the program from the "sum(X)" atoms.
#learn @mnist_labels_train, lr = 1., niters = 5, alg = "lagrange", batch = 500.
#semantics maxent.
% Ask for the probability of all groundings of sum(X).
#query sum(X).
