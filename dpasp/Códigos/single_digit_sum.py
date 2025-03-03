import pasp
import torchvision
import numpy as np

def mnist_labels():
  "Return the first and second digit values of the test set."
  Y = torchvision.datasets.MNIST(root="/tmp", train=False, download=True).targets
  return Y[:(h := len(Y)//2)].data.numpy(), Y[h:].data.numpy()

P = pasp.parse(...)                      # Parse the sum of digits program.
R = P(quiet=True)                        # Run program and capture output.
Y = np.argmax(R, axis=1).reshape(len(R)) # Retrieve what sum is likeliest.
D = mnist_labels()                       # D contains the digits of the test set.
T = D[0] + D[1]                          # The ground-truth in the test set.
accuracy = np.sum(Y == T)/len(T)         # Digit sum accuracy.
print(f"Accuracy: {100*accuracy}%")
