import pasp
import torchvision
import numpy as np

def mnist_labels():
  "Return the first and second digit values of the test set."
  Y = torchvision.datasets.MNIST(root="/tmp", train=False, download=True).targets
  h = len(Y)//4
  
  labels = []
  
  labels.append(Y[:h].data.numpy())
  labels.append(Y[h:2*h].data.numpy())
  labels.append(Y[2*h:3*h].data.numpy())
  labels.append(Y[3*h:4*h].data.numpy())
  
  D = []
  for i in range(h): D.append([labels[0][i], 
                               labels[1][i],
                               labels[2][i], 
                               labels[3][i]])

  T = np.array([a + b + c + d for a, b, c, d in D])
  return T
 

P = pasp.parse("sum4.lp")      # Parse the sum of digits program.
R = P(quiet=True)                        # Run program and capture output.
print("Calculating accuracy...")
Y = np.argmax(R, axis=1).reshape(len(R)) # Retrieve what sum is likeliest.
T = mnist_labels()                       # The ground-truth in the test set.

accuracy = np.sum(Y == T)/len(T)         # Digit sum accuracy.
print(f"Accuracy: {100*accuracy}%")