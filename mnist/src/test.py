import torch

model = torch.jit.load("models/digit_recon.pt")

example_input = torch.randn(1, 1, 28, 28)  # A random MNIST-like input
output = model(example_input)

# Process the output
_, predicted_class = output.max(dim=1)
print(f"Predicted Class: {predicted_class.item()}")