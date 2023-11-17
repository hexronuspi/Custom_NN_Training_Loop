from torchvision import datasets, transforms
import torch
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

print("Downloading and loading the training data...")
train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Initializing the model...")
model = SinglePass()

print("Starting training...")
model.fit(train_dataset, test_dataset , batch_size=64, epochs=5, device=device)
print("Training completed.")
