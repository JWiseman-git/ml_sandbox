import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# MNIST transforms: single channel 28x28 images, normalize to mean=0.1307 std=0.3081
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load MNIST
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=data_transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (preds.argmax(dim=1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = loss_fn(preds, y)

            total_loss += loss.item() * X.size(0)
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, correct / total


if __name__ == "__main__":
    print(f"Using device: {device}")

    optimizers = {
        "Adam": lambda params: torch.optim.Adam(params, lr=LEARNING_RATE),
        "GD (batch)": lambda params: torch.optim.SGD(params, lr=LEARNING_RATE),
        "SGD (momentum)": lambda params: torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9),
    }

    results = {}

    for opt_name, opt_fn in optimizers.items():
        print(f"\n{'=' * 50}")
        print(f"Training with: {opt_name}")
        print(f"{'=' * 50}")

        model = MNISTClassifier().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = opt_fn(model.parameters())

        train_losses = []
        test_losses = []

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_dataloader, loss_fn, optimizer)
            test_loss, test_acc = evaluate(model, test_dataloader, loss_fn)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

        results[opt_name] = {"train": train_losses, "test": test_losses}

    # Plot loss comparison
    epochs = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for opt_name, losses in results.items():
        axes[0].plot(epochs, losses["train"], label=opt_name)
        axes[1].plot(epochs, losses["test"], label=opt_name)

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle("Optimizer Comparison on MNIST")
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150)
    plt.show()
    print("\nFigure saved to optimizer_comparison.png")
