import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ToTensor() reshapes so each value is normalized between [0,1]
def download_mnist_datasets():
    # MNIST is a concrete implementation
    train_data = datasets.MNIST(
        root="data", # we are telling PyTorch where to store the dataset
        download=True,
        train=True,
        transform=ToTensor() # allows us to apply a transform directly to data
    )

    validation_data = datasets.MNIST(
        root="data", 
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data, validation_data


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.flatten = nn.Flatten() 
        # Sequential allows us to pack together multiple layers
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10), # 10 is the number of classes in MNIST
        )
        self.softmax = nn.Softmax(dim=1) # Basic transformation/normalization

    def forward(self, input_data):
        # Indicates PyTorch how to manipulate the dating/sequence
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs) # just pass inputs into model
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        # at each iteration, the optimizer will calculate gradients 

        optimizer.zero_grad() # resets gradients
        loss.backward() # applies backpropagation
        optimizer.step() # updates weights

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-------------------")
    print("Training is done!")

if __name__ == "__main__":
    # constants
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = .001

    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded!")

    # create a dataloader for the train data
    
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build the model
    # device can be CUDA or CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device: ", device)
    feed_forward_net = FeedForwardNet().to(device=device)

    # instantiate loss fn + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)
    
    # train the model
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    # store the model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")