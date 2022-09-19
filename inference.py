import torch

from train import FeedForwardNet, download_mnist_datasets

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def predict(model, input, target, class_mapping):
    model.eval() # call when you need to do inference
    with torch.no_grad():
        predictions = model(input) # Tensor (dim n samples, dim n classes) -> Tensor(1, 10)
        predicted_index = predictions[0].argmax(0) # gets index with highest val
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
        

if __name__ == "__main__":
    # load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a sample from the validation dataset for inference
    input, target = validation_data[50][0], validation_data[50][1]

    # make an inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

    print(f"Predicted: '{predicted}', Expected: '{expected}'")


    accuracy = 0
    for i in range(len(validation_data)):
        input, target = validation_data[i][0], validation_data[i][1]
        predicted, expected = predict(feed_forward_net, input, target, class_mapping)
        if predicted == expected:
            accuracy += 1
    
    print(f"Accuracy: {accuracy / len(validation_data)}")


