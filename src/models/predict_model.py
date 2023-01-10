import torch

def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0

    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)
        images = images.float()
        labels = labels.long()
        
        if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        pred = ps.max(1)[1]
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == pred)
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
  
    return test_loss, accuracy
