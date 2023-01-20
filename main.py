import torch
import wandb
from torch import optim, nn
import torchvision.models as models

from src.features.build_features import MyAwesomeModel as Mymodel
from src.models import train_model
import dill
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
from src.data.make_dataset import MyDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train(sweep=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Given model
    # model = fc_model.Network(784, 10, [512, 256, 128])
    
    # Own model
    model = Mymodel()
    
    model.to(device)
    
    if sweep is True:
        # Wandb sweep stuff
        wandb.init()
        wandb.watch(model, log_freq=100)
        bs = wandb.config.batch_size
        epochs = wandb.config.epochs
        lr  =  wandb.config.lr

    elif sweep is False:
        # # Old wandb config
        args = {"batch_size": 64,  # try log-spaced values from 1 to 50,000
            "num_workers": 0,  # try 0, 1, and 2
            "pin_memory": True,  # try False and True
            "precision": 32,  # try 16 and 32
            "optimizer": "Adadelta",  # try optim.Adadelta and optim.SGD
            "lr": 0.01,
            "epochs": 5,
            }
        wandb.init(config=args)
        wandb.watch(model, log_freq=100)
        bs = wandb.config.batch_size
        lr  = wandb.config.lr
        epochs = wandb.config.epochs
        num_workers = wandb.config.num_workers

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    # train_set, test_set = mnist(_PATH_DATA)

    # dataset = 'data'
    train_path = "data/train_"
    test_path = "data/test.npz"

    train_dataset = MyDataset(train_path, train=True)
    val_dataset = MyDataset(test_path, train=False)

    trainloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(dataset=val_dataset, batch_size=bs, shuffle=False, pin_memory=True, num_workers=num_workers)
   
    # Own training (The same as the given, since its pretty awesome)
    print("Training...")
    train_model.train(model, trainloader, testloader, criterion, optimizer, epochs, wandb_log=True)

    # Exporting table to wandb with sampled images, preditions and truths
    # wandb_table(model, testloader)

    # Saving model
    save_checkpoint(model)

def evaluate(model_checkpoint):
    print("Evaluating...")
    print(model_checkpoint)

    bs = wandb.config.batch_size
    test_set = torch.load("data/processed/test_tensor.pt")
    testloader = DataLoader(dataset=test_set, batch_size=bs, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = model_checkpoint
    model.eval()
                
    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = Mymodel.validation(model, testloader, criterion)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

# custom functions
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Mymodel()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_checkpoint(model):
    # Giving values but they are not used.
    checkpoint = {'input_size': 1,
              'output_size': 120,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, 'model_v1_0.pth')

def wandb_table(model, testloader):
    example_images = []
    example_pred = []
    example_truth = []

    for images, labels in testloader:
        images = images.float()
        labels = labels.long()

        if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

        output = model.forward(images)

        ps = torch.exp(output)
        pred = ps.max(1)[1]

        # would like to perform the addition to the table here but it wont work.
        example_pred.append(pred[0].item())
        example_truth.append(labels[0].item())
        example_images.append(wandb.Image(images[0], caption="Pred: {} Truth: {}".format(pred[0].item(), labels[0])))
                
    columns = ["Image", "Prediction", "Truth"]
    table = wandb.Table(columns=columns)

    # Adding to table here instead in a small loop
    for i in range(len(example_pred)):
        table.add_data(example_images[i], example_pred[i], example_truth[i])
    
    wandb.log({"Preditions": table})

def sweep_config():
    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'test_loss'
		},
    'parameters': {
        'batch_size': {'values': [32, 64, 128]},
        'epochs': {'values': [2, 5, 10]},
        'lr': {'max': 0.1, 'min': 0.0001}
    }
}

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration)
    # wandb.init()
    return sweep_id



if __name__ == "__main__":  
    
    ##################################
    #### Define sweep or no sweep ####
    ##################################
    sweep = False

    if sweep is False:
        train(sweep)

    elif sweep is True:
        sweep_id = sweep_config()
        wandb.agent(sweep_id, function=train, count=4)

    
    
    
    