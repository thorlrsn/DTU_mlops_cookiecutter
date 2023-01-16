import torch
import wandb

from src.models.predict_model import validation

def train(model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40, wandb_log=False):
    # if optimizer is None:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    steps = 0
    running_loss = 0
    loss_total = []    

    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()

        for images, labels in trainloader:
            steps += 1
            images = images.float()
            labels = labels.long()
            
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loss_total.append(loss.item())
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                if wandb_log:
                    # wandb logging
                    wandb.log({"Training Loss:": (running_loss/print_every),
                        "Test Loss:": (test_loss/len(testloader)),
                        "Test Accuracy:": (accuracy/len(testloader))})
                    
                running_loss = 0
                
                # Make suredropout and grads are on for training
                model.train()

    # fig, ax = plt.subplots()
    # ax.plot(loss_total, label="Training Loss")
    # legend = ax.legend(loc='upper center',fontsize='x-large')
    # legend.get_frame().set_facecolor('C0')
    # plt.show()
