import json
import torch.optim as optim

use_cuda = torch.cuda.is_available()

# create the network
network = PDnet()

# create the loss function and the optimizer
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(network.parameters(), lr=0.003)

# find the gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# putting the neural network on the gpus
if torch.cuda.device_count() > 1:
    network = nn.DataParallel(network, device_ids = [0,1,2,3])
    
network = network.to(device)

epochs = 100

train_losses, test_losses = [], []
losses = {}


for e in range(epochs):
    tot_train_loss = 0
    i = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        # move the images and the labels to the gpus
        images = images.to(device)
        images = images.float()
        labels = labels.to(device)
        labels = labels.float()
        
        #forward prop
        output = network.forward(images)
        
        # find the loss
        loss = criterion(output, labels)
        tot_train_loss += loss.item()

        # back propagation
        loss.backward()
        
        # change the weights according to the loss function
        optimizer.step()
        
    tot_test_loss = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for images, labels in testloader:

            images = images.to(device)
            images = images.float()
            labels = labels.to(device)
            labels = labels.float()

            output = network.forward(images)

            loss = criterion(output, labels)
            tot_test_loss += loss.item()

        # Get mean loss to enable comparison between train and test sets
        train_loss = tot_train_loss / len(trainloader.dataset)
        test_loss = tot_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss))


        losses[e+1] = [train_loss, test_loss]
        with open('losses.json', 'w') as fp:
            json.dump(losses, fp)

        ckpt_name = 'ckpt/checkpoint_'+str(e+1)+'.pth'
        torch.save(network.state_dict(), ckpt_name)
