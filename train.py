# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import argparse
#from workspace_utils import active_session


parser = argparse.ArgumentParser(description='Process training input')
parser.add_argument('data_dir', help='Directory with images', type=str)
parser.add_argument('-sd','--save_directory', help='Directory to save checkpoint file, Default is current directory', type=str, default='.')
parser.add_argument('-arch', '--arch', help='Choose architecture vgg16 or densenet121. Default arch=vgg16', default='vgg16')
parser.add_argument('-r', '--learning_rate', help='Learning rate. Default learnung_rate=0.001', type=float, default=0.001)
parser.add_argument('-e', '--epochs', help='Number of epochs. Default epochs=3', type=int, default=3)
parser.add_argument('-u', '--hidden_units', help='Number of hidden units in a layer. default hidden_units=1000', type=int, default=1000)
parser.add_argument('-gpu', '--gpu', help='Use gpu', action='store_true')

def main():
    global args, device
    args = parser.parse_args()
    data_dir = args.data_dir
    file = args.save_directory + "/" + "chk_" + args.arch + ".pth"
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units

    epochs = args.epochs
    if args.gpu:
        print("Use gpu")
        device = 'cuda'
    else:
        device = 'cpu'

    if args.arch == 'vgg16':
            #load VGG16 model
            print("Initializing vgg16 model")
            model = models.vgg16(pretrained=True)
            in_features = 25088
            #print(model)
    elif args.arch == 'densenet121':
            print("Initializing densenet121 model")
            model = models.densenet121(pretrained=True)
            in_features = 1024
            #print(model)
    else:
            print("Warning: ", args.arch, " model is unavailable, will use vgg16 model by default")
            print("Initializing vgg16 model")
            model = models.vgg16(pretrained=True)
            in_features = 25088
            #print(model)   

    #Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(90), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(0.2), transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_image_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_image_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_image_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)


    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_image_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_image_data, batch_size=32, shuffle=True)


    #Build and train network
    for param in model.parameters():
        param.requires_grad = False

    #new classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)), ('relu', nn.ReLU()), ('fc2', nn.Linear(hidden_units, 102)), ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    #print(model)

    #Train a model with a pre-trained network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    do_deep_learning(model,train_loader, valid_loader, criterion, optimizer, epochs, 40)
    check_accuracy_on_test(model, test_loader)
    save_checkpoint(model, train_image_data, file)
    
    
# Validation pass
def validation(model, valid_loader, criterion):
    valid_loss = 0
    accuracy = 0
    model = model.to(device)

    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def do_deep_learning(model, train_loader, valid_loader, criterion, optimizer, epochs, print_every=40):
    epochs = epochs
    print_every = print_every
    steps = 0
    model = model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs), "Training Loss: {:.3f}.. ".format(running_loss/print_every), "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)), "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                running_loss = 0

                model.train()


#Validation on the test set
def check_accuracy_on_test(model, testloader):
    print("Checking accuracy on test set:")
    correct = 0
    total = 0
    model = model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Total images: ", total)
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def save_checkpoint(model, train_image_data, file):
    print("Saving checkpoint file to ", file)
    check_point_file = file
    model.class_to_idx = train_image_data.class_to_idx
    
    checkpoint_dict = {
        'arch': args.arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint_dict, check_point_file)

if __name__ == '__main__':
    main()
