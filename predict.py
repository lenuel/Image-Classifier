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
from torch import nn
import json
try:
    from tabulate import tabulate
except:
    print("Warning: Please install tabulate to display pretty table with results by running: 'pip install tabulate'")


parser = argparse.ArgumentParser(description='Process training input')
parser.add_argument('input', help='Path to input image', type=str)
parser.add_argument('checkpoint', help='Path to checkpoint file', type=str)
parser.add_argument('-k', '--top_k', help='Top K classes to display. Default top_k=1 ', type=int, default=1)
parser.add_argument('-n', '--category_names', help='Category names to display. Default category_names = cat_to_name.json', default='cat_to_name.json')
parser.add_argument('-gpu', '--gpu', help='Use gpu', action='store_true')

def main():
    global args, device
    args = parser.parse_args()
    
    if args.gpu:
        print("Use gpu")
        device = 'cuda'
    else:
        device = 'cpu'

    args = parser.parse_args()

    image_path = args.input
    file = args.checkpoint

    model = load_checkpoint(file)
    image = Image.open(image_path)
    processed_im = process_image(image)

    ps, idx = predict(image_path, model, args.top_k)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[i] for i in idx]

    if args.top_k==1:
        print("\n Flower class is ", classes[0]," ( index=", idx[0],") with probability of",  "{0:.2f}".format(ps[0]*100) + "% \n")
    else:
        print("\n|------------------------")
        print("|The top ", args.top_k, " classes are:")
        print("|------------------------")
        line = []
        for i in range(args.top_k):
            line.append([classes[i], idx[i], "{0:.2f}".format(ps[i]*100) + "%"])
        try:
            print(tabulate(line, headers=['Name', 'Index', 'Probability'], tablefmt='orgtbl'))
        except:
            print("Warning: Please install tabulate to display pretty table with results by running: 'pip install tabulate'")
            print('Name', 'Index', 'Probability')
            [print(i) for i in line]            

# Load a checkpoint and rebuilds the model
def load_checkpoint(file):
    chkpt = torch.load(file,map_location='cpu')
    arch = chkpt['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    model.classifier = chkpt['classifier']
    model.load_state_dict(chkpt['state_dict'])
    model.class_to_idx = chkpt['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Process a PIL image for use in a PyTorch model    
    #resize with a shortest dimension being 256
    base = 256
    width, height = image.size
    if width <= height:
        new_height = int(float(height/width)*base)
        processed_im = image.resize((base, new_height), Image.ANTIALIAS)
    else:
        new_width = int(float(width/height)*base)
        processed_im = image.resize((new_width,base), Image.ANTIALIAS)
    
    #crop center image
    width, height = processed_im.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    processed_im = processed_im.crop((left, top, right, bottom))

    #convert values to 0-1
    np_image = np.array(processed_im)/255
    
    #normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.divide(np_image-mean, std)
    
    # Move color channels to first dimension (from third dimesion of PIL) as expected by PyTorch
    transposed_im = np_image.transpose((2, 0, 1))
    
    return transposed_im

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #predict the class from an image file
    image = Image.open(image_path)
    processed_im = process_image(image)
 
    model = model.to(device)
    
    model.eval()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():   
        #convert from double numpy to torch float type
        torch_im = torch.from_numpy(processed_im).float()
        torch_im.unsqueeze_(0) 
        torch_im = torch_im.to(device)
        
        outputs = model.forward(torch_im)
        ps, idx = outputs.topk(topk)
        ps = ps.cpu()
        ps = np.exp(ps.numpy())
        
        #invert mapping
        inv_map = {v: k for k, v in model.class_to_idx.items()}  
        idx = idx.cpu()
        idx = [inv_map[i] for i in idx.numpy()[0]]
        #classes = [cat_to_name[i] for i in idx]
        #print("Probabilites: ", ps[0])
        #print("Classes: ", idx)
        #print("Class_names: ", classes)
    
    return ps[0], idx

if __name__ == '__main__':
    main()




