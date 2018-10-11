# Data Scientist Nanodegree
# Supervised Learning
## Project: Image Classifier

Implemented an image classifier with PyTorch. Built and trained a deep neural network on the flower data set. Added the possibility to use GPU for neural network training and prediction.  The application was developped using image classification model. The application is a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [torch](http://pytorch.org)


You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

additionaly the code can be ran using GPU

### File Description

1. ImageClassifierProject.ipynb: the notebook that preprocesses the image dataset, traind the image classifier and use the trained classifier to predict image content.
2. train.py: part of image classifier application. Trains a new network on a dataset and save the model as a checkpoint.  
3. predict.py: uses a trained network to predict the class for an input image.
4. cat_to_name.json: the json file with a dictionary of flower names and corresponding number.

### Data
The data can not be uploaded here due to the size of flower library. However, an application can be trained on any set of labeled images.

## Run

To train a new network on a data set with train.py: 
                
        Basic usage: python train.py data_dir

--------------------------------------------------------------------------------------------------

positional arguments:
  data_dir              Directory with images (Must contain 'train', 'valid' and 'test' subfolders)

optional arguments:
  -h, --help            show this help message and exit
  -sd SAVE_DIRECTORY, --save_directory SAVE_DIRECTORY
                        Directory to save checkpoint file, Default is current
                        directory
  -arch ARCH, --arch ARCH
                        Choose architecture vgg16 or densenet121. Default
                        arch=vgg16
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate. Default learnung_rate=0.001
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs. Default epochs=3
  -u HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
                        Number of hidden units in a layer. default
                        hidden_units=1000
  -gpu, --gpu           Use gpu

  
--------------------------------------------------------------------------------------------------

To predict flower name from an image with predict.py along with the probability of that name:

    Basic usage: python predict.py input checkpoint

--------------------------------------------------------------------------------------------------

positional arguments:
  input                 Path to input image
  checkpoint            Path to checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  -k TOP_K, --top_k TOP_K
                        Top K classes to display. Default top_k=1
  -n CATEGORY_NAMES, --category_names CATEGORY_NAMES
                        Category names to display. Default category_names =
                        cat_to_name.json
 

## Licensing, Authors, Acknowledgements
This project is part of the Machine Learning Engineer Nanodegree Program at udacity.com


