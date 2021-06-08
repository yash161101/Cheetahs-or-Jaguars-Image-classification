# wild-cats-classification
An image processing project to classify between Cheetahs and Jaguars.

## About the Project

Cheetahs and Jaguars are two similar looking members of the feline kingdom. The major differences between the cats are:
- Cheetahs have a spotted coat design while Jaguars have rosettes on their coat.
- Jaguars are more muscular and stocky while Cheetahs have a leaner and longer frame.

This project uses Convolutional Neural Networks (CNNs) for classification, which is a Deep Learning Algorithm that assigns weights and biases to different aspects of the image and differentiates based on these differentiated aspects. The architecture of CNNs is analogous to that of the connectivity pattern of Neurons in the human brain.

## Requirements 
- Python (>3.6)
- TensorFlow 
- opencv
- Flask (optional)

## About the Dataset 
The [dataset](https://www.kaggle.com/iluvchicken/cheetah-jaguar-and-tiger) was found on Kaggle. Only the relevant parts of the same have been used in this project.
For each of the animals, we are using 900 images for training (stored in the 'train' folder) and 100 images for validation (stored in the val folder). The images we later want to test our model on are stored in the 'test' folder.

The necessary libraries are imported and the folders with the training and validation images are called.

### 1. Pre-Processing of Images 
- The images in both the training and testing dataset are re-sized to a size of 300*300. This ensures that the size of all the images are consistent.
- We also add labels to each of the image (Cheetah is assigned a label of 0 while Jaguar is assigned the label 1).
- The images are converted to an array and split into features and labels. 

### 2. Building the Model
We use the following layers in our model: 

- **Conv2D()** 
This is a filter to create a feature map that summarizes the presence of detected features or patterns. In our case, there are 32, 64, 128 and 128 filters or kernels in respective layers. We increase the filter size in subsequent layers to capture as many combinations as possible.

- **MaxPool2D():** 
It allows us to choose the maximum value within a matrix to reduce image size without losing the image information.

- **Flatten():** 
Converts the multi-dimensional image data array to a single dimensional array.

- **Dense():** 
Fully connected neural network layer where each input node is connected to each output node. We have used this layer twice, one in the hidden layer with 512 neurons and then for the output layer to make final predictions.

We use adam and binary_crossentropy since it's a binary classification project and adam is the best adaptive optimizer for most of the cases.

### 3. Training
We train our model with the model.fit() function in 10 epochs with 30 steps per epoch. After every epoch, we also validate the model on our validation dataset.

This leaves us with a model whose loss and accuracy for training and validation can be visualised as follows:

![image](https://user-images.githubusercontent.com/59526423/121019298-ea15ac00-c7bc-11eb-8bfa-c84d76f32c6b.png)

The downward trend in loss and upward trend in accuracy is necessary to show that the model improved after each epoch.

### 4. Testing
We test our saved model on the images in the 'test' folder which are images from the Internet that are present in neither the train, nor the validation folders. The model should be able to accurately make predictions and return 'Cheetah' or 'Jaguar' accordingly.

## Optional
This model can be used as a web application where you can upload an image and have it be classified. This web app was made using flask and the code for the same is stored in the app.py file of this repository. The static folder contains the .css files for the design of the web app, along with the image you need classified and the templates folder consists of the html templates. The web app output looks as follows:

![image](https://user-images.githubusercontent.com/59526423/121018860-868b7e80-c7bc-11eb-98b5-35569a2c377c.png)

You can find the link to the blog for the classification part of this project [here.](https://yash161101.medium.com/cheetah-or-jaguar-image-classification-convolutional-neural-network-437534643262)
