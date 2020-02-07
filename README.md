# Fashionmnist
Sample use of the fashionmnist in tensorflow
## Get started
The project is related to the fashion_mnist dataset that is extracted from keras and deals with manipulating that dataset. The dataset is a collection of clothes and shoes classified into different sections i.e. shirts, boots, sweaters among others.

According to the model in fashion.py, the dataset is loaded and split into training and test images. The two categories(training and test images) in which the images have been split into are normalized by dividing them by 255. The model is constructed using one hidden layer and a softmax activation function in the output layer with 10 categories in which the images are being classified. The model is trained using 10 epochs and used to make predictions of the test images by indexing the location of the item in the dataset that you would want to classify.

The model in mltrial.py more so operates the same way the model in fashion.py operates only that it has an additional callback class which helps terminate the model training process immmediately the model's accuracy exceeds 90%.
