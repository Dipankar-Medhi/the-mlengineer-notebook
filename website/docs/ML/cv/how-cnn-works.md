---
sidebar_label: How a CNN work?
title: How do Convulation Neural Networks work?
---

Convulation Neural Networks work by combining close by pixels into one pixel to get the value using some kind of method like $\sum{w_ip_i}$ where $w_i$ is weights and $p_i$ are pixels.

### Filter

- A filter is passed on the images (say a $K\times{}K$ filter).
- This $K\times{}K$  filter has  $w_i$, that is multiplied to each pixel values $p_i$ of the image.
- This filter is used to identify a pattern in the image.
- The summation of $w_ip_i$, gives a value that is the output of a convulation neural network.
  
- Then we decide stride and Padding.

### Stride

> Stride is the distance, or number of pixels, that the kernel moves over the input matrix. While stride values of two or greater is rare, a larger stride yields a smaller output.
The value that we decide to shift the filter over an image with every iteration.

- If stride increases, the output matrix decreases.
   
    
- Also the stride value must not be too big, other wise there might be gaps in between two parts of the image and we can miss those pixel during preproceessing.
- If the stride size is s, then output size is approximately, $\frac{n}{s}$ where n is size of image.


### Padding
Used to put some 0s around the matrix pixel.

Zero-padding is usually used when the filters do not fit the input image. This sets all elements that fall outside of the input matrix to zero, producing a larger or equally sized output. There are three types of padding:

- Valid padding: This is also known as no padding. In this case, the last convolution is dropped if dimensions do not align.
- Same padding: This padding ensures that the output layer has the same size as the input layer
- Full padding: This type of padding increases the size of the output by adding zeros to the border of the input.


### Channels
There are channels in an image, like in RGB, there are 3 channels. 

So our filter will be $K\times{}K\times{}3$.

Even if there are no channel, together they act as channels for the next convulation layer. 

### Pooling
Added after convulation layer to reduce the size of convulation layer.

Examples: Average pooling, Max pooling.

- Max Pooling → Takes the max value of pixel in the output.
- And if the pooling layer is 4x4 and image is 28x28, we will get 7x7.
- Average pooling: As the filter moves across the input, it calculates the average value within the receptive field to send to the output array.

### Fully Connected layer
The pixel values of the input image are not directly connected to the output layer in partially connected layers. 

However, in the fully-connected layer, each node in the output layer connects directly to a node in the previous layer.

This layer performs the task of classification based on the features extracted through the previous layers and their different filters. While convolutional and pooling layers tend to use ReLu functions, FC layers usually leverage a softmax activation function to classify inputs appropriately, producing a probability from 0 to 1.