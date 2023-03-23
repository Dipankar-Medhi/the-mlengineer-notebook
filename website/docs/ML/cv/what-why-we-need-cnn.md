---
sidebar_label: What and Why we need CNN
title: What and Why do we need Convulation Neural Network?
sidebar_position: 1
---

:::tip Resources
1. https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network
2. https://www.ibm.com/cloud/learn/convolutional-neural-networks

:::

When we look an image, we do not look directly into the pixels. We look at the combinations of pixels to process it as an image.

So the same principle is used in case of computers. We some how want to combine every pixel values of an image to get something meaningful and let our computers derive results from these combination of pixels.

A **Convulation Neural Network** is a technique of combining the pixels of an image and training machines to identify or detect images/objects for classification or computer vision tasks.

The architecture of a convolutional neural network is a multi-layered [feed-forward neural network](https://deepai.org/machine-learning-glossary-and-terms/feed-forward-neural-network), made by stacking many [hidden layers](https://iq.opengenus.org/hidden-layers/) on top of each other in sequence. It is this sequential design that allows convolutional neural networks to learn hierarchical features.

The hidden layers are typically convolutional layers followed by activation layers, some of them followed by pooling layers.

![image](https://images.deepai.org/user-content/7398488153-thumb-4335.svg)
`source: Deepai`

| key| Description |
| --- | --- |
| C1 | The first convolutional layer. This consists of six convolutional kernels of size 5x5, which ‘walk over’ the input image. C1 outputs six images of size 28x28. The first layer of a convolutional neural network normally identifies basic features such as straight edges and corners. |   
| S2 | A subsampling layer, also known as an average pooling layer. Each square of four pixels in the C1 output is averaged to a single pixel. S2 scales down the six 28x28 images by a factor of 2, producing six output images of size 14x14. |
| C3 | The second convolutional layer. This consists of 16 convolutional kernels, each of size 5x5, which take the six 14x14 images and walk over them again, producing 16 images of size 10x10.  
| S4 | The second average pooling layer. S4 scales down the sixteen 10x10 images to sixteen 5x5 images.  |
| C5 | A fully connected convolutional layer with 120 outputs. Each of the 120 output nodes is connected to all of the 400 nodes (5x5x16) that came from S4. At this point the output is no longer an image, but a 1D array of length 120.  |
| F6 | A fully connected layer mapping the 120-array to a new array of length 10. Each element of the array now corresponds to a handwritten digit 0-9.  |
| Output Layer | A softmax function which transforms the output of F6 into a  [probability distribution](https://deepai.org/machine-learning-glossary-and-terms/probability-distribution)  of 10 values which sum to 1. | 