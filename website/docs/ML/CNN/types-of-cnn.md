---
sidebar_label: Types of CNN
title: Types of Convulation Neural Networks
---
:::tip Resource
https://iq.opengenus.org/different-types-of-cnn-models/
:::

- LeNet-5 - LeNet is the most popular CNN architecture it is also the first CNN model which came in the year 1998.It is made up of seven layers, each with its own set of trainable parameters. It accepts a 32 × 32 pixel picture, which is rather huge in comparison to the images in the data sets used to train the network. RELU is the activation function that has been used. 
- AlexNet - Starting with an 11x11 kernel, Alexnet is built up of 5 conv layers. For the three massive linear layers, it was the first design to use max-pooling layers, ReLu activation functions, and dropout. The network was used to classify images into 1000 different categories.
  
  ![image](https://iq.opengenus.org/content/images/2021/11/2-3.png)
- GoogLeNet - Google created the model, which incorporates an improved implementation of the original LeNet design. This is based on the inception module concept. GoogLeNet is a variation of the Inception Network, which is a 22-layer deep convolutional neural network.
  ![image](https://iq.opengenus.org/content/images/2021/11/inception-module.png)
  The InceptionNet/GoogleLeNet design is made up of nine inception modules stacked on top of each other, with max-pooling layers between them (to halve the spatial dimensions). It is made up of 22 layers (27 with the pooling layers). After the last inception module, it employs global average pooling.
- ResNet - ResNet is one of the most widely used and effective deep learning models to date. ResNets are made up of what's known as a residual block.
This is built on the concept of "skip-connections" and uses a lot of batch-normalization to let it train hundreds of layers successfully without sacrificing speed over time.
- ZFNet - The architecture consists of five shared convolutional layers, as well as max-pooling layers, dropout layers, and three fully connected layers. In the first layer, it employed a 77 size filter and a lower stride value. The softmax layer is the ZFNet's last layer.
  ![image](https://iq.opengenus.org/content/images/2021/11/llustration-of-the-ZFNet-architecture-It-consisted-of-5-shareable-convolutional-layers.png)

- VGG - VGG is a convolutional neural network design that has been around for a long time. It was based on a study on how to make such networks more dense. Small 3 x 3 filters are used in the network. The network is otherwise defined by its simplicity, with simply pooling layers and a fully linked layer as additional components.
  
  VGG was created with 19 layers deep to replicate the relationship between depth and network representational capability.

  VGG replaced the 11x11 and 5x5 filters with a stack of 3x3 filters, demonstrating that the simultaneous placement of small size (3x3) filters may provide the effect of a big size filter (5x5 and 7x7). By lowering the number of parameters, the usage of tiny size filters gives an additional benefit of low computing complexity.

- PolyNet - PolyNet traverses the whole network and explores the entire space, making intelligent decisions about weights and structure so that it may automate improvements to increase performance and functionality, with better results for the end user. PolyNet is a first: a ready-to-use neural network training solution that you can implement in-house, ensuring that your data never leaves the premises. That is PolyNet's one-of-a-kind and distinguishing feature.
  ![image](https://iq.opengenus.org/content/images/2021/11/1_efoLd4MlOKTWnqh4oASFtA.png)