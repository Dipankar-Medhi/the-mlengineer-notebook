---
sidebar_label: Computer Vision Questions
title: Computer Vision Interview Questions
sidebar_position: 6
---

:::note Source
https://www.kaggle.com/getting-started/183949
::: 

Q1 Which of the following is a challenge when dealing with computer vision problems?
Variations due to geometric changes (like pose, scale, etc), Variations due to photometric factors (like illumination, appearance, etc) and Image occlusion. All the above-mentioned options are challenges in computer vision

Q2 Consider an image with width and height as 100×100. Each pixel in the image can have a color from Grayscale, i.e. values. How much space would this image require for storing?
The answer will be 8x100x100 because 8 bits will be required to represent a number from 0-256

Q3 Why do we use convolutions for images rather than just FC layers?
Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as it's own filter/feature detector.

Q4 What makes CNN’s translation-invariant?
As explained above, each convolution kernel acts as it's own filter/feature detector. So let's say you're doing object detection, it doesn't matter where in the image the object is since we're going to apply the convolution in a sliding window fashion across the entire image anyways.

Q5 Why do we have max-pooling in classification CNNs?
for a role in Computer Vision. Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNN’s more translation in-variance. Check out this great video from Andrew Ng on the benefits of max-pooling.

Q6 Why do segmentation CNN’s typically have an encoder-decoder style/structure?
The encoder CNN can basically be thought of as a feature extraction network, while the decoder uses that information to predict the image segments by "decoding" the features and upscaling to the original image size.
 

Q7 What is the significance of Residual Networks?
The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting paper about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

Q8 What is batch normalization and why does it work?
Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and a standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized. How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function and then feed it into the following layer (sub-network).

Q9 Why would you use many small convolutional kernels such as 3x3 rather than a few large ones?
This is very well explained in the VGGNet paper. There are 2 reasons: First, you can use several smaller kernels rather than few large ones to get the same receptive field and capture more spatial context, but with the smaller kernels you are using less parameters and computations. Secondly, because with smaller kernels you will be using more filters, you'll be able to use more activation functions and thus have a more discriminative mapping function being learned by your CNN.

Q10 What is Precision?
Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances
Precision = true positive / (true positive + false positive)

Q11 What is Recall?
Recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Recall = true positive / (true positive + false negative)



Q12 Define F1-score.
It is the weighted average of precision and recall. It considers both false positive and false
negatives into account. It is used to measure the model’s performance.
 
F1-Score = 2 * (precision * recall) / (precision + recall)

Q13 What is cost function?
The cost function is a scalar function that Quantifies the error factor of the Neural Network. Lower the cost function better than the Neural network. Eg: MNIST Data set to classify the image, the input image is digit 2 and the Neural network wrongly predicts it to be 3

Q14 List different activation neurons or functions
●	Linear Neuron
●	Binary Threshold Neuron
●	Stochastic Binary Neuron
●	Sigmoid Neuron
●	Tanh function
●	Rectified Linear Unit (ReLU)

Q15 Define Learning rate.
The learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient.

Q16 What is Momentum (w.r.t NN optimization)?
Momentum lets the optimization algorithm remembers its last step, and adds some proportion of it to the current step. This way, even if the algorithm is stuck in a flat region, or a small local minimum, it can get out and continue towards the true minimum.

Q17 What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?
Batch gradient descent computes the gradient using the whole dataset. This is great for convex or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution, either local or global. Additionally, batch gradient descent, given an annealed learning rate, will eventually find the minimum located in its basin of attraction.
Stochastic gradient descent (SGD) computes the gradient using a single sample. SGD works well (Not well, I suppose, but better than batch gradient descent) for error manifolds that have lots of local maxima/minima. In this case, the somewhat noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal.



Q18 Epoch vs Batch vs Iteration.
Epoch: one forward pass and one backward pass of all the training examples Batch: examples processed together in one pass (forward and backward) Iteration: number of training examples / Batch size
 
Q19 What is the vanishing gradient?
As we add more and more hidden layers, backpropagation becomes less and less useful in passing information to the lower layers. In effect, as information is passed back, the gradients begin to vanish and become small relative to the weights of the networks.

Q20 What are dropouts?
Dropout is a simple way to prevent a neural network from overfitting. It is the dropping out of some of the units in a neural network. It is similar to the natural reproduction process, where nature produces offsprings by combining distinct genes (dropping out others) rather than strengthening the co-adapting of them.

Q21 Can you explain the differences between supervised, unsupervised, and reinforcement learning?
In supervised learning, we train a model to learn the relationship between input data and output data. We need to have labeled data to be able to do supervised learning.
With unsupervised learning, we only have unlabeled data. The model learns a representation of the data. Unsupervised learning is frequently used to initialize the parameters of the model when we have a lot of unlabeled data and a small fraction of labeled data. We first train an unsupervised model and, after that, we use the weights of the model to train a supervised model. In reinforcement learning, the model has some input data and a reward depending on the output of the model. The model learns a policy that maximizes the reward. Reinforcement learning has been applied successfully to strategic games such as Go and even classic Atari video games.

Q22 What is data augmentation? Can you give some examples?
Data augmentation is a technique for synthesizing new data by modifying existing data in such a way that the target is not changed, or it is changed in a known way. Computer vision is one of the fields where data augmentation is very useful. There are many modifications that we can do to images:
●	Resize
●	Horizontal or vertical flip
●	Rotate, Add noise, Deform
●	Modify colors Each problem needs a customized data augmentation pipeline. For example, on OCR, doing flips will change the text and won’t be beneficial; however, resizes and small rotations may help.

Q23 What are the components of GAN?
●	Generator
●	Discriminator

Q24 What’s the difference between a generative and discriminative model?
A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.
 
Q25 What is Linear Filtering?
Linear filtering is a neighborhood operation, which means that the output of a pixel’s value is
decided by the weighted sum of the values of the input pixels.

Q26 How can you achieve Blurring through Gaussian Filter?
This is the most common technique for blurring or smoothing an image. This filter improves the resulting pixel found at the center and slowly minimizes the effects as pixels move away from the center. This filter can also help in removing noise in an image

Q27 What is Non-Linear Filtering? How it is used?
Linear filtering is easy to use and implement. In some cases, this method is enough to get the necessary output. However, an increase in performance can be obtained through non-linear filtering. Through non-linear filtering, we can have more control and achieve better results when we encounter a more complex computer vision task.

Q28 Explain Median Filtering.
The median filter is an example of a non-linear filtering technique. This technique is commonly used for minimizing the noise in an image. It operates by inspecting the image pixel by pixel and taking the place of each pixel’s value with the value of the neighboring pixel median.

Some techniques in detecting and matching features are:
●	Lucas-Kanade
●	Harris
●	Shi-Tomasi
●	SUSAN (smallest uni value segment assimilating nucleus)
●	MSER (maximally stable extremal regions)
●	SIFT (scale-invariant feature transform)
●	HOG (histogram of oriented gradients)
●	FAST (features from accelerated segment test)
●	SURF (speeded-up robust features)


Q29 Describe the Scale Invariant Feature Transform (SIFT) algorithm
SIFT solves the problem of detecting the corners of an object even if it is scaled. Steps to implement this algorithm:
●	Scale-space extrema detection – This step will identify the locations and scales that can still be recognized from different angles or views of the same object in an image.
●	Keypoint localization – When possible key points are located, they would be refined to get accurate results. This would result in the elimination of points that are low in contrast or points that have edges that are deficiently localized.
●	Orientation assignment – In this step, a consistent orientation is assigned to each key point to attain invariance when the image is being rotated.
●	Keypoint matching – In this step, the key points between images are now linked to recognizing their nearest neighbors.
 

Q30 Why Speeded-Up Robust Features (SURF) came into existence?
SURF was introduced to as a speed-up version of SIFT. Though SIFT can detect and describe key points of an object in an image, still this algorithm is slow.

Q31 What is Oriented FAST and rotated BRIEF (ORB)?
This algorithm is a great possible substitute for SIFT and SURF, mainly because it performs better in computation and matching. It is a combination of fast keypoint detector and brief descriptor, which contains a lot of alterations to improve performance. It is also a great alternative in terms of cost because the SIFT and SURF algorithms are patented, which means that you need to buy them for their utilization.

Q32 What is image segmentation?
In computer vision, segmentation is the process of extracting pixels in an image that is related. Segmentation algorithms usually take an image and produce a group of contours (the boundary of an object that has well-defined edges in an image) or a mask where a set of related pixels are assigned to a unique color value to identify it.
Popular image segmentation techniques:
●	Active contours
●	Level sets
●	Graph-based merging
●	Mean Shift
●	Texture and intervening contour-based normalized cuts

Q33 What is the purpose of semantic segmentation?
The purpose of semantic segmentation is to categorize every pixel of an image to a certain class or label. In semantic segmentation, we can see what is the class of a pixel by simply looking directly at the color, but one downside of this is that we cannot identify if two colored masks belong to a certain object.

Q34 Explain instance segmentation.
In semantic segmentation, the only thing that matters to us is the class of each pixel. This would somehow lead to a problem that we cannot identify if that class belongs to the same object or not. Semantic segmentation cannot identify if two objects in an image are separate entities. So to solve this problem, instance segmentation was created. This segmentation can identify two different objects of the same class. For example, if an image has two sheep in it, the sheep will be detected and masked with different colors to differentiate what instance of a class they belong to.

Q35	How	is	panoptic	segmentation	different	from	semantic/instance segmentation?
Panoptic segmentation is basically a union of semantic and instance segmentation. In panoptic segmentation, every pixel is classified by a certain class and those pixels that have several instances of a class are also determined. For example, if an image has two cars, these cars will
 
be masked with different colors. These colors represent the same class — car — but point to different instances of a certain class.

Q36 Explain the problem of recognition in computer vision.
Recognition is one of the toughest challenges in the concepts in computer vision. Why is recognition hard? For the human eyes, recognizing an object’s features or attributes would be very easy. Humans can recognize multiple objects with very small effort. However, this does not apply to a machine. It would be very hard for a machine to recognize or detect an object because these objects vary. They vary in terms of viewpoints, sizes, or scales. Though these things are still challenges faced by most computer vision systems, they are still making advancements or approaches for solving these daunting tasks.

Q37 What is Object Recognition?
Object recognition is used for indicating an object in an image or video. This is a product of machine learning and deep learning algorithms. Object recognition tries to acquire this innate human ability, which is to understand certain features or visual detail of an image.

Q38 What is Object Detection and it’s real-life use cases?
Object detection in computer vision refers to the ability of machines to pinpoint the location of an object in an image or video. A lot of companies have been using object detection techniques in their system. They use it for face detection, web images, and security purposes.




Q39 Describe Optical Flow, its uses, and assumptions.
Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. It is a 2D vector field where each vector is a displacement vector showing the movement of points from the first frame to the second
Optical flow has many applications in areas like :
●	Structure from Motion
●	Video Compression
●	Video Stabilization …
Optical flow works on several assumptions:
1.	The pixel intensities of an object do not change between consecutive frames.
2.	Neighboring pixels have similar motion.

Q40 What is Histogram of Oriented Gradients (HOG)?
HOG stands for Histograms of Oriented Gradients. HOG is a type of “feature descriptor”. The intent of a feature descriptor is to generalize the object in such a way that the same object (in this case a person) produces as close as possible to the same feature descriptor when viewed under different conditions. This makes the classification task easier.
 
Q41 What is BOV: Bag-of-visual-words (BOV)?
BOV also called the bag of keypoints, is based on vector quantization. Similar to HOG features, BOV features are histograms that count the number of occurrences of certain patterns within a patch of the image.

Q42 What is Poselets? Where are poselets used?
Poselets rely on manually added extra keypoints such as “right shoulder”, “left shoulder”, “right knee” and “left knee”. They were originally used for human pose estimation

Q43 Explain Textons in context of CNNs
A texton is the minimal building block of vision. The computer vision literature does not give a strict definition for textons, but edge detectors could be one example. One might argue that deep learning techniques with Convolution Neuronal Networks (CNNs) learn textons in the first filters.

Q44 What is Markov Random Fields (MRFs)?
MRFs are undirected probabilistic graphical models which are a wide-spread model in computer vision. The overall idea of MRFs is to assign a random variable for each feature and a random variable for each pixel


Q45 Explain the concept of superpixel?
A superpixel is an image patch that is better aligned with intensity edges than a rectangular patch. Superpixels can be extracted with any segmentation algorithm, however, most of them produce highly irregular superpixels, with widely varying sizes and shapes. A more regular space tessellation may be desired.

Q46 What is Non-maximum suppression(NMS) and where is it used?
NMS is often used along with edge detection algorithms. The image is scanned along the image gradient direction, and if pixels are not part of the local maxima they are set to zero. It is widely used in object detection algorithms.

Q47 Describe the use of Computer Vision in Healthcare.
Computer vision has also been an important part of advances in health-tech. Computer vision algorithms can help automate tasks such as detecting cancerous moles in skin images or finding symptoms in x-ray and MRI scans.

Q48 Describe the use of Computer Vision in Augmented Reality & Mixed Reality Computer vision also plays an important role in augmented and mixed reality, the technology that enables computing devices such as smartphones, tablets, and smart glasses to overlay and embed virtual objects on real-world imagery. Using computer vision, AR gear detects objects in the real world in order to determine the locations on a device’s display to place a virtual object. For instance, computer vision algorithms can help AR applications detect planes such as
 
tabletops, walls, and floors, a very important part of establishing depth and dimensions and placing virtual objects in the physical world.

Q49 Describe the use of Computer Vision in Facial Recognition
Computer vision also plays an important role in facial recognition applications, the technology that enables computers to match images of people’s faces to their identities. Computer vision algorithms detect facial features in images and compare them with databases of face profiles. Consumer devices use facial recognition to authenticate the identities of their owners. Social media apps use facial recognition to detect and tag users. Law enforcement agencies also rely on facial recognition technology to identify criminals in video feeds.

Q50 Describe the use of Computer Vision in Self-Driving Cars
Computer vision enables self-driving cars to make sense of their surroundings. Cameras capture video from different angles around the car and feed it to computer vision software, which then processes the images in real-time to find the extremities of roads, read traffic signs, detect other cars, objects, and pedestrians. The self-driving car can then steer its way on streets and highways, avoid hitting obstacles, and (hopefully) safely drive its passengers to their destination.


Q51 Explain famous Computer Vision tasks using a single image example.
Many popular computer vision applications involve trying to recognize things in photographs; for example:

Object Classification: What broad category of object is in this photograph? Object Identification: Which type of a given object is in this photograph?
Object Verification: Is the object in the photograph?
Object Detection: Where are the objects in the photograph?
Object Landmark Detection: What are the key points for the object in the photograph? Object Segmentation: What pixels belong to the object in the image?
Object Recognition: What objects are in this photograph and where are they?


Q52 Explain the distinction between Computer Vision and Image Processing.
Computer vision is distinct from image processing.
Image processing is the process of creating a new image from an existing image, typically simplifying or enhancing the content in some way. It is a type of digital signal processing and is not concerned with understanding the content of an image.
A given computer vision system may require image processing to be applied to raw input, e.g. pre-processing images.
Examples of image processing include:
●	Normalizing photometric properties of the image, such as brightness or color.
●	Cropping the bounds of the image, such as centering an object in a photograph.
●	Removing digital noise from an image, such as digital artifacts from low light levels.
 
Q53 Explain business use cases in computer vision.
●	Optical character recognition (OCR)
●	Machine inspection
●	Retail (e.g. automated checkouts)
●	3D model building (photogrammetry)
●	Medical imaging
●	Automotive safety
●	Match move (e.g. merging CGI with live actors in movies)
●	Motion capture (mocap)
●	Surveillance
●	Fingerprint recognition and biometrics




Q54 What is the Boltzmann Machine?
One of the most basic Deep Learning models is a Boltzmann Machine, resembling a simplified version of the Multi-Layer Perceptron. This model features a visible input layer and a hidden layer
-- just a two-layer neural net that makes stochastic decisions as to whether a neuron should be on or off. Nodes are connected across layers, but no two nodes of the same layer are connected.

Q56 What Is the Role of Activation Functions in a Neural Network?
At the most basic level, an activation function decides whether a neuron should be fired or not. It accepts the weighted sum of the inputs and bias as input to any activation function. Step function, Sigmoid, ReLU, Tanh, and Softmax are examples of activation functions.

Q57 What Is the Difference Between a Feedforward Neural Network and Recurrent Neural Network?
A Feedforward Neural Network signals travel in one direction from input to output. There are no feedback loops; the network considers only the current input. It cannot memorize previous inputs (e.g., CNN).

Q58 What Are the Applications of a Recurrent Neural Network (RNN)?
The RNN can be used for sentiment analysis, text mining, and image captioning. Recurrent Neural Networks can also address time series problems such as predicting the prices of stocks in a month or quarter.

Q59 What Are the Softmax and ReLU Functions?
Softmax is an activation function that generates the output between zero and one. It divides each output, such that the total sum of the outputs is equal to one. Softmax is often used for output layers.
 
Q60 What Are Hyperparameters?
With neural networks, you’re usually working with hyperparameters once the data is formatted correctly. A hyperparameter is a parameter whose value is set before the learning process begins. It determines how a network is trained and the structure of the network (such as the number of hidden units, the learning rate, epochs, etc.).





Q61 What Will Happen If the Learning Rate Is Set Too Low or Too High?
When your learning rate is too low, training of the model will progress very slowly as we are making minimal updates to the weights. It will take many updates before reaching the minimum point. If the learning rate is set too high, this causes undesirable divergent behavior to the loss function due to drastic updates in weights. It may fail to converge (model can give a good output) or even diverge (data is too chaotic for the network to train).

Q62 How Are Weights Initialized in a Network?
There are two methods here: we can either initialize the weights to zero or assign them randomly. Initializing all weights to 0: This makes your model similar to a linear model. All the neurons and every layer perform the same operation, giving the same output and making the deep net useless. Initializing all weights randomly: Here, the weights are assigned randomly by initializing them very close to 0. It gives better accuracy to the model since every neuron performs different computations. This is the most commonly used method.

Q63 What Are the Different Layers on CNN?
There are four layers in CNN:
1.	Convolutional Layer - the layer that performs a convolutional operation, creating several smaller picture windows to go over the data.
2.	ReLU Layer - it brings non-linearity to the network and converts all the negative pixels to zero. The output is a rectified feature map.
3.	Pooling Layer - pooling is a down-sampling operation that reduces the dimensionality of the feature map.
4.	Fully Connected Layer - this layer recognizes and classifies the objects in the image.

Q64 What is Pooling on CNN, and How Does It Work?
Pooling is used to reduce the spatial dimensions of a CNN. It performs down-sampling operations to reduce the dimensionality and creates a pooled feature map by sliding a filter matrix over the input matrix.

Q65 How Does an LSTM Network Work?
Long-Short-Term Memory (LSTM) is a special kind of recurrent neural network capable of learning long-term dependencies, remembering information for long periods as its default behavior. There are three steps in an LSTM network:
 
●	Step 1: The network decides what to forget and what to remember.
●	Step 2: It selectively updates cell state values.
●	Step 3: The network decides what part of the current state makes it to the output.



Q66 What Is the Difference Between Epoch, Batch, and Iteration in Deep Learning?
●	Epoch - Represents one iteration over the entire dataset (everything put into the training model).
●	Batch - Refers to when we cannot pass the entire dataset into the neural network at once, so we divide the dataset into several batches.
●	Iteration - if we have 10,000 images as data and a batch size of 200. then an epoch should run 50 iterations (10,000 divided by 50).

Q67 Why Is Tensorflow the Most Preferred Library in Deep Learning?
Tensorflow provides both C++ and Python APIs, making it easier to work on and has a faster compilation time compared to other Deep Learning libraries like Keras and Torch. Tensorflow supports both CPU and GPU computing devices.

Q68 What Do You Mean by Tensor in Tensorflow?
A tensor is a mathematical object represented as arrays of higher dimensions. These arrays of
data with different dimensions and ranks fed as input to the neural network are called “Tensors.”

Q69 Explain a Computational Graph.
Everything in TensorFlow is based on creating a computational graph. It has a network of nodes where each node operates, Nodes represent mathematical operations, and edges represent tensors. Since data flows in the form of a graph, it is also called a “DataFlow Graph.”

Q70 What Is an Auto-encoder?
This Neural Network has three layers in which the input neurons are equal to the output neurons. The network's target outside is the same as the input. It uses dimensionality reduction to restructure the input. It works by compressing the image input to a latent space representation then reconstructing the output from this representation.

Q71 Can we have the same bias for all neurons of a hidden layer?
Essentially, you can have a different bias value at each layer or at each neuron as well. However, it is best if we have a bias matrix for all the neurons in the hidden layers as well.
A point to note is that both these strategies would give you very different results.

Q72 In a neural network, what if all the weights are initialized with the same value? In simplest terms, if all the neurons have the same value of weights, each hidden unit will get exactly the same signal. While this might work during forward propagation, the derivative of the cost function during backward propagation would be the same every time.
 
In short, there is no learning happening by the network! What do you call the phenomenon of the model being unable to learn any patterns from the data? Yes, underfitting.
Therefore, if all weights have the same initial value, this would lead to underfitting.

Q73 What is the role of weights and bias in a neural network?
This is a question best explained with a real-life example. Consider that you want to go out today to play a cricket match with your friends. Now, a number of factors can affect your decision- making, like:
●	How many of your friends can make it to the game?
●	How much equipment can all of you bring?
●	What is the temperature outside?
And so on. These factors can change your decision greatly or not too much. For example, if it is raining outside, then you cannot go out to play at all. Or if you have only one bat, you can share it while playing as well. The magnitude by which these factors can affect the game is called the weight of that factor.
Factors like the weather or temperature might have a higher weight, and other factors like equipment would have a lower weight.

Q74 Why does a Convolutional Neural Network (CNN) work better with image data? The key to this question lies in the Convolution operation. Unlike humans, the machine sees the image as a matrix of pixel values. Instead of interpreting a shape like a petal or an ear, it just identifies curves and edges.
Thus, instead of looking at the entire image, it helps to just read the image in parts. Doing this for a 300 x 300-pixel image would mean dividing the matrix into smaller 3 x 3 matrices and dealing with them one by one. This is convolution.

Q75 Why do RNNs work better with text data?
The main component that differentiates Recurrent Neural Networks (RNN) from the other models is the addition of a loop at each node. This loop brings the recurrence mechanism in RNNs. In a basic Artificial Neural Network (ANN), each input is given the same weight and fed to the network at the same time. So, for a sentence like “I saw the movie and hated it”, it would be difficult to capture the information which associates “it” with the “movie”.

Q76 In a CNN, if the input size 5 X 5 and the filter size is 7 X 7, then what would be the size of the output?
This is a pretty intuitive answer. As we saw above, we perform the convolution on ‘x’ one step at a time, to the right, and in the end, we got Z with dimensions 2 X 2, for X with dimensions 3 X 3. Thus, to make the input size similar to the filter size, we make use of padding – adding 0s to the input matrix such that its new size becomes at least 7 X 7. Thus, the output size would be using the formula:
Dimension of image = (n, n) = 5 X 5 Dimension of filter = (f,f) = 7 X 7
Padding = 1 (adding 1 pixel with value 0 all around the edges) Dimension of output will be (n+2p-f+1) X (n+2p-f+1) = 1 X 1
 

Q77 What’s the difference between valid and same padding in a CNN?
This question has more chances of being a follow-up question to the previous one. Or if you have explained how you used CNNs in a computer vision task, the interviewer might ask this question along with the details of the padding parameters.
●	Valid Padding: When we do not use any padding. The resultant matrix after convolution will have dimensions (n – f + 1) X (n – f + 1)
●	Same padding: Adding padded elements all around the edges such that the output matrix will have the same dimensions as that of the input matrix

Q78 What are the applications of transfer learning in Deep Learning?
I am sure you would have a doubt as to why a relatively simple question was included in the Intermediate Level. The reason is the sheer volume of subsequent questions it can generate! The use of transfer learning has been one of the key milestones in deep learning. Training a large model on a huge dataset, and then using the final parameters on smaller simpler datasets has led to defining breakthroughs in the form of Pretrained Models. Be it Computer Vision or NLP, pretrained models have become the norm in research and in the industry.  Some popular examples include BERT, ResNet, GPT-2, VGG-16, etc, and many more.

Q79 Why is GRU faster as compared to LSTM?
As you can see, the LSTM model can become quite complex. In order to still retain the functionality of retaining information across time and yet not make a too complex model, we need GRUs. Basically, in GRUs, instead of having an additional Forget gate, we combine the input and Forget gates into a single Update Gate:

Q80 How is the transformer architecture better than RNN?
Advancements in deep learning have made it possible to solve many tasks in Natural Language Processing. Networks/Sequence models like RNNs, LSTMs, etc. are specifically used for this purpose – so as to capture all possible information from a given sentence, or a paragraph. However, sequential processing comes with its caveats:
●	It requires high processing power
●	It is difficult to execute in parallel because of its sequential nature

Q81 How Can We Scale GANs Beyond Image Synthesis?
Aside from applications like image-to-image translation and domain-adaptation most GAN successes have been in image synthesis. Attempts to use GANs beyond images have focused on three domains: Text, Structured Data and Audio



Q82 How Should we Evaluate GANs and When Should We Use Them?
When it comes to evaluating GANs, there are many proposals but little consensus. Suggestions include:
 
●	Inception Score and FID - Both these scores use a pre-trained image classifier and both have known issues. A common criticism is that these scores measure ‘sample quality’ and don’t really capture ‘sample diversity’.
●	MS-SSIM - propose using MS-SSIM to separately evaluate diversity, but this technique
has some issues and hasn’t really caught on.
●	AIS - propose putting a Gaussian observation model on the outputs of a GAN and using annealed importance sampling to estimate the log-likelihood under this model, but show that estimates computed this way are inaccurate in the case where the GAN generator is also a flow model The generator being a flow model allows for the computation of exact log-likelihoods in this case.
●	Geometry Score - suggest computing geometric properties of the generated data manifold and comparing those properties to the real data.
●	Precision and Recall - attempt to measure both the ‘precision’ and ‘recall’ of GANs.
●	Skill Rating - have shown that trained GAN discriminators can contain useful information with which evaluation can be performed.

Q83 What should we use GANs for?
If you want an actual density model, GANs probably isn’t the best choice. There is now good experimental evidence that GANs learn a ‘low support’ representation of the target dataset, which means there may be substantial parts of the test set to which a GAN (implicitly) assigns zero likelihood.

Q84 How should we evaluate GANs on these perceptual tasks?
Ideally, we would just use a human judge, but this is expensive. A cheap proxy is to see if a classifier can distinguish between real and fake examples. This is called a classifier two-sample test (C2STs). The main issue with C2STs is that if the Generator has even a minor defect that’s systematic across samples (e.g., ) this will dominate the evaluation.

Q85 Explain the problem of Vanishing Gradients in GANs
Research has suggested that if your discriminator is too good, then generator training can fail due to vanishing gradients. In effect, an optimal discriminator doesn't provide enough information for the generator to make progress.
Attempts to Remedy
●	Wasserstein loss: The Wasserstein loss is designed to prevent vanishing gradients even when you train the discriminator to optimality.
●	Modified minimax loss: The original GAN paper proposed a modification to minimax loss to deal with vanishing gradients.

Q86 What is Mode Collapse and why it is a big issue?
Usually, you want your GAN to produce a wide variety of outputs. You want, for example, a different face for every random input to your face generator.
However, if a generator produces an especially plausible output, the generator may learn to produce only that output. In fact, the generator is always trying to find the one output that seems most plausible to the discriminator.
 
If the generator starts producing the same output (or a small set of outputs) over and over again, the discriminator's best strategy is to learn to always reject that output. But if the next generation of discriminator gets stuck in a local minimum and doesn't find the best strategy, then it's too easy for the next generator iteration to find the most plausible output for the current discriminator.
Each iteration of generator over-optimizes for a particular discriminator and the discriminator never manages to learn its way out of the trap. As a result, the generators rotate through a small set of output types. This form of GAN failure is called mode collapse.

Q87 ExplainProgressive GANs
In a progressive GAN, the generator's first layers produce very low resolution images, and subsequent layers add details. This technique allows the GAN to train more quickly than comparable non-progressive GANs, and produces higher resolution images.

Q88 Explain Conditional GANs
Conditional GANs train on a labeled data set and let you specify the label for each generated instance. For example, an unconditional MNIST GAN would produce random digits, while a conditional MNIST GAN would let you specify which digit the GAN should generate.
Instead of modeling the joint probability P(X, Y), conditional GANs model the conditional probability P(X | Y).
For more information about conditional GANs, see Mirza et al, 2014.

Q89 Explain Image-to-Image Translation
Image-to-Image translation GANs take an image as input and map it to a generated output image with different properties. For example, we can take a mask image with blob of color in the shape of a car, and the GAN can fill in the shape with photorealistic car details.

Q90 Explain CycleGAN
CycleGANs learn to transform images from one set into images that could plausibly belong to another set. For example, a CycleGAN produced the righthand image below when given the lefthand image as input. It took an image of a horse and turned it into an image of a zebra.




Q91 What is Super-resolution?
Super-resolution GANs increase the resolution of images, adding detail where necessary to fill in blurry areas. For example, the blurry middle image below is a downsampled version of the original image on the left. Given the blurry image, a GAN produced the sharper image on the right:

Q92 Explain different problems in GANs
Many GAN models suffer the following major problems:
●	Non-convergence: the model parameters oscillate, destabilize and never converge,
●	Mode collapse: the generator collapses which produces limited varieties of samples,
 
●	Diminished gradient: the discriminator gets too successful that the generator gradient vanishes and learns nothing,
●	Unbalance between the generator and discriminator causing overfitting, &
●	Highly sensitive to the hyperparameter selections.

Q93 Describe Cost v.s. image quality in GANS?
In a discriminative model, the loss measures the accuracy of the prediction and we use it to monitor the progress of the training. However, the loss in GAN measures how well we are doing compared with our opponent. Often, the generator cost increases but the image quality is actually improving. We fall back to examine the generated images manually to verify the progress. This makes model comparison harder which leads to difficulties in picking the best model in a single run. It also complicates the tuning process.

Q94 Why Singular Value Decomposition (SVD) is used in Computer Vision?
The singular value decomposition is the most common and useful decomposition in computer vision. The goal of computer vision is to explain the three-dimensional world through two- dimensional pictures.

Q95 What Is Image Transform?
An image can be expanded in terms of a discrete set of basis arrays called basis images. Hence, these basis images can be generated by unitary matrices. An NxN image can be viewed as an N^2×1 vector. It provides a set of coordinates or basis vectors for vector space.

Q96 List The Hardware Oriented Color Models?
They are as follows.
–	RGB model
–	CMY model
–	YIQ model
–	HSI model



Q96 What Is The Need For Transform?
Answer: The need for transform is most of the signals or images are time-domain signal (ie) signals can be measured with a function of time. This representation is not always best. Any person of the mathematical transformations is applied to the signal or images to obtain further information from that signal. Particularly, for image processing.

Q97 What is FPN?
Feature Pyramid Network (FPN) is a feature extractor designed with a feature pyramid concept to improve accuracy and speed. Images are first to pass through the CNN pathway, yielding semantically rich final layers. Then to regain better resolution, it creates a top-down pathway by upsampling this feature map