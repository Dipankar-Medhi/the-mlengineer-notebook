---
sidebar_label: Transfer Learning 
title: Transfer Learning 
sidebar_position: 4 
---

Generally a machine learning model is trained to work on a specific task. Let's take the example of twitter sentiment analysis. The model is trained for this particular task and is optimized for performing well on this task alone. But there is something called transfer learning, which enable models, with proper fine-tuning, to work on different tasks.

## Requirements for transfer learning
1. Abundant data for pre-training
2. Fine-tuning should be done with similar data that is used for pre-training.

## Approach of transfer learning

:::note Source
https://machinelearningmastery.com/transfer-learning-for-deep-learning/
:::

### Develop Model Approach
- Select Source Task. You must select a related predictive modeling problem with an abundance of data where there is some relationship in the input data, output data, and/or concepts learned during the mapping from input to output data.
- Develop Source Model. Next, you must develop a skillful model for this first task. The model must be better than a naive model to ensure that some feature learning has been performed.
- Reuse Model. The model fit on the source task can then be used as the starting point for a model on the second task of interest. This may involve using all or parts of the model, depending on the modeling technique used.
- Tune Model. Optionally, the model may need to be adapted or refined on the input-output pair data available for the task of interest.
### Pre-trained Model Approach
- Select Source Model. A pre-trained source model is chosen from available models. Many research institutions release models on large and challenging datasets that may be included in the pool of candidate models from which to choose from.
- Reuse Model. The model pre-trained model can then be used as the starting point for a model on the second task of interest. This may involve using all or parts of the model, depending on the modeling technique used.
- Tune Model. Optionally, the model may need to be adapted or refined on the input-output pair data available for the task of interest.

## Pre-trained Models

-   [Oxford VGG Model](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
-   [Google Inception Model](https://github.com/tensorflow/models/tree/master/research/inception)
-   [Microsoft ResNet Model](https://github.com/KaimingHe/deep-residual-networks)
-   [Google’s word2vec Model](https://code.google.com/archive/p/word2vec/)
-   [Stanford’s GloVe Model](https://nlp.stanford.edu/projects/glove/)
-   [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)








