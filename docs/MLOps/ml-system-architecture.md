---
sidebar_label: ML System Architecture
title: Machine Learning System Architecture
---

Machine learning in production requires different compnents such as 
- infrastructure 
- data
- documentation
- environment configuration.

## Some best practices
- Proper model version control
- Effective reproducibility

## System Architecture approaches
:::**Formats** 
Serialization of model objects (pickle, PMML, ONNX, etc).
:::

- Embedded - Model packaged inside the application itself. 
  - Example: Tensorflow.js
- Model API - model as a service, deployed independently. Can be accessed using REST, gRPC, etc.
- Model published as data - Register new model with incomming data that can be accessed by application on run time.
  - Example: Kafka
- Offline Prediction - Offline predictions are stored in a DB that is accessed by an application from the database.

## High level layer
Evaluatino layer -> To evaluate the performance of the models in production.

Scoring layer -> This layer transform features into prediction. Eg: scikit-learn.

Feature layer -> Generate features.

Data layer -> Provides access to data.

:::tip Resources

Netflix's System Architectures for Personalization and Recommendation
https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8

:::