---
sidebar_label: Model Deployment
title: Model Deployment
sidebar_position: 4
---

It is the process of making our model available in production, where they can serve to other systems.

Deploying a model requires the coordination of data scientists, software developers, devops engineers and the business professinals.

## Machine Learning Pipelines

```yaml
Data 
    |
    ETL(Extract transform and load) 
        | 
        Data Preprocessing
            | 
            Feature selection 
                |
                Model building
                    |
                    Deployment
```

:::note
When we deploy the ML model in production, we also have to deploy the whole pipeline because the model will receive the raw data which has to be processed before feeding into the model for prediction.
:::

## Environment
It is the state of a machine where the machine learning model is developed or deployed for production.

In **Research environment**, data scientist develop the model and later is deployed in **Production environment**.

## Reproducible ML pipelines
The Research environment and the Production environment must produce the same output for some particular set of data.