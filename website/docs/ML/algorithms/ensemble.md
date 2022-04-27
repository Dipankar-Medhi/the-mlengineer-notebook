---
sidebar_label: Ensemble Techniques
title: Ensemble Techniques
---

:::tip Sources
https://www.analyticsvidhya.com/blog/2021/03/basic-ensemble-technique-in-machine-learning/

https://medium.com/analytics-vidhya/ensemble-methods-in-machine-learning-31084c3740be
:::

It is a technique of combining individual predictive models to produce the final model.

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-10-14-30-00.png)
`source: Analyticsvidhya`

## Classification of Ensemble methods
1. Bagging 

    Models are generated with ramdom sub-samples of dataset with bootstrap sampling method to reduce variation.

    ![image](https://miro.medium.com/max/1400/1*G279X-1YwECFstqqY_nbIg.jpeg)
    `source: medium`

    Random forest is an example of bagging algorithm.

2. Boosting

    In boosting, weak learners/models that perform better in different versions are fitted in a sequence.

    ![image](https://miro.medium.com/max/1400/1*DKDI6EZJHkw2YgjLVDohiQ.png)
    `source: medium`

    Algorithms that are based on boosting are:
    - XG Boost
    - Ada Boost
    - Light GBM
    - Cat Boost
    - Stacking


