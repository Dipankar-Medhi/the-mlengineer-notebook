---
slug: introducing-mlops-part1
title: What and Why do we need MLOps - Introducing MLOps part1
authors:
  name: Dipankar Medhi
  url: https://dipankarmedhi.hashnode.dev/
  image_url: https://dipankarmedhi.hashnode.dev/_next/image?url=https%3A%2F%2Fcdn.hashnode.com%2Fres%2Fhashnode%2Fimage%2Fupload%2Fv1646916330496%2FzzirE-6-6.png%3Fw%3D256%26h%3D256%26fit%3Dcrop%26crop%3Dentropy%26auto%3Dcompress%2Cformat%26format%3Dwebp&w=256&q=75
tags: [book, mlops]
---


A recap of Part 1 of the book "Introducing MLOps: How to Scale Machine Learning in the Enterprise by Mark Treveil and the Dataiku Team"


## What is MLOps?

MLOps stands for Machine Learning Operations. It is the standardization and streamlining of machine learning life cycle management.

<!--truncate-->


> MLOps is a core function of Machine Learning engineering, focused on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them. - Databricks

## Challenges faced in managing ML at scale

-   Data is continuously changing and Business keeps shifting with time.
-   Different people use different sets of tools in a production environment.
-   Managing models by data scientists due to sudden turnovers of staff is difficult.

**ML model life cycle inside an average organization**


```
"Data Engineers" : 
             "data acquisition"
             "data preparation" 
             "Prepare for production" : ["runtime environment", risk evaluation", QA]
"Data Scientists" : 
             "Feature engineering"
             "Model training/experimentation"
             "Model evaluation and comparison"
"DevOps engineer" : 
             "Development to production" : [ "Elastic scaling", "Containerization", "CI/CD"]
             "Monitoring and feedback loop" : ["Logging/alerting", "Input drift tracking", "Performance drift", "Online evaluation"]

```

MLOps is very similar to DevOps. Both deal with the end-to-end service life cycle of software and models. They prioritize the continuous delivery of high-quality results.

But the major difference is that DevOps deals with static software code whereas MLOps is based on continuously changing data that is fed to ML models and is constantly learning and adapting.

## Risks with ML models

-   Risk that it might be unavailable for a period of time.
-   Risk of bad prediction for a sample of data.
-   Risk of decrease in the accuracy of the models.
-   Risk of insufficient skills to maintain the ML model

ML models are only good if they are trained on good data. The training data must be a good reflection of data encountered in the production env.

ML model performance is sensitive to the production environment it is running in like the version of the software and the OS.

## Productionalization and deployment of ML models

Models can be deployed as REST API endpoints. Exporting models affects the functions of a model. So containerization is a very effective way of deploying ML models. Docker is a great example of containers.

**🚀Productionalization also includes CI/CD pipeline**.

-   Importing model
-   revalidating model accuracy
-   performing explainability checks
-   checking the quality of data artefacts
-   embedding more complex applications.

Once a model is deployed, it is important to monitor its performance.

MLOps isn't just important because it helps mitigate the risk of ML models, it is also an essential component to massively deploying ML models.

**🔥Good MLOps practices will help the team in**

-   version control, especially with experiments in the design phase.
-   comparing retrained models with previous versions.
-   ensuring model performance isn't degrading in production.

## Conclusion

MLOps can provide the transparency and accountability of model pipelines in production. It is a crucial part of transparent strategies for machine learning.

MLOps helps upper management and data scientists when machine learning models are deployed in production and the effect they are having on the business. And MLOps helps them understand the whole data pipeline behind a machine learning model in production.

----------

🌎Explore, 🎓Learn, 👷‍♂️Build. Happy Coding💛