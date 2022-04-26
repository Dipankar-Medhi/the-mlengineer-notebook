---
sidebar_position: 1
---

# Azure end-to-end Machine Learning

The whole process is
- Data Engineering
- ML engineering
- Backend process (REST API, etc)
- Visualization (analysts)

```yaml
Raw data -> Azure function (runs every minute) -> |
Databricks (clean and preprocess data) -> |
Azure datalake(clean-processed data stored in ADL)-> |
Azure machine learning pipeline (fetch data from ADL) -> |
Output stored in Azure Blob Storage -> |
Power Bi -> Insights dashboard.

```

## Azure Function 
#### *creates trigger for pipeline*


- Azure functions checks if data has arrieved in the COSMOS (NoSQL in Azure).
- RUns every hour or minute.
- If yes, it returns **True** boolean and triggers the remaining processes.

## Azure Databricks
*preprocess data*
- Once the azure functions trigger the process, Databricks cluster starts the notebook and it imports the data.
- It performs cleaning, sorting and all other necessay preprocessing jobs.
- Then the processed data is put into ADLS (Azure Data Lake Storage).

## Building and Training the model
*using the stored data*
- We have two option either use Pthon or R SDK or use Azure ML Studio for a code free drag and drop process.
- The pipeline reads data from ADL storage account.
- Runs all the training and model preparation scripts.
- The output is stored in Azure Blob storage.

> Another alternative is to use MLFlow.

## Azure Blob Storage
- It is cloud service that helps to create data lakes.
- It provides storage to build powerful cloud native apps.
- They are SSD based storage and recommended for severless applications like Azure Functions.

## Azure Data Factory 
*alters, monitoring and anomaly detection*
- It allows to set up several alterts on the pipeline.
- It helps in keeping track of the data and avoid any data staleness.

