---
sidebar_label: What is Kubernetes?
title: What is Kubernetes or K8s?
---

Kubernetes is a container orchastration tool. 

:::note
There are other orchestration tools too like Docker Swarm, HashiCorp Nomad, etc.
:::

> Kubernetes is a portable, extensible, open-source platform for managing containerized workloads and services, that facilitates both declarative configuration and automation. It has a large, rapidly growing ecosystem. Kubernetes services, support, and tools are widely available.


## Features Kubernetes offer
- Kubernetes can expose a container using the DNS name or using their own IP address and act as a load balancer.
- K8s allows automatic storage archestration.
- It can change the state of our containers like creating new containers, removing existing containers.
- K8s offers self-healing. It replaces a failed container and kills non-responding containers, etc.
- It lets us store and manage sensitive information like passwords, tokens, etc.

## Components of K8s

### Node
- **Control Plane** - This node makes all the global decisions like scheduling, detecting and responding to cluster events.
- **Worker Node** - This node runs the kubernetes workloads. It can be a VM and can host multiple pods. K8s nodes are managed by control plane.
- **kubelet** - It is like an agent that runs on each node and ensures that containers are running in a pod.
- **kube-proxy** - It runs on each node and maintains network rules on nodes.
- **Container runtime** - It is the software responsible for running the containers. Kubernetes supports several container runtimes: Docker, containerd, CRI-O, etc.
- ![image](https://github.com/MichaelCade/90DaysOfDevOps/raw/main/Days/Images/Day49_Kubernetes5.png)
  
### Cluster
:::tip Resource
https://github.com/MichaelCade/90DaysOfDevOps/blob/main/Days/day49.md#cluster
:::

Cluster is a group of nodes.

- **Kube API-Server** - It validates and configures data for the api objects which include pods, services, replicationcontrollers, and others. The API Server services REST operations and provides the frontend to the cluster's shared state through which all other components interact.
- **Scheduler** - It is control plane process that assigns Pods to nodes. The scheduler determines which Nodes are valid placements for each Pod in the scheduling queue according to constraints and available resources. The scheduler then ranks each valid Node and binds the Pod to a suitable Node.
- **Controller Manager** - It is a daemon that manages the core control loop of Kubernetes. In Kubernetes, a controller is a control loop that watches the shared state of the cluster through the apiserver and makes changes attempting to move the current state towards the desired state.
- **etcd** - It is a key-value store used as backup for cluster data.
- **kubectl** - Kubernetes CLI, kubectl manages all this processes.
![image](https://github.com/MichaelCade/90DaysOfDevOps/raw/main/Days/Images/Day49_Kubernetes7.png)

### Pods
A Pod is a group of containers that form an application. A Pod can also share common data volumes and they also share the same networking namespace.

Example: If you have a web application that is running a NodeJS container and also a MySQL container, then both these containers will be located in a single Pod.

- Pods could be brought up and down by the Master Controller.

- Kubernetes identify Pods based on labels (name - values)

- Pods handle Volumes, Secrets, and configuration for containers.​

- Pods are ephemeral. They are intended to be restarted automatically when they die.​

- Pods are replicated when the app is scaled horizontally by the ReplicationSet. Each Pod will run the same container code.​

- Pods live on Worker Nodes.​
![image](https://github.com/MichaelCade/90DaysOfDevOps/raw/main/Days/Images/Day49_Kubernetes8.png)

### Deployment
Deployment enables pod to run continuously. It allows to update a running app without downtime. ​​It also specify a strategy to restart Pods when they die.
![image](https://github.com/MichaelCade/90DaysOfDevOps/raw/main/Days/Images/Day49_Kubernetes9.png)

### ReplicaSets
- ReplicaSet ensures that our app has the desired number of Pods​.
- ReplicaSets will create and scale Pods based on the Deployment

:::tip
Deployment can also create the ReplicaSet
:::

### StatefulSets
- It maintains the state of the application.
  
### DaemonSets
- They are for continuous process.
- They run one pod per node.
  
### Services
- A single endpoint to access pods.
- Services help to bring up and down pods with ease.

:::tip Resources
- https://github.com/MichaelCade/90DaysOfDevOps/blob/main/Days/day49.md
- https://www.youtube.com/watch?v=X48VuDVv0do
- https://www.youtube.com/watch?v=KVBON1lA9N8

:::





