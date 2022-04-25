---
sidebar_position: 2
sidebar_label: Scalability
---

CS75 Scalability lecture

[![video](https://img.youtube.com/vi/-W9F__D3oY4/hqdefault.jpg)](https://www.youtube.com/watch?v=-W9F__D3oY4)


## Horizontal Scaling

We are not hitting the ceiling, instaead we scale horizontally. 

:::info
Instead of few powerful machines, why not get many slower or cheaper machines
:::

But now we have many machines running, and how do we handle that. For that we have load balancing.

## Load Balancer

:::note
Its like a black box to manage the incoming traffic.
:::

The load balancer has its own ip address and decides which one of the server to send this packet based on different factors. One of them can be how busy the server is. Load balancer sends the request to the least busy server in favour of optimization.

The server than gets the packet and decides "Oh! they want this html fle" and so it responds to the load balancer and the load balancer responds to the client.

So as per this, every time a user (OS) makes a request, it gets ip address on one of the servers that returns back with a response. But sending these requests everytime we visit a link. It wastes a lot number of milliseconds. Here comes Caching.

## Caching


## Clones

Full article: [Le Cloud Blog](https://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones).


Request are distributed by the load balancer to the group of cluster of application/server. So every time an user makes a request, any one of these hidden servers might be the one to respond.

:::tip 1st Goldern rule
Every server contains exactly the same codebase and does not store any user-related data, like sessions or profile pictures, on local disc or memory. 
:::

But with so many server running there comes the problem of deployment. How do we manage the code changes on all these server? One way is to create an image file from on of these servers and use it as "super clone". So all other instances are based on this super clone and any changes made to this will reflect in the rest.

## Database

Full article: [Le Cloud Blog](https://www.lecloud.net/post/7994751381/scalability-for-dummies-part-2-database).

Either stick to relational database management system like MySQL or choose NoSQL like MongoDB. 

But requesting large volume of data will slow the process. So we need cache like Memcached or Redis.

## Caching

:::tip
File-based caching is not a good option for cloning and auto-scaling the servers.
:::

A cache is key-value store and it holds the data in RAM for faster request handling. When an application makes a request, it first checks if the data is present in the cache and if not, only then it moves to the data source.

**Patterns of caching data**

Read full article on [Le Cloud Blog](https://www.lecloud.net/post/9246290032/scalability-for-dummies-part-3-cache) .

1. Cached Database Queries - Query to the database is stored in the cache as a hash. So the next time user makes a request, it first check on the cache (hash).
    
2. Cached Objects - A class based caching where class/instance data retrieved from the database is stored in the cache. It makes removing objects easier and faster. 
    :::info Example
    - User sessions
    - Blogs
    - streams
    :::

## Asynchronism
https://www.lecloud.net/tagged/scalability/chrono

## Perfomance vs Scalability
If the system is slow for a single user then it has got some perfomance issue. But if the system is fast for a single user but slow under heady load then there is some kind of scalability issue.

## Latency
https://community.cadence.com/cadence_blogs_8/b/sd/posts/understanding-latency-vs-throughput

The time required to do something is latency whereas the number of actions done per unit time is throughput.

:::info
Latency is unit of time -- hours, minutes, seconds, etc. 

Throughput is uints/time
:::

:::tip
Aim for maximum throughput with acceptable Latency
:::

