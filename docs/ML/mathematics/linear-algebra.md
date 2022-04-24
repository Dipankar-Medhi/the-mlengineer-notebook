---
sidebar_position: 1
title: Linear Algebra
---

## Scalar
A **scalar** is a matrix with 1 row and 1 column. All the numbers we know from algebra are referred to as scalars in linear algebra.
	
	
    Example: [15], [-5], [pie]

- A scalar has no dimension i.e **0D**.
- A scalar can be represented as a point.

## Vector
A **vector** is a single dimensional matrix.

	Dimension of vector is m x 1
- A matrix is a collection of vectors.
-  Vector is one-dimensional i.e **1D**.
-  A vector can be represented as a line.


```
Example: [5, 2, 4] is a vector of length = 3
```

There are **row vectors** and **column vectors**.

	v = np.array([5,-2,4])
	v
	-> array([ 5, -2, 4])

## Matrix
Dimension of **matrix** is m x n, where **m** is no of rows and **n** is no columns.
-  Matrix is 2 dimensional i.e. 2D
-  A matrix can be represented as a plane.

```
	m = np.array([[5,12,6],[-3,0,14]])
	m
	-> array([[ 5, 12, 6], [-3, 0, 14]])
```

## Linear Algebra and Geometry

vector [1 0] and [0 1] represents the 2D plane or 2 axis.


##  Tensor
Tensors are simply a generalized concept of scalar(rank 0), vector(rank 1), matrix(rank 2), or a collection of matrix (rank 3) of dimension k x m x n. Just like we see in **Convulation Neural Networks**.



	t = np.array([[[5,12,6], [3,0,14]],[[9,8,7], [1,3,-5]]])
	t
	array([[[ 5, 12,  6],
        [ 3,  0, 14]],

       [[ 9,  8,  7],
        [ 1,  3, -5]]])

---


	
	

