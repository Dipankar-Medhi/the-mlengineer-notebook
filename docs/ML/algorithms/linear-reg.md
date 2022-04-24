---
title: Linear Regression
---




To solve linear regression, we have to find the coefficients $\beta$ which minimize the sum of squared errors. 

Matrix Algebra method: Let’s say you have $x$, a matrix of features, and y, a vector with the values you want to predict. After going through the matrix algebra and minimization problem, you get this solution: $\beta = (X^TX)^{-1}X^T$

But solving this requires you to find an inverse, which can be time-consuming, if not impossible. Luckily, there are methods like **Singular Value Decomposition (SVD)** or QR Decomposition that can reliably calculate this part $\beta = (X^TX)^{-1}X^T$

(called the pseudo-inverse) without actually needing to find an inverse. The popular python ML library