---
title: Confidency Interval
---

A confidence interval is the range within which we expect the population parameter to be.


:::note Example
There is 95% confidence that price of pizza lie between 20$ to 25$. However, there is still 5% confidence that 5% price lie outside of this CI.
:::

## Confidence Level

`1 − α` 
where,α is the confidence level of the interval and 0 <= α <=1

- Common confidence levels = 90%, 95%, 99%
- *α* = 10%, 5%, 1% or 0.1, 0.05, 0.01

:::note Example
if confidence is 95% then the point is inside the interval and *α* is 5%.
:::

- is the mid point of the interval.

## Population variance known

Confidence Intervals: > [$\bar x - z_{\alpha/2}\frac{\alpha}{\sqrt{n}}$,$\bar x + z_{\alpha/2}\frac{\alpha}{\sqrt{n}}$] >z = z-score(having std normal distribution) of *α*/2 > $\frac{\alpha}{\sqrt{n}}$ is the [[Standard error]]

:::note Example
CI = 95% and *α* = 0.05 then *z*0.025 = we get from the z-score table.


we find the value of `(1-α)` in the table and add the corresponding row and column value. 

say, we get 1.9 and 0.06, so z0.025 = 1.9 + 0.06 = 1.96
:::

:::tip
z is also known as critical value.
:::

Putting the value of z in the CI formula we get the intervals.