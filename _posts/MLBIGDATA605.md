---
layout: post
title: 10-605 Deep Double Descent
tags: []
authors: Avadhani, Aashai, Carnegie Mellon University; Kim, Julia, Carnegie Mellon University
---

# This is a template file for a new blog post

As the world increasingly relies on more automated ways of improving processes using machine learning, we see more models rely on more data in order to capture various trends to create the most accurate predictions and classifications. Within applications like Natural Language Processing and Computer Vision, more complex models are required to understand intent, which results in large corpuses of texts being used coupled with very complex models to solve different tasks from language generation to image processing. However, with more data and complex deep learning models such as CNN’s and ResNets, we have to understand our limitations when it comes to the sample size we are using for training and the number of parameters we include within our model. Within this paper, we will cover the phenomenon of “Double Descent” which is a behavior that is noticed between the relationship between training error and model complexity in order to create the most optimized models for a wide variety of tough problems. 

This blog is based on the paper, “Deep Double Descent: Where Bigger Models and More Data Hurt,” which unravels the behavior between the complexity of models and increased training data samples. This could lead to a decrease in performance for models. The paper addresses different circumstances where model performance could either decrease or increase based on the complexity of the model through a new measurement called effective model complexity, which is defined as the maximum number of samples on which the model can achieve close to zero training error. Effective Model Complexity is defined mathematically below 

EMCD,(T) = max{n| ES~Dn[ErrorS(T(S))]  }

A common concept in statistics is that there is a bias-variance tradeoff: higher complexity models have lower bias and higher variance, while lower complexity models have higher bias and lower variance. When model complexity is low and the number of training samples is high, typically deep learning models exhibit the typical bias-variance tradeoff. This is called the under-parameterized regime. However, when model complexity is large compared to the number of samples, then increasing model complexity lowers training error. This phenomenon, first defined by Belkin et al. (2018), is called double descent. 


$ \sum_{i=0}^j \frac{1}{2^n} \times i $

$$\begin{equation}
a \times b \times c = 0 \\
j=1 \\
k=2 \\
\end{equation}$$

$$\begin{align}
i2 \times b \times c =0 \\
j=1 \\
k=2 \\
\end{align}$$

