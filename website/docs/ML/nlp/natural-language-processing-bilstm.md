---
sidebar_label: Natural Language Processing with BiLSTMs
title: Natural Language Processing with BiLSTMs
sidebar_position: 2
---

A common buidling block of both sentiment analysis and NER (Named Entity Recognition) models is Bi-directional RNN models. 

"Jeff Bezoz is the CEO of Amazon" is sentence where the relation between "Jeff Bezoz" and "Amazon" is "CEO". This is an example of relation extraction and NER is used to identify this in the sentence.

## Bi - directional LSTMs or BiLSTMs
An RNN uses previously generated output and the current item to generate the next ouput.

Mathematically, 

$ft(x_t) = f(f_{t-1}(x_{t-1}, x_t;\theta))$

This equation says to compute output at time t, and output at t - 1 is used as input along with input data $x_t$ at the same time step. Along with this, learned weights, $\theta$ is used in computing the output.

In regular LSTM network, tokens or words are fed in one direction. In English sentences, token are fed to LSTM unit from left to right.

However in BiLSTMs, the token are fed from both the ends (left and right). So, A BiLSTM can learn from tokens from the past and the future.

This allows the model to capture more dependencies between words and structure of the sentence and improve the accuracy.

:::note
In speech recognition, only sound spoken so far is available and in real time-series analytics, only data from the past is available. Here BiLSTM cannot be used.
:::




