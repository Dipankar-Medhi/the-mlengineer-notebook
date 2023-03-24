---
sidebar_label: Named Entitiy Recognition
title: Named Entity Recognition (NER)
sidebar_position: 3
---

The names of people, objects, products can be tagged in a peiece of text with NER, which is useful in chatbot applications and other information retrieval and extraction.

:::info Example: 

Sentence ->  Dipankar purchased a new Nvidia GFX card.

Tagged sentence -> $[Dipankar]_{PER}$ purchased a new $[Nvidia]_{ORG}$ GFX card.
:::

To build a NER model, BiLSTMs and CRFs are needed.