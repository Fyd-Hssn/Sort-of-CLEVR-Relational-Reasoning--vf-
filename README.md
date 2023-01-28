Artifical Intelligence CS-GY 6613 Final Project - Faiyaad Hossain (FH818) & Harsh Sonthalia (HS4226)

Pytorch implementation of Relational Networks - [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

Implemented & tested on Sort-of-CLEVR task.

## Sort-of-CLEVR

To replicate the results of the paper, we tuned the learning rate to 0.0002 (original implementation had 0.0001). We found this rate to produce the most accurate results, with ~92% accuracy for binary relational questions and ~99% for non-relational questions. 

However, given the inherent random nature present in the computations, we found some stochasticity with the results ranging from 88%-92% accuracy for binary relational questions. Still, we found this performance to be strong given that it demonstrates the RN's capabilities and that it outperformed the results of the original author of the repository. 

## State Description

We have implemented the state description dataset generator which composes the dataset using a python dictionary instead of using a numpy array representation of the image. The state was persisted in a Python dictionary ({}) and was dumped into the pickle file as such. We tweakd the model by first removing the CNN from the RN as the state descriptors already have the individual objects in factored representation form and hence the features of the image would not need to be convolved through the CNN. Therefore, the object features can be extracted from the state and fed directly into the gθ-MLP layers for object pair/question assessment to compute a relational score. The rest of the model would remain the same as the original.

## Miscellaneous

We have also modified the repo to run on JAX instead of numpy.