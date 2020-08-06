# Learning Chemistry With Transformers

This repository provides all the code I used for a project called "Learning Chemistry With Transformers", described in detail in a blog post I wrote [here](https://ytbai.net/2020/08/05/learning-chemistry-with-transformers/).

## Quick Guide

```data_factory/data/delaney.csv``` --- This is a dataset for the log solubility of 1144 different molecules provided by [Delaney's paper](https://pubs.acs.org/doi/10.1021/ci034243x). 

```data_factory/delaney_processed.csv``` and ```data_factory/excluded_molecules.csv``` --- These two files were taken from [Duvenaud's repository](https://github.com/HIPS/neural-fingerprint), which provide additional chemical properties for Delaney's molecules. 

```model_factory/models``` --- This directory provides all the models used in this project, including saved weights. In particular, __ChemTransformer__ is the main model for predicting log solubility. The other five models __MolWeightRegressor__, __NumHDonorRegressor__, __NumRingsRegressor__, __NumRotBondsRegressor__, and __PolarAreaRegressor__ are models for predicting solubility-related properties, as discussed in the blog post. 

```train.ipynb``` --- Notebook for training the main model

```train_ds.ipynb``` --- Notebook for training the other models

```test_ipynb``` --- Notebook for testing the main model

```test_ds.ipynb``` --- Notebook for testing the other models
