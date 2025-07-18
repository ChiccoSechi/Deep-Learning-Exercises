# Deep Learning Exercises ![License](https://img.shields.io/badge/license-MIT-green.svg)

## Index
- [Introduction](#Introduction)
- [Folders](#Folders)
- [Installation](#installation---jupyter-notebook)  

### Introduction
A series of exercises are developed to compare and implement various Deep Learning models.

There are three different folders containing three different dataset and tyopologies of Deep Learning approaches.

### Folders

**Fashion MNIST**
The first folder `Fashion_Mnist_Vision_Learning` contains a Jupyter Notebook where a step-by-step tuning of a Dense Neural Network and a simple Convolutional Neural Network are implemented for a simple comparison.

In the `Tuning` sub-folder there are all the tuning step saved.

The Dataset used is [Fashion MNIST from Zalando Research GitHub](https://github.com/zalandoresearch/fashion-mnist).

**WISDM AR**
The second folder `WISDM_ar_Temporal_Learning` contains a Jupyter Notebook with a hyperpameter tuning and a comparison of different types of Recurrent Neural Networks (LSTM, GRU, Bidirectional LSTM and Bidirectional GRU).

In the `Tuning` sub-folder there are all the tuning step saved.

The Dataset used is [Wireless Sensor Data Mining](https://www.cis.fordham.edu/wisdm/dataset.php). The license is avaliable in the Dataset folder: [readme.txt](/WISDM_ar_Temporal_Learning/WISDM_ar_v1.1/readme.txt).

The dataset is cleaned and saved as WISDM_ar_v1.1_cleaned.

**Yelp Sentiment Analysis**
The third and last folder `Yelp_Sentiment_Analysis` contains a Jupyter Notebook with a comparison of different Natural Language Processing Models for Text Classification (Binary Bag of Words, Frequency bag of Words, TF-IDF, Word Embedding).

The Dataset used is [Yelp review from Hugging Face](https://huggingface.co/datasets/Yelp/yelp_review_full). In one case of Word Embedding, [(GloVe 6B 50d)](https://nlp.stanford.edu/projects/glove/). is used as pre-computed word embedder.

### Installation - Jupyter Notebook
This guide will help you set up the environment for the project.
Python, Git and Jupyter Notebook are fundamental prerequisites.
First of all it is necessary to clone the Repository and move to the project directory:

```bash
git clone https://github.com/ChiccoSechi/Deep-Learning-Exercises.git
cd Deep-Learning-Exercises
```

Then, install the project dependencies listed in the `requirements.txt` file using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- scikit-learn
- matplotlib
- keras-tuner
- tensorflow
- datasets
- pandas
- numpy

Finally, open and run the project notebook:

```bash
jupyter notebook
```

This will start the Jupyter server and open a browser window. Navigate to the project notebook file (`.ipynb`) and open it.

Or use any other methods like VSCode, Google Colab, etc.