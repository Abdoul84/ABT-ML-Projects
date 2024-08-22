
# IMDB Sentiment Analysis with TensorFlow

This project demonstrates the implementation of a simple neural network using TensorFlow and Keras to perform sentiment analysis on the IMDB dataset. The IMDB dataset contains movie reviews along with their corresponding sentiment labels (positive or negative). The goal of the project is to build a model that can predict the sentiment of a movie review.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sentiment analysis is a common Natural Language Processing (NLP) task where the goal is to determine the sentiment expressed in a text. This project uses the IMDB dataset, which contains 50,000 movie reviews, to train a binary classification model to predict whether a review is positive or negative.

## Installation

To run this project, you need to have Python installed along with the following dependencies:

```bash
pip install tensorflow numpy matplotlib
```

You can also clone this repository and install the dependencies using:

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
```

## Dataset

The IMDB dataset is included with TensorFlow and can be easily loaded using:

```python
from tensorflow.keras.datasets import imdb
```

The dataset is preprocessed such that the words are encoded as integers, where each integer corresponds to a specific word in the dataset.

## Model Architecture

The model used in this project is a simple neural network with the following architecture:

- Input layer: The input data is a vector of integers representing the words in a review.
- Dense layer: 16 units with ReLU activation.
- Dense layer: 16 units with ReLU activation.
- Output layer: 1 unit with sigmoid activation to produce the probability of the review being positive.

The model is compiled using the `rmsprop` optimizer, `binary_crossentropy` loss function, and `accuracy` as a metric.

## Training

The model is trained on 40,000 reviews with a validation set of 10,000 reviews. The training process involves:

- Splitting the data into training and validation sets.
- Training the model for 20 epochs with a batch size of 512.

## Evaluation

After training, the model is evaluated on a test set of 10,000 reviews. The evaluation metrics include loss and accuracy.

## Results

The model achieves an accuracy of approximately 87.9% on the test set.

## Usage

To use the model for predictions, you can run the following:

```python
predictions = model.predict(x_test)
```

This will give you the probability of each review in the test set being positive.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.