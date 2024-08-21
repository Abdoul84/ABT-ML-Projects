# Reuters News Classification with TensorFlow and Keras

This project demonstrates how to classify Reuters newswire topics using a neural network with TensorFlow and Keras. The dataset used is the Reuters newswire classification dataset, which is included in Keras.

## Project Overview

The code provided showcases the following steps:

1. **Loading the Dataset**: The Reuters newswire dataset is loaded with the top 10,000 most frequently occurring words.
2. **Exploring the Data**: Basic exploration of the dataset is performed, including checking the size of the training and test datasets and decoding the first newswire in the dataset.
3. **Data Preprocessing**: The data is prepared for the neural network by vectorizing the sequences and one-hot encoding the labels.
4. **Preparing Data for Training**: The sequences are vectorized to binary matrices, and the labels are converted to categorical format using one-hot encoding.

## Code Breakdown

### 1. Loading the Dataset

```python
from tensorflow.keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
```

- The dataset is loaded, restricting the vocabulary to the top 10,000 most common words.

### 2. Exploring the Dataset

```python
len(train_data)
len(test_data)
train_data[10]
```

- The code checks the size of the training and test datasets and displays an example of the encoded newswire.

### 3. Decoding the Newswire

```python
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
```

- The code decodes one of the newswires back into human-readable text using the word index provided by Keras.

### 4. Vectorizing the Data

```python
import numpy as np

def vectorize_sequences(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension))   
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.                        
    return results

x_train = vectorize_sequences(train_data)             
x_test = vectorize_sequences(test_data)
```

- The sequences (news articles) are vectorized into binary matrices where each entry represents the presence or absence of a word.

### 5. One-Hot Encoding the Labels

```python
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1. 
    return results

y_train = to_one_hot(train_labels)     
y_test = to_one_hot(test_labels) 
```

- Labels are converted into one-hot encoded vectors for classification into 46 categories.

### 6. Using Keras Utility for One-Hot Encoding

```python
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
```

- The `to_categorical` function from Keras is used to one-hot encode the labels.

## Requirements

- Python 3.x
- TensorFlow
- NumPy

## How to Run

1. Clone the repository.
2. Install the required packages using `pip install tensorflow numpy`.
3. Run the Python script to execute the code.

## Conclusion

This script sets up the data for training a neural network to classify Reuters newswire topics. The next steps would involve building a neural network model, training it on the prepared data, and evaluating its performance on the test set.