# Reuters Newswire Classification

This repository contains a script for training a neural network model using the Reuters Newswire dataset to classify news articles into 46 different topics. The model is implemented using TensorFlow and Keras.

## Dataset

The script uses the Reuters Newswire dataset, which is preloaded in TensorFlow. The dataset contains 11,228 newswires from Reuters, labeled over 46 topics.

## Script Overview

### 1. Data Loading
```python
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
```
- The dataset is loaded with the top 10,000 most frequently occurring words in the training data.

### 2. Data Exploration
```python
len(train_data)
len(test_data)
train_data[10]
```
- Basic exploration is performed to understand the data structure.

### 3. Decoding Newswires
```python
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
```
- Converts the integer sequences back into readable text.

### 4. Data Preprocessing
```python
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
```
- Converts the data into one-hot encoded vectors.

### 5. Label Encoding
```python
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
```
- The labels are converted to categorical one-hot encoded vectors.

### 6. Model Definition
```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])
```
- The model is a Sequential neural network with two hidden layers and a softmax output layer.

### 7. Model Compilation
```python
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
```
- The model is compiled with the RMSprop optimizer, categorical cross-entropy loss, and accuracy as a metric.

### 8. Model Training
```python
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```
- The model is trained on the training data with a validation split.

### 9. Model Evaluation and Plotting
```python
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
- The script includes code for evaluating the model's performance on the test set and plotting training and validation loss and accuracy over epochs.

### 10. Making Predictions
```python
predictions = model.predict(x_test)
np.argmax(predictions[0])
```
- The model is used to make predictions on the test set, and the most likely class for each sample is identified.

### 11. Alternative Model Configurations
- The script also explores different configurations of the neural network, including varying the number of hidden units and output layers.

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Running the Script

1. Ensure all dependencies are installed.
2. Run the script to train the model and evaluate its performance.

## Notes

- The script includes multiple versions of the model with different configurations. Adjust the model architecture as needed based on your specific requirements.

## Acknowledgments

- The dataset used in this script is provided by Reuters and is included in TensorFlow's Keras Datasets.