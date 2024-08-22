# README: Boston Housing Price Prediction using TensorFlow and Keras

## Overview

This project demonstrates the implementation of a neural network to predict housing prices in Boston using the Boston Housing dataset. The model is built using TensorFlow and Keras and aims to predict the median value of owner-occupied homes in Boston suburbs based on various features. The project involves data preprocessing, model training with k-fold cross-validation, and model evaluation.

## Dataset

The dataset used in this project is the Boston Housing dataset, which contains 506 samples with 13 features each. The features describe various attributes of the homes and their surroundings, such as crime rate, number of rooms, and proximity to employment centers. The target variable is the median value of owner-occupied homes in $1000s.

## Project Structure

1. **Data Loading**
   - The Boston Housing dataset is loaded using TensorFlow's `keras.datasets` module.
   ```python
   from tensorflow.keras.datasets import boston_housing
   (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
   ```

2. **Data Preprocessing**
   - The data is normalized by subtracting the mean and dividing by the standard deviation.
   ```python
   mean = train_data.mean(axis=0)
   train_data -= mean
   std = train_data.std(axis=0)
   train_data /= std
   test_data -= mean
   test_data /= std
   ```

3. **Model Building**
   - A simple feedforward neural network is built using Keras' Sequential API. The model consists of two hidden layers with 64 units each and ReLU activation functions, followed by a single output layer.
   ```python
   from tensorflow import keras
   from tensorflow.keras import layers

   def build_model():
       model = keras.Sequential([
           layers.Dense(64, activation="relu"),
           layers.Dense(64, activation="relu"),
           layers.Dense(1)
       ])
       model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
       return model
   ```

4. **Model Training with K-Fold Cross-Validation**
   - The model is trained using 4-fold cross-validation to ensure robust performance. For each fold, the model is trained for 100 epochs, and the mean absolute error (MAE) on the validation set is recorded.
   ```python
   k = 4
   num_val_samples = len(train_data) // k
   num_epochs = 100
   all_scores = []
   for i in range(k):
       # Process each fold
   ```

5. **Model Evaluation**
   - The model's performance is evaluated by calculating the mean of the validation MAEs across all folds. Additionally, the model is trained for 500 epochs, and the MAE history is averaged and plotted to observe the model's performance over time.
   ```python
   num_epochs = 500
   all_mae_histories = []
   # Training process with k-fold validation
   average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
   ```

6. **Model Performance Visualization**
   - The average MAE history is plotted to visualize how the model's validation MAE evolves over epochs.
   ```python
   import matplotlib.pyplot as plt
   plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
   plt.xlabel("Epochs")
   plt.ylabel("Validation MAE")
   plt.show()
   ```

7. **Final Model Training and Testing**
   - The model is retrained on the entire training set and then evaluated on the test set to determine the final MAE.
   ```python
   model = build_model()
   model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
   test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
   ```

8. **Making Predictions**
   - The trained model is used to make predictions on the test data.
   ```python
   predictions = model.predict(test_data)
   ```

## Dependencies

- Python 3.x
- TensorFlow
- Keras (included in TensorFlow)
- NumPy
- Matplotlib

## How to Run

1. Install the necessary dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
2. Run the script in a Python environment to train the model and evaluate its performance:
   ```bash
   python boston_housing_prediction.py
   ```

## Results

- The final model achieved a test MAE score, indicating the average prediction error in thousands of dollars.
- The MAE history plot provides insight into how the model's performance improved during training.

## Conclusion

This project showcases how to build and evaluate a neural network model for regression tasks using TensorFlow and Keras. The use of k-fold cross-validation ensures that the model is well-validated and generalizes well to unseen data.