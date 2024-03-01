### Voice Gender Recognition and Smartphone Activity Detector Scripts

#### Overview
These scripts are designed for two distinct purposes: **Voice Gender Recognition** and **Smartphone Activity Detection**. The Voice Gender Recognition script analyzes voice and speech data to classify voices as either male or female. The Smartphone Activity Detector uses smartphone sensor data to predict human activities like walking, sitting, or standing. Both utilize deep learning models for accurate classification based on the respective datasets' properties.

#### Datasets
- **Voice Gender Recognition**: Utilizes 3,168 voice samples, including acoustic properties such as mean frequency, spectral entropy, and fundamental frequency measurements.
- **Smartphone Activity Detector**: Employs the "Smartphone-Based Recognition of Human Activities and Postural Transitions" dataset, featuring accelerometer and gyroscope data from smartphones.

#### Requirements
- Python 3.6+
- Libraries: numpy, pandas, scikit-learn, TensorFlow (or Keras for backend)

#### Setup
Ensure Python 3.6+ is installed. Install required libraries using pip:
```sh
pip install numpy pandas scikit-learn tensorflow
```
For TensorFlow's Keras backend, replace `tensorflow` with `keras` in the command above.

#### Steps to Run the Scripts
1. **Data Pre-Processing**: Ready the datasets by processing the specific properties required for each model.
2. **Model Building**: Create and compile the deep learning models tailored for each task.
3. **Model Training and Evaluation**: Train the models, then evaluate their performance on test datasets to gauge accuracy.
4. **Prediction**: Utilize the trained models to make predictions on new data samples.

#### How to Run
- For **Voice Gender Recognition**: Save the script and dataset in the same directory and run `python voice_gender_recognition.py`.
- For **Smartphone Activity Detection**: Place the script and dataset in one folder and execute `python smartphone_activity_detector.py`.
Ensure dataset paths are correctly set in each script.

#### Note
These scripts showcase the application of deep learning in analyzing complex datasets for specific recognition tasks. Adjust model parameters and preprocessing steps as needed to optimize performance for your datasets.