# ML_Kepler

### 
Exoplanet Classification Model

#### Overview
This script implements a machine learning model using Support Vector Classification (SVC) to classify candidate exoplanets from a provided dataset. The model uses GridSearchCV to find the optimal parameters for the SVC, aiming to achieve the highest accuracy in classifying potential exoplanets.

#### Requirements
- Python 3.6+
- scikit-learn
- numpy (if dataset preprocessing is necessary)

#### Setup
1. Ensure you have Python installed on your system.
2. Install the required Python packages using pip:
   ```sh
   pip install scikit-learn numpy
   ```

#### How to Run
1. Prepare your dataset with features (X) and targets (yy) representing candidate exoplanets.
2. Place the script in the same directory as your dataset.
3. Run the script using Python:
   ```sh
   python exoplanet_classification.py
   ```
   
#### Script Details
- The script starts by splitting the dataset into training and testing sets.
- It then defines a parameter grid for the SVC model with various values for 'C', 'gamma', and 'kernel'.
- GridSearchCV is applied to find the optimal parameters for the SVC model.
- The model is trained using the optimal parameters found.
- Finally, the script fits the model to the training data and can be used to predict new candidate exoplanets.

#### Note
- Ensure your dataset is preprocessed (if necessary) before running this script.
- Adjust the `param_grid` as needed to explore different parameters for the SVC model.