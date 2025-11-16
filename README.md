# Predicting Road Accident Risk using Deep Regression on Synthetic Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is the official submission for the **YAP470 Deep Learning** course. It explores the use of deep regression models to predict road accident risk scores based on various environmental and temporal factors. Due to the sparsity of real-world accident data, this model is trained on a synthetically generated dataset.



## 1. 🚀 Project Overview

The primary goal of this project is to build and evaluate a deep neural network (DNN) for **regression**, capable of predicting a continuous risk score for road accidents. Instead of relying on imbalanced and often incomplete real-world data, we utilize a synthetic dataset (inspired by a Kaggle dataset) to train a more robust and generalized model.

This model aims to answer: *Given a set of conditions (like time, weather, road type), what is the "risk level" of an accident occurring?*

## 2. 📊 Dataset

The model was trained on a synthetic dataset generated to mimic real-world driving scenarios and accident triggers. The original inspiration and feature structure were derived from a public Kaggle dataset.

* **Original Data (Inspiration):** [Link to the Kaggle Dataset you used]
* **Data Generation:** We employed specific techniques *(Buraya kullandığın yöntemi ekle, örn: statistical methods or a generative model like a VAE)* to create a balanced and comprehensive training set that covers a wide range of scenarios.

## 3. 🧠 Methodology & Model Architecture

Our approach involved three main stages:
1.  **Data Preprocessing:** Cleaning, scaling (e.g., standardization), and encoding (e.g., one-hot) the input features.
2.  **Model Architecture:** A sequential Deep Neural Network (DNN) built with Keras/TensorFlow. The architecture consists of multiple dense layers with 'ReLU' activation and dropout layers to prevent overfitting. The final layer is a single neuron with a 'Linear' (veya 'Sigmoid' - *eğer risk 0-1 arasındaysa*) activation to output the regression value.
3.  **Training & Validation:** The model was trained using the Adam optimizer and 'Mean Squared Error' (MSE) as the loss function.



## 4. 📈 Key Visualizations & Results

The model's performance was evaluated based on its ability to predict risk scores accurately on a hold-out test set.

### Training & Validation Loss
This plot shows the 'Mean Squared Error' (MSE) decreasing over epochs, indicating that the model is learning successfully.
*(Kendi grafiğini buraya ekle: `![Training Loss](visuals/loss_curve.png)`)*



### Feature Importance
An analysis (e.g., SHAP or permutation importance) showing which features (e.g., 'weather_condition', 'time_of_day') most significantly contributed to the accident risk prediction.
*(Kendi grafiğini buraya ekle: `![Feature Importance](visuals/feature_importance.png)`)*



### Predicted vs. Actual Risk
A scatter plot comparing the model's predictions against the true risk scores on the test set. A strong correlation (points clustering around the y=x line) indicates high accuracy.
*(Kendi grafiğini buraya ekle: `![Prediction Plot](visuals/predictions.png)`)*



## 5. 🛠️ Technologies Used

* **Python 3.9+**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn (sklearn):** For data preprocessing (like `StandardScaler`) and evaluation metrics (like `r2_score`, `mean_squared_error`).
* **TensorFlow / Keras:** For building and training the deep regression model.
* **Matplotlib / Seaborn:** For data visualization.
* **Jupyter Notebook:** For experimentation and analysis.

## 6. 📂 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/esmanurulu/Predicting-Road-Accident-Risk.git](https://github.com/esmanurulu/Predicting-Road-Accident-Risk.git)
    cd Predicting-Road-Accident-Risk
    ```
2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  Run the main training script or open the Jupyter Notebook:
    ```bash
    python train.py
    ```
    *(veya `jupyter notebook main_analysis.ipynb`)*

## 7. 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
