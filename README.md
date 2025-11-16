# Predicting Road Accident Risk using Deep Regression on Synthetic Data

Esmanur Ulu - 231101024
Nehir Tıraş - 231101065
Zeynep Yetkin - 231101042


This project is the official submission for the **YAP470 Deep Learning** course. It explores the use of deep regression models to predict road accident risk scores based on various environmental and temporal factors. Due to the sparsity of real-world accident data, this model is trained on a synthetically generated dataset.

## 1. 🚀 Project Aim (Overview)

The primary goal of this project is to build and evaluate a deep neural network (DNN) for **regression**, capable of predicting a continuous risk score for road accidents.

The key challenges addressed are:
* **Data Scarcity:** Real-world accident data is often sparse, imbalanced, and incomplete.
* **Regression Task:** Instead of a simple classification (accident/no accident), we aim to predict a nuanced "risk level", which is a more complex regression problem.

To overcome these challenges, we utilize a synthetic dataset (inspired by a Kaggle dataset) to train a more robust and generalized model. This model aims to answer the question: *Given a set of conditions (like time, weather, road type), what is the "risk level" of an accident occurring?*

## 2. 📊 Dataset

The model was trained on a synthetic dataset generated to mimic real-world driving scenarios and accident triggers. The original inspiration and feature structure were derived from a public Kaggle dataset.

* **Original Data (Inspiration):** (https://www.kaggle.com/competitions/playground-series-s5e10/data)
* **Data Generation:** We employed statistical sampling techniques based on the original dataset's properties. By analyzing the distributions (e.g., mean, variance, and feature correlations) of the source Kaggle data, we generated a new, larger, and more balanced dataset that covers a wider range of scenarios without replicating the original data's sparsity.

## 3. 🧠 Methodology & Model Architecture

Our approach involved three main stages:
1.  **Data Preprocessing:** Cleaning, scaling (e.g., standardization), and encoding (e.g., one-hot) the input features.
2.  **Model Architecture:** A sequential Deep Neural Network (DNN) built with Keras/TensorFlow. The architecture consists of multiple dense layers with 'ReLU' activation and dropout layers to prevent overfitting. The final layer is a single neuron with a 'Linear' activation to output the regression value.
3.  **Training & Validation:** The model was trained using the Adam optimizer and 'Mean Squared Error' (MSE) as the loss function.

## 4. 🛠️ Technologies Used

* **Python 3.9+**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn (sklearn):** For data preprocessing (like `StandardScaler`) and evaluation metrics.
* **TensorFlow / Keras:** For building and training the deep regression model.
* **Jupyter Notebook:** For experimentation and analysis.

