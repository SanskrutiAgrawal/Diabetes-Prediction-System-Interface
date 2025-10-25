# Diabetes Prediction System

A machine learning-based system to predict the onset of diabetes at an early stage.

## The Problem

Diabetes is a major cause of disability and premature death worldwide, leading to complications affecting the kidneys, eyes, and heart. The increasing number of deaths annually highlights the urgent need for an effective diagnostic system.

## The Solution

This project presents an efficient medical decision support system for diabetes prediction. By leveraging medical knowledge and machine learning, this system aims to enhance decision-making, effectiveness, adaptability, and transparency. This can lead to a reduction in time, effort, and labor in healthcare services while improving the accuracy of diagnoses.

To ensure the highest accuracy, several machine learning models were implemented and analyzed:

*   **Logistic Regression:** 0.81 accuracy
*   **Support Vector Machine (SVM):** 0.80 accuracy
*   **K-Nearest Neighbors (KNN):** 0.88 accuracy
*   **Random Forest Classifier:** 0.95 accuracy
*   **Naïve Bayes Theorem:** 0.77 accuracy
*   **Gradient Boost Classifier:** 0.91 accuracy

Based on these results, the **Random Forest Classifier** was selected for this prediction system.

## Features

*   **Early-Stage Diabetes Prediction:** Allows users to perform a preliminary check for diabetes by inputting relevant medical parameters.
*   **High Accuracy:** Utilizes the Random Forest Classifier, which demonstrated the highest accuracy (0.95) among the tested models.
*   **User-Friendly Interface:** Features a Graphical User Interface (GUI) designed for ease of use.
*   **Efficient Decision-Making:** Aids in faster and more accurate health assessments.

## Understanding the Random Forest Model

A Random Forest is a supervised machine learning algorithm that is widely used for both classification and regression tasks. It is an ensemble learning method, meaning it combines multiple individual models to produce a more accurate and stable prediction.

### How it Works

The "forest" in Random Forest is composed of numerous individual decision trees. Here's a breakdown of the process:

1.  **Bootstrap Sampling:** The algorithm starts by creating multiple random subsets of the original training data. This is done "with replacement," meaning the same data point can be selected more than once for a given subset.
2.  **Building Decision Trees:** A decision tree is built for each of these subsets. However, there's an added layer of randomness: at each node of the tree, instead of considering all features to make a split, the algorithm only considers a random subset of the features.
3.  **Voting for a Prediction:** For a new input, each decision tree in the forest provides a prediction (a "vote"). The final prediction is determined by the majority vote for classification tasks or the average of the predictions for regression tasks.

This method of combining multiple, slightly different decision trees helps to reduce the risk of overfitting, which is a common issue with single decision trees. The averaging of uncorrelated trees lowers the overall variance and improves the accuracy of the final prediction.

### Why Random Forest for Diabetes Prediction?

The Random Forest model was an excellent choice for this diabetes prediction system for several reasons:

*   **Robustness:** They are less prone to overfitting compared to single decision trees and can handle noise and outliers in the data effectively.
*   **Feature Importance:** Random Forest can provide insights into which medical parameters are most influential in predicting diabetes.
*   **Handles Missing Values:** The algorithm can maintain good accuracy even when some data is missing.
*   **Versatility:** It's a versatile algorithm that can be applied to various classification and regression problems.

## Conclusion

This project successfully created a Diabetes Prediction System using Supervised Machine Learning. It provides a valuable tool for individuals to conduct an initial self-assessment for diabetes. The development process provided significant learning experiences in applying advanced technologies like machine learning algorithms and designing user-friendly interfaces for complex models.

---

### Libraries Used
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
!pip install gradio
import gradio as gr
