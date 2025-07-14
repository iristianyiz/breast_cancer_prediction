#!/usr/bin/env python3
"""
Breast Cancer Prediction using Naive Bayes
This script implements both Gaussian and Categorical Naive Bayes approaches
"""

# Import all necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# Set seaborn style
sns.set_style("darkgrid")

def load_and_explore_data():
    """Load and explore the breast cancer dataset"""
    print("Loading dataset...")
    data = pd.read_csv("Breast_cancer_data.csv")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 10 rows:")
    print(data.head(10))
    
    print("\nDataset info:")
    print(data.info())
    
    print("\nBasic statistics:")
    print(data.describe())
    
    return data

def basic_eda(data):
    """Perform basic exploratory data analysis"""
    print("\n=== Basic EDA ===")
    
    # Distribution of diagnosis
    print("\nDiagnosis distribution:")
    print(data["diagnosis"].value_counts())
    
    # Plot diagnosis distribution
    plt.figure(figsize=(8, 6))
    data["diagnosis"].hist()
    plt.title("Distribution of Diagnosis")
    plt.xlabel("Diagnosis (0: Benign, 1: Malignant)")
    plt.ylabel("Count")
    plt.show()
    
    # Correlation matrix
    print("\nCalculating correlation matrix...")
    corr = data.iloc[:,:-1].corr(method="pearson")
    
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
    sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2, annot=True)
    plt.title("Feature Correlation Matrix")
    plt.show()
    
    # Feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.histplot(data, ax=axes[0], x="mean_radius", kde=True, color='r')
    axes[0].set_title("Mean Radius Distribution")
    
    sns.histplot(data, ax=axes[1], x="mean_smoothness", kde=True, color='b')
    axes[1].set_title("Mean Smoothness Distribution")
    
    sns.histplot(data, ax=axes[2], x="mean_texture", kde=True)
    axes[2].set_title("Mean Texture Distribution")
    
    plt.tight_layout()
    plt.show()

def prepare_data(data):
    """Prepare data for modeling by selecting relevant features"""
    print("\n=== Preparing Data ===")
    
    # Select relevant features
    data = data[["mean_radius", "mean_texture", "mean_smoothness", "diagnosis"]]
    print("Selected features:", list(data.columns[:-1]))
    print("Target variable:", data.columns[-1])
    
    print("\nFirst 10 rows of prepared data:")
    print(data.head(10))
    
    return data

def calculate_prior(df, Y):
    """Calculate P(Y=y) for all possible y"""
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    """Calculate P(X=x|Y=y) using Gaussian distribution"""
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val-mean)**2 / (2 * std**2)))
    return p_x_given_y

def naive_bayes_gaussian(df, X, Y):
    """Implement Gaussian Naive Bayes"""
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)

def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    """Calculate P(X=x|Y=y) categorically"""
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y

def naive_bayes_categorical(df, X, Y):
    """Implement Categorical Naive Bayes"""
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)

def prepare_categorical_data(data):
    """Convert continuous features to categorical features"""
    print("\n=== Preparing Categorical Data ===")
    
    # Create categorical features
    data["cat_mean_radius"] = pd.cut(data["mean_radius"].values, bins=3, labels=[0,1,2])
    data["cat_mean_texture"] = pd.cut(data["mean_texture"].values, bins=3, labels=[0,1,2])
    data["cat_mean_smoothness"] = pd.cut(data["mean_smoothness"].values, bins=3, labels=[0,1,2])

    # Drop original continuous features
    data = data.drop(columns=["mean_radius", "mean_texture", "mean_smoothness"])
    data = data[["cat_mean_radius", "cat_mean_texture", "cat_mean_smoothness", "diagnosis"]]
    
    print("Categorical features created:")
    print(data.head(10))
    
    return data

def evaluate_model(Y_test, Y_pred, model_name):
    """Evaluate the model and print results"""
    print(f"\n=== {model_name} Results ===")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))
    print(f"F1 Score: {f1_score(Y_test, Y_pred):.4f}")
    
    # Calculate accuracy
    accuracy = np.mean(Y_test == Y_pred)
    print(f"Accuracy: {accuracy:.4f}")

def main():
    """Main function to run the complete analysis"""
    print("Breast Cancer Prediction using Naive Bayes")
    print("=" * 50)
    
    # Load and explore data
    data = load_and_explore_data()
    
    # Basic EDA
    basic_eda(data)
    
    # Prepare data for Gaussian NB
    data_gaussian = prepare_data(data)
    
    # Test Gaussian model
    print("\n=== Testing Gaussian Naive Bayes ===")
    train, test = train_test_split(data_gaussian, test_size=.2, random_state=41)
    
    X_test = test.iloc[:,:-1].values
    Y_test = test.iloc[:,-1].values
    Y_pred_gaussian = naive_bayes_gaussian(train, X=X_test, Y="diagnosis")
    
    evaluate_model(Y_test, Y_pred_gaussian, "Gaussian Naive Bayes")
    
    # Prepare categorical data
    data_categorical = prepare_categorical_data(data_gaussian.copy())
    
    # Test Categorical model
    print("\n=== Testing Categorical Naive Bayes ===")
    train_cat, test_cat = train_test_split(data_categorical, test_size=.2, random_state=41)
    
    X_test_cat = test_cat.iloc[:,:-1].values
    Y_test_cat = test_cat.iloc[:,-1].values
    Y_pred_categorical = naive_bayes_categorical(train_cat, X=X_test_cat, Y="diagnosis")
    
    evaluate_model(Y_test_cat, Y_pred_categorical, "Categorical Naive Bayes")
    
    print("\n=== Summary ===")
    print("Both Gaussian and Categorical Naive Bayes models have been implemented and tested.")
    print("The Gaussian approach typically works better with continuous features,")
    print("while the categorical approach is useful when features are discretized.")

if __name__ == "__main__":
    main() 