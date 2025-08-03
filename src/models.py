"""
Naive Bayes model implementations for breast cancer prediction.
Includes both Gaussian and Categorical Naive Bayes approaches.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier implementation."""
    
    def __init__(self):
        """Initialize the Gaussian Naive Bayes classifier."""
        self.priors = None
        self.means = None
        self.stds = None
        self.classes = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GaussianNaiveBayes':
        """
        Fit the Gaussian Naive Bayes model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        self.classes = sorted(y.unique())
        
        # Calculate priors P(Y=y)
        self.priors = []
        for cls in self.classes:
            self.priors.append(len(y[y == cls]) / len(y))
        
        # Calculate means and standard deviations for each class and feature
        self.means = {}
        self.stds = {}
        
        for cls in self.classes:
            class_data = X[y == cls]
            self.means[cls] = class_data.mean().to_dict()
            self.stds[cls] = class_data.std().to_dict()
        
        logger.info(f"Fitted Gaussian Naive Bayes with {len(self.classes)} classes")
        return self
    
    def _calculate_likelihood(self, x: np.ndarray, cls: int) -> float:
        """
        Calculate likelihood P(X=x|Y=cls) using Gaussian distribution.
        
        Args:
            x: Feature values
            cls: Class label
            
        Returns:
            Likelihood value
        """
        likelihood = 1.0
        
        for i, feature in enumerate(self.feature_names):
            mean = self.means[cls][feature]
            std = self.stds[cls][feature]
            
            # Avoid division by zero
            if std == 0:
                std = 1e-10
                
            # Gaussian probability density function
            exponent = -((x[i] - mean) ** 2) / (2 * std ** 2)
            prob = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)
            likelihood *= prob
            
        return likelihood
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted class labels
        """
        predictions = []
        
        for _, row in X.iterrows():
            x = row.values
            
            # Calculate posterior probabilities for each class
            posteriors = []
            for i, cls in enumerate(self.classes):
                likelihood = self._calculate_likelihood(x, cls)
                posterior = likelihood * self.priors[i]
                posteriors.append(posterior)
            
            # Predict the class with highest posterior probability
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Class probabilities
        """
        probabilities = []
        
        for _, row in X.iterrows():
            x = row.values
            
            # Calculate posterior probabilities for each class
            posteriors = []
            for i, cls in enumerate(self.classes):
                likelihood = self._calculate_likelihood(x, cls)
                posterior = likelihood * self.priors[i]
                posteriors.append(posterior)
            
            # Normalize to get probabilities
            total = sum(posteriors)
            if total == 0:
                # If all posteriors are zero, assign equal probabilities
                probs = [1.0 / len(self.classes)] * len(self.classes)
            else:
                probs = [p / total for p in posteriors]
            
            probabilities.append(probs)
        
        return np.array(probabilities)


class CategoricalNaiveBayes:
    """Categorical Naive Bayes classifier implementation."""
    
    def __init__(self):
        """Initialize the Categorical Naive Bayes classifier."""
        self.priors = None
        self.likelihoods = None
        self.classes = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CategoricalNaiveBayes':
        """
        Fit the Categorical Naive Bayes model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        self.classes = sorted(y.unique())
        
        # Calculate priors P(Y=y)
        self.priors = []
        for cls in self.classes:
            self.priors.append(len(y[y == cls]) / len(y))
        
        # Calculate likelihoods P(X=x|Y=y) for each feature, value, and class
        self.likelihoods = {}
        
        for cls in self.classes:
            class_data = X[y == cls]
            class_likelihoods = {}
            
            for feature in self.feature_names:
                feature_likelihoods = {}
                value_counts = class_data[feature].value_counts()
                total_count = len(class_data)
                
                for value in X[feature].unique():
                    count = value_counts.get(value, 0)
                    # Add Laplace smoothing (add 1 to avoid zero probabilities)
                    prob = (count + 1) / (total_count + len(X[feature].unique()))
                    feature_likelihoods[value] = prob
                
                class_likelihoods[feature] = feature_likelihoods
            
            self.likelihoods[cls] = class_likelihoods
        
        logger.info(f"Fitted Categorical Naive Bayes with {len(self.classes)} classes")
        return self
    
    def _calculate_likelihood(self, x: np.ndarray, cls: int) -> float:
        """
        Calculate likelihood P(X=x|Y=cls) using categorical distribution.
        
        Args:
            x: Feature values
            cls: Class label
            
        Returns:
            Likelihood value
        """
        likelihood = 1.0
        
        for i, feature in enumerate(self.feature_names):
            value = x[i]
            feature_likelihoods = self.likelihoods[cls][feature]
            
            # Get probability for this feature value, default to small value if not seen
            prob = feature_likelihoods.get(value, 1e-10)
            likelihood *= prob
            
        return likelihood
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted class labels
        """
        predictions = []
        
        for _, row in X.iterrows():
            x = row.values
            
            # Calculate posterior probabilities for each class
            posteriors = []
            for i, cls in enumerate(self.classes):
                likelihood = self._calculate_likelihood(x, cls)
                posterior = likelihood * self.priors[i]
                posteriors.append(posterior)
            
            # Predict the class with highest posterior probability
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Class probabilities
        """
        probabilities = []
        
        for _, row in X.iterrows():
            x = row.values
            
            # Calculate posterior probabilities for each class
            posteriors = []
            for i, cls in enumerate(self.classes):
                likelihood = self._calculate_likelihood(x, cls)
                posterior = likelihood * self.priors[i]
                posteriors.append(posterior)
            
            # Normalize to get probabilities
            total = sum(posteriors)
            if total == 0:
                # If all posteriors are zero, assign equal probabilities
                probs = [1.0 / len(self.classes)] * len(self.classes)
            else:
                probs = [p / total for p in posteriors]
            
            probabilities.append(probs)
        
        return np.array(probabilities)


class ModelEvaluator:
    """Model evaluation utilities."""
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a model and return comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
        }
        
        # Log results
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        
        return metrics
    
    @staticmethod
    def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Model"):
        """
        Print a formatted evaluation summary.
        
        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
        """
        print(f"\n{'='*50}")
        print(f"{model_name} Evaluation Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(
            y_true=None, y_pred=None, 
            target_names=['Benign', 'Malignant'],
            digits=4
        ))


def train_and_evaluate_models(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate both Gaussian and Categorical Naive Bayes models.
    
    Args:
        train_data: Training data
        test_data: Test data
        
    Returns:
        Dictionary containing evaluation results for both models
    """
    # Prepare data
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    results = {}
    
    # Train and evaluate Gaussian Naive Bayes
    logger.info("Training Gaussian Naive Bayes...")
    gaussian_nb = GaussianNaiveBayes()
    gaussian_nb.fit(X_train, y_train)
    y_pred_gaussian = gaussian_nb.predict(X_test)
    
    results['gaussian'] = ModelEvaluator.evaluate_model(
        y_test, y_pred_gaussian, "Gaussian Naive Bayes"
    )
    
    # Train and evaluate Categorical Naive Bayes
    logger.info("Training Categorical Naive Bayes...")
    categorical_nb = CategoricalNaiveBayes()
    categorical_nb.fit(X_train, y_train)
    y_pred_categorical = categorical_nb.predict(X_test)
    
    results['categorical'] = ModelEvaluator.evaluate_model(
        y_test, y_pred_categorical, "Categorical Naive Bayes"
    )
    
    return results 