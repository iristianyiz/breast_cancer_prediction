"""
Unit tests for the models module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import GaussianNaiveBayes, CategoricalNaiveBayes, ModelEvaluator, train_and_evaluate_models


class TestGaussianNaiveBayes(unittest.TestCase):
    """Test cases for GaussianNaiveBayes class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic data with clear separation
        self.X_train = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.normal(0, 1, n_samples // 2),  # Class 0
                np.random.normal(3, 1, n_samples // 2)   # Class 1
            ]),
            'feature2': np.concatenate([
                np.random.normal(0, 1, n_samples // 2),  # Class 0
                np.random.normal(3, 1, n_samples // 2)   # Class 1
            ])
        })
        
        self.y_train = pd.Series([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Create test data
        self.X_test = pd.DataFrame({
            'feature1': [0.5, 3.5, 1.0, 2.5],
            'feature2': [0.5, 3.5, 1.0, 2.5]
        })
        
        self.y_test = np.array([0, 1, 0, 1])
    
    def test_fit(self):
        """Test model fitting."""
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        self.assertIsNotNone(model.priors)
        self.assertIsNotNone(model.means)
        self.assertIsNotNone(model.stds)
        self.assertIsNotNone(model.classes)
        self.assertIsNotNone(model.feature_names)
        
        # Check that priors sum to 1
        self.assertAlmostEqual(sum(model.priors), 1.0)
        
        # Check that we have means and stds for each class
        self.assertEqual(len(model.means), 2)
        self.assertEqual(len(model.stds), 2)
    
    def test_predict(self):
        """Test prediction functionality."""
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        probabilities = model.predict_proba(self.X_test)
        
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        
        # Check that probabilities sum to 1 for each sample
        for prob in probabilities:
            self.assertAlmostEqual(sum(prob), 1.0)
    
    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy on synthetic data."""
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        accuracy = np.mean(predictions == self.y_test)
        
        # Should achieve good accuracy on this synthetic data
        self.assertGreater(accuracy, 0.7)


class TestCategoricalNaiveBayes(unittest.TestCase):
    """Test cases for CategoricalNaiveBayes class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create categorical data
        self.X_train = pd.DataFrame({
            'feature1': np.random.choice([0, 1, 2], n_samples),
            'feature2': np.random.choice([0, 1, 2], n_samples)
        })
        
        # Create target with some correlation to features
        self.y_train = pd.Series([
            0 if (x1 == 0 and x2 == 0) else 1 
            for x1, x2 in zip(self.X_train['feature1'], self.X_train['feature2'])
        ])
        
        # Create test data
        self.X_test = pd.DataFrame({
            'feature1': [0, 1, 2, 0],
            'feature2': [0, 1, 2, 1]
        })
        
        self.y_test = np.array([0, 1, 1, 1])
    
    def test_fit(self):
        """Test model fitting."""
        model = CategoricalNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        self.assertIsNotNone(model.priors)
        self.assertIsNotNone(model.likelihoods)
        self.assertIsNotNone(model.classes)
        self.assertIsNotNone(model.feature_names)
        
        # Check that priors sum to 1
        self.assertAlmostEqual(sum(model.priors), 1.0)
        
        # Check that we have likelihoods for each class
        self.assertEqual(len(model.likelihoods), 2)
    
    def test_predict(self):
        """Test prediction functionality."""
        model = CategoricalNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        model = CategoricalNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        probabilities = model.predict_proba(self.X_test)
        
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        
        # Check that probabilities sum to 1 for each sample
        for prob in probabilities:
            self.assertAlmostEqual(sum(prob), 1.0)


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic predictions and true labels
        self.y_true = np.random.choice([0, 1], n_samples)
        self.y_pred = np.random.choice([0, 1], n_samples)
        
        # Create perfect predictions for testing
        self.y_true_perfect = np.array([0, 1, 0, 1, 0])
        self.y_pred_perfect = np.array([0, 1, 0, 1, 0])
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        metrics = ModelEvaluator.evaluate_model(
            self.y_true, self.y_pred, "Test Model"
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('specificity', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        
        # Check that metrics are in valid ranges
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['f1_score'], 0.0)
        self.assertLessEqual(metrics['f1_score'], 1.0)
    
    def test_evaluate_perfect_model(self):
        """Test evaluation of perfect predictions."""
        metrics = ModelEvaluator.evaluate_model(
            self.y_true_perfect, self.y_pred_perfect, "Perfect Model"
        )
        
        # Perfect predictions should have accuracy = 1.0
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
    
    def test_print_evaluation_summary(self):
        """Test evaluation summary printing."""
        metrics = ModelEvaluator.evaluate_model(
            self.y_true, self.y_pred, "Test Model"
        )
        
        # This should not raise an exception
        try:
            ModelEvaluator.print_evaluation_summary(metrics, "Test Model")
        except Exception as e:
            self.fail(f"print_evaluation_summary raised an exception: {e}")


class TestTrainAndEvaluateModels(unittest.TestCase):
    """Test cases for train_and_evaluate_models function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 50
        
        # Create synthetic training data
        self.train_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'diagnosis': np.random.choice([0, 1], n_samples)
        })
        
        # Create synthetic test data
        self.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20),
            'diagnosis': np.random.choice([0, 1], 20)
        })
    
    def test_train_and_evaluate_models(self):
        """Test training and evaluation of models."""
        results = train_and_evaluate_models(self.train_data, self.test_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('gaussian', results)
        self.assertIn('categorical', results)
        
        # Check that each model has evaluation metrics
        for model_name, model_results in results.items():
            self.assertIn('accuracy', model_results)
            self.assertIn('f1_score', model_results)
            self.assertIn('confusion_matrix', model_results)
            
            # Check that accuracy is in valid range
            self.assertGreaterEqual(model_results['accuracy'], 0.0)
            self.assertLessEqual(model_results['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main() 