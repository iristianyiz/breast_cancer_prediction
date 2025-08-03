"""
Unit tests for the preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing import DataPreprocessor, load_and_prepare_data


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'mean_radius': np.random.normal(15, 3, n_samples),
            'mean_texture': np.random.normal(20, 4, n_samples),
            'mean_smoothness': np.random.normal(0.1, 0.02, n_samples),
            'diagnosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        })
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        self.preprocessor = DataPreprocessor(self.temp_file.name)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        """Test data loading functionality."""
        data = self.preprocessor.load_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, self.sample_data.shape)
        self.assertTrue(all(col in data.columns for col in self.sample_data.columns))
    
    def test_get_basic_info(self):
        """Test basic info retrieval."""
        info = self.preprocessor.get_basic_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('shape', info)
        self.assertIn('columns', info)
        self.assertIn('target_distribution', info)
        self.assertEqual(info['shape'], self.sample_data.shape)
    
    def test_select_features(self):
        """Test feature selection."""
        data = self.preprocessor.load_data()
        selected_data = self.preprocessor.select_features()
        
        expected_features = ['mean_radius', 'mean_texture', 'mean_smoothness', 'diagnosis']
        self.assertEqual(list(selected_data.columns), expected_features)
        self.assertEqual(selected_data.shape[0], data.shape[0])
    
    def test_create_categorical_features(self):
        """Test categorical feature creation."""
        data = self.preprocessor.load_data()
        selected_data = self.preprocessor.select_features()
        categorical_data = self.preprocessor.create_categorical_features(selected_data)
        
        # Check that original features are replaced with categorical ones
        self.assertNotIn('mean_radius', categorical_data.columns)
        self.assertIn('cat_mean_radius', categorical_data.columns)
        self.assertEqual(categorical_data.shape[0], selected_data.shape[0])
    
    def test_prepare_data_for_modeling(self):
        """Test data preparation for modeling."""
        train_data, test_data = self.preprocessor.prepare_data_for_modeling()
        
        # Check that data is split
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)
        
        # Check that both datasets have the same columns
        self.assertEqual(list(train_data.columns), list(test_data.columns))
        
        # Check that target column is present
        self.assertIn('diagnosis', train_data.columns)
        self.assertIn('diagnosis', test_data.columns)
    
    def test_prepare_data_for_modeling_categorical(self):
        """Test categorical data preparation."""
        train_data, test_data = self.preprocessor.prepare_data_for_modeling(categorical=True)
        
        # Check that categorical features are created
        self.assertIn('cat_mean_radius', train_data.columns)
        self.assertIn('cat_mean_texture', train_data.columns)
        self.assertIn('cat_mean_smoothness', train_data.columns)
        
        # Check that original features are not present
        self.assertNotIn('mean_radius', train_data.columns)
        self.assertNotIn('mean_texture', train_data.columns)
        self.assertNotIn('mean_smoothness', train_data.columns)
    
    def test_get_feature_statistics(self):
        """Test feature statistics calculation."""
        stats = self.preprocessor.get_feature_statistics()
        
        self.assertIsInstance(stats, dict)
        for feature in ['mean_radius', 'mean_texture', 'mean_smoothness']:
            self.assertIn(feature, stats)
            self.assertIn('mean', stats[feature])
            self.assertIn('std', stats[feature])
            self.assertIn('min', stats[feature])
            self.assertIn('max', stats[feature])


class TestLoadAndPrepareData(unittest.TestCase):
    """Test cases for load_and_prepare_data function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 50
        
        self.sample_data = pd.DataFrame({
            'mean_radius': np.random.normal(15, 3, n_samples),
            'mean_texture': np.random.normal(20, 4, n_samples),
            'mean_smoothness': np.random.normal(0.1, 0.02, n_samples),
            'diagnosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_and_prepare_data_continuous(self):
        """Test loading and preparing continuous data."""
        train_data, test_data = load_and_prepare_data(self.temp_file.name, categorical=False)
        
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertIn('mean_radius', train_data.columns)
        self.assertIn('diagnosis', train_data.columns)
    
    def test_load_and_prepare_data_categorical(self):
        """Test loading and preparing categorical data."""
        train_data, test_data = load_and_prepare_data(self.temp_file.name, categorical=True)
        
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertIn('cat_mean_radius', train_data.columns)
        self.assertIn('diagnosis', train_data.columns)


if __name__ == '__main__':
    unittest.main() 