"""
Data preprocessing module for breast cancer prediction.
Handles data loading, cleaning, feature engineering, and data preparation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations for breast cancer prediction."""
    
    def __init__(self, data_path: str = "Breast_cancer_data.csv"):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the breast cancer dataset
        """
        self.data_path = data_path
        self.data = None
        self.feature_columns = ["mean_radius", "mean_texture", "mean_smoothness"]
        self.target_column = "diagnosis"
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the breast cancer dataset.
        
        Returns:
            Loaded dataset as pandas DataFrame
        """
        try:
            logger.info(f"Loading dataset from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        if self.data is None:
            self.load_data()
            
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "target_distribution": self.data[self.target_column].value_counts().to_dict()
        }
        
        logger.info(f"Dataset info: {info['shape']} samples, {info['shape'][1]} features")
        return info
    
    def select_features(self, features: List[str] = None) -> pd.DataFrame:
        """
        Select relevant features for modeling.
        
        Args:
            features: List of feature columns to select. If None, uses default features.
            
        Returns:
            DataFrame with selected features and target
        """
        if self.data is None:
            self.load_data()
            
        if features is None:
            features = self.feature_columns
            
        selected_data = self.data[features + [self.target_column]].copy()
        logger.info(f"Selected features: {features}")
        logger.info(f"Selected data shape: {selected_data.shape}")
        
        return selected_data
    
    def create_categorical_features(self, data: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
        """
        Convert continuous features to categorical features.
        
        Args:
            data: Input DataFrame with continuous features
            n_bins: Number of bins for discretization
            
        Returns:
            DataFrame with categorical features
        """
        categorical_data = data.copy()
        
        for feature in self.feature_columns:
            if feature in categorical_data.columns:
                cat_col = f"cat_{feature}"
                categorical_data[cat_col] = pd.cut(
                    categorical_data[feature].values, 
                    bins=n_bins, 
                    labels=list(range(n_bins))
                )
                categorical_data = categorical_data.drop(columns=[feature])
        
        # Reorder columns to put target last
        feature_cols = [col for col in categorical_data.columns if col != self.target_column]
        categorical_data = categorical_data[feature_cols + [self.target_column]]
        
        logger.info(f"Created categorical features with {n_bins} bins each")
        return categorical_data
    
    def prepare_data_for_modeling(self, categorical: bool = False, 
                                 n_bins: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for modeling with train/test split.
        
        Args:
            categorical: Whether to use categorical features
            n_bins: Number of bins for discretization (if categorical=True)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        from sklearn.model_selection import train_test_split
        
        # Select features
        data = self.select_features()
        
        # Convert to categorical if requested
        if categorical:
            data = self.create_categorical_features(data, n_bins)
        
        # Split data
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data[self.target_column]
        )
        
        logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
        return train_data, test_data
    
    def get_feature_statistics(self, data: pd.DataFrame = None) -> dict:
        """
        Calculate basic statistics for features.
        
        Args:
            data: DataFrame to analyze. If None, uses selected features.
            
        Returns:
            Dictionary with feature statistics
        """
        if data is None:
            data = self.select_features()
            
        stats = {}
        for feature in self.feature_columns:
            if feature in data.columns:
                stats[feature] = {
                    "mean": data[feature].mean(),
                    "std": data[feature].std(),
                    "min": data[feature].min(),
                    "max": data[feature].max(),
                    "median": data[feature].median()
                }
        
        return stats


def load_and_prepare_data(data_path: str = "Breast_cancer_data.csv", 
                         categorical: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and prepare data.
    
    Args:
        data_path: Path to the dataset
        categorical: Whether to use categorical features
        
    Returns:
        Tuple of (train_data, test_data)
    """
    preprocessor = DataPreprocessor(data_path)
    return preprocessor.prepare_data_for_modeling(categorical=categorical) 