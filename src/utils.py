"""
Utility functions for breast cancer prediction.
Includes data validation, model persistence, and other helper functions.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def validate_data(data: pd.DataFrame, required_columns: list, 
                 target_column: str = "diagnosis") -> bool:
    """
    Validate input data for modeling.
    
    Args:
        data: Input DataFrame
        required_columns: List of required feature columns
        target_column: Name of the target column
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check if all required columns exist
        missing_columns = set(required_columns + [target_column]) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        missing_values = data[required_columns + [target_column]].isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found missing values: {missing_values.to_dict()}")
        
        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                logger.error(f"Column {col} is not numeric")
                return False
        
        # Check target column values
        unique_targets = data[target_column].unique()
        if not all(val in [0, 1] for val in unique_targets):
            logger.error(f"Target column contains invalid values: {unique_targets}")
            return False
        
        logger.info("Data validation passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


def save_model(model: Any, filepath: str, model_info: Dict[str, Any] = None) -> bool:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        model_info: Additional model information to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model info if provided
        if model_info:
            info_path = filepath.replace('.pkl', '_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
        
        logger.info(f"Model saved successfully to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False


def load_model(filepath: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Tuple of (model, model_info)
    """
    try:
        # Load model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        # Try to load model info
        info_path = filepath.replace('.pkl', '_info.json')
        model_info = None
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        logger.info(f"Model loaded successfully from {filepath}")
        return model, model_info
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def calculate_feature_importance(model: Any, feature_names: list) -> Dict[str, float]:
    """
    Calculate feature importance for a model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        # For Naive Bayes, we can calculate importance based on likelihood differences
        if hasattr(model, 'means') and hasattr(model, 'stds'):
            # Gaussian Naive Bayes
            importance = {}
            for i, feature in enumerate(feature_names):
                # Calculate importance based on mean difference between classes
                means_diff = abs(model.means[1][feature] - model.means[0][feature])
                std_avg = (model.stds[1][feature] + model.stds[0][feature]) / 2
                importance[feature] = means_diff / std_avg if std_avg > 0 else 0
            
            return importance
        else:
            # For other models, return equal importance
            return {feature: 1.0 / len(feature_names) for feature in feature_names}
            
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {e}")
        return {feature: 1.0 / len(feature_names) for feature in feature_names}


def generate_model_report(results: Dict[str, Dict[str, Any]], 
                         feature_names: list,
                         save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive model evaluation report.
    
    Args:
        results: Model evaluation results
        feature_names: List of feature names
        save_path: Optional path to save the report
        
    Returns:
        Report text
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("BREAST CANCER PREDICTION - MODEL EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Features used: {', '.join(feature_names)}")
    report_lines.append("")
    
    # Model comparison
    report_lines.append("MODEL PERFORMANCE COMPARISON")
    report_lines.append("-" * 40)
    
    # Create comparison table
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'specificity']
    header = f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    for model_name, model_results in results.items():
        model_display = model_name.replace('_', ' ').title()
        row = f"{model_display:<20}"
        for metric in metrics:
            value = model_results.get(metric, 0)
            row += f"{value:<10.4f}"
        report_lines.append(row)
    
    report_lines.append("")
    
    # Detailed results for each model
    for model_name, model_results in results.items():
        report_lines.append(f"DETAILED RESULTS - {model_name.upper().replace('_', ' ')}")
        report_lines.append("-" * 50)
        
        for metric, value in model_results.items():
            if metric not in ['confusion_matrix', 'classification_report']:
                report_lines.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        report_lines.append("")
        report_lines.append("Confusion Matrix:")
        conf_matrix = model_results['confusion_matrix']
        report_lines.append("                Predicted")
        report_lines.append("Actual    Benign  Malignant")
        report_lines.append(f"Benign    {conf_matrix[0,0]:<8} {conf_matrix[0,1]:<8}")
        report_lines.append(f"Malignant {conf_matrix[1,0]:<8} {conf_matrix[1,1]:<8}")
        report_lines.append("")
    
    # Summary and recommendations
    report_lines.append("SUMMARY AND RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    report_lines.append(f"Best performing model: {best_model.replace('_', ' ').title()}")
    report_lines.append(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    if best_accuracy > 0.95:
        report_lines.append("Excellent model performance achieved!")
    elif best_accuracy > 0.90:
        report_lines.append("Good model performance achieved.")
    elif best_accuracy > 0.85:
        report_lines.append("Acceptable model performance.")
    else:
        report_lines.append("Model performance needs improvement.")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        try:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Model report saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return report_text


def create_experiment_log(experiment_name: str, 
                         parameters: Dict[str, Any],
                         results: Dict[str, Any],
                         save_dir: str = "experiments") -> str:
    """
    Create an experiment log for reproducibility.
    
    Args:
        experiment_name: Name of the experiment
        parameters: Experiment parameters
        results: Experiment results
        save_dir: Directory to save experiment logs
        
    Returns:
        Path to the saved experiment log
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    experiment_data = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "parameters": parameters,
        "results": results
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        logger.info(f"Experiment log saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save experiment log: {e}")
        return ""


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_report = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicate_rows": data.duplicated().sum(),
        "data_types": data.dtypes.to_dict(),
        "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Check for outliers in numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    
    quality_report["outliers"] = outliers
    
    return quality_report 