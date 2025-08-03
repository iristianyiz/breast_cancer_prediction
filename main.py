#!/usr/bin/env python3
"""
Breast Cancer Prediction - Main Pipeline
End-to-end machine learning pipeline for breast cancer prediction using Naive Bayes.
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing import DataPreprocessor, load_and_prepare_data
from src.models import train_and_evaluate_models, ModelEvaluator
from src.visualize import create_visualization_report, DataVisualizer
from src.utils import validate_data, generate_model_report, create_experiment_log, check_data_quality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breast_cancer_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the complete breast cancer prediction pipeline."""
    
    print("=" * 60)
    print("BREAST CANCER PREDICTION PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Data Loading and Validation
        logger.info("Step 1: Loading and validating data...")
        preprocessor = DataPreprocessor("Breast_cancer_data.csv")
        
        # Load data
        data = preprocessor.load_data()
        
        # Get basic info
        info = preprocessor.get_basic_info()
        print(f"\nDataset Info:")
        print(f"  Shape: {info['shape']}")
        print(f"  Features: {len(info['columns']) - 1}")
        print(f"  Target distribution: {info['target_distribution']}")
        
        # Validate data
        feature_cols = ["mean_radius", "mean_texture", "mean_smoothness"]
        if not validate_data(data, feature_cols):
            logger.error("Data validation failed. Exiting.")
            return
        
        # Check data quality
        quality_report = check_data_quality(data)
        print(f"\nData Quality Report:")
        print(f"  Memory usage: {quality_report['memory_usage_mb']:.2f} MB")
        print(f"  Duplicate rows: {quality_report['duplicate_rows']}")
        print(f"  Missing values: {sum(quality_report['missing_values'].values())}")
        
        # Step 2: Data Preparation
        logger.info("Step 2: Preparing data for modeling...")
        
        # Prepare data for both Gaussian and Categorical models
        train_data_gaussian, test_data_gaussian = preprocessor.prepare_data_for_modeling(categorical=False)
        train_data_categorical, test_data_categorical = preprocessor.prepare_data_for_modeling(categorical=True)
        
        print(f"\nData Split:")
        print(f"  Gaussian model - Train: {len(train_data_gaussian)}, Test: {len(test_data_gaussian)}")
        print(f"  Categorical model - Train: {len(train_data_categorical)}, Test: {len(test_data_categorical)}")
        
        # Step 3: Model Training and Evaluation
        logger.info("Step 3: Training and evaluating models...")
        
        # Train and evaluate Gaussian model
        print("\nTraining Gaussian Naive Bayes...")
        results_gaussian = train_and_evaluate_models(train_data_gaussian, test_data_gaussian)
        
        # Train and evaluate Categorical model
        print("\nTraining Categorical Naive Bayes...")
        results_categorical = train_and_evaluate_models(train_data_categorical, test_data_categorical)
        
        # Combine results
        all_results = {
            'gaussian': results_gaussian['gaussian'],
            'categorical': results_categorical['categorical']
        }
        
        # Step 4: Visualization
        logger.info("Step 4: Creating visualizations...")
        
        # Create visualization report
        create_visualization_report(
            data=data,
            results=all_results,
            feature_cols=feature_cols,
            save_dir="plots"
        )
        
        # Step 5: Generate Reports
        logger.info("Step 5: Generating reports...")
        
        # Generate comprehensive model report
        report_text = generate_model_report(
            results=all_results,
            feature_names=feature_cols,
            save_path="model_evaluation_report.txt"
        )
        
        print("\n" + report_text)
        
        # Create experiment log
        experiment_params = {
            "features": feature_cols,
            "test_size": 0.2,
            "random_state": 42,
            "models": ["gaussian_naive_bayes", "categorical_naive_bayes"]
        }
        
        create_experiment_log(
            experiment_name="breast_cancer_prediction",
            parameters=experiment_params,
            results=all_results
        )
        
        # Step 6: Summary
        logger.info("Step 6: Pipeline completed successfully!")
        
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        
        best_model = max(all_results.keys(), key=lambda x: all_results[x]['accuracy'])
        best_accuracy = all_results[best_model]['accuracy']
        
        print(f"✅ Best Model: {best_model.replace('_', ' ').title()}")
        print(f"✅ Best Accuracy: {best_accuracy:.4f}")
        print(f"✅ Models Trained: {len(all_results)}")
        print(f"✅ Visualizations: Saved to 'plots/' directory")
        print(f"✅ Reports: Saved to 'model_evaluation_report.txt'")
        print(f"✅ Logs: Saved to 'breast_cancer_prediction.log'")
        
        if best_accuracy > 0.95:
            print("Excellent performance achieved!")
        elif best_accuracy > 0.90:
            print("Good performance achieved!")
        else:
            print("Performance could be improved.")
        
        print("\nPipeline completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: Could not find the dataset file. Please ensure 'Breast_cancer_data.csv' is in the current directory.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

