#!/usr/bin/env python3
"""
Training script for Breast Cancer Prediction models.
Trains and saves models for deployment.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing import DataPreprocessor
from src.models import GaussianNaiveBayes, CategoricalNaiveBayes
from src.utils import save_model, generate_model_report, create_experiment_log

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_save_models():
    """Train and save models for deployment."""
    
    print("=" * 60)
    print("TRAINING BREAST CANCER PREDICTION MODELS")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        logger.info("Loading and preparing data...")
        preprocessor = DataPreprocessor("Breast_cancer_data.csv")
        
        # Prepare data for both model types
        train_data_gaussian, test_data_gaussian = preprocessor.prepare_data_for_modeling(categorical=False)
        train_data_categorical, test_data_categorical = preprocessor.prepare_data_for_modeling(categorical=True)
        
        print(f"Data prepared:")
        print(f"  Gaussian model - Train: {len(train_data_gaussian)}, Test: {len(test_data_gaussian)}")
        print(f"  Categorical model - Train: {len(train_data_categorical)}, Test: {len(test_data_categorical)}")
        
        # Step 2: Train Gaussian Naive Bayes
        logger.info("Training Gaussian Naive Bayes...")
        gaussian_model = GaussianNaiveBayes()
        X_train_gaussian = train_data_gaussian.iloc[:, :-1]
        y_train_gaussian = train_data_gaussian.iloc[:, -1]
        gaussian_model.fit(X_train_gaussian, y_train_gaussian)
        
        # Evaluate Gaussian model
        X_test_gaussian = test_data_gaussian.iloc[:, :-1]
        y_test_gaussian = test_data_gaussian.iloc[:, -1]
        y_pred_gaussian = gaussian_model.predict(X_test_gaussian)
        
        from sklearn.metrics import accuracy_score, f1_score
        gaussian_accuracy = accuracy_score(y_test_gaussian, y_pred_gaussian)
        gaussian_f1 = f1_score(y_test_gaussian, y_pred_gaussian)
        
        print(f"Gaussian Naive Bayes Results:")
        print(f"  Accuracy: {gaussian_accuracy:.4f}")
        print(f"  F1 Score: {gaussian_f1:.4f}")
        
        # Step 3: Train Categorical Naive Bayes
        logger.info("Training Categorical Naive Bayes...")
        categorical_model = CategoricalNaiveBayes()
        X_train_categorical = train_data_categorical.iloc[:, :-1]
        y_train_categorical = train_data_categorical.iloc[:, -1]
        categorical_model.fit(X_train_categorical, y_train_categorical)
        
        # Evaluate Categorical model
        X_test_categorical = test_data_categorical.iloc[:, :-1]
        y_test_categorical = test_data_categorical.iloc[:, -1]
        y_pred_categorical = categorical_model.predict(X_test_categorical)
        
        categorical_accuracy = accuracy_score(y_test_categorical, y_pred_categorical)
        categorical_f1 = f1_score(y_test_categorical, y_pred_categorical)
        
        print(f"Categorical Naive Bayes Results:")
        print(f"  Accuracy: {categorical_accuracy:.4f}")
        print(f"  F1 Score: {categorical_f1:.4f}")
        
        # Step 4: Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Step 5: Save models with metadata
        logger.info("Saving models...")
        
        # Save Gaussian model
        gaussian_info = {
            "model_type": "gaussian_naive_bayes",
            "accuracy": gaussian_accuracy,
            "f1_score": gaussian_f1,
            "features": list(X_train_gaussian.columns),
            "training_samples": len(X_train_gaussian),
            "test_samples": len(X_test_gaussian)
        }
        
        save_model(gaussian_model, "models/gaussian_model.pkl", gaussian_info)
        
        # Save Categorical model
        categorical_info = {
            "model_type": "categorical_naive_bayes",
            "accuracy": categorical_accuracy,
            "f1_score": categorical_f1,
            "features": list(X_train_categorical.columns),
            "training_samples": len(X_train_categorical),
            "test_samples": len(X_test_categorical)
        }
        
        save_model(categorical_model, "models/categorical_model.pkl", categorical_info)
        
        # Step 6: Generate comprehensive report
        logger.info("Generating model report...")
        
        results = {
            'gaussian': {
                'accuracy': gaussian_accuracy,
                'f1_score': gaussian_f1,
                'confusion_matrix': None,  # Will be calculated in report
                'classification_report': None
            },
            'categorical': {
                'accuracy': categorical_accuracy,
                'f1_score': categorical_f1,
                'confusion_matrix': None,
                'classification_report': None
            }
        }
        
        # Add confusion matrices and classification reports
        from sklearn.metrics import confusion_matrix, classification_report
        
        results['gaussian']['confusion_matrix'] = confusion_matrix(y_test_gaussian, y_pred_gaussian)
        results['gaussian']['classification_report'] = classification_report(y_test_gaussian, y_pred_gaussian, output_dict=True)
        
        results['categorical']['confusion_matrix'] = confusion_matrix(y_test_categorical, y_pred_categorical)
        results['categorical']['classification_report'] = classification_report(y_test_categorical, y_pred_categorical, output_dict=True)
        
        # Generate report
        report_text = generate_model_report(
            results=results,
            feature_names=list(X_train_gaussian.columns),
            save_path="model_training_report.txt"
        )
        
        # Step 7: Create experiment log
        experiment_params = {
            "features": list(X_train_gaussian.columns),
            "test_size": 0.2,
            "random_state": 42,
            "models": ["gaussian_naive_bayes", "categorical_naive_bayes"]
        }
        
        create_experiment_log(
            experiment_name="model_training",
            parameters=experiment_params,
            results=results
        )
        
        # Step 8: Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        best_model = "gaussian" if gaussian_accuracy > categorical_accuracy else "categorical"
        best_accuracy = max(gaussian_accuracy, categorical_accuracy)
        
        print(f"✅ Best Model: {best_model.replace('_', ' ').title()}")
        print(f"✅ Best Accuracy: {best_accuracy:.4f}")
        print(f"✅ Models Saved: models/gaussian_model.pkl, models/categorical_model.pkl")
        print(f"✅ Report Generated: model_training_report.txt")
        print(f"✅ Experiment Logged: experiments/")
        
        if best_accuracy > 0.95:
            print("🎉 Excellent model performance achieved!")
        elif best_accuracy > 0.90:
            print("👍 Good model performance achieved!")
        else:
            print("⚠️  Model performance could be improved.")
        
        print("\nModels are ready for deployment! 🚀")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: Could not find the dataset file. Please ensure 'Breast_cancer_data.csv' is in the current directory.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Error: Training failed with error: {e}")
        raise


if __name__ == "__main__":
    train_and_save_models() 