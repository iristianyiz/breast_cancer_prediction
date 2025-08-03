#!/usr/bin/env python3
"""
Flask API for Breast Cancer Prediction
Deploy trained models as a REST API for real-time predictions.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import GaussianNaiveBayes, CategoricalNaiveBayes
from src.preprocessing import DataPreprocessor
from src.utils import load_model, validate_data, calculate_feature_importance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
gaussian_model = None
categorical_model = None
feature_columns = ["mean_radius", "mean_texture", "mean_smoothness"]
model_info = {}


def load_models():
    """Load trained models from disk."""
    global gaussian_model, categorical_model, model_info
    
    try:
        # Try to load saved models
        if os.path.exists("models/gaussian_model.pkl"):
            gaussian_model, gaussian_info = load_model("models/gaussian_model.pkl")
            model_info['gaussian'] = gaussian_info
            logger.info("Gaussian model loaded successfully")
        
        if os.path.exists("models/categorical_model.pkl"):
            categorical_model, categorical_info = load_model("models/categorical_model.pkl")
            model_info['categorical'] = categorical_info
            logger.info("Categorical model loaded successfully")
        
        # If no saved models, train new ones
        if gaussian_model is None or categorical_model is None:
            logger.info("Training new models...")
            train_models()
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Training new models...")
        train_models()


def train_models():
    """Train models if not already loaded."""
    global gaussian_model, categorical_model
    
    try:
        # Load and prepare data
        preprocessor = DataPreprocessor("Breast_cancer_data.csv")
        train_data_gaussian, test_data_gaussian = preprocessor.prepare_data_for_modeling(categorical=False)
        train_data_categorical, test_data_categorical = preprocessor.prepare_data_for_modeling(categorical=True)
        
        # Train Gaussian model
        if gaussian_model is None:
            gaussian_model = GaussianNaiveBayes()
            X_train = train_data_gaussian.iloc[:, :-1]
            y_train = train_data_gaussian.iloc[:, -1]
            gaussian_model.fit(X_train, y_train)
            logger.info("Gaussian model trained successfully")
        
        # Train Categorical model
        if categorical_model is None:
            categorical_model = CategoricalNaiveBayes()
            X_train = train_data_categorical.iloc[:, :-1]
            y_train = train_data_categorical.iloc[:, -1]
            categorical_model.fit(X_train, y_train)
            logger.info("Categorical model trained successfully")
            
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise


@app.route('/')
def home():
    """Home page with prediction form."""
    best_accuracy = 0.0
    if model_info:
        accuracies = [info.get('accuracy', 0) for info in model_info.values() if info]
        best_accuracy = max(accuracies) * 100 if accuracies else 0.0
    
    return render_template('index.html', 
                         features=feature_columns,
                         best_accuracy=f"{best_accuracy:.2f}")


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained models."""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        missing_fields = [field for field in feature_columns if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Create feature DataFrame
        features = pd.DataFrame([{
            'mean_radius': data['mean_radius'],
            'mean_texture': data['mean_texture'],
            'mean_smoothness': data['mean_smoothness']
        }])
        
        # Validate data
        if not validate_data(features, feature_columns):
            return jsonify({'error': 'Invalid data provided'}), 400
        
        # Get model type (default to gaussian)
        model_type = data.get('model_type', 'gaussian').lower()
        
        # Make prediction
        if model_type == 'gaussian' and gaussian_model is not None:
            prediction = gaussian_model.predict(features)[0]
            probabilities = gaussian_model.predict_proba(features)[0]
            probability = probabilities[prediction]
        elif model_type == 'categorical' and categorical_model is not None:
            prediction = categorical_model.predict(features)[0]
            probabilities = categorical_model.predict_proba(features)[0]
            probability = probabilities[prediction]
        else:
            return jsonify({'error': f'Model type "{model_type}" not available'}), 400
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'model_type': model_type,
            'features': data,
            'diagnosis': 'Benign' if prediction == 0 else 'Malignant'
        }
        
        logger.info(f"Prediction made: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    models_loaded = {
        'gaussian': gaussian_model is not None,
        'categorical': categorical_model is not None
    }
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'feature_columns': feature_columns
    })


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about trained models."""
    return jsonify({
        'models': model_info,
        'feature_columns': feature_columns,
        'available_models': list(model_info.keys())
    })


@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance for the models."""
    try:
        model_type = request.args.get('model_type', 'gaussian').lower()
        
        if model_type == 'gaussian' and gaussian_model is not None:
            importance = calculate_feature_importance(gaussian_model, feature_columns)
        elif model_type == 'categorical' and categorical_model is not None:
            importance = calculate_feature_importance(categorical_model, feature_columns)
        else:
            return jsonify({'error': f'Model type "{model_type}" not available'}), 400
        
        return jsonify({
            'model_type': model_type,
            'feature_importance': importance
        })
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return jsonify({'error': f'Failed to get feature importance: {str(e)}'}), 500


if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 