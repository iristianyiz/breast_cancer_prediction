# Breast Cancer Prediction using Naive Bayes

A comprehensive machine learning project that implements both Gaussian and Categorical Naive Bayes classifiers for breast cancer prediction using the Wisconsin Breast Cancer dataset. This project demonstrates professional software engineering practices including modular code design, comprehensive testing, and API deployment.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for breast cancer prediction with the following key features:

- **Modular Architecture**: Clean separation of concerns with dedicated modules for preprocessing, modeling, visualization, and utilities
- **Multiple Model Types**: Both Gaussian and Categorical Naive Bayes implementations
- **Comprehensive Testing**: Unit tests for all major components
- **Professional Visualization**: EDA plots and model explainability visualizations
- **API Deployment**: Flask-based REST API for real-time predictions
- **Production-Ready**: Logging, error handling, and data validation

## 📊 Performance Results

- **Gaussian Naive Bayes**: 97.37% accuracy
- **Categorical Naive Bayes**: 95.17% accuracy
- **Features Used**: mean_radius, mean_texture, mean_smoothness

## 🏗️ Project Structure

```
breast_cancer_prediction/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading and preprocessing
│   ├── models.py                 # Naive Bayes implementations
│   ├── visualize.py              # Visualization and EDA
│   └── utils.py                  # Utility functions
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py     # Tests for preprocessing
│   └── test_models.py            # Tests for models
├── data/                         # Data directory
│   └── Breast_cancer_data.csv    # Dataset
├── models/                       # Saved models (created after training)
├── plots/                        # Generated visualizations
├── experiments/                  # Experiment logs
├── main.py                       # Main pipeline script
├── app.py                        # Flask API
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd breast_cancer_prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run the main pipeline
python main.py
```

This will:
- Load and validate the dataset
- Train both Gaussian and Categorical Naive Bayes models
- Generate comprehensive visualizations
- Create evaluation reports
- Save experiment logs

### 3. Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with verbose output
python -m pytest tests/ -v
```

### 4. Deploy the API

```bash
# Start the Flask API
python app.py
```

The API will be available at `http://localhost:5000`

## 📈 Usage Examples

### Running the Main Pipeline

```python
from src.preprocessing import DataPreprocessor
from src.models import train_and_evaluate_models
from src.visualize import create_visualization_report

# Load and prepare data
preprocessor = DataPreprocessor("Breast_cancer_data.csv")
train_data, test_data = preprocessor.prepare_data_for_modeling()

# Train and evaluate models
results = train_and_evaluate_models(train_data, test_data)

# Create visualizations
create_visualization_report(data, results, feature_cols)
```

### Using the API

```python
import requests
import json

# Make a prediction
data = {
    "mean_radius": 15.0,
    "mean_texture": 20.0,
    "mean_smoothness": 0.1,
    "model_type": "gaussian"
}

response = requests.post("http://localhost:5000/predict", json=data)
result = response.json()

print(f"Prediction: {result['diagnosis']}")
print(f"Confidence: {result['probability']:.2%}")
```

### API Endpoints

- `GET /` - Web interface for predictions
- `POST /predict` - Make predictions
- `GET /health` - Health check
- `GET /model_info` - Model information
- `GET /feature_importance` - Feature importance scores

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=src

# Run specific test
python -m pytest tests/test_models.py::TestGaussianNaiveBayes::test_fit
```

### Test Coverage

- **Preprocessing**: Data loading, validation, feature engineering
- **Models**: Gaussian and Categorical Naive Bayes implementations
- **Evaluation**: Model evaluation metrics and utilities
- **Utilities**: Data validation, model persistence, reporting

## 📊 Model Details

### Gaussian Naive Bayes
- Assumes features follow Gaussian (normal) distribution
- Calculates likelihood using probability density function
- Suitable for continuous features
- Achieves 97.37% accuracy

### Categorical Naive Bayes
- Discretizes continuous features into bins
- Calculates likelihood using frequency counts
- Suitable for categorical or discretized features
- Achieves 95.17% accuracy

## 🔧 Configuration

### Model Parameters

```python
# Gaussian Naive Bayes parameters
gaussian_params = {
    "features": ["mean_radius", "mean_texture", "mean_smoothness"],
    "test_size": 0.2,
    "random_state": 42
}

# Categorical Naive Bayes parameters
categorical_params = {
    "features": ["mean_radius", "mean_texture", "mean_smoothness"],
    "n_bins": 3,
    "test_size": 0.2,
    "random_state": 42
}
```

### API Configuration

```python
# Flask app configuration
app_config = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False
}
```

## 📈 Results and Visualizations

The pipeline generates several outputs:

### Reports
- `model_evaluation_report.txt` - Comprehensive model evaluation
- `breast_cancer_prediction.log` - Detailed execution logs

### Visualizations
- `plots/target_distribution.png` - Target variable distribution
- `plots/feature_distributions.png` - Feature distributions by class
- `plots/correlation_matrix.png` - Feature correlation matrix
- `plots/model_comparison.png` - Model performance comparison
- `plots/gaussian_confusion_matrix.png` - Gaussian model confusion matrix
- `plots/categorical_confusion_matrix.png` - Categorical model confusion matrix

### Experiment Logs
- `experiments/` - JSON files containing experiment parameters and results

## 🛠️ Development

### Adding New Features

1. **New Model Type**: Add to `src/models.py`
2. **New Preprocessing Step**: Add to `src/preprocessing.py`
3. **New Visualization**: Add to `src/visualize.py`
4. **New Utility Function**: Add to `src/utils.py`

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Write unit tests for new functionality

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run all tests to ensure they pass
6. Submit a pull request

## 📚 Dependencies

- **numpy>=1.21.0** - Numerical computing
- **pandas>=1.3.0** - Data manipulation
- **matplotlib>=3.4.0** - Plotting
- **seaborn>=0.11.0** - Statistical visualization
- **scikit-learn>=1.0.0** - Machine learning utilities
- **flask>=2.0.0** - Web framework
- **pytest>=6.0.0** - Testing framework
- **jupyter>=1.0.0** - Jupyter notebooks

## 🎓 Educational Value

This project demonstrates:

- **Software Engineering**: Modular design, testing, documentation
- **Machine Learning**: Algorithm implementation, evaluation, deployment
- **Data Science**: EDA, feature engineering, visualization
- **Production Practices**: Logging, error handling, API design

## 📄 License

This project is for educational purposes. The dataset is from the Wisconsin Breast Cancer dataset.

## 🤝 Acknowledgments

- Wisconsin Breast Cancer dataset contributors
- Scikit-learn community for machine learning utilities
- Flask community for web framework

## 📞 Support

For questions or issues:
1. Check the documentation
2. Run the tests to verify your setup
3. Create an issue with detailed information

---

**Note**: This is a demonstration project for educational purposes. For real medical applications, consult with healthcare professionals and follow appropriate regulatory guidelines.