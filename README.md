# Breast Cancer Prediction using Naive Bayes

This project implements both Gaussian and Categorical Naive Bayes algorithms for breast cancer prediction using the Wisconsin Breast Cancer dataset.

## Features

- **Gaussian Naive Bayes**: Uses continuous features with Gaussian distribution assumption
- **Categorical Naive Bayes**: Uses discretized features for categorical classification
- **Exploratory Data Analysis**: Comprehensive data visualization and analysis
- **Model Evaluation**: Confusion matrix, F1 score, and accuracy metrics

## Setup Instructions

### 1. Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### 2. Virtual Environment Setup (Recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Alternative: Direct Installation

If you prefer not to use a virtual environment:

```bash
# Install packages directly (may require --user flag on some systems)
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Usage

### Option 1: Run the Complete Python Script

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run the main script
python breast_cancer_naive_bayes.py
```

### Option 2: Run the Jupyter Notebook

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

**After running the command above:**

1. **Wait for Jupyter to start** - You'll see output like:
   ```
   Jupyter Server 2.16.0 is running at:
   http://localhost:8888/tree?token=...
   ```

2. **Open your web browser** and go to:
   - `http://localhost:8888` (most common)
   - Or copy the exact URL shown in your terminal

3. **Navigate to your notebook**:
   - You'll see a file browser showing all files in your project
   - Click on `naive bayes.ipynb` to open it

4. **Run the notebook**:
   - **Run all cells**: Go to `Cell` → `Run All` in the menu
   - **Run individual cells**: Click on a cell and press `Shift + Enter`
   - **Run and stay**: Press `Ctrl + Enter` to run a cell and stay on it

5. **Trust the notebook** (if prompted):
   - If you see a "Not Trusted" warning, click "Trust" to allow outputs to display

### Option 3: Test Imports First

```bash
# Test if everything is set up correctly
python test_imports.py
```

## Project Structure

```
breast_cancer_prediction-main/
├── Breast_cancer_data.csv          # Dataset
├── naive bayes.ipynb               # Jupyter notebook
├── breast_cancer_naive_bayes.py    # Complete Python script
├── test_imports.py                 # Import test script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── venv/                          # Virtual environment (created during setup)
```

## Dataset

The dataset contains the following features:
- `mean_radius`: Mean radius of the tumor
- `mean_texture`: Mean texture of the tumor
- `mean_perimeter`: Mean perimeter of the tumor
- `mean_area`: Mean area of the tumor
- `mean_smoothness`: Mean smoothness of the tumor
- `diagnosis`: Target variable (0: Benign, 1: Malignant)

## Algorithm Implementation

### Gaussian Naive Bayes
- Assumes features follow Gaussian (normal) distribution
- Calculates likelihood using probability density function
- Suitable for continuous features

### Categorical Naive Bayes
- Discretizes continuous features into bins
- Calculates likelihood using frequency counts
- Suitable for categorical or discretized features

## Model Performance

The script will output:
- Confusion matrices for both models
- F1 scores for classification performance
- Accuracy metrics
- Visualizations of data distributions and correlations

## Viewing the Notebook

### On GitHub
- **Rendered View**: Click on `naive bayes.ipynb` in your GitHub repository to see the notebook with all outputs
- **Interactive View**: Use [nbviewer](https://nbviewer.org/) to view the notebook with all visualizations
- **Graphs and Plots**: All visualizations will be displayed as long as you run all cells before committing

### Locally
- **Jupyter Interface**: Use the instructions above to run the notebook locally
- **VS Code**: You can also open `.ipynb` files directly in VS Code with the Python extension

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've activated the virtual environment and installed all requirements
2. **Dataset Not Found**: Ensure `Breast_cancer_data.csv` is in the same directory
3. **Permission Errors**: On macOS/Linux, you might need to use `python3` instead of `python`
4. **Jupyter Won't Start**: Try running `jupyter notebook --no-browser` and manually open the URL
5. **Notebook Not Trusted**: Click "Trust" in the Jupyter interface to allow outputs to display
6. **Port Already in Use**: If port 8888 is busy, Jupyter will automatically use the next available port

### Getting Help

1. Run `python test_imports.py` to verify your setup
2. Check that all files are in the correct directory
3. Ensure you're using Python 3.7 or higher
4. Make sure your virtual environment is activated (`source venv/bin/activate`)

## Dependencies

- numpy >= 2.3.1
- pandas >= 2.3.1
- matplotlib >= 3.10.3
- seaborn >= 0.13.2
- scikit-learn >= 1.7.0
- jupyter >= 1.1.1

## License

This project is for educational purposes. The dataset is from the Wisconsin Breast Cancer dataset.