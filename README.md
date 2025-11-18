# Stars Classifier

A machine learning project for classifying stars into different types using multiple classification algorithms. This project includes both a Streamlit web application for interactive predictions and standalone scripts for training individual models.

## Features

- **Interactive Report UI**: Streamlit interface branded “Stars Classifier Interactive Report” for quick exploratory model analysis.
- **Smart Metrics Deck**: Each run surfaces Accuracy, cross-validation mean, best hyperparameters, confusion matrix, learning curve, and validation curve in a compact report.
- **Dropdown Model Switcher**: Generate professional reports for Logistic Regression, KNN, or Decision Tree with a single click.
- **Star Type Classification**: Classify stars into 6 different types:
  - Red Dwarf (0)
  - Brown Dwarf (1)
  - White Dwarf (2)
  - Main Sequence (3)
  - Super Giants (4)
  - Hyper Giants (5)

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Stars-Classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application

Run the Streamlit app for an interactive classification interface:

```bash
streamlit run app.py
```

Once it opens in your browser you can:
- Choose **Logistic Regression**, **KNN**, or **Decision Tree** from the dropdown
- Launch training + hyperparameter search via **Generate Report**
- Review the automatically generated report (metrics, best params, confusion matrix, learning/validation curves)
- Iterate quickly to compare model behavior side by side

### Standalone Scripts

You can also run individual classifier scripts to train and evaluate models:

**Logistic Regression:**
```bash
python logreg.py
```

**Decision Tree:**
```bash
python dt.py
```

**KNN:**
```bash
python knn.py
```

Each script will:
- Load and preprocess the data
- Train the model
- Evaluate on test data
- Display accuracy and confusion matrix

## Dataset

The project uses `Stars.csv` which contains the following features:
- **Temperature** (K): Star temperature in Kelvin
- **L** (L☉): Luminosity relative to the Sun
- **R** (R☉): Radius relative to the Sun
- **A_M**: Absolute Magnitude
- **Color**: Star color (categorical)
- **Spectral_Class**: Spectral classification (categorical)
- **Type**: Star type (target variable)

## Model Details

### Data Preprocessing
- Uses a `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder` for categorical fields
- 80/20 stratified train/test split with cached dataset loading
- Pipelines keep preprocessing consistent across models

### Models
- **Logistic Regression**: Multinomial with up to 2000 iterations; tuned over regularization strengths
- **Decision Tree**: Grid-searches criterion, depth, and split sizes to find the best structure
- **KNN**: Explores neighbor counts and weighting strategies

Each model is evaluated with 5-fold cross-validation (GridSearchCV), and its best performer is used to produce the downstream diagnostics shown in the report.

## Project Structure

```
Stars-Classifier/
├── app.py              # Streamlit web application
├── dt.py               # Decision Tree classifier script
├── knn.py              # KNN classifier script
├── logreq.py           # Logistic Regression classifier script
├── Stars.csv           # Dataset
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Technologies Used

- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## License

This project is open source and available for educational purposes.

## Contributing

Contributions, issues, and feature requests are welcome!

