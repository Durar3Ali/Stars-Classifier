# ⭐ Stars Classifier

A machine learning project for classifying stars into different types using multiple classification algorithms. This project includes both a Streamlit web application for interactive predictions and standalone scripts for training individual models.

## 🌟 Features

- **Interactive Web Application**: Streamlit-based UI for testing star classifications
- **Multiple Classifiers**: Compare predictions from three different algorithms:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- **Star Type Classification**: Classify stars into 6 different types:
  - Red Dwarf (0)
  - Brown Dwarf (1)
  - White Dwarf (2)
  - Main Sequence (3)
  - Super Giants (4)
  - Hyper Giants (5)

## 📋 Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Stars-Classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Web Application

Run the Streamlit app for an interactive classification interface:

```bash
streamlit run app.py
```

The application will open in your browser where you can:
- Input star features (Temperature, Luminosity, Radius, Absolute Magnitude, Color, Spectral Class)
- Get predictions from all three classifiers simultaneously
- View confidence scores for each prediction

### Standalone Scripts

You can also run individual classifier scripts to train and evaluate models:

**Logistic Regression:**
```bash
python logreq.py
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

## 📊 Dataset

The project uses `Stars.csv` which contains the following features:
- **Temperature** (K): Star temperature in Kelvin
- **L** (L☉): Luminosity relative to the Sun
- **R** (R☉): Radius relative to the Sun
- **A_M**: Absolute Magnitude
- **Color**: Star color (categorical)
- **Spectral_Class**: Spectral classification (categorical)
- **Type**: Star type (target variable)

## 🔧 Model Details

### Data Preprocessing
- Categorical features (Color, Spectral_Class) are encoded using Label Encoding
- Numerical features (Temperature, L, R) are scaled using StandardScaler for Decision Tree
- Data is split 80/20 for training/testing with stratification

### Models
- **Logistic Regression**: Multinomial logistic regression with 1000 max iterations
- **Decision Tree**: Entropy-based with max depth of 3
- **KNN**: K-Nearest Neighbors with k=5

## 📁 Project Structure

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

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

