import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Set page config
st.set_page_config(page_title="Stars Classifier", page_icon="☆", layout="wide")

# Title
st.title("☆ Stars Classifier Demo ☆")

# Star type mapping
STAR_TYPE_NAMES = {
    0: "Red Dwarf",
    1: "Brown Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Super Giants",
    5: "Hyper Giants"
}

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Stars.csv")
    return df

df = load_data()

# Preprocessing function
def preprocess_data(X, categorical_cols):
    """Encode categorical columns"""
    X_processed = X.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        label_encoders[col] = le
    return X_processed, label_encoders

# Prepare data
X = df.drop("Type", axis=1)
y = df["Type"]
categorical_cols = ["Color", "Spectral_Class"]

st.markdown("---")

# Demo Section
st.header("🔮 Test Classifiers")

# Input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Star Features")
    temperature = st.number_input("Temperature (K)", min_value=0.0, value=3000.0, step=100.0)
    luminosity = st.number_input("L (L☉)", min_value=0.0, value=0.001, format="%.6f")
    radius = st.number_input("R (R☉)", min_value=0.0, value=0.1, format="%.3f")
    
with col2:
    st.subheader("Additional Features")
    abs_magnitude = st.number_input("A_M", value=15.0, step=0.1)
    color = st.selectbox("Color", df["Color"].unique())
    spectral_class = st.selectbox("Spectral Class", df["Spectral_Class"].unique())

predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict_button:
    # Prepare input
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "L": [luminosity],
        "R": [radius],
        "A_M": [abs_magnitude],
        "Color": [color],
        "Spectral_Class": [spectral_class]
    })
    
    # Use 80/20 split for demo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess
    X_train_processed, label_encoders = preprocess_data(X_train, categorical_cols)
    input_processed = input_data.copy()
    for col in categorical_cols:
        if col in label_encoders:
            try:
                input_processed[col] = label_encoders[col].transform([input_data[col].iloc[0]])[0]
            except:
                input_processed[col] = 0
    
    st.markdown("---")
    st.subheader("Predictions")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    # Logistic Regression
    with pred_col1:
        st.write("**Logistic Regression**")
        lr_model = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)
        lr_model.fit(X_train_processed, y_train)
        lr_pred = lr_model.predict(input_processed)[0]
        lr_pred_name = STAR_TYPE_NAMES.get(lr_pred, f"Type {lr_pred}")
        st.success(f"**{lr_pred_name}** ({lr_pred})")
        proba = lr_model.predict_proba(input_processed)[0]
        st.write(f"Confidence: {max(proba)*100:.1f}%")
    
    # Decision Tree
    with pred_col2:
        st.write("**Decision Tree**")
        cols_to_scale = ["Temperature", "L", "R"]
        scaler = StandardScaler()
        X_train_scaled = X_train_processed.copy()
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_processed[cols_to_scale])
        input_scaled = input_processed.copy()
        input_scaled[cols_to_scale] = scaler.transform(input_processed[cols_to_scale])
        
        dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
        dt_model.fit(X_train_scaled, y_train)
        dt_pred = dt_model.predict(input_scaled)[0]
        dt_pred_name = STAR_TYPE_NAMES.get(dt_pred, f"Type {dt_pred}")
        st.success(f"**{dt_pred_name}** ({dt_pred})")
        proba = dt_model.predict_proba(input_scaled)[0]
        st.write(f"Confidence: {max(proba)*100:.1f}%")
    
    # KNN
    with pred_col3:
        st.write("**KNN**")
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train_processed, y_train)
        knn_pred = knn_model.predict(input_processed)[0]
        knn_pred_name = STAR_TYPE_NAMES.get(knn_pred, f"Type {knn_pred}")
        st.success(f"**{knn_pred_name}** ({knn_pred})")
        proba = knn_model.predict_proba(input_processed)[0]
        st.write(f"Confidence: {max(proba)*100:.1f}%")
