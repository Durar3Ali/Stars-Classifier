import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Set page config
st.set_page_config(page_title="Stars Classifier", page_icon="☆", layout="wide")

# Title
st.title("☆ Stars Classifier Interactive Report")

# Star type mapping
STAR_TYPE_NAMES = {
    0: "Red Dwarf",
    1: "Brown Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Super Giants",
    5: "Hyper Giants",
}

NUMERIC_FEATURES = ["Temperature", "L", "R", "A_M"]
CATEGORICAL_FEATURES = ["Color", "Spectral_Class"]

MODEL_CONFIGS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            multi_class="multinomial",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l2"],
        },
        "validation": {
            "param_name": "model__C",
            "param_label": "Regularization strength (C)",
            "param_range": np.logspace(-2, 2, 6),
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
        },
        "validation": {
            "param_name": "model__n_neighbors",
            "param_label": "Neighbors",
            "param_range": np.arange(3, 16, 2),
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None, 3, 5, 7, 9],
            "model__min_samples_split": [2, 4, 6],
        },
        "validation": {
            "param_name": "model__max_depth",
            "param_label": "Tree depth",
            "param_range": [3, 5, 7, 9, 11, 13],
        },
    },
}


@st.cache_data
def load_data():
    return pd.read_csv("Stars.csv")


df = load_data()
X = df.drop("Type", axis=1)
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ]
)


def build_pipeline(estimator):
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clone(estimator)),
        ]
    )


def create_confusion_matrix_figure(cm):
    class_labels = [STAR_TYPE_NAMES[label] for label in sorted(STAR_TYPE_NAMES.keys())]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def create_learning_curve_figure(model_name, estimator):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, "o-", label="Training accuracy")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, valid_mean, "o-", label="Validation accuracy")
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve · {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def create_validation_curve_figure(model_name, base_pipeline, validation_cfg):
    param_name = validation_cfg["param_name"]
    param_range = validation_cfg["param_range"]
    train_scores, valid_scores = validation_curve(
        base_pipeline,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(param_range, train_mean, marker="o", label="Training accuracy")
    ax.plot(param_range, valid_mean, marker="o", label="Validation accuracy")
    ax.set_xlabel(validation_cfg["param_label"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Validation Curve · {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def generate_report(model_name):
    config = MODEL_CONFIGS[model_name]
    base_pipeline = build_pipeline(config["estimator"])

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=config["param_grid"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(STAR_TYPE_NAMES.keys()),
    )

    learning_curve_fig = create_learning_curve_figure(model_name, best_pipeline)
    validation_curve_fig = create_validation_curve_figure(
        model_name,
        base_pipeline,
        config["validation"],
    )

    return {
        "accuracy": accuracy,
        "cv_mean": grid.best_score_,
        "best_params": grid.best_params_,
        "confusion_fig": create_confusion_matrix_figure(cm),
        "learning_fig": learning_curve_fig,
        "validation_fig": validation_curve_fig,
    }

st.markdown("---")

model_choice = st.selectbox("Choose a classifier", list(MODEL_CONFIGS.keys()))
go = st.button("Generate Report", type="primary", use_container_width=True)

if go:
    with st.spinner(f"Training {model_choice}..."):
        report = generate_report(model_choice)

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Accuracy (hold-out)", f"{report['accuracy']*100:.2f}%")
    metric_col2.metric("Cross-validation mean", f"{report['cv_mean']*100:.2f}%")

    st.subheader("Best Hyperparameters")
    st.code(json.dumps(report["best_params"], indent=2))

    st.subheader("Confusion Matrix")
    st.pyplot(report["confusion_fig"], clear_figure=True)

    lc_col, vc_col = st.columns(2)
    with lc_col:
        st.subheader("Learning Curve")
        st.pyplot(report["learning_fig"], clear_figure=True)
    with vc_col:
        st.subheader("Validation Curve")
        st.pyplot(report["validation_fig"], clear_figure=True)
else:
    st.info("Select a classifier and click **Generate Report** to get started.")
