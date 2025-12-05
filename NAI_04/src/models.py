"""
Model Training Utilities
Author: Gabriel Francke, Adrian Kopczyński
Description:
Utility functions for training ML models such as Decision Trees and SVM.
These functions are intended to be imported and used inside Jupyter notebooks.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC


def train_decision_tree(
        X_train, X_test, y_train, y_test,
        criterion: str = "gini", max_depth: int | None = None, random_state: int = 42) -> tuple[DecisionTreeClassifier, dict]:
    """
    Train a Decision Tree classifier and return evaluation metrics.

    Parameters:
    X_train      : Training features.
    X_test       : Testing features.
    y_train      : Training labels.
    y_test       : Testing labels. 
    criterion    : Splitting criterion, either 'gini' or 'entropy'. (default: 'gini')
    max_depth    : Maximum depth of the tree. If None, the tree grows until all leaves are pure. (default: None)
    random_state : Seed for reproducibility. (default: 42)

    Returns:
    model : DecisionTreeClassifier - Fitted decision tree model.
    metrics : Dictionary containing accuracy, classification_report (as dict), and confusion_matrix.
    """

    # Initialize the DecisionTreeClassifier model
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return model, metrics


def train_svm(
        X_train, X_test, y_train, y_test,
        kernel: str = "rbf", C: float = 1.0, gamma: str | float = "scale",
        degree: int = 3, random_state: int = 42) -> tuple[SVC, dict]:
    """
    Train an SVM classifier using the specified kernel and hyperparameters.

    Parameters:
    X_train      : Training features.
    X_test       : Testing features.
    y_train      : Training labels.
    y_test       : Testing labels. 
    kernel       : Kernel type: 'linear', 'rbf', 'poly', or 'sigmoid'. (default: 'rbf')
    C            : Regularization strength. Higher values reduce regularization. (default: 1.0)
    gamma        : Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. (default: 'scale')
    degree       : Degree of the polynomial kernel (only used when kernel='poly'). (default: 3)
    random_state : Seed for reproducibility. (default: 42)

    Returns:
    model  : SVC - Fitted SVM classifier.
    metrics : Dictionary containing accuracy, classification_report (as dict), and confusion_matrix.
    """

    # Initialize the SVM model
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return model, metrics

def evaluate_svm_kernels(
        X_train, X_test, y_train, y_test,
        kernels: list[str] = None,
        C: float = 1.0, gamma: str | float = "scale", 
        degree: int = 3, random_state: int = 42) -> list[dict]:
    """
    Evaluate multiple SVM kernels and return accuracy results for comparison.

    Parameters:
    X_train      : Training features.
    X_test       : Testing features.
    y_train      : Training labels.
    y_test       : Testing labels. 
    kernels      : List of kernel names to test. (default: ['linear', 'rbf', 'poly', 'sigmoid'])
    C            : Regularization parameter. (default: 1.0)
    gamma        : Kernel coefficient. (default: 'scale')
    degree       : Degree of polynomial kernel. (default: 3)
    random_state : Seed for reproducibility. (default: 42)

    Returns:
    results : Dictionary containing kernel name and accuracy.
    """

    # If kernels is not provided, use the default kernels
    if kernels is None:
        kernels = ["linear", "rbf", "poly", "sigmoid"]

    results = []

    # Train the model for each kernel
    for k in kernels:
        model = SVC(kernel=k, C=C, gamma=gamma, degree=degree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            "kernel": k,
            "accuracy": acc
        })

    return results
