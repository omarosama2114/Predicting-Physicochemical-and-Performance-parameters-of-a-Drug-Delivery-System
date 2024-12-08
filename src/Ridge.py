import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../Data/Drug Dataset - Cleaned & Encoded.csv')

# Define label columns
labels = [
    "Drug loading (EE%)", 
    "Vesicular  Size (nm)", 
    "Vesicles zeta potential (mV)", 
    "Drug released percentage"
]

# Define a function to prepare data for a specific label
def prepare_data_for_label(df, label_column):
    # Drop rows where the target label has missing values
    df_cleaned = df.dropna(subset=[label_column])

    # Separate features and target label
    X = df_cleaned.drop(columns=labels)  # Drop all label columns
    y = df_cleaned[label_column]         # Select the specific label column as target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


results = []

# Prepare data for each label and train the model
for label in labels:
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data_for_label(df, label)

    # Define Ridge model
    ridge_model = Ridge()

    # Define hyperparameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1, 5, 10, 50, 100],
        'max_iter': [50, 100, 200, 300, 500],
        'tol': [0.001, 0.01, 0.1, 1]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=ridge_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_

    # Train the best model on the full training data
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate Ridge
    mae = mean_absolute_error(y_test, y_test_pred)
    mse_test_ridge = mean_squared_error(y_test, y_test_pred)
    mse_train_ridge = mean_squared_error(y_train, y_train_pred)
    r2_test_ridge = r2_score(y_test, y_test_pred)

    results.append({
        "Label": label,
        "Model": "Ridge",
        "Best Parameters": best_params,
        "MAE": mae,
        "Training MSE": mse_train_ridge,
        "Testing MSE": mse_test_ridge,
        "R^2 Score": r2_test_ridge,
    })

    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors="k")
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {label}")
    plt.legend()
    plt.show()

    # Plot predictions vs actual values for training data
    plt.figure(figsize=(12, 6))

    # Training data plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolors="k", label="Predictions")
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values (Training)")
    plt.ylabel("Predicted Values")
    plt.title(f"Training Data: {label}")
    plt.legend()

    # Testing data plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values (Testing)")
    plt.ylabel("Predicted Values")
    plt.title(f"Testing Data: {label}")
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Save results to a DataFrame
results_df = pd.DataFrame(results)
