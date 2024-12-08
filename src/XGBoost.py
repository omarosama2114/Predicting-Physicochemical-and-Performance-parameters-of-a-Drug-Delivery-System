from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/Drug Dataset - Cleaned & Encoded.csv')

# Define label columns
labels = [
    "Drug loading (EE%)", 
    "Vesicular  Size (nm)", 
    "Vesicles zeta potential (mV)", 
    "Drug released percentage"
]

# Function to prepare data for a specific label
def prepare_data_for_label(df, label_column):
    # Drop rows where the target label has missing values
    df_cleaned = df.dropna(subset=[label_column])

    # Separate features and target label
    X = df_cleaned.drop(columns=labels)  # Drop all label columns
    y = df_cleaned[label_column]         # Select the specific label column as target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

results = []

# Train and tune XGBoost for each label
for label in labels:
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data_for_label(df, label)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
        'subsample': [0.5, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
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

    # Calculate error metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Feature Importance
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    results.append({
        "Label": label,
        "Best Parameters": best_params,
        "MAE": mae_test,
        "Training MSE": mse_train,
        "Testing MSE": mse_test,
        "R^2 Score": r2_test
    })

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)

    # Residual plot
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors="k")
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {label} (XGBoost)")
    plt.legend()
    plt.show()

    # Plot predictions vs actual values for training and testing data
    plt.figure(figsize=(12, 6))

    # Training data plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolors="k", label="Predictions")
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values (Training)")
    plt.ylabel("Predicted Values")
    plt.title(f"Training Data: {label} (XGBoost)")
    plt.legend()

    # Testing data plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values (Testing)")
    plt.ylabel("Predicted Values")
    plt.title(f"Testing Data: {label} (XGBoost)")
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()
