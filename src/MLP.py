import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
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

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


results = []

# Loop through each label
for label in labels:
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_label(df, label)

    # Initialize and train the MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50),  # Two layers with 100 and 50 neurons
                       activation='relu',             # Activation function
                       solver='adam',                 # Optimizer
                       max_iter=500,                  # Maximum iterations
                       random_state=42)
    mlp.fit(X_train, y_train)

    # Make predictions
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    results.append({
        "Label": label,
        "MAE": mae,
        "Training MSE": mse_train,
        "Testing MSE": mse_test,
        "R^2 Score": r2_test,
    })

    # Residual plot
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors="k")
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {label} (MLP)")

    plt.legend()
    plt.show()

    # Plot predictions vs actual values
    plt.figure(figsize=(12, 6))

    # Training data plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolors="k", label="Predictions")
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values (Training)")
    plt.ylabel("Predicted Values")
    plt.title(f"Training Data: {label} (MLP)")
    plt.legend()

    # Testing data plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values (Testing)")
    plt.ylabel("Predicted Values")
    plt.title(f"Testing Data: {label} (MLP)")
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


results_df = pd.DataFrame(results)
