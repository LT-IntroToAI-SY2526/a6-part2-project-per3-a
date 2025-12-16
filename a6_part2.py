"""
Project: Calories Burned Prediction (Multivariable Regression)

This assignment predicts calories burned using MULTIPLE features.
Complete all the functions below following the in-class car price example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_and_explore_data(filename):
    """
    Load the calories burned data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # TODO: Load the CSV file using pandas
    data = pd.read_csv(filename)
    # TODO: Print the first 5 rows
    print("=== Calories Burned Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    # TODO: Print the shape of the dataset
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    # TODO: Print basic statistics for ALL columns
    print(f"\nBasic statistics:")
    print(data.describe())
    # TODO: Print the column names
    print(f"\nColumn names: {list(data.columns)}")
    # TODO: Return the dataframe
    return data

    import matplotlib.pyplot as plt

def visualize_features(data):
    """
    Create 4 scatter plots (one for each feature vs Calories)

    Args:
        data: pandas DataFrame with Age, Duration, HeartRate, Calories
    """

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Main title
    fig.suptitle('Exercise Factors vs Calories Burned',
                 fontsize=16, fontweight='bold')

    # Plot 1 (top left): Age vs Calories
    axes[0, 0].scatter(data['Age'], data['Calories'],
                       color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Calories Burned')
    axes[0, 0].set_title('Age vs Calories Burned')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2 (top right): Duration vs Calories
    axes[0, 1].scatter(data['Duration'], data['Calories'],
                       color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Duration (minutes)')
    axes[0, 1].set_ylabel('Calories Burned')
    axes[0, 1].set_title('Duration vs Calories Burned')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3 (bottom left): HeartRate vs Calories
    axes[1, 0].scatter(data['HeartRate'], data['Calories'],
                       color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Heart Rate (bpm)')
    axes[1, 0].set_ylabel('Calories Burned')
    axes[1, 0].set_title('Heart Rate vs Calories Burned')
    axes[1, 0].grid(True, alpha=0.3)

    # Layout & save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_features(data):
    # TODO: Create a list of feature column names
    #       ['Age', 'Duration', 'HeartRate']
    feature_columns = ['Age', 'Duration', 'HeartRate']
    X = data[feature_columns]
    y = data['Calories Burned']
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    return X, y

def split_data(X,y):
    X_train = X.iloc[:15]
    X_test = X.iloc[15:]
    y_train = y.iloc[:15]
    y_test = y.iloc[15:]
    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples (first 15 excercise features)")
    print(f"Testing set: {len(X_test)} samples (last 3 excercise features - your holdout set!)")
    print(f"\nNOTE: We're NOT scaling features here so coefficients are easy to interpret!")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, feature_names):
    # TODO: Create a LinearRegression model
    model = LinearRegression()
    # TODO: Train the model using fit()
    model.fit(X_train, y_train)
    # TODO: Print the intercept
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    # TODO: Print each coefficient with its feature name
    #       Hint: use zip(feature_names, model.coef_)
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    print(f"\nEquation:")
    equation = f"Calories Burned = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    # TODO: Print the full equation in readable format
    print(equation)
    # TODO: Return the trained model
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    # Makes predictions on X_test
    predictions = model.predict(X_test)
    # Calculates R² score
    r2 = r2_score(y_test, predictions)
    # Calculates MSE and RMSE
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of calories burned variation")
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    # Calculates and prints feature importance
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    # Returns predictions
    return predictions

def compare_predictions(y_test, predicyions, num_examples=5):
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Calories Burned':<15} {'Predicted Calories Burned':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    # TODO: For the first num_examples:
    #       - Get actual and predicted calories burned
    #       - Calculate error (actual - predicted)
    #       - Calculate percentage error
    #       - Print in a nice formatted table
    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100

        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")

def make_prediction(model, age, duration, heartrate):
    input_df = pd.DataFrame([[age, duration, heartrate]],
                            columns=['Age', 'Duration', 'HeartRate'])

    prediction = model.predict(input_df)[0]

    print("\n=== New Prediction ===")
    print(f"Age: {age}, Duration: {duration} min, Heart Rate: {heartrate} bpm")
    print(f"Predicted Calories Burned: {prediction:.2f}")

    return prediction

if __name__ == "__main__":
    print("=" * 70)
    print("CALORIES BURNED PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore
    # TODO: Call load_and_explore_data() with 'house_prices.csv'
    data = load_and_explore_data('house_prices.csv')
    # Step 2: Visualize features
    # TODO: Call visualize_features() with the data
    visualize_features(data)
    # Step 3: Prepare features
    # TODO: Call prepare_features() and store X and y
    X, y = prepare_features(data)
    # Step 4: Split data
    # TODO: Call split_data() and store X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(x, y)
    # Step 5: Train model
    # TODO: Call train_model() with training data and feature names (X.columns)
    model = train_model(X_train, y_train, X.columns)
    # Step 6: Evaluate model
    # TODO: Call evaluate_model() with model, test data, and feature names
    predictions = evaluate_model(model, X_test, y_test, X.columns)
    # Step 7: Compare predictions
    # TODO: Call compare_predictions() showing first 10 examples
    compare_predictions(y_test, predictions)
    # Step 8: Make a new prediction
    # TODO: Call make_prediction() for a house of your choice
    make_predictions(model, 45, 3, 0)

    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part2_writeup.md!")
    print("=" * 70)