"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Ashley Quiroz

Dataset: [Calories dataset]
Predicting: [How many calories you burn based on different features]
Features: [Age, Duration, and Heartrate]"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'calories.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    data = pd.read_csv(filename)
    # Prints the first 5 rows
    print("=== Calories Burned Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    # Prints the shape of the dataset
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    # Prints basic statistics for ALL columns
    print(f"\nBasic statistics:")
    print(data.describe())
    # Prints the column names
    print(f"\nColumn names: {list(data.columns)}")
    return data

    import matplotlib.pyplot as plt



def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
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


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)

    # Define features and target
    feature_columns = ['Age', 'Duration', 'HeartRate']
    target_column = 'Calories'

    # Separate X and y
    X = data[feature_columns]
    y = data[target_column]

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Print sizes
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing target shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)

    feature_names = X_train.columns

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n=== Model Training Complete ===")
    print(f"Intercept: {model.intercept_:.2f}")

    print("\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")

    print("\nEquation:")
    equation = "Calories = "
    equation += " + ".join(
        [f"{coef:.2f}×{name}" for name, coef in zip(feature_names, model.coef_)]
    )
    equation += f" + {model.intercept_:.2f}"
    print(equation)

    return model



def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    feature_names = X_test.columns
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
    
    


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    input_df = pd.DataFrame([[age, duration, heartrate]],
                            columns=['Age', 'Duration', 'HeartRate'])

    prediction = model.predict(input_df)[0]

    print("\n=== New Prediction ===")
    print(f"Age: {age}, Duration: {duration} min, Heart Rate: {heartrate} bpm")
    print(f"Predicted Calories Burned: {prediction:.2f}")

    return prediction



if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

