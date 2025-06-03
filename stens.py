import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Define the EnsembleModel class
class EnsembleModel:
    def __init__(self, rf_model, xgb_model, scaler):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.scaler = scaler

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        rf_pred = self.rf_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        # Ensemble predictions (simple average)
        ensemble_pred = (rf_pred + xgb_pred) / 2
        return ensemble_pred

# Training and model creation logic
def train_and_save_model(data_path='augmented_arecanut_dataset5k.csv', model_path='ensemble_model.pkl'):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Preprocessing function to convert scientific notation
    def convert_scientific_notation(x):
        if isinstance(x, str) and 'x 10' in x:
            parts = x.split('x 10^')
            return float(parts[0]) * (10 ** int(parts[1]))
        return float(x)

    # Convert scientific notation to float
    df['Beneficial_Microbes (CFU/g)'] = df['Beneficial_Microbes (CFU/g)'].apply(convert_scientific_notation)
    df['Harmful_Microbes (CFU/g)'] = df['Harmful_Microbes (CFU/g)'].apply(convert_scientific_notation)
    df['Soil_Organic_Carbon'] = df['Soil_Organic_Carbon'].str.replace('%', '').astype(float) / 100.0

    # Define numeric columns
    numeric_cols = [
        'Soil_pH', 'N (Nitrogen)', 'P (Phosphorus)', 'K (Potassium)', 
        'Organic_Matter (kg compost)', 'Temperature (°C)', 'Rainfall (mm)', 
        'Elevation (m)', 'Beneficial_Microbes (CFU/g)', 'Harmful_Microbes (CFU/g)', 
        'Microbial_Biomass_C (g/kg)', 'Soil_Organic_Carbon'
    ]

    # Convert to numeric and handle errors
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Fill missing values with column means
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Split data into features and target variable, excluding Sample_ID
    X = df.drop(['Crop_Yield (kg/palm)', 'Sample_ID'], axis=1)
    y = df['Crop_Yield (kg/palm)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)

    # Define and train the XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)

    # Predicting the yields for test data using both models
    rf_pred = rf_model.predict(X_test_scaled)
    xgb_pred = xgb_model.predict(X_test_scaled)

    # Ensemble predictions (simple average)
    ensemble_pred = (rf_pred + xgb_pred) / 2

    # Calculate and print performance metrics for ensemble
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    print(f"Ensemble Mean Squared Error: {ensemble_mse:.2f}")
    print(f"Ensemble Mean Absolute Error: {ensemble_mae:.2f}")
    print(f"Ensemble R^2 Score: {ensemble_r2:.2f}")

    # Feature Importances
    rf_feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    xgb_feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plotting feature importances for both models
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.barh(rf_feature_importances['Feature'], rf_feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from Random Forest')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.barh(xgb_feature_importances['Feature'], xgb_feature_importances['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from XGBoost')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig("feature_importances.png")  # Save instead of show for server compatibility
    print("Feature importance chart saved as feature_importances.png")

    # Create an instance of the ensemble model
    ensemble_model = EnsembleModel(rf_model, xgb_model, scaler)

    # Dump the ensemble model to a file
    joblib.dump(ensemble_model, model_path)
    print(f"Ensemble model dumped successfully to {model_path}!")

# Run training only if the script is executed directly
if __name__ == "__main__":
    train_and_save_model()