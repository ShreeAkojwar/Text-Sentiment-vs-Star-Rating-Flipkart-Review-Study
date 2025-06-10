import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Load and prepare data
df = pd.read_csv("processed_data/aspect_sentiment_vader.csv")
df = df.dropna(subset=['quality', 'cost', 'delivery', 'flexibility', 'Rate'])

# Create polynomial features and interactions
poly = PolynomialFeatures(degree=2, include_bias=False)
features = ['quality', 'cost', 'delivery', 'flexibility']
X = df[features]
poly_features = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(features)

# Prepare target
y = df['Rate']

# Split data
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for better balance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'ElasticNet': ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.5)
}

# Train and evaluate models
print("\nModel Performance:")
print("-" * 50)

results = {}
predictions = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n{name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

# Create ensemble
weights = [results[model]['R2'] for model in models.keys()]
weights = np.array(weights)
weights = weights / weights.sum()  # Normalize weights

ensemble_pred = np.zeros_like(y_test, dtype=float)
for (name, pred), weight in zip(predictions.items(), weights):
    ensemble_pred += weight * pred

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

print("\nWeighted Ensemble:")
print(f"RMSE: {ensemble_rmse:.4f}")
print(f"MAE: {ensemble_mae:.4f}")
print(f"R² Score: {ensemble_r2:.4f}")

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Save results
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': ensemble_pred,
    'Error': y_test - ensemble_pred
})
results_df.to_csv('outputs/regression_results.csv', index=False)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ensemble_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Aspect Sentiment vs Rating: Prediction Analysis")
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/prediction_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot error distribution
plt.figure(figsize=(10, 6))
sns.histplot(results_df['Error'], bins=50, kde=True)
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Predicted - Actual Rating)')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Save summary
with open('outputs/model_summary.txt', 'w') as f:
    f.write("Aspect Sentiment vs Rating Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write("Individual Model Performance:\n")
    for name, metrics in results.items():
        f.write(f"\n{name}:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    f.write("\nEnsemble Model Performance:\n")
    f.write(f"RMSE: {ensemble_rmse:.4f}\n")
    f.write(f"MAE: {ensemble_mae:.4f}\n")
    f.write(f"R² Score: {ensemble_r2:.4f}\n")

# Save feature importance analysis
if hasattr(models['Random Forest'], 'feature_importances_'):
    importance = models['Random Forest'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
