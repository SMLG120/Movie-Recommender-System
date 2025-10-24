import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from offline_eval import evaluate_xgboost_regression

def test_regression_metrics():
    """Test XGBoost regression metrics with synthetic data"""
    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get metrics
    metrics = evaluate_xgboost_regression(model, X_test, y_test)
    
    # Test metric properties
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['r2'] <= 1.0
    
    return metrics

if __name__ == '__main__':
    # Run quick test
    metrics = test_regression_metrics()
    print("\nXGBoost Regression Metrics on Test Data:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
