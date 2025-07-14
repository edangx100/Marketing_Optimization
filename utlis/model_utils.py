import pandas as pd
import xgboost as xgb
import optuna
from typing import Tuple, List
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, f1_score
import numpy as np


def predict_propensity(model: BaseEstimator, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Predict propensity scores using a trained model.
    
    Parameters:
    -----------
    model : BaseEstimator
        Trained classification model
    df : pd.DataFrame
        Input dataframe containing features
    feature_cols : List[str]
        List of feature column names to use for prediction
        
    Returns:
    --------
    np.ndarray
        Array of propensity scores (probability of positive class)
    """
    X = df[feature_cols].fillna(0)
    return model.predict_proba(X)[:,1] 


def objective_revenue_xgb(trial: optuna.Trial, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                         y_train: pd.Series, y_val: pd.Series, random_state: int = 42) -> float:
    """
    Optuna objective function for XGBoost regression hyperparameter optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object for hyperparameter suggestions
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : pd.Series
        Training target values
    y_val : pd.Series
        Validation target values
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    float
        R² score on validation set
    """
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': random_state
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    
    return r2


def train_revenue_model_xgb_optuna(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                  y_train: pd.Series, y_val: pd.Series, 
                                  n_trials: int = 100, random_state: int = 42) -> Tuple[xgb.XGBRegressor, float, float, dict, optuna.Study]:
    """
    Train XGBoost regression model with Optuna hyperparameter optimization.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : pd.Series
        Training target values
    y_val : pd.Series
        Validation target values
    n_trials : int, default=100
        Number of Optuna trials for hyperparameter optimization
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (trained_model, r2_score, rmse_score, best_parameters, optuna_study)
    """
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(
        lambda trial: objective_revenue_xgb(trial, X_train, X_val, y_train, y_val, random_state),
        n_trials=n_trials
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    best_params['random_state'] = random_state
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate final model
    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    return model, r2, rmse, best_params, study


def objective_sales_xgb_f1(trial: optuna.Trial, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                           y_train: pd.Series, y_val: pd.Series, random_state: int = 42) -> float:
    """
    Optuna objective function for XGBoost classification hyperparameter optimization using F1 score.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object for hyperparameter suggestions
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : pd.Series
        Training target values
    y_val : pd.Series
        Validation target values
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    float
        F1 score on validation set
    """
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': random_state
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate using F1 score
    val_pred = model.predict(X_val)
    f1 = f1_score(y_val, val_pred)
    
    return f1


def train_sales_model_xgb_optuna_f1(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                   y_train: pd.Series, y_val: pd.Series, 
                                   n_trials: int = 100, random_state: int = 42) -> Tuple[xgb.XGBClassifier, float, float, dict, optuna.Study]:
    """
    Train XGBoost classification model with Optuna hyperparameter optimization using F1 score.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : pd.Series
        Training target values
    y_val : pd.Series
        Validation target values
    n_trials : int, default=100
        Number of Optuna trials for hyperparameter optimization
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (trained_model, f1_score, roc_auc_score, best_parameters, optuna_study)
    """
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(
        lambda trial: objective_sales_xgb_f1(trial, X_train, X_val, y_train, y_val, random_state),
        n_trials=n_trials
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    best_params['random_state'] = random_state
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate final model using both F1 and ROC-AUC for comparison
    val_pred = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)[:,1]
    f1 = f1_score(y_val, val_pred)
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    
    return model, f1, roc_auc, best_params, study 