import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, f1_score
import numpy as np
# from data_utils import scale_features
import xgboost as xgb
import optuna


# def predict_propensity(model, df, feature_cols, scaler=None):
def predict_propensity(model, df, feature_cols):
    X = df[feature_cols].fillna(0)
    # if scaler is not None:
    #     X = scaler.transform(X)
    return model.predict_proba(X)[:,1] 


# def train_revenue_model_dt(X_train_scaled, X_val_scaled, y_train, y_val, random_state=42):
def train_revenue_model_dt(X_train, X_val, y_train, y_val, random_state=42):
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return model, r2, rmse


# def train_sales_model_dt(X_train_scaled, X_val_scaled, y_train, y_val, random_state=42):
def train_sales_model_dt(X_train, X_val, y_train, y_val, random_state=42):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    val_pred_proba = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    return model, roc_auc 


def train_revenue_model_xgb(X_train, X_val, y_train, y_val, random_state=42):
    model = xgb.XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return model, r2, rmse 


def train_sales_model_xgb(X_train, X_val, y_train, y_val, random_state=42):
    model = xgb.XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    val_pred_proba = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    return model, roc_auc 


def train_revenue_model_rf(X_train, X_val, y_train, y_val, random_state=42):
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return model, r2, rmse 


def train_sales_model_rf(X_train, X_val, y_train, y_val, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    val_pred_proba = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    return model, roc_auc 


def objective_revenue_xgb(trial, X_train, X_val, y_train, y_val, random_state=42):
    """Optuna objective function for XGBoost regression hyperparameter optimization"""
    
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


def objective_sales_xgb(trial, X_train, X_val, y_train, y_val, random_state=42):
    """Optuna objective function for XGBoost classification hyperparameter optimization"""
    
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
    
    # Evaluate
    val_pred_proba = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    
    return roc_auc


def train_revenue_model_xgb_optuna(X_train, X_val, y_train, y_val, n_trials=100, random_state=42):
    """Train XGBoost regression model with Optuna hyperparameter optimization"""
    
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


def train_sales_model_xgb_optuna(X_train, X_val, y_train, y_val, n_trials=100, random_state=42):
    """Train XGBoost classification model with Optuna hyperparameter optimization"""
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(
        lambda trial: objective_sales_xgb(trial, X_train, X_val, y_train, y_val, random_state),
        n_trials=n_trials
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    best_params['random_state'] = random_state
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate final model
    val_pred_proba = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    
    return model, roc_auc, best_params, study


def objective_sales_xgb_f1(trial, X_train, X_val, y_train, y_val, random_state=42):
    """Optuna objective function for XGBoost classification hyperparameter optimization using F1 score"""
    
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


def train_sales_model_xgb_optuna_f1(X_train, X_val, y_train, y_val, n_trials=100, random_state=42):
    """Train XGBoost classification model with Optuna hyperparameter optimization using F1 score"""
    
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