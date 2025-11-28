"""
MLflow Tracker Utility for Titan V8
Multi-Agent Volatility Prediction Research Project

Provides a robust wrapper class for experiment tracking and model logging.

Usage:
    from src.utils.tracker import MLTracker
"""

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MLTracker:
    """
    A wrapper class for MLflow experiment tracking.
    
    Handles experiment setup, parameter logging, metric calculation,
    and model artifact storage for volatility prediction experiments.
    
    Attributes:
        experiment_name (str): Name of the MLflow experiment.
        experiment_id (str): ID of the created/retrieved experiment.
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize the MLTracker with an experiment name.
        
        Args:
            experiment_name: Name for the MLflow experiment.
                           Creates new experiment if doesn't exist.
        """
        self.experiment_name = experiment_name
        
        # Set or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
    
    def log_params(self, params: dict) -> None:
        """
        Log hyperparameters to the active MLflow run.
        
        Args:
            params: Dictionary of parameter names and values.
        
        Example:
            tracker.log_params({
                'learning_rate': 0.01,
                'n_estimators': 100,
                'max_depth': 5
            })
        """
        mlflow.log_params(params)
    
    def log_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        step: int = None
    ) -> dict:
        """
        Calculate and log prediction metrics to MLflow.
        
        Computes and logs:
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - R2: R-Squared Score
            - Directional_Accuracy: % of times sign(y_pred_diff) == sign(y_true_diff)
        
        Args:
            y_true: Ground truth target values.
            y_pred: Predicted values.
            step: Optional step number for metric versioning.
        
        Returns:
            Dictionary containing all calculated metrics.
        
        Example:
            metrics = tracker.log_metrics(y_test, predictions, step=epoch)
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Calculate standard regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate directional accuracy
        # Measures how often the predicted direction matches actual direction
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
        
        # Log to MLflow
        mlflow.log_metrics(metrics, step=step)
        
        return metrics
    
    def _calculate_directional_accuracy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy between predictions and ground truth.
        
        Directional accuracy measures the percentage of times the predicted
        direction of change matches the actual direction of change.
        
        Args:
            y_true: Ground truth target values.
            y_pred: Predicted values.
        
        Returns:
            Directional accuracy as a percentage (0-100).
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate differences (direction of change)
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        
        # Get signs of differences
        true_signs = np.sign(y_true_diff)
        pred_signs = np.sign(y_pred_diff)
        
        # Calculate percentage of matching signs
        matches = np.sum(true_signs == pred_signs)
        total = len(true_signs)
        
        directional_accuracy = (matches / total) * 100.0
        
        return directional_accuracy
    
    def log_model(self, model, artifact_path: str) -> None:
        """
        Log a trained model to MLflow.
        
        Args:
            model: A trained sklearn-compatible model.
            artifact_path: Path within the artifact store for the model.
        
        Example:
            tracker.log_model(trained_model, 'lightgbm_volatility')
        """
        mlflow.sklearn.log_model(model, artifact_path)
    
    def start_run(self, run_name: str = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run.
        
        Returns:
            Active MLflow run context.
        """
        return mlflow.start_run(run_name=run_name)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

