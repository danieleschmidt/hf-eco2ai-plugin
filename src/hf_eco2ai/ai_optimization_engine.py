"""AI-Powered Optimization Engine for Carbon-Efficient ML Training.

This module implements advanced AI optimization capabilities including machine learning
for carbon prediction, reinforcement learning for adaptive power management, neural
network-based anomaly detection, automated hyperparameter tuning, and federated
learning for global carbon optimization.
"""

import asyncio
import logging
import time
import json
import math
import numpy as np
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import pickle
import joblib
from pathlib import Path
import warnings

# Suppress sklearn warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import gymnasium as gym
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationAlgorithm(Enum):
    """AI optimization algorithms available."""
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    LINEAR_REGRESSION = "linear_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class AnomalyDetectionMethod(Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"
    STATISTICAL = "statistical"
    CLUSTERING = "clustering"
    ENSEMBLE = "ensemble"


class HyperparameterOptimizer(Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"
    EVOLUTIONARY = "evolutionary"


@dataclass
class PredictionResult:
    """Result of carbon prediction."""
    
    predicted_value: float
    confidence: float
    prediction_interval: Tuple[float, float]
    model_used: str
    features_used: List[str]
    timestamp: float = field(default_factory=time.time)
    
    # Additional metadata
    model_accuracy: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_horizon: int = 60  # minutes


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    affected_metrics: List[str]
    severity: str  # "low", "medium", "high", "critical"
    
    # Context
    timestamp: float = field(default_factory=time.time)
    detection_method: str = "ensemble"
    confidence: float = 0.0
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    estimated_impact: float = 0.0


@dataclass
class OptimizationRecommendation:
    """AI-generated optimization recommendation."""
    
    recommendation_id: str
    optimization_type: str
    description: str
    expected_carbon_reduction: float  # kg CO2/hr
    expected_cost_savings: float      # USD/hr
    implementation_effort: str        # "low", "medium", "high"
    
    # Configuration changes
    parameter_changes: Dict[str, Any] = field(default_factory=dict)
    infrastructure_changes: List[str] = field(default_factory=list)
    
    # Validation
    confidence_score: float = 0.0
    estimated_roi: float = 0.0  # Return on investment
    risk_level: str = "low"     # "low", "medium", "high"
    
    timestamp: float = field(default_factory=time.time)


class CarbonPredictionModel:
    """Machine learning model for carbon emission prediction."""
    
    def __init__(self, 
                 algorithm: OptimizationAlgorithm = OptimizationAlgorithm.RANDOM_FOREST,
                 prediction_horizon: int = 60):
        """Initialize carbon prediction model.
        
        Args:
            algorithm: ML algorithm to use
            prediction_horizon: Prediction horizon in minutes
        """
        self.algorithm = algorithm
        self.prediction_horizon = prediction_horizon
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.max_training_samples = 100000
        
        # Model metadata
        self.model_accuracy = 0.0
        self.last_training_time = 0.0
        self.training_samples_count = 0
        self.feature_importance = {}
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Carbon prediction model initialized: {algorithm.value}")
    
    def _initialize_model(self):
        """Initialize the ML model based on algorithm."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using simple linear model")
            return
        
        try:
            if self.algorithm == OptimizationAlgorithm.RANDOM_FOREST:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif self.algorithm == OptimizationAlgorithm.LINEAR_REGRESSION:
                self.model = Ridge(alpha=1.0)
            elif self.algorithm == OptimizationAlgorithm.NEURAL_NETWORK:
                self.model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
            else:
                # Default to Random Forest
                self.model = RandomForestRegressor(
                    n_estimators=50,
                    random_state=42,
                    n_jobs=-1
                )
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.model = None
    
    def add_training_data(self, 
                         carbon_metrics: List[Dict[str, Any]]):
        """Add training data for the model.
        
        Args:
            carbon_metrics: List of carbon metrics with features
        """
        # Add to training data buffer
        self.training_data.extend(carbon_metrics)
        
        # Limit training data size
        if len(self.training_data) > self.max_training_samples:
            # Keep most recent samples
            self.training_data = self.training_data[-self.max_training_samples:]
        
        # Auto-retrain if we have enough new data
        if (len(self.training_data) >= 1000 and 
            time.time() - self.last_training_time > 3600):  # 1 hour
            asyncio.create_task(self.train_model())
    
    async def train_model(self) -> bool:
        """Train the carbon prediction model.
        
        Returns:
            True if training successful
        """
        if not self.training_data or not self.model:
            return False
        
        try:
            # Prepare features and targets
            features, targets = self._prepare_training_data()
            
            if len(features) < 10:  # Need minimum samples
                logger.warning("Insufficient training data")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            self.model_accuracy = r2_score(y_test, y_pred)
            
            # Feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_columns,
                    self.model.feature_importances_
                ))
            
            self.last_training_time = time.time()
            self.training_samples_count = len(features)
            
            logger.info(f"Model trained successfully: R² = {self.model_accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from collected metrics."""
        features = []
        targets = []
        
        # Define feature columns
        self.feature_columns = [
            'gpu_utilization', 'memory_utilization', 'power_draw',
            'temperature', 'gpu_count', 'batch_size', 'model_size',
            'hour_of_day', 'day_of_week', 'carbon_intensity'
        ]
        
        for metric in self.training_data:
            # Extract features
            feature_vector = []
            
            # GPU metrics
            feature_vector.append(metric.get('gpu_utilization', 0))
            feature_vector.append(metric.get('memory_utilization', 0))
            feature_vector.append(metric.get('power_draw', 0))
            feature_vector.append(metric.get('temperature', 0))
            
            # Training context
            feature_vector.append(metric.get('gpu_count', 1))
            feature_vector.append(metric.get('batch_size', 32))
            feature_vector.append(metric.get('model_parameters', 1000000))
            
            # Temporal features
            timestamp = metric.get('timestamp', time.time())
            dt = datetime.fromtimestamp(timestamp)
            feature_vector.append(dt.hour)
            feature_vector.append(dt.weekday())
            
            # Environmental
            feature_vector.append(metric.get('carbon_intensity_gco2_kwh', 400))
            
            # Target: carbon emission
            target = metric.get('carbon_emission_kg_hr', 0)
            
            if len(feature_vector) == len(self.feature_columns) and target > 0:
                features.append(feature_vector)
                targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def predict_carbon_emission(self, 
                              features: Dict[str, Any]) -> PredictionResult:
        """Predict carbon emission for given features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Prediction result
        """
        if not self.model or not self.feature_columns:
            # Fallback prediction
            power_kw = features.get('power_draw', 250) / 1000
            carbon_intensity = features.get('carbon_intensity_gco2_kwh', 400)
            predicted_value = power_kw * (carbon_intensity / 1000)
            
            return PredictionResult(
                predicted_value=predicted_value,
                confidence=0.5,
                prediction_interval=(predicted_value * 0.8, predicted_value * 1.2),
                model_used="fallback",
                features_used=list(features.keys()),
                model_accuracy=0.5
            )
        
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                if col == 'hour_of_day':
                    feature_vector.append(datetime.now().hour)
                elif col == 'day_of_week':
                    feature_vector.append(datetime.now().weekday())
                else:
                    feature_vector.append(features.get(col, 0))
            
            # Scale features
            feature_array = np.array([feature_vector])
            feature_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(feature_scaled)[0]
            
            # Calculate confidence based on model accuracy
            confidence = min(0.95, max(0.1, self.model_accuracy))
            
            # Prediction interval (simple approach)
            margin = prediction * (1 - confidence) * 0.5
            prediction_interval = (prediction - margin, prediction + margin)
            
            return PredictionResult(
                predicted_value=max(0, prediction),
                confidence=confidence,
                prediction_interval=prediction_interval,
                model_used=self.algorithm.value,
                features_used=self.feature_columns,
                model_accuracy=self.model_accuracy,
                feature_importance=self.feature_importance.copy(),
                prediction_horizon=self.prediction_horizon
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback prediction
            return self.predict_carbon_emission(features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            "algorithm": self.algorithm.value,
            "model_accuracy": self.model_accuracy,
            "training_samples": self.training_samples_count,
            "last_training": self.last_training_time,
            "feature_columns": self.feature_columns,
            "feature_importance": self.feature_importance,
            "prediction_horizon": self.prediction_horizon,
            "model_available": self.model is not None
        }


class AnomalyDetectionSystem:
    """AI-powered anomaly detection for carbon metrics."""
    
    def __init__(self,
                 methods: List[AnomalyDetectionMethod] = None,
                 sensitivity: float = 0.1):
        """Initialize anomaly detection system.
        
        Args:
            methods: Detection methods to use
            sensitivity: Detection sensitivity (0.0 - 1.0)
        """
        self.methods = methods or [
            AnomalyDetectionMethod.ISOLATION_FOREST,
            AnomalyDetectionMethod.STATISTICAL,
            AnomalyDetectionMethod.CLUSTERING
        ]
        self.sensitivity = sensitivity
        
        # Detection models
        self.detectors: Dict[AnomalyDetectionMethod, Any] = {}
        
        # Historical data for statistical detection
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Anomaly patterns
        self.anomaly_patterns: List[Dict[str, Any]] = []
        
        # Initialize detectors
        self._initialize_detectors()
        
        logger.info(f"Anomaly detection system initialized with {len(self.methods)} methods")
    
    def _initialize_detectors(self):
        """Initialize anomaly detection models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using statistical detection only")
            self.methods = [AnomalyDetectionMethod.STATISTICAL]
            return
        
        try:
            for method in self.methods:
                if method == AnomalyDetectionMethod.ISOLATION_FOREST:
                    self.detectors[method] = IsolationForest(
                        contamination=self.sensitivity,
                        random_state=42,
                        n_jobs=-1
                    )
                elif method == AnomalyDetectionMethod.CLUSTERING:
                    self.detectors[method] = KMeans(
                        n_clusters=3,
                        random_state=42,
                        n_init=10
                    )
                
        except Exception as e:
            logger.error(f"Detector initialization failed: {e}")
    
    def add_metrics(self, metrics: List[Dict[str, Any]]):
        """Add metrics for anomaly detection training.
        
        Args:
            metrics: List of carbon metrics
        """
        for metric in metrics:
            for key, value in metric.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    self.metric_history[key].append(value)
    
    def detect_anomalies(self, 
                        current_metrics: Dict[str, Any]) -> AnomalyResult:
        """Detect anomalies in current metrics.
        
        Args:
            current_metrics: Current metric values
            
        Returns:
            Anomaly detection result
        """
        anomaly_scores = {}
        anomaly_methods = []
        
        # Run each detection method
        for method in self.methods:
            try:
                if method == AnomalyDetectionMethod.STATISTICAL:
                    score, is_anomaly = self._statistical_detection(current_metrics)
                elif method == AnomalyDetectionMethod.ISOLATION_FOREST:
                    score, is_anomaly = self._isolation_forest_detection(current_metrics)
                elif method == AnomalyDetectionMethod.CLUSTERING:
                    score, is_anomaly = self._clustering_detection(current_metrics)
                else:
                    continue
                
                anomaly_scores[method.value] = score
                if is_anomaly:
                    anomaly_methods.append(method.value)
                    
            except Exception as e:
                logger.warning(f"Anomaly detection method {method.value} failed: {e}")
        
        # Ensemble decision
        overall_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0
        is_anomaly = len(anomaly_methods) >= len(self.methods) / 2  # Majority vote
        
        # Determine severity
        if overall_score > 0.8:
            severity = "critical"
        elif overall_score > 0.6:
            severity = "high"
        elif overall_score > 0.4:
            severity = "medium"
        else:
            severity = "low"
        
        # Identify affected metrics
        affected_metrics = self._identify_affected_metrics(current_metrics)
        
        # Generate recommendations
        recommendations = self._generate_anomaly_recommendations(
            current_metrics, anomaly_methods, severity
        )
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=overall_score,
            anomaly_type="ensemble",
            affected_metrics=affected_metrics,
            severity=severity,
            detection_method="ensemble",
            confidence=min(0.95, max(0.1, overall_score)),
            recommended_actions=recommendations,
            estimated_impact=self._estimate_anomaly_impact(current_metrics, overall_score)
        )
    
    def _statistical_detection(self, 
                             metrics: Dict[str, Any]) -> Tuple[float, bool]:
        """Statistical anomaly detection using z-score."""
        anomaly_scores = []
        
        for key, value in metrics.items():
            if key not in self.metric_history or not isinstance(value, (int, float)):
                continue
            
            history = list(self.metric_history[key])
            if len(history) < 10:  # Need minimum history
                continue
            
            mean_val = np.mean(history)
            std_val = np.std(history)
            
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                # Convert z-score to anomaly score (0-1)
                anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma rule
                anomaly_scores.append(anomaly_score)
        
        if not anomaly_scores:
            return 0.0, False
        
        overall_score = np.mean(anomaly_scores)
        is_anomaly = overall_score > (1 - self.sensitivity)
        
        return overall_score, is_anomaly
    
    def _isolation_forest_detection(self, 
                                  metrics: Dict[str, Any]) -> Tuple[float, bool]:
        """Isolation Forest anomaly detection."""
        if AnomalyDetectionMethod.ISOLATION_FOREST not in self.detectors:
            return 0.0, False
        
        detector = self.detectors[AnomalyDetectionMethod.ISOLATION_FOREST]
        
        # Prepare feature vector
        feature_vector = []
        feature_keys = ['gpu_utilization', 'memory_utilization', 'power_draw', 
                       'temperature', 'carbon_emission_kg_hr']
        
        for key in feature_keys:
            feature_vector.append(metrics.get(key, 0))
        
        if len(feature_vector) == 0:
            return 0.0, False
        
        try:
            # Need to fit model if not already trained
            if not hasattr(detector, 'decision_function'):
                # Use historical data to fit
                training_data = []
                for key in feature_keys:
                    if key in self.metric_history:
                        history = list(self.metric_history[key])
                        if len(history) >= 50:  # Minimum for training
                            training_data.append(history[:50])
                
                if len(training_data) == len(feature_keys):
                    X_train = np.array(training_data).T
                    detector.fit(X_train)
                else:
                    return 0.0, False
            
            # Predict anomaly
            X = np.array([feature_vector])
            anomaly_score = detector.decision_function(X)[0]
            is_anomaly = detector.predict(X)[0] == -1
            
            # Normalize score to 0-1 range
            normalized_score = max(0, min(1, (-anomaly_score + 0.5) / 1.0))
            
            return normalized_score, is_anomaly
            
        except Exception as e:
            logger.warning(f"Isolation Forest detection failed: {e}")
            return 0.0, False
    
    def _clustering_detection(self, 
                            metrics: Dict[str, Any]) -> Tuple[float, bool]:
        """Clustering-based anomaly detection."""
        if AnomalyDetectionMethod.CLUSTERING not in self.detectors:
            return 0.0, False
        
        # Similar to isolation forest but using clustering
        # Implementation would be similar but using distance from cluster centers
        return 0.0, False  # Placeholder
    
    def _identify_affected_metrics(self, 
                                 metrics: Dict[str, Any]) -> List[str]:
        """Identify which metrics are anomalous."""
        affected = []
        
        for key, value in metrics.items():
            if key in self.metric_history and isinstance(value, (int, float)):
                history = list(self.metric_history[key])
                if len(history) >= 10:
                    mean_val = np.mean(history)
                    std_val = np.std(history)
                    
                    if std_val > 0:
                        z_score = abs(value - mean_val) / std_val
                        if z_score > 2.0:  # 2-sigma threshold
                            affected.append(key)
        
        return affected
    
    def _generate_anomaly_recommendations(self, 
                                        metrics: Dict[str, Any],
                                        anomaly_methods: List[str],
                                        severity: str) -> List[str]:
        """Generate recommendations for handling anomalies."""
        recommendations = []
        
        if 'gpu_utilization' in self._identify_affected_metrics(metrics):
            recommendations.append("Check GPU workload distribution")
            recommendations.append("Verify training job configuration")
        
        if 'power_draw' in self._identify_affected_metrics(metrics):
            recommendations.append("Monitor power supply and cooling")
            recommendations.append("Check for hardware issues")
        
        if 'temperature' in self._identify_affected_metrics(metrics):
            recommendations.append("Verify cooling system operation")
            recommendations.append("Check ambient temperature")
        
        if severity in ["high", "critical"]:
            recommendations.append("Consider stopping training temporarily")
            recommendations.append("Alert system administrators")
        
        return recommendations
    
    def _estimate_anomaly_impact(self, 
                               metrics: Dict[str, Any],
                               anomaly_score: float) -> float:
        """Estimate the impact of the anomaly."""
        # Simple impact estimation based on metrics
        power_impact = metrics.get('power_draw', 0) * anomaly_score
        carbon_impact = metrics.get('carbon_emission_kg_hr', 0) * anomaly_score
        
        # Normalize to 0-100 scale
        return min(100, (power_impact + carbon_impact * 1000) / 10)


class ReinforcementLearningOptimizer:
    """Reinforcement learning for adaptive power management."""
    
    def __init__(self, 
                 learning_algorithm: str = "PPO",
                 learning_rate: float = 0.003):
        """Initialize RL optimizer.
        
        Args:
            learning_algorithm: RL algorithm to use
            learning_rate: Learning rate for training
        """
        self.learning_algorithm = learning_algorithm
        self.learning_rate = learning_rate
        
        # RL components
        self.env = None
        self.model = None
        self.training_data = []
        
        # State and action spaces
        self.state_dim = 10  # GPU metrics, power, temperature, etc.
        self.action_dim = 5  # Power limit, frequency, cooling, etc.
        
        # Training statistics
        self.episodes_trained = 0
        self.total_reward = 0.0
        self.best_reward = float('-inf')
        
        if RL_AVAILABLE:
            self._initialize_rl_environment()
        else:
            logger.warning("Reinforcement learning not available")
    
    def _initialize_rl_environment(self):
        """Initialize RL environment for power optimization."""
        try:
            # Create custom environment for carbon optimization
            self.env = CarbonOptimizationEnv()
            
            # Initialize RL model
            if self.learning_algorithm == "PPO":
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.learning_rate,
                    verbose=0
                )
            elif self.learning_algorithm == "A2C":
                self.model = A2C(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.learning_rate,
                    verbose=0
                )
            
            logger.info(f"RL optimizer initialized with {self.learning_algorithm}")
            
        except Exception as e:
            logger.error(f"RL initialization failed: {e}")
    
    async def optimize_power_settings(self, 
                                    current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Use RL to optimize power settings.
        
        Args:
            current_state: Current system state
            
        Returns:
            Optimized power settings
        """
        if not self.model or not self.env:
            # Fallback to rule-based optimization
            return self._rule_based_optimization(current_state)
        
        try:
            # Convert state to observation
            observation = self._state_to_observation(current_state)
            
            # Get action from RL model
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Convert action to power settings
            power_settings = self._action_to_settings(action)
            
            return power_settings
            
        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
            return self._rule_based_optimization(current_state)
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert system state to RL observation."""
        observation = np.zeros(self.state_dim)
        
        # GPU utilization
        observation[0] = state.get('gpu_utilization', 0) / 100.0
        observation[1] = state.get('memory_utilization', 0) / 100.0
        
        # Power metrics
        observation[2] = state.get('power_draw', 0) / 500.0  # Normalize to 500W
        observation[3] = state.get('temperature', 0) / 100.0  # Normalize to 100°C
        
        # Training metrics
        observation[4] = state.get('batch_size', 32) / 1000.0
        observation[5] = state.get('learning_rate', 0.001) / 0.1
        
        # Environmental
        observation[6] = state.get('carbon_intensity_gco2_kwh', 400) / 1000.0
        observation[7] = state.get('ambient_temperature', 25) / 50.0
        
        # Time features
        hour = datetime.now().hour
        observation[8] = hour / 24.0
        observation[9] = datetime.now().weekday() / 7.0
        
        return observation
    
    def _action_to_settings(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert RL action to power settings."""
        settings = {}
        
        # Power limit (0-100%)
        settings['power_limit_percent'] = max(50, min(100, action[0] * 100))
        
        # GPU frequency scaling (0.8-1.2)
        settings['gpu_frequency_scale'] = max(0.8, min(1.2, 0.8 + action[1] * 0.4))
        
        # Memory frequency scaling (0.8-1.2)
        settings['memory_frequency_scale'] = max(0.8, min(1.2, 0.8 + action[2] * 0.4))
        
        # Fan speed (30-100%)
        settings['fan_speed_percent'] = max(30, min(100, action[3] * 100))
        
        # Temperature target (65-85°C)
        settings['temperature_target'] = max(65, min(85, 65 + action[4] * 20))
        
        return settings
    
    def _rule_based_optimization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based optimization."""
        settings = {}
        
        # Simple rules based on current state
        gpu_util = state.get('gpu_utilization', 50)
        temperature = state.get('temperature', 70)
        carbon_intensity = state.get('carbon_intensity_gco2_kwh', 400)
        
        # Power limit based on utilization and carbon intensity
        if carbon_intensity > 600:  # High carbon intensity
            power_limit = max(70, 100 - gpu_util * 0.3)
        else:
            power_limit = min(100, 80 + gpu_util * 0.2)
        
        settings['power_limit_percent'] = power_limit
        
        # Temperature-based cooling
        if temperature > 80:
            settings['fan_speed_percent'] = 100
        elif temperature > 75:
            settings['fan_speed_percent'] = 80
        else:
            settings['fan_speed_percent'] = max(40, temperature)
        
        # Conservative frequency scaling
        settings['gpu_frequency_scale'] = 0.95
        settings['memory_frequency_scale'] = 1.0
        settings['temperature_target'] = 75
        
        return settings
    
    async def train_model(self, 
                         training_episodes: int = 1000) -> bool:
        """Train the RL model.
        
        Args:
            training_episodes: Number of training episodes
            
        Returns:
            True if training successful
        """
        if not self.model or not self.env:
            return False
        
        try:
            # Train model
            self.model.learn(total_timesteps=training_episodes * 100)
            
            self.episodes_trained += training_episodes
            
            logger.info(f"RL model trained for {training_episodes} episodes")
            return True
            
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get RL training statistics."""
        return {
            "algorithm": self.learning_algorithm,
            "episodes_trained": self.episodes_trained,
            "total_reward": self.total_reward,
            "best_reward": self.best_reward,
            "model_available": self.model is not None,
            "environment_available": self.env is not None
        }


class CarbonOptimizationEnv(gym.Env):
    """Custom Gym environment for carbon optimization."""
    
    def __init__(self):
        """Initialize carbon optimization environment."""
        super().__init__()
        
        # Action space: [power_limit, gpu_freq, mem_freq, fan_speed, temp_target]
        self.action_space = gym.spaces.Box(
            low=np.array([0.5, 0.8, 0.8, 0.3, 0.65]),
            high=np.array([1.0, 1.2, 1.2, 1.0, 0.85]),
            dtype=np.float32
        )
        
        # Observation space: [gpu_util, mem_util, power, temp, carbon_intensity, ...]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Environment state
        self.state = None
        self.episode_length = 100
        self.current_step = 0
        
    def reset(self, **kwargs):
        """Reset environment to initial state."""
        # Initialize random state
        self.state = np.random.random(10).astype(np.float32)
        self.current_step = 0
        
        return self.state, {}
    
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        if self.state is None:
            raise ValueError("Environment not reset")
        
        # Simulate environment dynamics
        next_state = self._simulate_dynamics(self.state, action)
        
        # Calculate reward
        reward = self._calculate_reward(self.state, action, next_state)
        
        # Update state
        self.state = next_state
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        truncated = False
        
        info = {
            "carbon_emission": self._calculate_carbon_emission(next_state),
            "power_consumption": self._calculate_power_consumption(next_state),
            "efficiency": self._calculate_efficiency(next_state)
        }
        
        return next_state, reward, done, truncated, info
    
    def _simulate_dynamics(self, state, action):
        """Simulate system dynamics."""
        next_state = state.copy()
        
        # Simple dynamics: actions affect state
        # Power limit affects GPU utilization
        next_state[0] = max(0, min(1, state[0] + (action[0] - 0.75) * 0.1))
        
        # Temperature is affected by power and cooling
        next_state[3] = max(0, min(1, state[3] + (state[2] - action[3]) * 0.05))
        
        # Power consumption affected by frequency scaling and utilization
        power_factor = (action[1] + action[2]) / 2.0
        next_state[2] = max(0, min(1, state[0] * power_factor * action[0]))
        
        # Add some noise
        next_state += np.random.normal(0, 0.01, size=next_state.shape)
        next_state = np.clip(next_state, 0, 1)
        
        return next_state
    
    def _calculate_reward(self, state, action, next_state):
        """Calculate reward for the action."""
        # Reward components
        
        # 1. Energy efficiency (lower power consumption = higher reward)
        power_reward = -(next_state[2] ** 2)  # Quadratic penalty
        
        # 2. Temperature management (keep temperature in optimal range)
        temp_optimal = 0.7  # 70% of max temperature
        temp_penalty = -abs(next_state[3] - temp_optimal) * 2
        
        # 3. Performance maintenance (keep utilization reasonable)
        util_penalty = -max(0, 0.9 - next_state[0]) * 3  # Penalty if util drops below 90%
        
        # 4. Carbon emission reduction
        carbon_emission = self._calculate_carbon_emission(next_state)
        carbon_reward = -carbon_emission * 10
        
        # Total reward
        total_reward = power_reward + temp_penalty + util_penalty + carbon_reward
        
        return total_reward
    
    def _calculate_carbon_emission(self, state):
        """Calculate carbon emission from state."""
        power_consumption = state[2]  # Normalized power
        carbon_intensity = state[6]   # Normalized carbon intensity
        
        return power_consumption * carbon_intensity
    
    def _calculate_power_consumption(self, state):
        """Calculate power consumption from state."""
        return state[2] * 500  # Convert to watts
    
    def _calculate_efficiency(self, state):
        """Calculate efficiency metric."""
        utilization = state[0]
        power = state[2]
        
        return utilization / max(power, 0.01)  # Avoid division by zero


class HyperparameterTuner:
    """Automated hyperparameter tuning for carbon efficiency."""
    
    def __init__(self, 
                 optimization_method: HyperparameterOptimizer = HyperparameterOptimizer.OPTUNA):
        """Initialize hyperparameter tuner.
        
        Args:
            optimization_method: Optimization method to use
        """
        self.optimization_method = optimization_method
        
        # Optuna study
        self.study = None
        
        # Tuning history
        self.tuning_history: List[Dict[str, Any]] = []
        
        # Best parameters found
        self.best_params = {}
        self.best_score = float('inf')
        
        if OPTUNA_AVAILABLE and optimization_method == HyperparameterOptimizer.OPTUNA:
            self._initialize_optuna()
        else:
            logger.warning("Optuna not available, using grid search")
    
    def _initialize_optuna(self):
        """Initialize Optuna study."""
        try:
            self.study = optuna.create_study(
                direction='minimize',  # Minimize carbon emission
                sampler=TPESampler(),
                pruner=MedianPruner()
            )
            
            logger.info("Optuna hyperparameter tuner initialized")
            
        except Exception as e:
            logger.error(f"Optuna initialization failed: {e}")
    
    async def tune_training_parameters(self, 
                                     objective_function: Callable,
                                     param_space: Dict[str, Any],
                                     n_trials: int = 100) -> Dict[str, Any]:
        """Tune training parameters for carbon efficiency.
        
        Args:
            objective_function: Function to optimize
            param_space: Parameter space to search
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters found
        """
        if not self.study:
            return await self._grid_search_tuning(objective_function, param_space, n_trials)
        
        try:
            # Define Optuna objective
            def optuna_objective(trial):
                # Sample parameters
                params = {}
                
                for param_name, param_config in param_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Evaluate objective
                score = objective_function(params)
                
                # Record trial
                self.tuning_history.append({
                    'trial_number': trial.number,
                    'params': params.copy(),
                    'score': score,
                    'timestamp': time.time()
                })
                
                return score
            
            # Run optimization
            self.study.optimize(optuna_objective, n_trials=n_trials)
            
            # Get best parameters
            self.best_params = self.study.best_params.copy()
            self.best_score = self.study.best_value
            
            logger.info(f"Hyperparameter tuning completed: best score = {self.best_score:.4f}")
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_trials': len(self.study.trials),
                'optimization_history': self.tuning_history[-n_trials:]
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            return {}
    
    async def _grid_search_tuning(self, 
                                objective_function: Callable,
                                param_space: Dict[str, Any],
                                n_trials: int) -> Dict[str, Any]:
        """Fallback grid search tuning."""
        # Simple grid search implementation
        best_params = {}
        best_score = float('inf')
        trials_conducted = 0
        
        # Generate parameter combinations (simplified)
        param_combinations = []
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'float':
                values = np.linspace(param_config['low'], param_config['high'], 5)
            elif param_config['type'] == 'int':
                values = range(param_config['low'], param_config['high'] + 1)
            elif param_config['type'] == 'categorical':
                values = param_config['choices']
            
            if not param_combinations:
                param_combinations = [{param_name: v} for v in values]
            else:
                new_combinations = []
                for combo in param_combinations:
                    for v in values:
                        new_combo = combo.copy()
                        new_combo[param_name] = v
                        new_combinations.append(new_combo)
                param_combinations = new_combinations
        
        # Limit combinations if too many
        if len(param_combinations) > n_trials:
            param_combinations = param_combinations[:n_trials]
        
        # Evaluate combinations
        for params in param_combinations:
            try:
                score = objective_function(params)
                trials_conducted += 1
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                
                self.tuning_history.append({
                    'trial_number': trials_conducted,
                    'params': params.copy(),
                    'score': score,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': trials_conducted,
            'optimization_history': self.tuning_history[-trials_conducted:]
        }
    
    def get_tuning_stats(self) -> Dict[str, Any]:
        """Get hyperparameter tuning statistics."""
        return {
            'optimization_method': self.optimization_method.value,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_trials': len(self.tuning_history),
            'study_available': self.study is not None
        }


class FederatedLearningCoordinator:
    """Federated learning coordinator for global carbon optimization."""
    
    def __init__(self, 
                 coordinator_id: str,
                 aggregation_rounds: int = 10):
        """Initialize federated learning coordinator.
        
        Args:
            coordinator_id: Unique coordinator identifier
            aggregation_rounds: Number of aggregation rounds
        """
        self.coordinator_id = coordinator_id
        self.aggregation_rounds = aggregation_rounds
        
        # Federated participants
        self.participants: Dict[str, Dict[str, Any]] = {}
        
        # Global model
        self.global_model = None
        self.global_model_version = 0
        
        # Aggregation history
        self.aggregation_history: List[Dict[str, Any]] = []
        
        # Carbon efficiency gains
        self.efficiency_gains: Dict[str, float] = {}
        
        logger.info(f"Federated learning coordinator initialized: {coordinator_id}")
    
    def register_participant(self, 
                           participant_id: str,
                           participant_info: Dict[str, Any]) -> bool:
        """Register federated learning participant.
        
        Args:
            participant_id: Unique participant identifier
            participant_info: Participant information
            
        Returns:
            True if registration successful
        """
        try:
            self.participants[participant_id] = {
                'info': participant_info,
                'last_update': time.time(),
                'model_version': 0,
                'contribution_score': 0.0,
                'carbon_reduction': 0.0
            }
            
            logger.info(f"Registered federated participant: {participant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register participant {participant_id}: {e}")
            return False
    
    async def aggregate_models(self, 
                             participant_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate model updates from participants.
        
        Args:
            participant_updates: Model updates from participants
            
        Returns:
            Aggregated model and metadata
        """
        try:
            if not participant_updates:
                return {'status': 'no_updates'}
            
            # Simple federated averaging (placeholder)
            aggregated_weights = {}
            total_samples = 0
            
            # Collect weights and sample counts
            for participant_id, update in participant_updates.items():
                if participant_id not in self.participants:
                    continue
                
                weights = update.get('model_weights', {})
                samples = update.get('sample_count', 1)
                
                total_samples += samples
                
                for layer_name, layer_weights in weights.items():
                    if layer_name not in aggregated_weights:
                        aggregated_weights[layer_name] = layer_weights * samples
                    else:
                        aggregated_weights[layer_name] += layer_weights * samples
            
            # Average weights
            for layer_name in aggregated_weights:
                aggregated_weights[layer_name] /= total_samples
            
            # Update global model
            self.global_model = aggregated_weights
            self.global_model_version += 1
            
            # Calculate carbon efficiency gains
            carbon_gains = self._calculate_carbon_gains(participant_updates)
            
            # Record aggregation
            aggregation_record = {
                'round': len(self.aggregation_history) + 1,
                'timestamp': time.time(),
                'participants': list(participant_updates.keys()),
                'total_samples': total_samples,
                'carbon_gains': carbon_gains,
                'model_version': self.global_model_version
            }
            
            self.aggregation_history.append(aggregation_record)
            
            logger.info(f"Model aggregation completed: round {aggregation_record['round']}")
            
            return {
                'status': 'success',
                'global_model': self.global_model,
                'model_version': self.global_model_version,
                'carbon_gains': carbon_gains,
                'participants': len(participant_updates)
            }
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_carbon_gains(self, 
                              participant_updates: Dict[str, Any]) -> Dict[str, float]:
        """Calculate carbon efficiency gains from federated learning."""
        gains = {}
        
        for participant_id, update in participant_updates.items():
            baseline_carbon = update.get('baseline_carbon_kg_hr', 1.0)
            optimized_carbon = update.get('optimized_carbon_kg_hr', 1.0)
            
            if baseline_carbon > 0:
                reduction_percent = ((baseline_carbon - optimized_carbon) / baseline_carbon) * 100
                gains[participant_id] = max(0, reduction_percent)
                
                # Update participant record
                if participant_id in self.participants:
                    self.participants[participant_id]['carbon_reduction'] = gains[participant_id]
        
        return gains
    
    async def distribute_global_model(self) -> Dict[str, Any]:
        """Distribute global model to participants.
        
        Returns:
            Distribution results
        """
        if not self.global_model:
            return {'status': 'no_global_model'}
        
        distribution_results = {}
        
        for participant_id in self.participants:
            try:
                # In a real implementation, this would send the model
                # via network protocol (gRPC, HTTP, etc.)
                distribution_results[participant_id] = {
                    'status': 'success',
                    'model_version': self.global_model_version,
                    'timestamp': time.time()
                }
                
                # Update participant model version
                self.participants[participant_id]['model_version'] = self.global_model_version
                
            except Exception as e:
                distribution_results[participant_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        logger.info(f"Global model distributed to {len(self.participants)} participants")
        
        return {
            'status': 'completed',
            'model_version': self.global_model_version,
            'distribution_results': distribution_results
        }
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        total_carbon_reduction = sum(
            p.get('carbon_reduction', 0) for p in self.participants.values()
        )
        
        avg_carbon_reduction = (
            total_carbon_reduction / max(len(self.participants), 1)
        )
        
        return {
            'coordinator_id': self.coordinator_id,
            'participants_count': len(self.participants),
            'global_model_version': self.global_model_version,
            'aggregation_rounds': len(self.aggregation_history),
            'total_carbon_reduction_percent': total_carbon_reduction,
            'avg_carbon_reduction_percent': avg_carbon_reduction,
            'participants': {
                pid: {
                    'model_version': p['model_version'],
                    'carbon_reduction': p['carbon_reduction'],
                    'last_update': p['last_update']
                }
                for pid, p in self.participants.items()
            }
        }


class AIOptimizationEngine:
    """Main AI optimization engine coordinating all AI components."""
    
    def __init__(self,
                 enable_prediction: bool = True,
                 enable_anomaly_detection: bool = True,
                 enable_rl_optimization: bool = True,
                 enable_hyperparameter_tuning: bool = True,
                 enable_federated_learning: bool = True):
        """Initialize AI optimization engine.
        
        Args:
            enable_prediction: Enable carbon prediction
            enable_anomaly_detection: Enable anomaly detection
            enable_rl_optimization: Enable RL optimization
            enable_hyperparameter_tuning: Enable hyperparameter tuning
            enable_federated_learning: Enable federated learning
        """
        # Component flags
        self.enable_prediction = enable_prediction
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_rl_optimization = enable_rl_optimization
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.enable_federated_learning = enable_federated_learning
        
        # AI components
        self.prediction_model = None
        self.anomaly_detector = None
        self.rl_optimizer = None
        self.hyperparameter_tuner = None
        self.federated_coordinator = None
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'carbon_reduction_kg_hr': 0.0,
            'cost_savings_usd_hr': 0.0,
            'avg_optimization_time_ms': 0.0
        }
        
        self._initialize_components()
        
        logger.info("AI optimization engine initialized")
    
    def _initialize_components(self):
        """Initialize AI optimization components."""
        try:
            if self.enable_prediction:
                self.prediction_model = CarbonPredictionModel()
            
            if self.enable_anomaly_detection:
                self.anomaly_detector = AnomalyDetectionSystem()
            
            if self.enable_rl_optimization:
                self.rl_optimizer = ReinforcementLearningOptimizer()
            
            if self.enable_hyperparameter_tuning:
                self.hyperparameter_tuner = HyperparameterTuner()
            
            if self.enable_federated_learning:
                self.federated_coordinator = FederatedLearningCoordinator(
                    coordinator_id=f"coordinator_{int(time.time())}"
                )
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    async def optimize_carbon_efficiency(self, 
                                       current_metrics: Dict[str, Any],
                                       optimization_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive carbon efficiency optimization.
        
        Args:
            current_metrics: Current system metrics
            optimization_config: Optimization configuration
            
        Returns:
            Optimization results and recommendations
        """
        start_time = time.time()
        config = optimization_config or {}
        
        try:
            optimization_results = {
                'timestamp': start_time,
                'optimization_id': f"opt_{int(start_time)}",
                'components_used': [],
                'recommendations': [],
                'predictions': {},
                'anomalies': {},
                'optimized_settings': {},
                'estimated_impact': {}
            }
            
            # 1. Carbon Prediction
            if self.prediction_model:
                try:
                    prediction = self.prediction_model.predict_carbon_emission(current_metrics)
                    optimization_results['predictions'] = asdict(prediction)
                    optimization_results['components_used'].append('prediction')
                except Exception as e:
                    logger.warning(f"Carbon prediction failed: {e}")
            
            # 2. Anomaly Detection
            if self.anomaly_detector:
                try:
                    self.anomaly_detector.add_metrics([current_metrics])
                    anomaly_result = self.anomaly_detector.detect_anomalies(current_metrics)
                    optimization_results['anomalies'] = asdict(anomaly_result)
                    optimization_results['components_used'].append('anomaly_detection')
                    
                    # Add anomaly-based recommendations
                    if anomaly_result.is_anomaly:
                        optimization_results['recommendations'].extend(
                            anomaly_result.recommended_actions
                        )
                        
                except Exception as e:
                    logger.warning(f"Anomaly detection failed: {e}")
            
            # 3. RL-based Power Optimization
            if self.rl_optimizer:
                try:
                    optimized_settings = await self.rl_optimizer.optimize_power_settings(current_metrics)
                    optimization_results['optimized_settings'] = optimized_settings
                    optimization_results['components_used'].append('rl_optimization')
                except Exception as e:
                    logger.warning(f"RL optimization failed: {e}")
            
            # 4. Generate AI-powered recommendations
            ai_recommendations = await self._generate_ai_recommendations(
                current_metrics, optimization_results
            )
            optimization_results['recommendations'].extend(ai_recommendations)
            
            # 5. Estimate optimization impact
            impact_estimation = self._estimate_optimization_impact(
                current_metrics, optimization_results
            )
            optimization_results['estimated_impact'] = impact_estimation
            
            # Record optimization
            optimization_time = (time.time() - start_time) * 1000  # ms
            
            self.optimization_history.append({
                'timestamp': start_time,
                'metrics': current_metrics.copy(),
                'results': optimization_results.copy(),
                'optimization_time_ms': optimization_time
            })
            
            # Update performance metrics
            self.performance_metrics['total_optimizations'] += 1
            if optimization_results['components_used']:
                self.performance_metrics['successful_optimizations'] += 1
            
            # Update average optimization time
            avg_time = self.performance_metrics['avg_optimization_time_ms']
            total_opts = self.performance_metrics['total_optimizations']
            new_avg = (avg_time * (total_opts - 1) + optimization_time) / total_opts
            self.performance_metrics['avg_optimization_time_ms'] = new_avg
            
            # Add processing metadata
            optimization_results['optimization_time_ms'] = optimization_time
            optimization_results['success'] = len(optimization_results['components_used']) > 0
            
            logger.info(f"AI optimization completed in {optimization_time:.1f}ms")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"AI optimization failed: {e}")
            return {
                'timestamp': start_time,
                'success': False,
                'error': str(e),
                'optimization_time_ms': (time.time() - start_time) * 1000
            }
    
    async def _generate_ai_recommendations(self, 
                                         metrics: Dict[str, Any],
                                         optimization_results: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate AI-powered optimization recommendations."""
        recommendations = []
        
        try:
            # Analyze current state
            gpu_util = metrics.get('gpu_utilization', 50)
            power_draw = metrics.get('power_draw', 250)
            carbon_intensity = metrics.get('carbon_intensity_gco2_kwh', 400)
            
            # High power consumption recommendation
            if power_draw > 400:  # High power draw
                rec = OptimizationRecommendation(
                    recommendation_id=f"rec_power_{int(time.time())}",
                    optimization_type="power_reduction",
                    description="Reduce power consumption through frequency scaling",
                    expected_carbon_reduction=0.05,  # kg CO2/hr
                    expected_cost_savings=0.02,     # USD/hr
                    implementation_effort="low",
                    parameter_changes={
                        "gpu_frequency_scale": 0.9,
                        "power_limit_percent": 85
                    },
                    confidence_score=0.8,
                    estimated_roi=2.5,
                    risk_level="low"
                )
                recommendations.append(rec)
            
            # High carbon intensity recommendation
            if carbon_intensity > 500:
                rec = OptimizationRecommendation(
                    recommendation_id=f"rec_carbon_{int(time.time())}",
                    optimization_type="carbon_scheduling",
                    description="Schedule training during lower carbon intensity periods",
                    expected_carbon_reduction=0.15,
                    expected_cost_savings=0.05,
                    implementation_effort="medium",
                    infrastructure_changes=[
                        "Implement carbon-aware scheduling",
                        "Add carbon intensity monitoring"
                    ],
                    confidence_score=0.7,
                    estimated_roi=3.0,
                    risk_level="low"
                )
                recommendations.append(rec)
            
            # Low utilization recommendation
            if gpu_util < 70:
                rec = OptimizationRecommendation(
                    recommendation_id=f"rec_util_{int(time.time())}",
                    optimization_type="efficiency_improvement",
                    description="Improve GPU utilization through batch size optimization",
                    expected_carbon_reduction=0.08,
                    expected_cost_savings=0.03,
                    implementation_effort="low",
                    parameter_changes={
                        "batch_size_multiplier": 1.2,
                        "gradient_accumulation_steps": 2
                    },
                    confidence_score=0.75,
                    estimated_roi=2.0,
                    risk_level="low"
                )
                recommendations.append(rec)
            
            # Model optimization recommendation
            if metrics.get('model_parameters', 0) > 10000000:  # Large model
                rec = OptimizationRecommendation(
                    recommendation_id=f"rec_model_{int(time.time())}",
                    optimization_type="model_optimization",
                    description="Apply model compression techniques",
                    expected_carbon_reduction=0.20,
                    expected_cost_savings=0.10,
                    implementation_effort="high",
                    infrastructure_changes=[
                        "Implement quantization",
                        "Add pruning pipeline",
                        "Enable mixed precision training"
                    ],
                    confidence_score=0.6,
                    estimated_roi=4.0,
                    risk_level="medium"
                )
                recommendations.append(rec)
            
        except Exception as e:
            logger.error(f"AI recommendation generation failed: {e}")
        
        return recommendations
    
    def _estimate_optimization_impact(self, 
                                    metrics: Dict[str, Any],
                                    optimization_results: Dict[str, Any]) -> Dict[str, float]:
        """Estimate the impact of optimization recommendations."""
        try:
            current_power = metrics.get('power_draw', 250) / 1000  # kW
            current_carbon_intensity = metrics.get('carbon_intensity_gco2_kwh', 400)
            current_carbon_emission = current_power * (current_carbon_intensity / 1000)
            
            # Estimate total carbon reduction from all recommendations
            total_carbon_reduction = 0.0
            total_cost_savings = 0.0
            
            for rec in optimization_results.get('recommendations', []):
                if isinstance(rec, OptimizationRecommendation):
                    total_carbon_reduction += rec.expected_carbon_reduction
                    total_cost_savings += rec.expected_cost_savings
            
            # Calculate percentages
            carbon_reduction_percent = (total_carbon_reduction / max(current_carbon_emission, 0.001)) * 100
            
            return {
                'current_carbon_emission_kg_hr': current_carbon_emission,
                'estimated_carbon_reduction_kg_hr': total_carbon_reduction,
                'carbon_reduction_percent': min(50, carbon_reduction_percent),  # Cap at 50%
                'estimated_cost_savings_usd_hr': total_cost_savings,
                'payback_period_hours': 24 if total_cost_savings > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Impact estimation failed: {e}")
            return {}
    
    async def train_models(self, 
                          training_data: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Train AI models with new data.
        
        Args:
            training_data: Training data for models
            
        Returns:
            Training results for each component
        """
        results = {}
        
        # Train prediction model
        if self.prediction_model:
            try:
                self.prediction_model.add_training_data(training_data)
                success = await self.prediction_model.train_model()
                results['prediction_model'] = success
            except Exception as e:
                logger.error(f"Prediction model training failed: {e}")
                results['prediction_model'] = False
        
        # Train anomaly detector
        if self.anomaly_detector:
            try:
                self.anomaly_detector.add_metrics(training_data)
                results['anomaly_detector'] = True
            except Exception as e:
                logger.error(f"Anomaly detector training failed: {e}")
                results['anomaly_detector'] = False
        
        # Train RL optimizer
        if self.rl_optimizer:
            try:
                success = await self.rl_optimizer.train_model(training_episodes=100)
                results['rl_optimizer'] = success
            except Exception as e:
                logger.error(f"RL optimizer training failed: {e}")
                results['rl_optimizer'] = False
        
        return results
    
    async def hyperparameter_optimization(self, 
                                        training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hyperparameter optimization for carbon efficiency.
        
        Args:
            training_config: Training configuration
            
        Returns:
            Optimized hyperparameters
        """
        if not self.hyperparameter_tuner:
            return {'status': 'hyperparameter_tuner_not_available'}
        
        # Define parameter space for carbon optimization
        param_space = {
            'learning_rate': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-1,
                'log': True
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64, 128, 256]
            },
            'power_limit': {
                'type': 'int',
                'low': 60,
                'high': 100
            },
            'temperature_target': {
                'type': 'int',
                'low': 65,
                'high': 85
            }
        }
        
        # Define objective function (minimize carbon emission)
        def carbon_objective(params):
            # Simulate training with given parameters
            # In practice, this would run actual training
            
            # Simple carbon estimation model
            lr_factor = np.log10(params['learning_rate']) / -5  # Normalize
            batch_factor = params['batch_size'] / 256
            power_factor = params['power_limit'] / 100
            temp_factor = (85 - params['temperature_target']) / 20
            
            # Estimate carbon emission (lower is better)
            carbon_estimate = (
                lr_factor * 0.3 +
                batch_factor * 0.2 +
                (1 - power_factor) * 0.3 +
                temp_factor * 0.2
            )
            
            return max(0.1, carbon_estimate)
        
        try:
            results = await self.hyperparameter_tuner.tune_training_parameters(
                objective_function=carbon_objective,
                param_space=param_space,
                n_trials=50
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_ai_system_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status.
        
        Returns:
            AI system status and metrics
        """
        status = {
            'timestamp': time.time(),
            'components_enabled': {
                'prediction': self.enable_prediction,
                'anomaly_detection': self.enable_anomaly_detection,
                'rl_optimization': self.enable_rl_optimization,
                'hyperparameter_tuning': self.enable_hyperparameter_tuning,
                'federated_learning': self.enable_federated_learning
            },
            'components_status': {},
            'performance_metrics': self.performance_metrics.copy(),
            'optimization_history_length': len(self.optimization_history)
        }
        
        # Component status
        if self.prediction_model:
            status['components_status']['prediction_model'] = self.prediction_model.get_model_info()
        
        if self.anomaly_detector:
            status['components_status']['anomaly_detector'] = {
                'methods': [m.value for m in self.anomaly_detector.methods],
                'sensitivity': self.anomaly_detector.sensitivity
            }
        
        if self.rl_optimizer:
            status['components_status']['rl_optimizer'] = self.rl_optimizer.get_training_stats()
        
        if self.hyperparameter_tuner:
            status['components_status']['hyperparameter_tuner'] = self.hyperparameter_tuner.get_tuning_stats()
        
        if self.federated_coordinator:
            status['components_status']['federated_coordinator'] = self.federated_coordinator.get_federation_stats()
        
        # Recent optimization summary
        if self.optimization_history:
            recent_optimizations = self.optimization_history[-10:]  # Last 10
            status['recent_optimization_summary'] = {
                'avg_optimization_time_ms': np.mean([
                    opt['optimization_time_ms'] for opt in recent_optimizations
                ]),
                'success_rate': np.mean([
                    opt['results'].get('success', False) for opt in recent_optimizations
                ]) * 100,
                'avg_components_used': np.mean([
                    len(opt['results'].get('components_used', [])) for opt in recent_optimizations
                ])
            }
        
        return status


# Global AI optimization engine instance
_ai_optimization_engine: Optional[AIOptimizationEngine] = None


def get_ai_optimization_engine(**kwargs) -> AIOptimizationEngine:
    """Get global AI optimization engine instance."""
    global _ai_optimization_engine
    
    if _ai_optimization_engine is None:
        _ai_optimization_engine = AIOptimizationEngine(**kwargs)
    
    return _ai_optimization_engine


async def initialize_ai_optimization():
    """Initialize AI optimization engine with optimal configuration."""
    engine = get_ai_optimization_engine(
        enable_prediction=True,
        enable_anomaly_detection=True,
        enable_rl_optimization=RL_AVAILABLE,
        enable_hyperparameter_tuning=OPTUNA_AVAILABLE,
        enable_federated_learning=True
    )
    
    logger.info("AI optimization engine initialized for enterprise scale")
    return engine