import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

logs_dir = Path("./.logs")
logs_dir.mkdir(exist_ok=True)
LOG_FILE = logs_dir / f"control_system_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


class SharedBuffer:
    def __init__(self):
        self.history = []
        self.best_params = None
        self.best_metrics = {'mse': float('inf')}
        self.controller_type = None
        self.scenario = None
        self.param_ranges = None
        self.latest_juror_feedback = None
        self.system_name = None
        self.system_description = None
        self.control_objective = None
        self.current_scenario_metrics = None
        self.total_metrics = {'tokens_in': 0, 'tokens_out': 0, 'time': 0.0, 'cost': 0.0}

        # NEW: Store target metrics for scoring
        self.target_metrics = None

        # NEW: Configurable weights for performance scoring
        self.performance_weights = {
            'mse': 0.25,
            'settling_time': 0.30,
            'overshoot': 0.15,
            'ss_error': 0.20,
            'control_effort': 0.10
        }

    def calculate_performance_score(self, metrics: dict, target_metrics: dict = None) -> float:
        """
        Calculate a weighted performance score for ranking attempts.
        Lower score is better (like a cost function).

        Args:
            metrics: Performance metrics from simulation
            target_metrics: Optional target thresholds for normalization

        Returns:
            float: Weighted performance score (lower is better)
        """
        # If unstable, return worst possible score
        if not metrics.get('stable', False):
            return float('inf')

        # Use stored target_metrics if not provided
        if target_metrics is None:
            target_metrics = self.target_metrics or {}

        weights = self.performance_weights
        score = 0.0

        # MSE component
        mse = metrics.get('mse', float('inf'))
        if np.isfinite(mse):
            score += weights['mse'] * mse
        else:
            return float('inf')

        # Settling time component (penalize infinite settling time heavily)
        ts = metrics.get('settling_time', float('inf'))
        if np.isfinite(ts):
            # Normalize by target if available
            ts_target = target_metrics.get('settling_time', 5.0)
            score += weights['settling_time'] * (ts / ts_target)
        else:
            # Massive penalty for infinite settling time (non-convergent response)
            return float('inf')  # Treat as completely unacceptable

        # Overshoot component
        overshoot = metrics.get('overshoot', 0.0)
        if np.isfinite(overshoot):
            os_target = max(target_metrics.get('overshoot', 10.0), 1.0)
            score += weights['overshoot'] * (overshoot / os_target)

        # Steady-state error component
        ss_error = metrics.get('ss_error', 0.0)
        if np.isfinite(ss_error):
            score += weights['ss_error'] * abs(ss_error) * 10.0  # Scale up for visibility

        # Control effort component
        control_effort = metrics.get('control_effort', 0.0)
        if np.isfinite(control_effort):
            score += weights['control_effort'] * (control_effort / 10.0)

        return score

    def add_entry(self, params, metrics, trajectory, control_signals, errors, feedback=None):
        entry = {
            'timestamp': time.time(),
            'iteration': len(self.history) + 1,
            'params': params,
            'metrics': metrics,
            'trajectory': trajectory,
            'control_signals': control_signals,
            'errors': errors,
            'feedback': feedback,
            'param_ranges': self.param_ranges,
            'performance_score': self.calculate_performance_score(metrics, self.target_metrics)
            # Use stored target_metrics
        }
        self.history.append(entry)

        # Update best based on performance score with proper filtering
        # Only update if finite settling time and better score
        if (np.isfinite(metrics.get('settling_time', float('inf'))) and
                entry['performance_score'] < self.best_metrics.get('performance_score', float('inf'))):
            self.best_params = params
            self.best_metrics = metrics.copy()
            self.best_metrics['performance_score'] = entry['performance_score']

    def get_last_entry(self):
        return self.history[-1] if self.history else None

    def get_entries(self, n=3):
        return self.history[-n:] if len(self.history) >= n else self.history

    def get_best_entries(self, n=3, target_metrics: dict = None):
        """
        Get n best performing entries based on weighted performance score.

        Args:
            n: Number of best entries to return
            target_metrics: Optional target thresholds for score calculation

        Returns:
            list: Best n entries sorted by performance score (best first)
        """
        if not self.history:
            return []

        # Use stored target_metrics if not provided
        if target_metrics is None:
            target_metrics = self.target_metrics

        # Calculate scores for all entries
        scored_entries = []
        for entry in self.history:
            # Recalculate score with current target_metrics if needed
            if target_metrics != self.target_metrics or 'performance_score' not in entry:
                score = self.calculate_performance_score(entry['metrics'], target_metrics)
            else:
                score = entry.get('performance_score', float('inf'))
            scored_entries.append((score, entry))

        # Sort by score (lower is better) and return top n
        sorted_entries = sorted(scored_entries, key=lambda x: x[0])
        return [entry for score, entry in sorted_entries[:n]]

    def clear_history(self):
        """Clear history for new scenario or controller type"""
        self.history = []
        self.best_params = None
        self.best_metrics = {'mse': float('inf')}

    def set_performance_weights(self, weights: dict):
        """
        Set custom weights for performance scoring.

        Args:
            weights: Dictionary with keys: mse, settling_time, overshoot, ss_error, control_effort
        """
        self.performance_weights.update(weights)


def log_to_file(message, also_print=False):
    """Write message to log file and optionally print to console"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

    if also_print:
        print(message)


def set_global_seed(seed: int):
    """Set random seed for all random number generators."""
    np.random.seed(seed)
    # Add other RNG seeds if needed (e.g., random.seed(seed))


def calculate_performance_score(self, metrics: dict, target_metrics: dict = None) -> float:
    """
    Calculate a weighted performance score for ranking attempts.
    Lower score is better (like a cost function).

    Args:
        metrics: Performance metrics from simulation
        target_metrics: Optional target thresholds for normalization

    Returns:
        float: Weighted performance score (lower is better)
    """
    # If unstable, return worst possible score
    if not metrics.get('stable', False):
        return float('inf')

    # NEW: If settling time is infinite, heavily penalize (indicates non-convergence)
    if not np.isfinite(metrics.get('settling_time', 0)):
        return float('inf')

    # Default weights (can be adjusted based on your priorities)
    weights = {
        'mse': 0.25,
        'settling_time': 0.30,
        'overshoot': 0.15,
        'ss_error': 0.20,
        'control_effort': 0.10
    }

    # Normalize and calculate weighted score
    score = 0.0

    # MSE component
    mse = metrics.get('mse', float('inf'))
    if np.isfinite(mse):
        score += weights['mse'] * mse
    else:
        return float('inf')

    # Settling time component (penalize infinite settling time heavily)
    ts = metrics.get('settling_time', float('inf'))
    if np.isfinite(ts):
        # Normalize by target if available, otherwise use raw value
        ts_target = target_metrics.get('settling_time', 5.0) if target_metrics else 5.0
        score += weights['settling_time'] * (ts / ts_target)
    else:
        # Heavy penalty for infinite settling time
        score += weights['settling_time'] * 10.0

    # Overshoot component
    overshoot = metrics.get('overshoot', 0.0)
    if np.isfinite(overshoot):
        os_target = target_metrics.get('overshoot', 10.0) if target_metrics else 10.0
        score += weights['overshoot'] * (overshoot / max(os_target, 1.0))

    # Steady-state error component
    ss_error = metrics.get('ss_error', 0.0)
    if np.isfinite(ss_error):
        score += weights['ss_error'] * abs(ss_error)

    # Control effort component (optional, can help avoid aggressive control)
    control_effort = metrics.get('control_effort', 0.0)
    if np.isfinite(control_effort):
        score += weights['control_effort'] * (control_effort / 10.0)  # Normalize by typical value

    return score


def get_best_entries(self, n=3, target_metrics: dict = None):
    """
    Get n best performing entries based on weighted performance score.
    Filters out entries with infinite settling time or other poor characteristics.

    Args:
        n: Number of best entries to return
        target_metrics: Optional target thresholds for score calculation

    Returns:
        list: Best n entries sorted by performance score (best first)
    """
    if not self.history:
        return []

    # Use stored target_metrics if not provided
    if target_metrics is None:
        target_metrics = self.target_metrics

    # Calculate scores for all entries, filtering out poor performers
    scored_entries = []
    for entry in self.history:
        metrics = entry['metrics']

        # CRITICAL FILTER: Skip entries with infinite settling time
        # These indicate controllers that never reach steady state
        if not np.isfinite(metrics.get('settling_time', float('inf'))):
            continue

        # CRITICAL FILTER: Skip unstable controllers
        if not metrics.get('stable', False):
            continue

        # Recalculate score with current target_metrics if needed
        if target_metrics != self.target_metrics or 'performance_score' not in entry:
            score = self.calculate_performance_score(entry['metrics'], target_metrics)
        else:
            score = entry.get('performance_score', float('inf'))

        # Only include entries with finite scores
        if np.isfinite(score):
            scored_entries.append((score, entry))

    # If no valid entries found, return empty list
    if not scored_entries:
        return []

    # Sort by score (lower is better) and return top n
    sorted_entries = sorted(scored_entries, key=lambda x: x[0])
    return [entry for score, entry in sorted_entries[:n]]