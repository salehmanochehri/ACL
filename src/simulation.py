import numpy as np
from src.systems import GeneralDynamicalSystem
from typing import Optional


class SimulationRunner:
    def __init__(self, system_class: type[GeneralDynamicalSystem]):
        self.system_class = system_class
        self.system: Optional[GeneralDynamicalSystem] = None
        # Store configuration parameters
        self._config = {}

    def set_config(self, **kwargs):
        """Store configuration parameters to apply to all system instances"""
        self._config.update(kwargs)

    def set_scenario(self, scenario: dict) -> GeneralDynamicalSystem:
        self.system = self.system_class(scenario)
        # Apply stored configuration
        for key, value in self._config.items():
            if hasattr(self.system, key):
                setattr(self.system, key, value)
        return self.system

    def calculate_metrics(self, errors, control_signals):
        """Calculate performance metrics using the configured output channel"""
        metrics = {}
        t = np.arange(0, len(errors) * self.system.dt, self.system.dt)

        # errors is already the error for the output channel from simulations
        output_errors = errors

        # Mean Squared Error
        metrics['mse'] = np.mean(output_errors ** 2)
        metrics['rmse'] = np.sqrt(np.mean(output_errors ** 2))

        # Get initial error magnitude
        initial_error = np.abs(output_errors[0])

        # Define thresholds based on initial error
        # If initial error is very small, use absolute threshold
        if initial_error > 1e-6:
            settling_threshold = 0.05 * initial_error  # 5% of initial error
            stability_threshold = 0.20 * initial_error  # 20% of initial error for stability check
        else:
            settling_threshold = 0.05  # Absolute threshold
            stability_threshold = 0.20  # Absolute threshold

        # Settling time - time when system enters and stays within settling threshold
        settled_indices = np.where(np.abs(output_errors) < settling_threshold)[0]

        if len(settled_indices) > 0:
            # Find consecutive settled indices
            consecutive_groups = []
            current_group = [settled_indices[0]]

            for i in range(1, len(settled_indices)):
                if settled_indices[i] == settled_indices[i - 1] + 1:
                    current_group.append(settled_indices[i])
                else:
                    if len(current_group) > 0:
                        consecutive_groups.append(current_group)
                    current_group = [settled_indices[i]]

            if len(current_group) > 0:
                consecutive_groups.append(current_group)

            # Find the longest consecutive sequence that continues to the end
            for group in consecutive_groups:
                if group[-1] == len(output_errors) - 1:
                    metrics['settling_time'] = t[group[0]]
                    break
            else:
                metrics['settling_time'] = np.inf
        else:
            metrics['settling_time'] = np.inf

        # Calculate percentage overshoot
        # For regulation problems starting away from target
        if initial_error > 1e-6:
            # Track the actual output trajectory approaching the target
            # Overshoot occurs when we cross the target and go beyond

            # Find when we first get close to target (within 90% of initial error)
            approach_threshold = 0.1 * initial_error
            approach_indices = np.where(np.abs(output_errors) < approach_threshold)[0]

            if len(approach_indices) > 0:
                # Look for overshoot after first approach
                first_approach = approach_indices[0]

                # Check if error changes sign (crosses target)
                if first_approach < len(output_errors) - 1:
                    errors_after_approach = output_errors[first_approach:]

                    # If initial error was positive (output below target)
                    # Overshoot means error becomes negative (output above target)
                    # If initial error was negative (output above target)
                    # Overshoot means error becomes positive (output below target)

                    initial_sign = np.sign(output_errors[0])
                    crossed_indices = np.where(np.sign(errors_after_approach) == -initial_sign)[0]

                    if len(crossed_indices) > 0:
                        # Find maximum deviation beyond target
                        max_overshoot_error = np.max(np.abs(errors_after_approach[crossed_indices]))
                        # Express as percentage of initial error
                        metrics['overshoot'] = (max_overshoot_error / initial_error) * 100
                    else:
                        metrics['overshoot'] = 0.0
                else:
                    metrics['overshoot'] = 0.0
            else:
                metrics['overshoot'] = 0.0
        else:
            metrics['overshoot'] = 0.0

        # REDEFINED STABILITY CRITERIA:
        # System is stable if:
        # 1. Simulation completed without early termination (divergence check)
        # 2. Error remains bounded within acceptable limits for the final portion of simulation
        # 3. No extreme values that indicate numerical instability

        expected_steps = int(self.system.max_time / self.system.dt)
        simulation_completed = len(output_errors) >= expected_steps
        

        
        # Check if error remains within stability threshold for last 20% of simulation
        final_portion_length = max(int(0.2 * len(output_errors)), 10)  # At least 10 samples
        final_errors = output_errors[-final_portion_length:]

        # Stability conditions:
        # 1. All errors in final portion within stability threshold
        errors_bounded = np.all(np.abs(final_errors) < stability_threshold)
        
        
        # 2. No extreme values (numerical instability indicator)
        no_extreme_values = np.all(np.abs(output_errors) < 1000)  # Reasonable bound
        
        
        # 3. Check control signals are also bounded (not saturating wildly)
        control_bounded = np.all(np.abs(control_signals) < 1000)
        
        
        # Combined stability check
        if errors_bounded and no_extreme_values and control_bounded:
            metrics['stable'] = True
        else:
            metrics['stable'] = False
        
        # Optional: Add stability margin metric (how close to stability threshold)
        if len(final_errors) > 0:
            max_final_error = np.max(np.abs(final_errors))
            if initial_error > 1e-6:
                metrics['stability_margin'] = (stability_threshold - max_final_error) / stability_threshold * 100
            else:
                metrics['stability_margin'] = (stability_threshold - max_final_error) * 100

        # Rise time - time to reach within 5% of target
        rise_threshold = 0.05 * initial_error if initial_error > 1e-6 else 0.05
        rise_indices = np.where(np.abs(output_errors) < rise_threshold)[0]
        metrics['rise_time'] = t[rise_indices[0]] if rise_indices.size > 0 else np.inf

        # Zero-crossings (oscillations around target)
        zero_crossings = np.where(np.diff(np.signbit(output_errors)))[0]
        metrics['zero_crossings'] = len(zero_crossings)

        # Control effort (non-dimensional: fraction of maximum possible effort)
        max_abs_u = max(abs(self.system.min_control), abs(self.system.max_control))
        num_steps = len(control_signals)
        max_possible_effort = max_abs_u * num_steps
        metrics['control_effort'] = np.sum(np.abs(control_signals)) / max_possible_effort

        # Control signal zero-crossings
        control_zero_crossings = np.where(np.diff(np.signbit(control_signals)))[0]
        metrics['control_zero_crossings'] = len(control_zero_crossings)

        # Steady-state error
        metrics['ss_error'] = np.abs(np.mean(output_errors[-int(0.1 * len(output_errors)):]))  # Last 10% of simulation

        return metrics

    def evaluate_parameters(self, params, initial_state=None):
        """Evaluate controller parameters on the system

        Args:
            params: Controller parameters dict
            initial_state: Optional fixed initial state for reproducible simulations
        """
        try:
            # Check controller type and call appropriate simulation
            if isinstance(params, dict):
                fsf_keys = [f"K{i + 1}" for i in range(self.system.num_states)]
                if any(key in params for key in fsf_keys):
                    # FSF controller
                    K_values = [params.get(f"K{i + 1}", 0.0)
                                for i in range(self.system.num_states)]
                    trajectory, control_signals, errors = self.system.run_fsf_simulation(
                        K_values, initial_state=initial_state
                    )
                else:
                    # PID controller
                    Kp = params.get('Kp', 0.0)
                    Ki = params.get('Ki', 0.0)
                    Kd = params.get('Kd', 0.0)
                    trajectory, control_signals, errors = self.system.run_pid_simulation(
                        Kp, Ki, Kd, initial_state=initial_state
                    )

            metrics = self.calculate_metrics(errors, control_signals)
            return {
                'success': True,
                'metrics': metrics,
                'trajectory': trajectory,
                'control_signals': control_signals,
                'errors': errors,
                'initial_state': initial_state if initial_state is not None else None  # Store IC used
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_monte_carlo(self, params, num_runs=100):
        """Run Monte Carlo simulations and return aggregated metrics"""
        metrics_list = []
        for _ in range(num_runs):
            result = self.evaluate_parameters(params)
            if result['success']:
                metrics_list.append(result['metrics'])

        # Calculate statistics
        stats = {}
        if metrics_list:
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                stats[key] = {
                    'mean': np.nanmean(values),
                    'std': np.nanstd(values)
                }
        return stats
