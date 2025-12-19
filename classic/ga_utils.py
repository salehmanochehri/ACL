import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import pygad
from scipy.linalg import solve_continuous_are
import copy

# Import from your existing modules
from src.systems import (
    create_system, InvertedPendulum, GeneralDynamicalSystem,
    CustomDynamicalSystem, OctaveSISOSystem
)
from src.simulation import SimulationRunner


class GAOptimizer:
    """Genetic Algorithm optimizer for controller design"""

    def __init__(
            self,
            system_name: str,
            controller_type: str,
            ga_config: Dict,
            param_ranges: Dict,
            scenario_config: Dict,
            num_evaluation_runs: int = 10,
            weights: Optional[Dict] = None,
            custom_dynamics_path: Optional[str] = None,
            file_type: str = "Python (.py)",
            matlab_func_name: Optional[str] = None,
            num_states: Optional[int] = None,
            dt: float = 0.01,
            max_time: float = 5.0,
            target: float = 0.0,
            num_inputs: int = 1,
            input_channel: int = 0,
            output_channel: int = 0,
            trim_values: Optional[List[float]] = None,
            min_ctrl: float = -10.0,
            max_ctrl: float = 10.0
    ):
        self.system_name = system_name
        self.controller_type = controller_type
        self.ga_config = ga_config
        self.param_ranges = param_ranges
        self.scenario_config = scenario_config
        self.num_evaluation_runs = num_evaluation_runs
        self.weights = weights or {
            "mse": 1.0,
            "settling_time": 0.1,
            "overshoot": 0.01,
            "control_effort": 0.001
        }

        # System configuration
        self.custom_dynamics_path = custom_dynamics_path
        self.file_type = file_type
        self.matlab_func_name = matlab_func_name
        self.num_states_input = num_states
        self.dt = dt
        self.max_time = max_time
        self.target = target
        self.num_inputs = num_inputs
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.trim_values = trim_values if trim_values else [0.0] * num_inputs
        self.min_ctrl = min_ctrl
        self.max_ctrl = max_ctrl

        # Create system
        self.system = self._create_system()
        self.num_states = self.system.num_states

        # Setup simulator
        self.simulator = self._setup_simulator()

        # Setup parameter space
        self.param_names, self.gene_space = self._setup_parameter_space()

        # Optimization history
        self.history = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "params": {}
        }
        for param in self.param_names:
            self.history["params"][param] = []

        # ADD THIS: Track fitness evaluations
        self.fitness_evaluation_counter = 0

    def _create_system(self) -> GeneralDynamicalSystem:
        """Create the dynamical system"""
        system = create_system(
            self.system_name,
            scenario=self.scenario_config,
            custom_dynamics_path=self.custom_dynamics_path,
            file_type=self.file_type,
            matlab_func_name=self.matlab_func_name,
            num_states=self.num_states_input,
            num_inputs=self.num_inputs
        )

        # Configure system parameters
        system.dt = self.dt
        system.max_time = self.max_time
        system.target = self.target
        system.num_inputs = self.num_inputs
        system.input_channel = self.input_channel
        system.output_channel = self.output_channel
        system.trim_values = np.array(self.trim_values)
        system.min_control = self.min_ctrl
        system.max_control = self.max_ctrl

        return system

    def _setup_simulator(self) -> SimulationRunner:
        """Setup the simulation runner"""
        if self.system_name == "custom" and self.custom_dynamics_path:
            if self.file_type == "MATLAB/Octave (.m)":
                def octave_factory(scenario=None):
                    return OctaveSISOSystem(
                        self.custom_dynamics_path,
                        self.matlab_func_name,
                        self.num_states,
                        scenario,
                        self.num_inputs
                    )

                simulator = SimulationRunner(octave_factory)
            else:
                def custom_factory(scenario=None):
                    return CustomDynamicalSystem(
                        self.custom_dynamics_path,
                        scenario,
                        self.num_inputs
                    )

                simulator = SimulationRunner(custom_factory)
        elif self.system_name == "inverted_pendulum":
            simulator = SimulationRunner(InvertedPendulum)
        else:
            # Default to the system class
            def default_factory(scenario=None):
                return create_system(self.system_name, scenario, num_inputs=self.num_inputs)

            simulator = SimulationRunner(default_factory)

        # Configure simulator
        simulator.set_config(
            dt=self.dt,
            max_time=self.max_time,
            target=self.target,
            num_inputs=self.num_inputs,
            input_channel=self.input_channel,
            output_channel=self.output_channel,
            trim_values=np.array(self.trim_values),
            min_control=self.min_ctrl,
            max_control=self.max_ctrl
        )
        simulator.set_scenario(self.scenario_config)

        return simulator

    def _setup_parameter_space(self) -> Tuple[List[str], List[Dict]]:
        """Setup the parameter space for GA"""
        if self.controller_type == "PID":
            param_names = ["Kp", "Ki", "Kd"]
            gene_space = [
                {"low": self.param_ranges["PID"]["Kp"][0], "high": self.param_ranges["PID"]["Kp"][1]},
                {"low": self.param_ranges["PID"]["Ki"][0], "high": self.param_ranges["PID"]["Ki"][1]},
                {"low": self.param_ranges["PID"]["Kd"][0], "high": self.param_ranges["PID"]["Kd"][1]}
            ]

        elif self.controller_type == "FSF":
            param_names = [f"K{i + 1}" for i in range(self.num_states)]

            # Setup FSF ranges if not provided
            if not self.param_ranges.get("FSF"):
                self.param_ranges["FSF"] = {f"K{i + 1}": [0.1, 100.0] for i in range(self.num_states)}

            gene_space = [
                {"low": self.param_ranges["FSF"][f"K{i + 1}"][0],
                 "high": self.param_ranges["FSF"][f"K{i + 1}"][1]}
                for i in range(self.num_states)
            ]

        elif self.controller_type == "LQR":
            # LQR optimizes Q diagonal elements and R
            param_names = [f"Q{i + 1}" for i in range(self.num_states)] + ["R"]

            # Setup Q ranges
            Q_range = self.param_ranges["LQR"].get("Q_diag", [[0.1, 100.0]] * self.num_states)
            if len(Q_range) != self.num_states:
                Q_range = [[0.1, 100.0]] * self.num_states

            gene_space = [{"low": q[0], "high": q[1]} for q in Q_range]

            # Add R range
            R_range = self.param_ranges["LQR"].get("R", [0.01, 10.0])
            gene_space.append({"low": R_range[0], "high": R_range[1]})

        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

        return param_names, gene_space

    def _params_from_solution(self, solution: np.ndarray) -> Dict[str, float]:
        """Convert GA solution to parameter dictionary"""
        return {name: float(val) for name, val in zip(self.param_names, solution)}

    def _evaluate_controller(self, params: Dict[str, float]) -> float:
        """Evaluate controller performance (returns cost to minimize)"""
        total_cost = 0.0
        valid_runs = 0
        failure_reasons = []

        for run_idx in range(self.num_evaluation_runs):
            try:
                # For LQR, first solve for gains
                if self.controller_type == "LQR":
                    controller_params = self._solve_lqr(params)
                else:
                    controller_params = params

                # Evaluate
                result = self.simulator.evaluate_parameters(controller_params)

                if result['success']:
                    metrics = result['metrics']

                    # Sanitize metrics: replace inf/NaN with reasonable large/finite values to avoid blanket penalties
                    for k, v in list(metrics.items()):
                        if np.isnan(v):
                            if k in ['overshoot', 'zero_crossings', 'control_zero_crossings']:
                                metrics[k] = 0.0  # Non-negative metrics default to zero
                            else:  # e.g., mse, rmse, ss_error
                                metrics[k] = 10000.0  # High penalty for invalid errors
                        elif np.isinf(v):
                            if k in ['settling_time', 'rise_time']:
                                metrics[k] = self.max_time * 2  # Large but finite (e.g., 10.0s for max_time=5.0)
                            else:
                                metrics[k] = 10000.0  # High penalty for other infinities (e.g., unstable mse)

                    # Now check for remaining invalid metrics
                    if any(np.isnan(v) or np.isinf(v) for v in metrics.values()):
                        total_cost += 10000
                        if run_idx == 0:
                            failure_reasons.append("Invalid metrics after sanitization (NaN/Inf)")
                    else:
                        cost = (
                                self.weights["mse"] * metrics['mse'] +
                                self.weights["settling_time"] * metrics['settling_time'] +
                                self.weights["overshoot"] * metrics['overshoot'] / 100.0 +
                                self.weights["control_effort"] * metrics['control_effort'] / 1000.0
                        )
                        total_cost += cost
                        valid_runs += 1
                else:
                    total_cost += 10000
                    if run_idx == 0:
                        failure_reasons.append("Simulation failed")
            except Exception as e:
                total_cost += 10000
                if run_idx == 0:
                    failure_reasons.append(f"Exception: {str(e)}")

        # Debug output for first generation
        if hasattr(self, '_first_eval'):
            if not self._first_eval and valid_runs == 0 and failure_reasons:
                print(f"\n⚠️  DEBUG: All simulations failing! Reasons: {failure_reasons[0]}")
                print(f"   Params tested: {params}")
                print(f"   System target: {self.target}")
                print(f"   Initial condition range: {self.scenario_config['initial_condition_range']}")
                self._first_eval = True
        else:
            self._first_eval = False

        if valid_runs == 0:
            return 10000

        return total_cost / valid_runs

    def _solve_lqr(self, params: Dict[str, float]) -> Dict[str, float]:
        """Solve LQR problem given Q and R parameters"""
        # Get linearized system matrices
        A, B = self._get_linearized_system()

        # Construct Q matrix (diagonal)
        Q_diag = [params[f"Q{i + 1}"] for i in range(self.num_states)]
        Q = np.diag(Q_diag)

        # Construct R matrix
        R = np.array([[params["R"]]])

        # Solve ARE
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            K = K.flatten()

            # Return as FSF gains
            return {f"K{i + 1}": K[i] for i in range(self.num_states)}
        except Exception as e:
            print(f"LQR solve failed: {e}")
            return {f"K{i + 1}": 0.0 for i in range(self.num_states)}

    def _get_linearized_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linearized system matrices A and B"""
        if isinstance(self.system, InvertedPendulum):
            # Analytical linearization for inverted pendulum
            m = self.system.m
            l = self.system.l
            b = self.system.b
            g = self.system.g

            A = np.array([
                [0, 1],
                [g / l, -b / (m * l ** 2)]
            ])
            B = np.array([
                [0],
                [1 / (m * l ** 2)]
            ])
            return A, B
        elif isinstance(self.system, BallBeam):  # Analytical linearization for BallBeam
            m = self.system.m
            R = self.system.R
            J = self.system.J
            g = self.system.g

            H = -m * g / (J / R ** 2 + m)
            A = np.array([
                [0, 1, 0, 0],
                [0, 0, H, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ])
            B = np.array([
                [0],
                [0],
                [0],
                [1]
            ])
            return A, B
        else:
            # Numerical linearization
            return self._numerical_linearization()

    def _numerical_linearization(self) -> Tuple[np.ndarray, np.ndarray]:
        """Numerically compute linearized A and B matrices"""
        # Equilibrium point (target at output channel, others zero)
        x_eq = np.zeros(self.num_states)
        x_eq[self.output_channel] = self.target if not callable(self.target) else self.target(0)
        u_eq = self.trim_values[self.input_channel]

        # Small perturbation
        epsilon = 1e-6

        # Compute A matrix (df/dx)
        A = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            x_plus = x_eq.copy()
            x_plus[i] += epsilon
            x_minus = x_eq.copy()
            x_minus[i] -= epsilon

            u_full = self.trim_values.copy()
            u_full[self.input_channel] = u_eq

            try:
                f_plus = self.system.system_dynamics(x_plus, u_full)
                f_minus = self.system.system_dynamics(x_minus, u_full)
                A[:, i] = (f_plus - f_minus) / (2 * epsilon)
            except:
                pass

        # Compute B matrix (df/du)
        B = np.zeros((self.num_states, 1))
        u_plus = self.trim_values.copy()
        u_plus[self.input_channel] = u_eq + epsilon
        u_minus = self.trim_values.copy()
        u_minus[self.input_channel] = u_eq - epsilon

        try:
            f_plus = self.system.system_dynamics(x_eq, u_plus)
            f_minus = self.system.system_dynamics(x_eq, u_minus)
            B[:, 0] = (f_plus - f_minus) / (2 * epsilon)
        except:
            pass

        return A, B

    def fitness_function(self, ga_instance, solution, solution_idx):
        """Fitness function for PyGAD (maximize = minimize negative cost)"""
        # ADD THIS: Increment counter
        self.fitness_evaluation_counter += 1

        params = self._params_from_solution(solution)
        cost = self._evaluate_controller(params)

        # Convert cost to fitness (minimize cost = maximize negative cost)
        if np.isnan(cost) or np.isinf(cost):
            return -10000
        return -cost

    def on_generation(self, ga_instance):
        """Callback for tracking optimization progress"""
        generation = ga_instance.generations_completed
        solution, solution_fitness, _ = ga_instance.best_solution()
        params = self._params_from_solution(solution)
        cost = -solution_fitness

        # Store history
        self.history["generation"].append(generation)
        self.history["best_fitness"].append(cost)

        # Store mean fitness
        all_fitness = ga_instance.last_generation_fitness
        mean_cost = -np.mean(all_fitness)
        self.history["mean_fitness"].append(mean_cost)

        # Store parameters
        for param_name, param_value in params.items():
            self.history["params"][param_name].append(param_value)

        # Print progress
        print(f"Generation {generation}/{self.ga_config['num_generations']}: "
              f"Best Cost = {cost:.6f}, Mean Cost = {mean_cost:.6f}")

    def optimize(self) -> Tuple[Dict[str, float], float, Dict]:
        """Run the genetic algorithm optimization"""
        # ADD THIS: Reset counter before optimization
        self.fitness_evaluation_counter = 0

        # Create PyGAD instance
        ga_instance = pygad.GA(
            num_generations=self.ga_config["num_generations"],
            num_parents_mating=self.ga_config["num_parents_mating"],
            fitness_func=self.fitness_function,
            sol_per_pop=self.ga_config["population_size"],
            num_genes=len(self.param_names),
            gene_space=self.gene_space,
            parent_selection_type="sss",
            keep_parents=self.ga_config["keep_parents"],
            crossover_type="single_point",
            crossover_probability=self.ga_config.get("crossover_probability", 0.8),
            mutation_type="random",
            mutation_probability=self.ga_config.get("mutation_probability", 0.1),
            mutation_num_genes=self.ga_config.get("mutation_num_genes", 1),
            on_generation=self.on_generation,
            random_seed=self.ga_config.get("random_seed", 42)
        )

        # Run GA
        ga_instance.run()

        # Get best solution
        solution, solution_fitness, _ = ga_instance.best_solution()
        best_params = self._params_from_solution(solution)

        # For LQR, convert to actual gains
        if self.controller_type == "LQR":
            best_params = self._solve_lqr(best_params)

        # ADD THIS: Store counter in history
        self.history["total_fitness_evaluations"] = self.fitness_evaluation_counter

        return best_params, solution_fitness, self.history

    def plot_optimization_history(self, history: Dict):
        """Plot optimization progress"""
        n_params = len(self.param_names)
        # Total plots: 1 (cost) + n_params (parameters)
        total_plots = n_params + 1
        n_cols = min(3, total_plots)
        n_rows = (total_plots + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()

        # Plot cost evolution
        axes[0].plot(history["generation"], history["best_fitness"],
                     'b-', linewidth=2, label='Best Cost')
        axes[0].plot(history["generation"], history["mean_fitness"],
                     'r--', linewidth=1.5, label='Mean Cost')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Cost')
        axes[0].set_title('Cost Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot parameter evolutions
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        for i, param_name in enumerate(self.param_names):
            ax_idx = i + 1
            if ax_idx < len(axes):  # Safety check
                ax = axes[ax_idx]
                ax.plot(history["generation"], history["params"][param_name],
                        color=colors[i % len(colors)], linewidth=2)
                ax.set_xlabel('Generation')
                ax.set_ylabel(param_name)
                ax.set_title(f'{param_name} Evolution')
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'ga_optimization_history_{self.controller_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_time_response(self, params: Dict[str, float], num_runs: int = 50):
        """Plot Monte Carlo time response"""
        all_outputs = []
        all_controls = []
        all_metrics = []  # NEW: Collect metrics for stats

        for _ in range(num_runs):
            try:
                result = self.simulator.evaluate_parameters(params)
                if result['success']:
                    all_outputs.append(result['trajectory'])
                    all_controls.append(result['control_signals'])
                    all_metrics.append(result['metrics'])  # NEW: Append metrics
            except:
                continue

        if not all_outputs:
            print("All simulations failed!")
            return

        # Process trajectories
        max_len = max(len(out) for out in all_outputs)
        time_vector = np.linspace(0, (max_len - 1) * self.dt, max_len)  # Using linspace as per previous fix

        output_matrix = np.full((len(all_outputs), max_len), np.nan)
        control_matrix = np.full((len(all_controls), max_len), np.nan)

        for i in range(len(all_outputs)):
            output_matrix[i, :len(all_outputs[i])] = all_outputs[i]
            control_matrix[i, :len(all_controls[i])] = all_controls[i]

        # Statistics
        output_mean = np.nanmean(output_matrix, axis=0)
        output_std = np.nanstd(output_matrix, axis=0)
        control_mean = np.nanmean(control_matrix, axis=0)
        control_std = np.nanstd(control_matrix, axis=0)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Output plot
        ax1.plot(time_vector, output_mean, 'b-', linewidth=2, label='Mean Output')
        ax1.fill_between(time_vector, output_mean - output_std, output_mean + output_std,
                         color='blue', alpha=0.2, label='±1σ')
        target_val = self.target if not callable(self.target) else self.target(0)
        ax1.axhline(target_val, color='r', linestyle='--', alpha=0.7, label='Target')
        ax1.set_ylabel('System Output')
        ax1.set_title(f'Closed-Loop Response ({num_runs} runs) | {self.controller_type}')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend()

        # Control plot
        ax2.plot(time_vector, control_mean, 'g-', linewidth=2, label='Mean Control')
        ax2.fill_between(time_vector, control_mean - control_std, control_mean + control_std,
                         color='green', alpha=0.2, label='±1σ')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Input')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'ga_time_response_{self.controller_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Metrics
        if all_metrics:
            avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
            std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}
            print("Performance Metrics (Mean ± Std):")
            for key in avg_metrics:
                print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
            stable_pct = sum(1 for m in all_metrics if m.get('stable', False)) / len(all_metrics) * 100
            print(f"  Stability: {stable_pct:.0f}% stable")

    def get_performance_metrics(self, params: Dict[str, float], num_runs: int = 20) -> Dict[str, float]:
        all_metrics = []
        successful_runs = 0
        for _ in range(num_runs):
            try:
                result = self.simulator.evaluate_parameters(params)
                if result['success']:
                    all_metrics.append(result['metrics'])
                    successful_runs += 1
            except Exception:
                continue
        if successful_runs == 0:
            return {"mse": float('inf'), "settling_time": float('inf'), "overshoot": float('inf')}
        return {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}