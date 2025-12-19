from typing import Dict, List, Optional, TypedDict, Any
import json
import queue
import numpy as np
import time
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from src.systems import GeneralDynamicalSystem, create_system
from src.systems import CustomDynamicalSystem
from src.systems import InvertedPendulum
from src.simulation import SimulationRunner
from src.utils import SharedBuffer, set_global_seed

# Minimal logging directory
logs_dir = Path("./.logs")
logs_dir.mkdir(exist_ok=True)


def initialize_state(
    llm_model: str = "deepseek-r1-distill-llama-70b",
    run_id: int = 1,
    seed: int = 42,
    system_name: str = "ball_beam",
    max_scenarios: int = 3,
    max_iter: int = 10,
    controllers: Optional[List[str]] = None,
    custom_scenarios: Optional[List[Dict]] = None,
    param_ranges: Optional[Dict[str, Dict[str, List[float]]]] = None,
    target_metrics: Optional[Dict[str, float]] = None,
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
    trim_values: Optional[np.ndarray] = None,
    min_ctrl: float = -10.0,
    max_ctrl: float = 10.0,
    max_tokens: int = 100000,
    update_queue: Optional[queue.Queue] = None,
    **kwargs
) -> Dict:
    # === CORRECT SYSTEM & SIMULATOR CREATION ===
    if system_name == "custom" and custom_dynamics_path:
        path = Path(custom_dynamics_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom dynamics file not found: {custom_dynamics_path}")

        if file_type == "Python (.py)":
            def factory(scenario=None):
                return CustomDynamicalSystem(str(path), scenario, num_inputs=num_inputs)
            simulator = SimulationRunner(factory)
        else:
            # For MATLAB/Octave or Simulink — fall back to InvertedPendulum for mock simplicity
            # (In real app this would use OctaveSISOSystem or SimulinkSISOSystem)
            simulator = SimulationRunner(InvertedPendulum)
    else:
        simulator = SimulationRunner(InvertedPendulum)

    # Configure simulator
    simulator.set_config(
        dt=dt,
        max_time=max_time,
        target=target,
        num_inputs=num_inputs,
        input_channel=input_channel,
        output_channel=output_channel,
        trim_values=trim_values or np.zeros(num_inputs),
        min_control=min_ctrl,
        max_control=max_ctrl
    )

    # === Rest of initialization (unchanged) ===
    controllers_list = controllers if controllers else ["FSF"]

    default_param_ranges = {
        "FSF": {"K1": [0.01, 100.0], "K2": [0.01, 100.0], "K3": [0.01, 100.0], "K4": [0.01, 100.0]},
        "PID": {"Kp": [0.01, 50.0], "Ki": [0.01, 50.0], "Kd": [0.01, 50.0]}
    }
    param_ranges = param_ranges if param_ranges else default_param_ranges

    default_target_metrics = {"mse": 0.2, "settling_time": 4.0, "overshoot": 0}
    target_metrics = target_metrics if target_metrics else default_target_metrics

    set_global_seed(seed)

    return {
        "llm_model": llm_model,
        "run_id": run_id,
        "seed": seed,
        "temperature": 0.0,
        "system": simulator.system_class(),  # dummy system instance
        "system_name": system_name,
        "display_system_name": system_name.replace("_", " ").title(),
        "system_description": "Mock system for testing",
        "iteration": 0,
        "max_iterations": max_iter,
        "scenario_level": 1,
        "max_scenarios": max_scenarios if custom_scenarios is None else len(custom_scenarios),
        "controllers_list": controllers_list,
        "current_controller_index": 0,
        "buffer": SharedBuffer(),
        "current_params": None,
        "simulator": simulator,
        "results": None,
        "feedback": None,
        "should_continue": True,
        "should_continue_outer": True,
        "scenario": None,
        "controller_type": None,
        "target_metrics": target_metrics,
        "param_ranges": param_ranges,
        "inner_loop_completed": False,
        "redesign_requested": False,
        "all_scenario_history": [],
        "custom_scenarios": custom_scenarios,
        "scenario_controller_history": {},
        "update_queue": update_queue,
        "max_tries": 4,
        "scenario_start_time": None,
        "max_tokens": max_tokens,
        "control_objective": "Design a stable controller"
    }


def suggest_controller(state: Dict) -> Dict:
    print(f"\n=== 🎛️ SELECTING CONTROLLER FOR SCENARIO {state['scenario_level']} ===")
    controller_type = state["controllers_list"][state["current_controller_index"]]
    state["buffer"].controller_type = controller_type

    # Simplified: use custom ranges or defaults
    param_ranges = state.get("param_ranges", {}).get(controller_type)
    if not param_ranges or not any(param_ranges.values()):
        param_ranges = {"K1": [0.01, 100.0], "K2": [0.01, 100.0],
                        "K3": [0.01, 100.0], "K4": [0.01, 100.0]}

    state["buffer"].param_ranges = param_ranges
    initial_params = {key: (rng[0] + rng[1]) / 2 for key, rng in param_ranges.items()}
    initial_params["reasoning"] = f"Initial midpoint for {controller_type}"

    target_metrics = state.get("target_metrics", {"mse": 0.2, "settling_time": 4.0, "overshoot": 0})
    state["buffer"].target_metrics = target_metrics
    state["buffer"].clear_history()

    print(f"Selected Controller: {controller_type} | Initial Params: {initial_params}")
    return {
        "controller_type": controller_type,
        "current_params": initial_params,
        "target_metrics": target_metrics,
        "iteration": 0
    }


def design_scenario(state: Dict) -> Dict:
    if state["scenario_level"] > state["max_scenarios"]:
        print("🏁 Exceeded maximum number of scenarios. Terminating workflow.")
        return {"should_continue_outer": False, "scenario": None}

    print(f"\n=== 🎭 DESIGNING SCENARIO LEVEL {state['scenario_level']}/{state['max_scenarios']} ===")

    # Use custom scenarios or simple defaults
    if state.get("custom_scenarios"):
        scenario_data = state["custom_scenarios"][state["scenario_level"] - 1]
    else:
        scenario_data = {
            "id": f"Scenario_{state['scenario_level']}",
            "randomness_level": 0.0,
            "param_uncertainty": 0.0,
            "initial_condition_range": [1.0, 1.0],
            "disturbance_level": 0.0,
        }

    state["simulator"].set_scenario(scenario_data)
    state["buffer"].scenario = scenario_data
    state["buffer"].current_scenario_metrics = {'tokens_in': 0, 'tokens_out': 0, 'time': 0.0, 'cost': 0.0}
    state["scenario_start_time"] = time.time()

    print(f"Scenario {scenario_data['id']}: IC Range {scenario_data['initial_condition_range']}")
    return {"scenario": scenario_data, "inner_loop_completed": False, "should_continue_outer": True}


def propose_parameters(state: Dict) -> Dict:
    """Propose controller parameters using mock LLM"""
    from src.llm_agents_mock import LLMActor

    actor = LLMActor(model=state["llm_model"], seed=state["seed"])
    params = actor.generate_parameters(
        state["buffer"], state["controller_type"],
        state["iteration"], state["max_iterations"], state["system"]
    )

    # Check token limit
    total_tokens = state["buffer"].total_metrics['tokens_in'] + state["buffer"].total_metrics['tokens_out']
    if total_tokens > state["max_tokens"]:
        return {"current_params": params, "should_continue": False}

    return {"current_params": params}


def run_simulation(state: Dict) -> Dict:
    """Run simulation with current parameters"""
    params = state["current_params"]
    simulator = state["simulator"]

    if not params:
        return {
            "results": {
                "success": False,
                "error": "Invalid parameters",
                "metrics": {"mse": float('inf'), "settling_time": float('inf'),
                            "overshoot": float('inf'), "stable": False},
                "trajectory": [], "control_signals": [], "errors": []
            },
            "should_continue": False
        }

    result = simulator.evaluate_parameters(params)

    if not result['success']:
        return {
            "results": {
                "success": False,
                "error": result['error'],
                "metrics": {"mse": float('inf'), "settling_time": float('inf'),
                            "overshoot": float('inf'), "stable": False},
                "trajectory": [], "control_signals": [], "errors": []
            },
            "should_continue": False
        }

    return {"results": result}


def evaluate_performance(state: Dict) -> Dict:
    """Evaluate controller performance using mock critic"""
    from src.llm_agents_mock import LLMCritic

    results = state["results"]

    if not results['success']:
        feedback = json.dumps({
            "result_analysis": f"Simulation failed: {results['error']}",
            "suggested_improvements": ["Adjust parameters"]
        })
    else:
        critic = LLMCritic(model=state["llm_model"], seed=state["seed"])
        feedback = critic.analyze_results(
            state["current_params"], results['metrics'],
            {'trajectory': results['trajectory'], 'control_signals': results['control_signals']},
            state["buffer"], state["target_metrics"],
            state["iteration"], state["max_iterations"]
        )

    # Check token limit
    total_tokens = state["buffer"].total_metrics['tokens_in'] + state["buffer"].total_metrics['tokens_out']
    if total_tokens > state["max_tokens"]:
        return {"feedback": feedback, "should_continue": False}

    return {"feedback": feedback}


def update_buffer(state: Dict) -> Dict:
    """Update history buffer with results"""
    state["buffer"].add_entry(
        state["current_params"],
        state["results"]["metrics"],
        state["results"]["trajectory"],
        state["results"]["control_signals"],
        state["results"]["errors"],
        state["feedback"]
    )

    # Build performance summary for UI
    metrics = state["results"]["metrics"]
    params = state["current_params"]
    metrics_line = f"#{state['iteration'] + 1}/{state['max_iterations']} | "
    metrics_line += f"Type:{state['controller_type']} | "

    # Add parameters
    param_strings = [f"{k}:{v:.3f}" for k, v in params.items() if k != 'reasoning']
    if param_strings:
        metrics_line += " | ".join(param_strings) + " | "

    # Add metrics
    metrics_line += f"MSE:{metrics['mse']:.4f} | "
    metrics_line += f"Ts:{metrics['settling_time']:.2f} | "
    metrics_line += f"%OS:{metrics['overshoot']:.2f} | "
    metrics_line += f"Stable:{metrics['stable']}"

    print(metrics_line)

    # Send to UI queue
    if state.get('update_queue'):
        state['update_queue'].put(metrics_line)

    state["iteration"] += 1
    return {"iteration": state["iteration"]}


def judge_termination(state: Dict) -> Dict:
    """Judge whether to terminate inner loop"""
    from src.llm_agents_mock import LLMTerminator, LLMJuror

    termination_judge = LLMTerminator(model=state["llm_model"], seed=state["seed"])
    juror = LLMJuror(model=state["llm_model"], seed=state["seed"], max_tries=state["max_tries"])

    current_metrics = state["results"]["metrics"] if state["buffer"].history else {
        "mse": float('inf'), "settling_time": float('inf'), "stable": False
    }

    # Get termination decision
    termination_data, _ = termination_judge.judge_termination(
        state["buffer"], current_metrics, state["target_metrics"],
        state["max_iterations"], state["controller_type"],
        state["system_description"], state["iteration"]
    )

    decision = termination_data.get("decision", "CONTINUE")
    print(f"=== 🔍 TERMINATOR DECISION: {decision} ===")

    # Handle TERMINATE_SUCCESS
    if decision == "TERMINATE_SUCCESS":
        return {
            "should_continue": False,
            "inner_loop_completed": True,
            "redesign_requested": False,
            "iteration": state["iteration"]
        }

    # Handle TERMINATE_REDESIGN with juror
    elif decision == "TERMINATE_REDESIGN":
        juror_decision = juror.decide(state)
        print(f"=== 👨‍⚖️ JUROR DECISION: {juror_decision['decision']} ===")

        if juror_decision['decision'] == "RECONSIDER_RANGE":
            state["buffer"].param_ranges = juror_decision['new_range']
            state["iteration"] = 0
            return {
                "should_continue": True,
                "inner_loop_completed": False,
                "redesign_requested": False,
                "iteration": 0
            }
        elif juror_decision['decision'] == "REDESIGN_APPROVED":
            return {
                "should_continue": False,
                "inner_loop_completed": False,
                "redesign_requested": True,
                "iteration": state["iteration"]
            }
        else:  # EXPLORE_FURTHER
            return {
                "should_continue": True,
                "inner_loop_completed": False,
                "redesign_requested": False,
                "iteration": state["iteration"]
            }

    # Handle max iterations
    if state["iteration"] >= state["max_iterations"]:
        juror_decision = juror.decide(state)

        if juror_decision['decision'] == "RECONSIDER_RANGE":
            state["buffer"].param_ranges = juror_decision['new_range']
            return {
                "should_continue": True,
                "inner_loop_completed": False,
                "redesign_requested": False,
                "iteration": 0
            }
        elif juror_decision['decision'] == "REDESIGN_APPROVED":
            return {
                "should_continue": False,
                "inner_loop_completed": False,
                "redesign_requested": True,
                "iteration": state["iteration"]
            }
        else:  # EXPLORE_FURTHER
            return {
                "should_continue": True,
                "inner_loop_completed": False,
                "redesign_requested": False,
                "iteration": 0
            }

    # Check token limit
    total_tokens = state["buffer"].total_metrics['tokens_in'] + state["buffer"].total_metrics['tokens_out']
    if total_tokens > state["max_tokens"]:
        return {
            "should_continue": False,
            "inner_loop_completed": True,
            "redesign_requested": False,
            "iteration": state["iteration"]
        }

    # Default: continue
    return {
        "should_continue": True,
        "inner_loop_completed": False,
        "redesign_requested": False,
        "iteration": state["iteration"]
    }


def evaluate_scenario_completion(state: Dict) -> Dict:
    """Evaluate whether to move to next scenario or try different controller"""
    # Compute wall clock time
    if state.get("scenario_start_time"):
        duration = time.time() - state["scenario_start_time"]
        if state["buffer"].current_scenario_metrics:
            state["buffer"].current_scenario_metrics["time"] = duration

    if state["inner_loop_completed"]:
        print(f"✅ Scenario {state['scenario_level']} completed successfully!")

        completed_level = state["scenario_level"]

        # Get best entry (simplified)
        best_entries = state["buffer"].get_best_entries(1, state["target_metrics"])
        best_entry = best_entries[0] if best_entries else None
        best_success_rate = 0.5 if best_entry else 0.0  # Simplified success rate

        state["scenario_level"] += 1

        # Store in history
        state["all_scenario_history"].append({
            'scenario_level': completed_level,
            'controller_type': state["controller_type"],
            'history': state["buffer"].history.copy(),
            'scenario_metrics': state["buffer"].current_scenario_metrics.copy() if state[
                "buffer"].current_scenario_metrics else {},
            'best_entry': best_entry,
            'best_success_rate': best_success_rate
        })

        # Send update to UI
        if state.get('update_queue'):
            state['update_queue'].put(f"Scenario {completed_level} completed!")

        if state["scenario_level"] > state["max_scenarios"]:
            print(f"🎉 All scenarios completed!")
            state["should_continue_outer"] = False
        else:
            state["should_continue_outer"] = True

    elif state["redesign_requested"]:
        print(f"🔄 Redesign requested for Scenario {state['scenario_level']}")
        state["current_controller_index"] += 1

        if state["current_controller_index"] >= len(state["controllers_list"]):
            print(f"⚠️ All controllers tried without success")
            state["should_continue_outer"] = False
        else:
            print(f"⭐ Trying next controller: {state['controllers_list'][state['current_controller_index']]}")
            state["should_continue_outer"] = True
            state["scenario_level"] = 1

    # Reset iteration if completed or redesigning
    if state.get("inner_loop_completed") or state.get("redesign_requested"):
        state["iteration"] = 0
        state["scenario_start_time"] = None

    return state


def generate_final_report(state: Dict) -> Dict:
    """Generate simplified final report"""
    print(f"\n=== MOCK FINAL REPORT ===")
    print(f"Completed {state['scenario_level'] - 1}/{state['max_scenarios']} scenarios")

    # Calculate simplified metrics
    total_tokens = state["buffer"].total_metrics['tokens_in'] + state["buffer"].total_metrics['tokens_out']
    print(f"Total tokens: {total_tokens}")

    return {"report_generated": True, "charts": []}


def should_continue_outer(state) -> str:
    """Decision node for outer loop"""
    return "continue_outer" if state["should_continue_outer"] else "exit_outer"


# State schema
class OptimizationState(TypedDict):
    llm_model: str
    run_id: int
    seed: int
    temperature: float
    system: GeneralDynamicalSystem
    system_name: str
    system_description: str
    iteration: int
    max_iterations: int
    scenario_level: int
    max_scenarios: int
    controllers_list: List[str]
    current_controller_index: int
    buffer: SharedBuffer
    current_params: Optional[Dict]
    simulator: SimulationRunner
    results: Optional[Dict]
    feedback: Optional[str]
    should_continue: bool
    should_continue_outer: bool
    scenario: Optional[Dict]
    controller_type: Optional[str]
    target_metrics: Optional[Dict]
    param_ranges: Optional[Dict]
    custom_scenarios: Optional[List[Dict]]
    inner_loop_completed: bool
    redesign_requested: bool
    all_scenario_history: List
    update_queue: Optional[queue.Queue]
    max_tries: int
    scenario_start_time: Optional[float]
    max_tokens: int
    control_objective: str


def create_optimization_graph(max_scenarios=3, max_iter=10):
    """Create simplified LangGraph workflow"""
    builder = StateGraph(OptimizationState)

    # Add nodes
    builder.add_node("suggest_controller", suggest_controller)
    builder.add_node("design_scenario", design_scenario)
    builder.add_node("evaluate_scenario_completion", evaluate_scenario_completion)
    builder.add_node("generate_final_report", generate_final_report)
    builder.add_node("propose_parameters", propose_parameters)
    builder.add_node("run_simulation", run_simulation)
    builder.add_node("evaluate_performance", evaluate_performance)
    builder.add_node("update_buffer", update_buffer)
    builder.add_node("judge_termination", judge_termination)

    # Add edges
    builder.add_edge(START, "suggest_controller")
    builder.add_edge("suggest_controller", "design_scenario")
    builder.add_edge("design_scenario", "propose_parameters")
    builder.add_edge("propose_parameters", "run_simulation")
    builder.add_edge("run_simulation", "evaluate_performance")
    builder.add_edge("evaluate_performance", "update_buffer")
    builder.add_edge("update_buffer", "judge_termination")

    # Conditional edges
    builder.add_conditional_edges(
        "judge_termination",
        lambda state: "continue_inner" if state["should_continue"] else "evaluate_completion",
        {
            "continue_inner": "propose_parameters",
            "evaluate_completion": "evaluate_scenario_completion"
        }
    )

    builder.add_conditional_edges(
        "evaluate_scenario_completion",
        should_continue_outer,
        {"continue_outer": "suggest_controller", "exit_outer": "generate_final_report"}
    )

    builder.add_edge("generate_final_report", END)

    graph = builder.compile()
    config = {"max_scenarios": max_scenarios, "max_iterations": max_iter}
    return graph, config


def run_optimization(
        llm_model: str = "deepseek-r1-distill-llama-70b",
        run_id: int = 1,
        seed: int = 42,
        system_name: str = "ball_beam",
        max_scenarios: int = 3,
        max_iter: int = 10,
        controllers: Optional[List[str]] = None,
        custom_scenarios: Optional[List[Dict]] = None,
        param_ranges: Optional[Dict[str, Dict[str, List[float]]]] = None,
        target_metrics: Optional[Dict[str, float]] = None,
        dt: float = 0.01,
        max_time: float = 5.0,
        target: float = 0.0,
        num_inputs: int = 1,
        min_ctrl: float = -10.0,
        max_ctrl: float = 10.0,
        update_queue: Optional[queue.Queue] = None,
        max_tokens: int = 100000,
        **kwargs  # Catch all other unused params
):
    """Run optimization workflow"""
    graph, config = create_optimization_graph(max_scenarios, max_iter)

    initial_state = initialize_state(
        llm_model=llm_model,
        run_id=run_id,
        seed=seed,
        system_name=system_name,
        max_scenarios=max_scenarios,
        max_iter=max_iter,
        controllers=controllers,
        custom_scenarios=custom_scenarios,
        param_ranges=param_ranges,
        target_metrics=target_metrics,
        dt=dt,
        max_time=max_time,
        target=target,
        num_inputs=num_inputs,
        min_ctrl=min_ctrl,
        max_ctrl=max_ctrl,
        update_queue=update_queue,
        max_tokens=max_tokens,
        **kwargs
    )

    # Run workflow
    for _ in graph.stream(initial_state, config={"recursion_limit": 1000}):
        pass

    # Calculate simplified success rates
    success_rates = {}
    for scen_hist in initial_state["all_scenario_history"]:
        level = scen_hist.get('scenario_level', 0)
        success_rates[level] = scen_hist.get('best_success_rate', 0.0)

    overall_sr = success_rates[max(success_rates.keys())] if success_rates else 0.0

    final_total_tokens = initial_state["buffer"].total_metrics['tokens_in'] + \
                         initial_state["buffer"].total_metrics['tokens_out']
    final_cost = initial_state["buffer"].total_metrics['cost']

    print(f"🏁 Workflow complete. Tokens={final_total_tokens}, Cost=${final_cost:.4f}")

    return {
        "success_rates_per_scenario": success_rates,
        "overall_success_rate": overall_sr,
        "all_scenario_history": initial_state["all_scenario_history"],
        "final_total_tokens": final_total_tokens,
        "final_cost": final_cost
    }