import streamlit as st
import numpy as np
from pathlib import Path
import textwrap
import re
import ast
from classic.ga_utils import GAOptimizer


CSS_STYLES = """
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-complete {
        color: #007bff;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .agent-response {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .session-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .session-item:hover {
        background-color: #f0f0f0;
    }
    .session-item-active {
        background-color: #e3f2fd;
        border-left: 3px solid #1f77b4;
    }
    .speech-bubble-user {
        background-color: #e3f2fd;
        border-radius: 18px;
        padding: 15px 20px;
        margin: 10px 0;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .speech-bubble-assistant {
        background-color: #f5f5f5;
        border-radius: 18px;
        padding: 15px 20px;
        margin: 10px 0;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .file-preview {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
    }
</style>
"""


def display_logo():
    """Display logo on home page"""
    # Load logo from file
    logo_path = Path("assets/logo.svg")

    if logo_path.exists():
        with open(logo_path, 'r') as f:
            logo_svg = f.read()
        # Ensure the SVG scales with container width while keeping aspect ratio
        logo_svg = logo_svg.replace('<svg', '<svg style="width:100%; height:auto; max-width:200px;"')
    else:
        # Fallback to default logo if file doesn't exist
        logo_svg = """
        <svg style="width:100%; height:auto; max-width:200px;" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <circle cx="100" cy="100" r="80" fill="#1f77b4" opacity="0.2"/>
            <circle cx="100" cy="100" r="60" fill="#1f77b4" opacity="0.4"/>
            <circle cx="100" cy="100" r="40" fill="#1f77b4" opacity="0.6"/>
            <text x="100" y="110" font-size="48" font-weight="bold" 
                  text-anchor="middle" fill="#1f77b4">🎛️</text>
        </svg>
        """

    html_content = textwrap.dedent(f"""
        <div style='text-align: center; margin: 2rem 0;'>
            <div style='max-width: 160px; margin: 0 auto;'>
                {logo_svg}
            </div>
            <h2 style='color: #1f77b4; margin-top: 1rem;'>Agentic Control Design</h1>
            <p style='color: #6c757d; font-size: 1.1rem;'>AI-Powered Control System Design Studio</p>
        </div>
    """).strip()

    # De-indent all lines to avoid Markdown code-block parsing
    lines = html_content.splitlines()
    html_content = '\n'.join(line.lstrip() for line in lines)

    st.markdown(
        html_content,
        unsafe_allow_html=True
    )


def process_objective(user_input: str) -> str:
    """Mock version – returns a fixed, refined control objective."""
    return "Design a stable controller with minimal settling time, negligible overshoot, and zero steady-state error."


def _get_default_param_ranges_for_all():
    """Get default parameter ranges for all controller types"""
    defaults = {}
    for controller in ["P", "PI", "PD", "PID", "FSF"]:
        defaults[controller] = _get_default_param_ranges(controller)
    return defaults


def build_config_from_session():
    """Build configuration from session state"""

    # Get advanced settings if available
    if 'temp_config' in st.session_state and st.session_state.temp_config:
        base_config = st.session_state.temp_config
    else:
        # Use defaults
        # Use defaults
        base_config = {
            'llm_model': 'meta/llama-4-maverick-17b-128e-instruct',
            'controllers': ['PID', 'FSF'],
            'max_scenarios': 2,
            'max_iter': 20,
            'seed': 42,
            'max_tries': 0,
            'target_metrics': {
                'mse': 0.15,
                'settling_time': 3.5,
                'overshoot': 0.0,
                'max_iterations': 20
            },
            'dt': 0.01,
            'max_time': 5.0,
            'target': 0.0,
            'num_inputs': 1,
            'input_channel': 0,
            'output_channel': 0,
            'min_ctrl': -10.0,
            'max_ctrl': 10.0,
            'trim_values': None,
            'num_states': None,
            'matlab_func_name': None,
            'param_ranges': None,
            'custom_scenarios': None,
            'enable_ga': False,
            'ga_config': None,
            'display_system_name': 'Custom System',
            'system_description': 'A custom dynamical system for control design.',
            'max_tokens': 200000
        }

    # Add system-specific configuration
    config = {
        **base_config,
        'run_id': 1,
        'system_name': 'custom',
        'custom_dynamics_path': st.session_state.get('custom_dynamics_path'),
        'file_type': st.session_state.get('file_type', 'Python (.py)'),
        'control_objective': st.session_state.get('control_objective', ''),
        'param_ranges': base_config.get('param_ranges')
    }

    # Add MATLAB-specific fields if needed
    if config['file_type'] == "MATLAB/Octave (.m)":
        config['matlab_func_name'] = base_config.get('matlab_func_name')
        config['num_states'] = base_config.get('num_states')
    elif config['file_type'] == "Simulink (.slx)":
        config['simulink_model_name'] = st.session_state.get('simulink_model_name')
        # CRITICAL: num_states is REQUIRED for Simulink
        config['num_states'] = st.session_state.get('num_states') or base_config.get('num_states')
        if config['num_states'] is None:
            raise ValueError("num_states must be specified for Simulink (.slx) files")
    # NEW: For Python files with FSF, also get num_states if available
    elif 'FSF' in config.get('controllers', []):
        config['num_states'] = base_config.get('num_states')

    # Add trim_values
    config['trim_values'] = base_config.get('trim_values')
    config['custom_scenarios'] = base_config.get('custom_scenarios')

    # Add GA config
    config['enable_ga'] = base_config.get('enable_ga', False)
    config['ga_config'] = base_config.get('ga_config')

    # NEW: Add display fields
    config['display_system_name'] = base_config.get('display_system_name', 'Custom System')
    config['system_description'] = base_config.get('system_description',
                                                   'A custom dynamical system for control design.')
    config['max_tokens'] = base_config.get('max_tokens', 200000)

    return config


def create_advanced_settings():
    """Create advanced settings UI (extracted from sidebar logic)"""

    # LLM Model selection
    # LLM Model selection
    st.subheader("LLM Model")
    llm_model = st.selectbox(
        "Model",
        [
            "meta/llama-4-maverick-17b-128e-instruct",
            "./deepseek/deepseek-chat",
            "./openai_chat_completion/gpt-4o-mini",
            "./openai_chat_completion/gpt-5-nano",
            "./grok/grok-3-mini-beta"
        ],
        index=0,
        key="adv_llm_model"
    )

    # Controllers selection
    st.subheader("Controllers")
    available_controllers = ["P", "PI", "PD", "PID", "FSF"]
    selected_controllers_unsorted = st.multiselect(
        "Select Controllers",
        available_controllers,
        default=["PID", "FSF"],
        key="adv_controllers"
    )
    selected_controllers = sorted(selected_controllers_unsorted, key=lambda x: available_controllers.index(x))

    # Design parameters
    st.subheader("Design Parameters")
    col1, col2 = st.columns(2)
    with col1:
        max_scenarios = st.slider("Max Scenarios", 1, 5, 2, key="adv_scenarios")
        seed = st.number_input("Random Seed", 1, 10000, 42, key="adv_seed")
    with col2:
        max_iter = st.slider("Max Iterations", 5, 30, 20, key="adv_iter")
        max_tries = st.slider("Max Tries for Juror", 0, 10, 0, key="adv_tries")

    # Target metrics
    st.subheader("Target Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_mse = st.number_input("Target MSE", 0.01, 1.0, 0.15, format="%.3f", key="adv_mse")
    with col2:
        target_settling = st.number_input("Settling Time (s)", 0.5, 10.0, 3.5, key="adv_settling")
    with col3:
        target_overshoot = st.number_input("Overshoot (%)", 0.0, 50.0, 0.0, key="adv_overshoot")

    # Simulation params
    st.subheader("Simulation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        dt = st.number_input("Sample Time (dt)", 0.001, 1.0, 0.01, format="%.3f", key="adv_dt")
        target = st.number_input("Target Setpoint", -100.0, 100.0, 0.0, format="%.2f", key="adv_target")
        input_channel = st.number_input("Input Channel", 0, 10, 0, step=1, key="adv_input_ch")
    with col2:
        max_time = st.number_input("Max Simulation Time (s)", 0.1, 100.0, 5.0, format="%.1f", key="adv_max_time")
        num_inputs = st.number_input("Number of Inputs", 1, 10, 1, step=1, key="adv_num_inputs")
        output_channel = st.number_input("Output Channel", 0, 10, 0, step=1, key="adv_output_ch")

    # Control limits
    st.subheader("Control Limits")
    col1, col2 = st.columns(2)
    with col1:
        min_ctrl = st.number_input("Min Control Input", -100.0, 0.0, -10.0, format="%.2f", key="adv_min_ctrl")
    with col2:
        max_ctrl = st.number_input("Max Control Input", 0.0, 100.0, 10.0, format="%.2f", key="adv_max_ctrl")

    # Custom system params
    st.subheader("Custom System Parameters")
    matlab_func_name = st.text_input("MATLAB Function Name", "dynamics", key="adv_matlab_func")
    simulink_model_name = st.text_input("Simulink Model Name (without .slx)", "model", key="adv_simulink_model")
    num_states = st.number_input("Number of States", 1, 20, 4, step=1, key="adv_num_states")
    trim_values_str = st.text_input("Trim Values (comma-separated, e.g., 0.0,0.0)", "0.0", key="adv_trim_values")
    trim_values = [float(v.strip()) for v in trim_values_str.split(',') if v.strip()]

    # NEW: Display System Name and Description
    st.subheader("System Display Information")
    display_system_name = st.text_input(
        "Display System Name",
        "Custom System",
        key="adv_display_system_name",
        help="Friendly name for your system (e.g., 'Two-link robotic manipulator')"
    )
    system_description = st.text_area(
        "System Description",
        "A custom dynamical system for control design.",
        key="adv_system_description",
        help="Detailed description of your system"
    )

    # NEW: Token Limit
    st.subheader("LLM Token Limit")
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=10000,
        max_value=500000,
        value=200000,
        step=10000,
        key="adv_max_tokens",
        help="Maximum number of tokens for LLM API calls"
    )

    # Custom Parameter Ranges
    st.subheader("Custom Parameter Ranges")
    custom_param_ranges = {}
    default_ranges = _get_default_param_ranges_for_all()

    pid_controllers_selected = any(c in ["P", "PI", "PD", "PID"] for c in selected_controllers)
    if pid_controllers_selected and st.checkbox("Customize PID-like gains (Kp, Ki, Kd)", key="custom_toggle_pid"):
        with st.expander("PID-like Parameter Ranges", expanded=True):
            unified_ranges = {}
            col_min, col_max = st.columns(2)
            with col_min:
                kp_min = st.number_input("Kp Min", value=0.0, step=0.1, format="%.1f", key="pid_Kp_min")
            with col_max:
                kp_max = st.number_input("Kp Max", value=200.0, step=0.1, format="%.1f", key="pid_Kp_max")
            unified_ranges["Kp"] = [kp_min, kp_max]

            if any(c in ["PI", "PID"] for c in selected_controllers):
                col_min, col_max = st.columns(2)
                with col_min:
                    ki_min = st.number_input("Ki Min", value=0.0, step=0.1, format="%.1f", key="pid_Ki_min")
                with col_max:
                    ki_max = st.number_input("Ki Max", value=50.0, step=0.1, format="%.1f", key="pid_Ki_max")
                unified_ranges["Ki"] = [ki_min, ki_max]

            if any(c in ["PD", "PID"] for c in selected_controllers):
                col_min, col_max = st.columns(2)
                with col_min:
                    kd_min = st.number_input("Kd Min", value=0.0, step=0.1, format="%.1f", key="pid_Kd_min")
                with col_max:
                    kd_max = st.number_input("Kd Max", value=100.0, step=0.1, format="%.1f", key="pid_Kd_max")
                unified_ranges["Kd"] = [kd_min, kd_max]

            for controller in selected_controllers:
                if controller in ["P", "PI", "PD", "PID"]:
                    controller_ranges = {}
                    if "Kp" in unified_ranges:
                        controller_ranges["Kp"] = unified_ranges["Kp"]
                    if controller == "PI" and "Ki" in unified_ranges:
                        controller_ranges["Ki"] = unified_ranges["Ki"]
                    if controller == "PD" and "Kd" in unified_ranges:
                        controller_ranges["Kd"] = unified_ranges["Kd"]
                    if controller == "PID" and "Ki" in unified_ranges and "Kd" in unified_ranges:
                        controller_ranges["Ki"] = unified_ranges["Ki"]
                        controller_ranges["Kd"] = unified_ranges["Kd"]
                    if controller_ranges:
                        custom_param_ranges[controller] = controller_ranges

    if "FSF" in selected_controllers and st.checkbox("Customize FSF gains", key="custom_toggle_fsf"):
        with st.expander("FSF Parameter Ranges", expanded=True):
            fsf_ranges = {}
            for i in range(num_states):
                param_name = f"K{i + 1}"
                default_min, default_max = -50.0, 50.0
                col_min, col_max = st.columns(2)
                with col_min:
                    min_val = st.number_input(f"{param_name} Min", value=default_min, step=0.1,
                                              format="%.1f", key=f"fsf_{param_name}_min")
                with col_max:
                    max_val = st.number_input(f"{param_name} Max", value=default_max, step=0.1,
                                              format="%.1f", key=f"fsf_{param_name}_max")
                fsf_ranges[param_name] = [min_val, max_val]
            if fsf_ranges:
                custom_param_ranges["FSF"] = fsf_ranges

    param_ranges = custom_param_ranges if custom_param_ranges else None

    # Scenario Configurations
    st.subheader("Scenario Configurations")
    custom_scenarios = []
    for i in range(1, max_scenarios + 1):
        with st.expander(f"Scenario {i} (ID: {chr(64 + i)})", expanded=(i == 1)):
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            with col_sc1:
                ic_range = st.slider(f"Initial condition range", -2.0, 2.0, (-1.0, 1.0), key=f"adv_ic_range_{i}")
            with col_sc2:
                randomness_level = st.slider("Measurement noise level", 0.0, 0.5, 0.0, key=f"adv_rand_{i}")
            with col_sc3:
                disturbance_level = st.slider("Input disturbance level", 0.0, 2.0, 0.0, key=f"adv_dist_{i}")
            custom_scenarios.append({
                'id': chr(64 + i),
                'initial_condition_range': ic_range,
                'randomness_level': randomness_level,
                'disturbance_level': disturbance_level,
            })

    # NEW: GA Configuration
    st.divider()
    st.subheader("🧬 Genetic Algorithm (GA) Settings")
    enable_ga = st.checkbox("Enable GA Optimization", value=False, key="enable_ga_checkbox",
                            help="Run GA optimization alongside agentic workflow for comparison")

    ga_config = None
    if enable_ga:
        with st.expander("GA Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ga_population = st.slider("Population Size", 10, 100, 50, key="ga_population")
                ga_generations = st.slider("Generations", 20, 200, 100, key="ga_generations")
                ga_parents = st.slider("Parents Mating", 2, 20, 10, key="ga_parents")
            with col2:
                ga_keep_parents = st.slider("Keep Parents", 0, 10, 2, key="ga_keep_parents")
                ga_crossover_prob = st.slider("Crossover Probability", 0.0, 1.0, 0.8, step=0.05, key="ga_crossover")
                ga_mutation_prob = st.slider("Mutation Probability", 0.0, 1.0, 0.1, step=0.05, key="ga_mutation")

            ga_eval_runs = st.slider("Evaluation Runs (Monte Carlo)", 5, 50, 10, key="ga_eval_runs",
                                     help="Number of Monte Carlo runs for fitness evaluation")

            # Optimization weights
            st.markdown("**Optimization Weights**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                w_mse = st.number_input("MSE Weight", 0.0, 10.0, 1.0, step=0.1, key="ga_w_mse")
            with col2:
                w_settling = st.number_input("Settling Weight", 0.0, 1.0, 0.1, step=0.01, key="ga_w_settling")
            with col3:
                w_overshoot = st.number_input("Overshoot Weight", 0.0, 1.0, 0.01, step=0.001, key="ga_w_overshoot")
            with col4:
                w_control = st.number_input("Control Effort Weight", 0.0, 0.01, 0.001, step=0.0001,
                                            format="%.4f", key="ga_w_control")

            ga_config = {
                'num_generations': ga_generations,
                'population_size': ga_population,
                'num_parents_mating': ga_parents,
                'keep_parents': ga_keep_parents,
                'crossover_probability': ga_crossover_prob,
                'mutation_probability': ga_mutation_prob,
                'mutation_num_genes': 1,
                'random_seed': seed,
                'num_evaluation_runs': ga_eval_runs,
                'weights': {
                    'mse': w_mse,
                    'settling_time': w_settling,
                    'overshoot': w_overshoot,
                    'control_effort': w_control
                }
            }

    return {
        'llm_model': llm_model,
        'controllers': selected_controllers if selected_controllers else None,
        'max_scenarios': max_scenarios,
        'max_iter': max_iter,
        'seed': seed,
        'max_tries': max_tries,
        'target_metrics': {
            'mse': target_mse,
            'settling_time': target_settling,
            'overshoot': target_overshoot,
            'max_iterations': max_iter
        },
        'dt': dt,
        'max_time': max_time,
        'target': target,
        'num_inputs': num_inputs,
        'input_channel': input_channel,
        'output_channel': output_channel,
        'min_ctrl': min_ctrl,
        'max_ctrl': max_ctrl,
        'matlab_func_name': matlab_func_name if matlab_func_name != "dynamics" else None,
        'num_states': num_states,
        'trim_values': trim_values if len(trim_values) == num_inputs else None,
        'param_ranges': param_ranges,
        'custom_scenarios': custom_scenarios if custom_scenarios else None,
        'enable_ga': enable_ga,
        'ga_config': ga_config,
        'matlab_func_name': matlab_func_name if matlab_func_name != "dynamics" else None,
        'simulink_model_name': simulink_model_name if simulink_model_name != "model" else None,
        'display_system_name': display_system_name,
        'system_description': system_description,
        'max_tokens': max_tokens
    }


def run_ga_optimization(config, ga_results_container):
    """Run GA optimization in background thread"""
    try:
        # Extract config
        system_name = config.get('system_name', 'custom')

        # Determine controller type from selected controllers
        controllers = config.get('controllers', ['PID'])
        controller_type = controllers[0] if controllers else 'PID'

        # Get GA config
        ga_config = config.get('ga_config', {})

        # NEW: Get num_states - either from config or by creating a temporary system
        num_states = config.get('num_states')

        if num_states is None and controller_type == 'FSF':
            # Need to determine num_states from the actual system
            try:
                # Create a temporary system to get num_states
                from src.controllers import initialize_state
                temp_init_kwargs = {
                    'system_name': system_name,
                    'custom_dynamics_path': config.get('custom_dynamics_path'),
                    'file_type': config.get('file_type', 'Python (.py)'),
                    'matlab_func_name': config.get('matlab_func_name'),
                    'simulink_model_name': st.session_state.saved_config.get('simulink_model_name'),
                    'num_states': None,  # Let it auto-detect
                    'dt': config.get('dt', 0.01),
                    'max_time': config.get('max_time', 5.0),
                    'target': config.get('target', 0.0),
                    'num_inputs': config.get('num_inputs', 1),
                    'input_channel': config.get('input_channel', 0),
                    'output_channel': config.get('output_channel', 0),
                    'trim_values': config.get('trim_values'),
                    'min_ctrl': config.get('min_ctrl', -10.0),
                    'max_ctrl': config.get('max_ctrl', 10.0),
                    'monitor': None
                }
                temp_state = initialize_state(**temp_init_kwargs)
                if temp_state.get('system'):
                    num_states = temp_state['system'].num_states
                else:
                    raise ValueError("Failed to create system to determine num_states")
            except Exception as e:
                raise ValueError(f"Cannot determine num_states for FSF controller: {e}")

        # Build param_ranges for GA
        param_ranges = config.get('param_ranges', {})
        if not param_ranges:
            # Use defaults
            if controller_type == 'PID':
                param_ranges = {'PID': {'Kp': [0.0, 200.0], 'Ki': [0.0, 50.0], 'Kd': [0.0, 100.0]}}
            elif controller_type == 'FSF':
                if num_states is None:
                    raise ValueError("num_states is required for FSF controller but could not be determined")
                param_ranges = {'FSF': {f'K{i + 1}': [-50.0, 50.0] for i in range(num_states)}}

        # Get scenario config (use first scenario)
        scenarios = config.get('custom_scenarios', [])
        scenario_config = scenarios[0] if scenarios else {
            'initial_condition_range': (-1.0, 1.0),
            'randomness_level': 0.0,
            'disturbance_level': 0.0
        }

        # Create optimizer
        optimizer = GAOptimizer(
            system_name=system_name,
            controller_type=controller_type,
            ga_config=ga_config,
            param_ranges=param_ranges,
            scenario_config=scenario_config,
            num_evaluation_runs=ga_config.get('num_evaluation_runs', 10),
            weights=ga_config.get('weights', {}),
            custom_dynamics_path=config.get('custom_dynamics_path'),
            file_type=config.get('file_type', 'Python (.py)'),
            matlab_func_name=config.get('matlab_func_name'),
            # simulink_model_name=config.get('simulink_model_name'),
            num_states=config.get('num_states'),
            dt=config.get('dt', 0.01),
            max_time=config.get('max_time', 5.0),
            target=config.get('target', 0.0),
            num_inputs=config.get('num_inputs', 1),
            input_channel=config.get('input_channel', 0),
            output_channel=config.get('output_channel', 0),
            trim_values=config.get('trim_values'),
            min_ctrl=config.get('min_ctrl', -10.0),
            max_ctrl=config.get('max_ctrl', 10.0)
        )

        # Run optimization
        best_params, best_fitness, history = optimizer.optimize()

        # Get final metrics
        final_metrics = optimizer.get_performance_metrics(best_params, num_runs=20)

        # Store results
        ga_results_container['best_params'] = best_params
        ga_results_container['best_fitness'] = -best_fitness  # Convert back to cost
        ga_results_container['history'] = history
        ga_results_container['final_metrics'] = final_metrics
        ga_results_container['controller_type'] = controller_type
        ga_results_container['status'] = 'complete'
        ga_results_container['optimizer'] = optimizer  # Store for plotting

    except Exception as e:
        ga_results_container['status'] = 'error'
        ga_results_container['error'] = str(e)
        import traceback
        ga_results_container['traceback'] = traceback.format_exc()


def display_time_response():
    """Display latest system response with interactive gain sliders and GA comparison"""

    # Initialize session state for gains if not present
    if 'manual_gains' not in st.session_state:
        st.session_state.manual_gains = {}
    if 'optimal_gains' not in st.session_state:
        st.session_state.optimal_gains = {}
    if 'test_mode' not in st.session_state:
        st.session_state.test_mode = False
    if 'ga_results' not in st.session_state:
        st.session_state.ga_results = {}

    # Get latest gains and controller type from monitor
    latest_controller_type = None
    latest_gains = {}
    current_system = None

    if st.session_state.monitor.state_history:
        latest_state = st.session_state.monitor.state_history[-1]['state']
        latest_controller_type = latest_state.get('controller_type', None)

        if latest_state.get('current_params'):
            latest_gains = {k: v for k, v in latest_state['current_params'].items()
                            if k != 'reasoning'}

        current_system = latest_state.get('system', None)

    # Scenario selection
    selected_scenario = None
    if 'scenarios' in st.session_state and st.session_state.scenarios:
        scenario_options = [f"Scenario {s['id']} ({s['initial_condition_range']})" for s in st.session_state.scenarios]
        selected_scenario_idx = st.selectbox(
            "Select Scenario for Design/Tuning",
            options=range(len(st.session_state.scenarios)),
            format_func=lambda i: scenario_options[i],
            index=0
        )
        selected_scenario = st.session_state.scenarios[selected_scenario_idx]
        st.markdown(f"**Selected:** {scenario_options[selected_scenario_idx]}")

    # Create columns for sliders and plot
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("**🎚️ Gain Controls**")

        # Show status
        if st.session_state.monitor.is_running:
            st.info("🔒 Sliders locked during optimization")
        else:
            st.success("✅ Optimization complete - sliders active")

            if latest_gains and not st.session_state.optimal_gains:
                st.session_state.optimal_gains = latest_gains.copy()

        # Display current controller type
        if latest_controller_type:
            st.markdown(f"**Controller:** {latest_controller_type}")

            # Get parameter ranges
            param_ranges = {}
            if current_system and hasattr(current_system, 'get_control_param_schema'):
                try:
                    schema = current_system.get_control_param_schema(latest_controller_type)
                    param_ranges = {k: [v["min"], v["max"]] for k, v in schema.items()}
                except:
                    param_ranges = _get_default_param_ranges(latest_controller_type, current_system)
            else:
                param_ranges = _get_default_param_ranges(latest_controller_type, current_system)

            # Create sliders for each gain
            updated_gains = {}
            for param_name in sorted(latest_gains.keys()):
                if param_name == 'reasoning':
                    continue

                param_range = param_ranges.get(param_name, [0.0, 100.0])

                if st.session_state.test_mode and param_name in st.session_state.manual_gains:
                    current_value = st.session_state.manual_gains[param_name]
                else:
                    current_value = latest_gains[param_name]

                optimal_marker = ""
                if param_name in st.session_state.optimal_gains:
                    optimal_val = st.session_state.optimal_gains[param_name]
                    if abs(current_value - optimal_val) > 0.01:
                        optimal_marker = f" (Optimal: {optimal_val:.2f})"

                new_value = st.slider(
                    f"{param_name}{optimal_marker}",
                    min_value=float(param_range[0]),
                    max_value=float(param_range[1]),
                    value=float(current_value),
                    step=0.01,
                    disabled=st.session_state.monitor.is_running,
                    key=f"gain_slider_{param_name}"
                )

                updated_gains[param_name] = new_value

            # Add control buttons
            if not st.session_state.monitor.is_running and latest_gains:
                st.divider()

                col_btn1, col_btn2 = st.columns(2)

                with col_btn1:
                    if st.button("🎯 Reset to Optimal", use_container_width=True,
                                 disabled=not st.session_state.optimal_gains):
                        st.session_state.manual_gains = st.session_state.optimal_gains.copy()
                        st.session_state.test_mode = True
                        st.rerun()

                with col_btn2:
                    if st.button("🧪 Test Current", use_container_width=True):
                        st.session_state.manual_gains = updated_gains
                        st.session_state.test_mode = True
                        st.rerun()

                if st.session_state.test_mode:
                    st.info("📊 Showing response with manual gains")

                    if st.session_state.optimal_gains:
                        st.markdown("**Δ from optimal:**")
                        for param_name in sorted(updated_gains.keys()):
                            if param_name in st.session_state.optimal_gains:
                                diff = updated_gains[param_name] - st.session_state.optimal_gains[param_name]
                                if abs(diff) > 0.01:
                                    st.markdown(f"- {param_name}: {diff:+.2f}")

                # NEW: Show GA results if available
                if st.session_state.ga_results.get('status') == 'complete':
                    st.divider()
                    st.markdown("**🧬 GA Results**")
                    ga_params = st.session_state.ga_results.get('best_params', {})
                    for param_name, param_value in ga_params.items():
                        st.markdown(f"- {param_name}: {param_value:.2f}")

    with col2:
        if st.session_state.monitor.state_history:
            _plot_system_response(
                st.session_state.monitor.state_history,
                test_gains=st.session_state.manual_gains if st.session_state.test_mode else None,
                latest_controller_type=latest_controller_type,
                selected_scenario=selected_scenario,
                ga_results=st.session_state.ga_results if st.session_state.ga_results.get(
                    'status') == 'complete' else None
            )
        else:
            st.info("No simulation data available yet. Start the design process to see results.")


def _get_default_param_ranges(controller_type, system=None):
    """Get default parameter ranges for a controller type"""
    if controller_type == "P":
        return {"Kp": [0.0, 100.0]}
    elif controller_type == "PI":
        return {"Kp": [0.0, 100.0], "Ki": [0.0, 10.0]}
    elif controller_type == "PD":
        return {"Kp": [0.0, 100.0], "Kd": [0.0, 50.0]}
    elif controller_type == "PID":
        return {"Kp": [0.0, 200.0], "Ki": [0.0, 50.0], "Kd": [0.0, 100.0]}
    elif controller_type == "FSF":
        num_states = 4
        if system and hasattr(system, 'num_states'):
            num_states = system.num_states
        return {f"K{i + 1}": [-50.0, 50.0] for i in range(num_states)}
    else:
        return {}


def str_to_array(s):
    """Convert string representation of numpy array back to np.array"""
    if isinstance(s, str):
        if s.startswith('array(') and s.endswith(')'):
            content = s[6:-1]
            content_clean = re.sub(r'\s+', '', content)
            try:
                list_vals = ast.literal_eval(content_clean)
                return np.array(list_vals)
            except (ValueError, SyntaxError):
                pass
    return s


def _plot_system_response(state_history, test_gains=None, latest_controller_type=None, selected_scenario=None,
                          ga_results=None):
    """Plot system response comparing optimal vs manual gains vs GA with same initial conditions"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if not state_history:
        st.info("No simulation data available yet. Start the design process to see results.")
        return

    # Extract from final state
    final_state = state_history[-1]['state']
    target = final_state.get('target', 0.0)
    dt = final_state.get('dt', 0.01)
    max_time = final_state.get('max_time', 5.0)

    latest_results = final_state.get('results') or {}
    optimal_params = {k: v for k, v in final_state.get('current_params', {}).items() if k != 'reasoning'}
    latest_system = final_state.get('system')
    latest_simulator = final_state.get('simulator')
    latest_controller_type = final_state.get('controller_type', latest_controller_type)

    # Fallback to session_state
    if latest_simulator is None and hasattr(st.session_state, 'current_simulator'):
        latest_simulator = st.session_state.current_simulator
    if latest_system is None and hasattr(st.session_state, 'current_system'):
        latest_system = st.session_state.current_system
    if not latest_controller_type and hasattr(st.session_state, 'current_controller_type'):
        latest_controller_type = st.session_state.current_controller_type

    # Fix simulator's internal system reference if needed
    if latest_simulator and hasattr(latest_simulator, 'system') and latest_simulator.system is None:
        if latest_system:
            latest_simulator.system = latest_system

    if not latest_simulator or not optimal_params:
        st.info("No simulation data available yet.")
        return

    # Set controller type
    if latest_controller_type:
        latest_simulator.controller_type = latest_controller_type

    # Set selected scenario on simulator
    if selected_scenario:
        latest_simulator.set_scenario(selected_scenario)
        latest_system = latest_simulator.system
        scenario_str = f" | Scenario {selected_scenario['id']}"
    else:
        default_scenario = {'initial_condition_range': (-1.0, 1.0), 'randomness_level': 0.0, 'disturbance_level': 0.0}
        latest_simulator.set_scenario(default_scenario)
        latest_system = latest_simulator.system
        scenario_str = ""

    # Generate fixed initial condition
    initial_state = np.zeros(latest_system.num_states)
    ic_min, ic_max = latest_system.initial_condition_range
    fixed_ic_value = (ic_min + ic_max) / 2
    initial_state[latest_system.output_channel] = fixed_ic_value

    # Run simulation with optimal parameters
    try:
        optimal_result = latest_simulator.evaluate_parameters(optimal_params, initial_state=initial_state)

        if not optimal_result['success']:
            st.error(f"❌ Optimal simulation failed: {optimal_result.get('error', 'Unknown')}")
            return

        optimal_trajectory = optimal_result['trajectory']
        optimal_control = optimal_result['control_signals']

    except Exception as e:
        st.error(f"❌ Optimal simulation error: {e}")
        return

    # Run simulation with test gains if provided
    test_trajectory = None
    test_control = None

    if test_gains:
        try:
            test_result = latest_simulator.evaluate_parameters(test_gains, initial_state=initial_state)

            if test_result['success']:
                test_trajectory = test_result['trajectory']
                test_control = test_result['control_signals']
            else:
                st.warning(f"⚠️ Test simulation failed: {test_result.get('error', 'Unknown')}")
        except Exception as test_e:
            st.error(f"❌ Test simulation error: {test_e}")

    # NEW: Run simulation with GA parameters if available
    ga_trajectory = None
    ga_control = None

    if ga_results and ga_results.get('best_params'):
        try:
            ga_params = ga_results['best_params']
            ga_result = latest_simulator.evaluate_parameters(ga_params, initial_state=initial_state)

            if ga_result['success']:
                ga_trajectory = ga_result['trajectory']
                ga_control = ga_result['control_signals']
        except Exception as ga_e:
            st.warning(f"⚠️ GA simulation error: {ga_e}")

    # Create time points
    expected_steps = int(max_time / dt) + 1
    time_points = np.arange(0, max_time + dt, dt)[:expected_steps]

    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['System Response', 'Control Input'],
        vertical_spacing=0.15
    )

    # Plot optimal trajectory
    if optimal_trajectory is not None and len(optimal_trajectory) > 0:
        traj_time = time_points[:len(optimal_trajectory)]
        fig.add_trace(
            go.Scatter(x=traj_time, y=optimal_trajectory,
                       mode='lines', name='Agentic (Optimal)',
                       line=dict(color='rgb(255, 99, 132)', width=2)),
            row=1, col=1
        )

    # Plot test trajectory if available
    if test_trajectory is not None and len(test_trajectory) > 0:
        test_traj_time = time_points[:len(test_trajectory)]
        fig.add_trace(
            go.Scatter(x=test_traj_time, y=test_trajectory,
                       mode='lines', name='Manual Test',
                       line=dict(color='rgb(54, 162, 235)', width=2, dash='dash')),
            row=1, col=1
        )

    # NEW: Plot GA trajectory if available
    if ga_trajectory is not None and len(ga_trajectory) > 0:
        ga_traj_time = time_points[:len(ga_trajectory)]
        fig.add_trace(
            go.Scatter(x=ga_traj_time, y=ga_trajectory,
                       mode='lines', name='GA Optimized',
                       line=dict(color='rgb(75, 192, 192)', width=2.5, dash='dot')),
            row=1, col=1
        )

    # Add reference line at target
    fig.add_hline(y=target, line_dash="dash", line_color="green",
                  row=1, col=1, annotation_text="Target")

    # Plot optimal control signals
    if optimal_control is not None and len(optimal_control) > 0:
        ctrl_time = time_points[:len(optimal_control)]
        fig.add_trace(
            go.Scatter(x=ctrl_time, y=optimal_control,
                       mode='lines', name='Agentic Control',
                       line=dict(color='rgb(255, 159, 64)', width=2)),
            row=2, col=1
        )

    # Plot test control signals if available
    if test_control is not None and len(test_control) > 0:
        test_ctrl_time = time_points[:len(test_control)]
        fig.add_trace(
            go.Scatter(x=test_ctrl_time, y=test_control,
                       mode='lines', name='Manual Control',
                       line=dict(color='rgb(153, 102, 255)', width=2, dash='dash')),
            row=2, col=1
        )

    # NEW: Plot GA control signals if available
    if ga_control is not None and len(ga_control) > 0:
        ga_ctrl_time = time_points[:len(ga_control)]
        fig.add_trace(
            go.Scatter(x=ga_ctrl_time, y=ga_control,
                       mode='lines', name='GA Control',
                       line=dict(color='rgb(255, 205, 86)', width=2.5, dash='dot')),
            row=2, col=1
        )

    # Format title
    ic_str = f"IC: x[{latest_system.output_channel}]={fixed_ic_value:.3f} (fixed from range {latest_system.initial_condition_range})"

    if test_gains:
        gains_str = ', '.join([f"{k}:{v:.2f}" for k, v in test_gains.items()])
        title_text = f"Controller: {latest_controller_type} | Test Gains: {gains_str} | {ic_str}{scenario_str}"
    else:
        gains_str = ', '.join([f"{k}:{v:.2f}" for k, v in optimal_params.items()])
        title_text = f"Controller: {latest_controller_type} | Optimized Gains: {gains_str} | {ic_str}{scenario_str}"

    # NEW: Add GA info to title if available
    if ga_results:
        ga_gains_str = ', '.join([f"{k}:{v:.2f}" for k, v in ga_results['best_params'].items()])
        title_text += f" | GA: {ga_gains_str}"

    fig.update_layout(
        height=700,
        showlegend=True,
        title_text=title_text
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Output", row=1, col=1)
    fig.update_yaxes(title_text="Control Signal", row=2, col=1)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "svg", "filename": "time_response"}
        }
    )