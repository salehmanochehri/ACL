import os
import streamlit as st
import threading
import time
import json
from datetime import datetime
from typing import Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import textwrap

from src.controllers_mock import run_optimization, initialize_state, create_optimization_graph
from src.utils import log_to_file
from src.st_utils import (display_logo, CSS_STYLES, create_advanced_settings,
                          process_objective, display_time_response, build_config_from_session,
                          run_ga_optimization)
from session_manager import SessionManager

# Configure Streamlit page
st.set_page_config(
    page_title="Control System Designer",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(CSS_STYLES, unsafe_allow_html=True)


class DummyMonitor:
    """Dummy monitor for safe object recreation without side effects"""

    def __init__(self):
        self.progress_history = []
        self.state_history = []
        self.llm_responses = []
        self.current_state = {}
        self.is_running = False

    def add_progress(self, message: str, data: Dict = None):
        pass

    def add_llm_response(self, agent_name: str, prompt: str, response: str):
        pass

    def update_state(self, update: Dict):
        pass


class DesignMonitor:
    """Monitor class to capture real-time design progress"""

    def __init__(self):
        self.progress_history = []
        self.state_history = []
        self.llm_responses = []
        self.current_state = {}
        self.is_running = False
        self.scenario_metrics_history = []  # NEW: Track per-scenario computational metrics

    def add_progress(self, message: str, data: Dict = None):
        """Add progress update to history list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_history.append({
            'timestamp': timestamp,
            'message': message,
            'data': data or {}
        })

    def add_llm_response(self, agent_name: str, prompt: str, response: str):
        """Add LLM response for monitoring"""
        self.llm_responses.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'agent': agent_name,
            'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
            'response': response
        })

    def update_state(self, update: Dict):
        """Update current state"""
        self.current_state.update(update)
        if 'iteration' in update:
            self.state_history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'state': self.current_state.copy()
            })

    def add_scenario_metrics(self, scenario_level: int, metrics: Dict):
        """NEW: Add per-scenario computational metrics to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.scenario_metrics_history.append({
            'scenario_level': scenario_level,
            'timestamp': timestamp,
            'metrics': metrics  # {'tokens_in': int, 'tokens_out': int, 'time': float (wall-clock + LLM), 'cost': float}
        })
        self.add_progress(f"📊 Scenario {scenario_level} profiling: {metrics['tokens_in']} in, {metrics['tokens_out']} out, {metrics['time']:.1f}s, ${metrics['cost']:.4f}")


# Initialize session manager and state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()

if 'user_id' not in st.session_state:
    st.session_state.user_id = st.session_state.session_manager._get_user_id()

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

if 'monitor' not in st.session_state:
    st.session_state.monitor = DesignMonitor()

if 'design_thread' not in st.session_state:
    st.session_state.design_thread = None

if 'design_results' not in st.session_state:
    st.session_state.design_results = []

if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'show_advanced_settings' not in st.session_state:
    st.session_state.show_advanced_settings = False

if 'pending_objective' not in st.session_state:
    st.session_state.pending_objective = None

if 'pending_file' not in st.session_state:
    st.session_state.pending_file = None

if 'design_auto_started' not in st.session_state:
    st.session_state.design_auto_started = False

# NEW: GA-related session state
if 'ga_results' not in st.session_state:
    st.session_state.ga_results = {}

if 'ga_thread' not in st.session_state:
    st.session_state.ga_thread = None


# $$$
def load_current_session():
    """Load current session data into streamlit state"""
    session_data = st.session_state.session_manager.load_session(
        st.session_state.user_id,
        st.session_state.current_session_id
    )

    if session_data:
        st.session_state.chat_history = session_data.get('chat_history', [])
        st.session_state.control_objective = session_data.get('control_objective', '')
        st.session_state.saved_config = session_data.get('config', {})

        # Restore monitor state
        monitor_state = session_data.get('monitor_state', {})
        st.session_state.monitor.state_history = monitor_state.get('state_history', [])
        st.session_state.monitor.llm_responses = monitor_state.get('llm_responses', [])
        st.session_state.monitor.current_state = monitor_state.get('current_state', {})
        st.session_state.monitor.progress_history = monitor_state.get('progress_history', [])
        st.session_state.monitor.scenario_metrics_history = monitor_state.get('scenario_metrics_history', [])  # NEW: Restore profiling history

        # Reset manual tuning state for the loaded session
        st.session_state.optimal_gains = {}
        st.session_state.manual_gains = {}
        st.session_state.test_mode = False

        # Load custom dynamics if exists
        dynamics_path = st.session_state.session_manager.load_custom_dynamics(
            st.session_state.user_id,
            st.session_state.current_session_id
        )
        if dynamics_path:
            st.session_state.custom_dynamics_path = dynamics_path
            st.session_state.uploaded_file_name = os.path.basename(dynamics_path).split('_', 1)[-1]
            file_extension = dynamics_path.split('.')[-1]
            st.session_state.file_type = "Python (.py)" if file_extension == "py" else "MATLAB/Octave (.m)"

            if 'custom_dynamics_path' not in st.session_state.saved_config:
                st.session_state.saved_config['custom_dynamics_path'] = dynamics_path

        # Recreate non-serializable objects using saved config
        if st.session_state.saved_config:
            try:
                dummy_monitor = DummyMonitor()

                # NEW: Filter out GA-specific keys before passing to initialize_state
                ga_keys = {'enable_ga', 'ga_config'}
                filtered_config = {k: v for k, v in st.session_state.saved_config.items() if k not in ga_keys}

                # Extract simulation parameters from config with defaults
                init_kwargs = {
                    **filtered_config,
                    'dt': st.session_state.saved_config.get('dt', 0.01),
                    'max_time': st.session_state.saved_config.get('max_time', 5.0),
                    'target': st.session_state.saved_config.get('target', 0.1),
                    'num_inputs': st.session_state.saved_config.get('num_inputs', 2),
                    'input_channel': st.session_state.saved_config.get('input_channel', 0),
                    'output_channel': st.session_state.saved_config.get('output_channel', 0),
                    # NEW: Add trim_values, num_states, matlab_func_name, min_ctrl, max_ctrl
                    'trim_values': st.session_state.saved_config.get('trim_values'),
                    'num_states': st.session_state.saved_config.get('num_states'),
                    'matlab_func_name': st.session_state.saved_config.get('matlab_func_name'),
                    'simulink_model_name': st.session_state.saved_config.get('simulink_model_name'),
                    'min_ctrl': st.session_state.saved_config.get('min_ctrl', -10.0),
                    'max_ctrl': st.session_state.saved_config.get('max_ctrl', 10.0),
                    'monitor': dummy_monitor
                }

                fresh_state = initialize_state(**init_kwargs)
                if fresh_state.get('simulator') is None:
                    st.error("❌ Failed to create simulator")
                    return

                simulator = fresh_state.get('simulator')
                system = fresh_state.get('system')

                # Ensure simulator's internal system reference is set
                if simulator and system:
                    if hasattr(simulator, 'system') and simulator.system is None:
                        simulator.system = system

                # Get loaded current state for merging
                loaded_current = st.session_state.monitor.current_state

                # Merge loaded data into fresh state
                merge_keys = ['controller_type', 'current_params', 'results']
                for key in merge_keys:
                    if key in loaded_current:
                        fresh_state[key] = loaded_current[key]

                # Update with fresh objects
                loaded_current['system'] = system
                loaded_current['simulator'] = simulator

                # Store in session_state for direct access
                st.session_state.current_system = system
                st.session_state.current_simulator = simulator
                st.session_state.current_controller_type = loaded_current.get('controller_type')

                # Update last history entry
                if st.session_state.monitor.state_history:
                    last_state = st.session_state.monitor.state_history[-1]['state']
                    last_state['system'] = system
                    last_state['simulator'] = simulator

                    for key in merge_keys:
                        if key in loaded_current:
                            last_state[key] = loaded_current[key]

                # Set controller_type on simulator
                controller_type = loaded_current.get('controller_type')

                if simulator and controller_type:
                    simulator.controller_type = controller_type

                    # Prime the simulator with loaded optimal params
                    if 'current_params' in loaded_current:
                        optimal_params = {k: v for k, v in loaded_current['current_params'].items()
                                          if k != 'reasoning'}

                        try:
                            init_result = simulator.evaluate_parameters(optimal_params)
                            if init_result['success']:
                                fresh_state['results'] = init_result
                                loaded_current['results'] = init_result
                                if st.session_state.monitor.state_history:
                                    last_state['results'] = init_result
                        except Exception as init_e:
                            st.warning(f"⚠️ Simulator priming failed: {init_e}")

            except Exception as e:
                st.error(f"❌ Failed to recreate system/simulator: {e}")
                import traceback
                st.code(traceback.format_exc(), language='python')


def save_current_session():
    """Save current session data"""
    monitor_state = get_serializable_monitor_state(st.session_state.monitor)

    # Ensure config includes trim_values if available
    if 'saved_config' in st.session_state:
        if 'trim_values' in st.session_state.saved_config:
            # Convert to list for serialization if numpy array
            trim_vals = st.session_state.saved_config['trim_values']
            if hasattr(trim_vals, 'tolist'):
                st.session_state.saved_config['trim_values'] = trim_vals.tolist()

        # NEW: Ensure min_ctrl and max_ctrl are serialized as floats
        if 'min_ctrl' in st.session_state.saved_config:
            st.session_state.saved_config['min_ctrl'] = float(st.session_state.saved_config['min_ctrl'])
        if 'max_ctrl' in st.session_state.saved_config:
            st.session_state.saved_config['max_ctrl'] = float(st.session_state.saved_config['max_ctrl'])

    st.session_state.session_manager.update_session(
        st.session_state.user_id,
        st.session_state.current_session_id,
        {
            'chat_history': st.session_state.get('chat_history', []),
            'control_objective': st.session_state.get('control_objective', ''),
            'config': st.session_state.get('saved_config', {}),
            'monitor_state': monitor_state
        }
    )


def get_serializable_monitor_state(monitor):
    """Create a JSON-serializable copy of monitor state by converting non-serializable objects"""
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert non-serializable objects to string representation
            return repr(obj)

    return {
        'state_history': [
            {
                'timestamp': entry['timestamp'],
                'state': to_serializable(entry['state'])
            } for entry in monitor.state_history
        ],
        'llm_responses': monitor.llm_responses,  # Already serializable
        'current_state': to_serializable(monitor.current_state),
        'progress_history': monitor.progress_history,  # Already serializable
        'scenario_metrics_history': [  # NEW: Serialize per-scenario metrics
            {
                'scenario_level': entry['scenario_level'],
                'timestamp': entry['timestamp'],
                'metrics': to_serializable(entry['metrics'])
            } for entry in monitor.scenario_metrics_history
        ]
    }


def display_session_sidebar_home():
    """Display simplified session history sidebar for home page"""

    # Sidebar logo (small version)
    logo_path = Path("assets/logo.svg")
    if logo_path.exists():
        with open(logo_path, 'r') as f:
            logo_svg = f.read()
        logo_svg = logo_svg.replace('<svg', '<svg style="width:100%; height:auto; max-width:80px;"')
    else:
        # Fallback to default logo (scaled down)
        logo_svg = """
        <svg style="width:100%; height:auto; max-width:80px;" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <circle cx="100" cy="100" r="80" fill="#1f77b4" opacity="0.2"/>
            <circle cx="100" cy="100" r="60" fill="#1f77b4" opacity="0.4"/>
            <circle cx="100" cy="100" r="40" fill="#1f77b4" opacity="0.6"/>
            <text x="100" y="110" font-size="24" font-weight="bold" 
                  text-anchor="middle" fill="#1f77b4">🎛️</text>
        </svg>
        """

    html_content = textwrap.dedent(f"""
        <div style='text-align: center; padding: 0.5rem 0;'>
            <div style='max-width: 80px; margin: 0 auto;'>
                {logo_svg}
            </div>
        </div>
    """).strip()
    lines = html_content.splitlines()
    html_content = '\n'.join(line.lstrip() for line in lines)
    st.sidebar.markdown(html_content, unsafe_allow_html=True)

    st.sidebar.title("💬 Session History")

    # Get all sessions
    sessions = st.session_state.session_manager.get_all_sessions(st.session_state.user_id)

    if sessions:
        for session in sessions:
            session_id = session['session_id']

            # Session button
            if st.sidebar.button(
                    session['title'][:30] + ("..." if len(session['title']) > 30 else ""),
                    key=f"home_session_{session_id}",
                    width='stretch'
            ):
                # Reset monitor first
                st.session_state.monitor = DesignMonitor()
                st.session_state.test_mode = False
                st.session_state.manual_gains = {}
                st.session_state.optimal_gains = {}
                # Load session and switch to project page
                st.session_state.current_session_id = session_id
                load_current_session()
                st.session_state.page = 'project'
                st.rerun()
    else:
        st.sidebar.info("No previous sessions")


def display_session_sidebar_project():
    """Display improved session sidebar for project page"""

    # Sidebar logo (small version) - at the very top of the sidebar
    logo_path = Path("assets/logo.svg")
    if logo_path.exists():
        with open(logo_path, 'r') as f:
            logo_svg = f.read()
        logo_svg = logo_svg.replace('<svg', '<svg style="width:100%; height:auto; max-width:80px;"')
    else:
        # Fallback to default logo (scaled down)
        logo_svg = """
        <svg style="width:100%; height:auto; max-width:80px;" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <circle cx="100" cy="100" r="80" fill="#1f77b4" opacity="0.2"/>
            <circle cx="100" cy="100" r="60" fill="#1f77b4" opacity="0.4"/>
            <circle cx="100" cy="100" r="40" fill="#1f77b4" opacity="0.6"/>
            <text x="100" y="110" font-size="24" font-weight="bold" 
                  text-anchor="middle" fill="#1f77b4">🎛️</text>
        </svg>
        """

    html_content = textwrap.dedent(f"""
        <div style='text-align: center; padding: 0.5rem 0 1rem 0;'>
            <div style='max-width: 80px; margin: 0 auto;'>
                {logo_svg}
            </div>
        </div>
    """).strip()
    lines = html_content.splitlines()
    html_content = '\n'.join(line.lstrip() for line in lines)
    st.sidebar.markdown(html_content, unsafe_allow_html=True)

    # Add toggle state
    if 'show_session_actions' not in st.session_state:
        st.session_state.show_session_actions = False

    # Fixed header section
    with st.sidebar.container():
        if st.button("🏠 New Project", width='stretch', type="primary"):
            save_current_session()
            st.session_state.page = 'home'
            st.session_state.pending_objective = None
            st.session_state.pending_file = None
            st.session_state.show_advanced_settings = False
            st.session_state.objective_streamed = False
            st.rerun()

        st.divider()

    # Sessions section header with toggle
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.subheader("💬 Recent Sessions")
    with col2:
        if st.button("⚙️", help="Toggle edit actions"):
            st.session_state.show_session_actions = not st.session_state.show_session_actions
            st.rerun()

    # Scrollable sessions container
    with st.sidebar.container():
        sessions = st.session_state.session_manager.get_all_sessions(st.session_state.user_id)

        if sessions:
            for session in sessions:
                session_id = session['session_id']
                is_active = session_id == st.session_state.current_session_id

                # Session item
                if st.session_state.show_session_actions:
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col1:
                        if st.button(
                                session['title'][:30] + ("..." if len(session['title']) > 30 else ""),
                                key=f"session_{session_id}",
                                width='stretch',
                                disabled=is_active
                        ):
                            if not is_active:
                                save_current_session()
                                # Reset monitor first
                                st.session_state.monitor = DesignMonitor()
                                # Load session
                                st.session_state.current_session_id = session_id
                                load_current_session()
                                st.session_state.objective_streamed = True
                                st.rerun()

                    with col2:
                        if st.button("✏️", key=f"rename_{session_id}", help="Rename"):
                            st.session_state.renaming_session = session_id

                    with col3:
                        if st.button("🗑️", key=f"delete_{session_id}", help="Delete"):
                            if not is_active:
                                st.session_state.session_manager.delete_session(
                                    st.session_state.user_id,
                                    session_id
                                )
                                st.rerun()
                else:
                    # Simple button without actions
                    if st.button(
                            session['title'][:30] + ("..." if len(session['title']) > 30 else ""),
                            key=f"session_{session_id}",
                            width='stretch',
                            disabled=is_active
                    ):
                        if not is_active:
                            save_current_session()
                            # Reset monitor first
                            st.session_state.monitor = DesignMonitor()
                            # Load session
                            st.session_state.current_session_id = session_id
                            load_current_session()
                            st.session_state.objective_streamed = True
                            st.rerun()

                # Handle rename dialog
                if st.session_state.get('renaming_session') == session_id:
                    new_title = st.sidebar.text_input(
                        "New title:",
                        value=session['title'],
                        key=f"rename_input_{session_id}"
                    )
                    col_ok, col_cancel = st.sidebar.columns(2)
                    with col_ok:
                        if st.button("✔", key=f"confirm_rename_{session_id}"):
                            st.session_state.session_manager.rename_session(
                                st.session_state.user_id,
                                session_id,
                                new_title
                            )
                            del st.session_state.renaming_session
                            st.rerun()
                    with col_cancel:
                        if st.button("✗", key=f"cancel_rename_{session_id}"):
                            del st.session_state.renaming_session
                            st.rerun()
        else:
            st.sidebar.info("No previous sessions")

    st.sidebar.divider()

    # Export/Import buttons (fixed at bottom)
    st.sidebar.subheader("💾 Backup & Restore")

    col_exp, col_imp = st.sidebar.columns(2)

    with col_exp:
        if st.button("📤 Export", width='stretch'):
            export_data = st.session_state.session_manager.export_session(
                st.session_state.user_id,
                st.session_state.current_session_id
            )
            if export_data:
                st.sidebar.download_button(
                    "⬇️ Download",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"session_{st.session_state.current_session_id}.json",
                    mime="application/json",
                    width='stretch'
                )

    with col_imp:
        uploaded_session = st.file_uploader(
            "Import Session",
            type=['json'],
            key="import_session",
            label_visibility="collapsed"
        )
        if uploaded_session:
            try:
                import_data = json.load(uploaded_session)
                new_session_id = st.session_state.session_manager.import_session(
                    st.session_state.user_id,
                    import_data
                )
                st.sidebar.success("Imported session!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Import failed: {str(e)}")


def run_design_with_monitoring(config: Dict, monitor: DesignMonitor):
    """Run the design process with real-time monitoring"""
    monitor.is_running = True
    monitor.add_progress("🚀 Starting Control System Design Process...")

    try:
        # Create modified versions of the functions to include monitoring
        original_run_optimization = run_optimization

        def monitored_run_optimization(**kwargs):
            monitor.add_progress(f"📋 Initializing design with system: {kwargs.get('system_name', 'unknown')}")

            # Create graph and initial state
            graph, graph_config = create_optimization_graph(
                kwargs.get('max_scenarios', 3),
                kwargs.get('max_iter', 10)
            )

            # NEW: Filter out GA-specific keys before passing to initialize_state
            ga_keys = {'enable_ga', 'ga_config'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ga_keys}

            init_kwargs = {
                **filtered_kwargs,
                'dt': filtered_kwargs.get('dt', 0.01),
                'max_time': filtered_kwargs.get('max_time', 5.0),
                'target': filtered_kwargs.get('target', 0.0),
                'num_inputs': filtered_kwargs.get('num_inputs', 1),
                'input_channel': filtered_kwargs.get('input_channel', 0),
                'output_channel': filtered_kwargs.get('output_channel', 0),
                # NEW: Pass trim_values, num_states, matlab_func_name, min_ctrl, max_ctrl
                'trim_values': filtered_kwargs.get('trim_values'),
                'num_states': filtered_kwargs.get('num_states'),
                'matlab_func_name': filtered_kwargs.get('matlab_func_name'),
                'simulink_model_name': filtered_kwargs.get('simulink_model_name'),
                'min_ctrl': filtered_kwargs.get('min_ctrl', -10.0),
                'max_ctrl': filtered_kwargs.get('max_ctrl', 10.0),
                'monitor': monitor
            }

            initial_state = initialize_state(**init_kwargs)
            monitor.current_state = initial_state.copy()

            log_to_file(f"=== CONTROL DESIGN LOG - {datetime.now()} ===\n\n", True)

            # Stream the graph execution with monitoring
            step_count = 0
            for step_output in graph.stream(initial_state, config={"recursion_limit": 1000}):
                step_count += 1

                # Extract state information from step output
                if step_output:
                    for node_name, node_output in step_output.items():
                        monitor.add_progress(f"⚙️ Executing: {node_name}")

                        # Update state with node output
                        if isinstance(node_output, dict):
                            monitor.update_state(node_output)

                            # Special handling for different node types
                            if node_name == "propose_parameters" and "current_params" in node_output:
                                params = node_output["current_params"]
                                monitor.add_progress(f"🎯 Proposed Parameters: {params}")

                            elif node_name == "run_simulation" and "results" in node_output:
                                results = node_output["results"]
                                if results and results.get("success"):
                                    metrics = results.get("metrics", {})
                                    monitor.add_progress(
                                        f"📊 Simulation Results - MSE: {metrics.get('mse', 'N/A'):.4f}, "
                                        f"Settling Time: {metrics.get('settling_time', 'N/A'):.2f}s")

                            elif node_name == "evaluate_performance" and "feedback" in node_output:
                                feedback = node_output["feedback"]
                                try:
                                    feedback_data = json.loads(feedback) if isinstance(feedback, str) else feedback
                                    monitor.add_progress(f"🔍 Performance Analysis Complete")
                                except:
                                    monitor.add_progress(f"🔍 Performance Analysis: {str(feedback)[:100]}...")

                        if step_count > 1000:  # Safety break
                            break

            monitor.add_progress("✅ Design process completed!")

        # Run the monitored optimization
        monitored_run_optimization(**config)

    except Exception as e:
        monitor.add_progress(f"❌ Error during design: {str(e)}")
        st.error(f"Design process failed: {str(e)}")
    finally:
        monitor.is_running = False

def display_llm_responses():
    """Display recent LLM responses"""
    if st.session_state.monitor.llm_responses:
        # Show last 10 responses
        for response in st.session_state.monitor.llm_responses[-10:]:
            with st.expander(f"[{response['timestamp']}] {response['agent']}", expanded=False):
                st.text(f"Prompt: {response['prompt']}")
                st.code(response['response'], language='json')

def display_current_metrics():
    """Display current design metrics"""
    if st.session_state.monitor.current_state:
        state = st.session_state.monitor.current_state

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Iteration", state.get('iteration', 0))

        with col2:
            st.metric("Scenario Level", state.get('scenario_level', 0))

        with col3:
            # Fixed: Better handling of controller type
            controller_type = state.get('controller_type', None)
            if controller_type:
                current_controller = controller_type
            else:
                controller_idx = state.get('current_controller_index', 0)
                controllers = state.get('controllers_list', [])
                current_controller = controllers[controller_idx] if controllers and controller_idx < len(
                    controllers) else 'Unknown'
            st.metric("Current Controller", current_controller)

        with col4:
            if state.get('results') and state['results'].get('metrics'):
                mse = state['results']['metrics'].get('mse', float('inf'))
                if mse != float('inf'):
                    st.metric("Current MSE", f"{mse:.4f}")
                else:
                    st.metric("Current MSE", "∞")

        # NEW: Simulation parameters display (unchanged)
        st.divider()
        st.subheader("Simulation Configuration")

        # FIXED: Get min_ctrl and max_ctrl from saved_config if not in state
        saved_config = st.session_state.get('saved_config', {})
        min_ctrl = state.get('min_ctrl', saved_config.get('min_ctrl', -10.0))
        max_ctrl = state.get('max_ctrl', saved_config.get('max_ctrl', 10.0))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Time (dt)", f"{state.get('dt', 0.01):.3f} s")
            st.metric("Target Setpoint", f"{state.get('target', 0.0):.2f}")
            st.metric("Min Control", f"{min_ctrl:.2f}")  # FIXED
        with col2:
            st.metric("Max Time", f"{state.get('max_time', 5.0):.1f} s")
            st.metric("Input Channel", state.get('input_channel', 0))
            st.metric("Max Control", f"{max_ctrl:.2f}")  # FIXED
        with col3:
            st.metric("Number of Inputs", state.get('num_inputs', 1))
            st.metric("Output Channel", state.get('output_channel', 0))

        # NEW: Computational Profiling Section
        st.divider()
        st.subheader("🖥️ Computational Profiling")
        monitor = st.session_state.monitor
        if monitor.scenario_metrics_history:
            # Cumulative totals
            total_tokens_in = sum(m['metrics']['tokens_in'] for m in monitor.scenario_metrics_history)
            total_tokens_out = sum(m['metrics']['tokens_out'] for m in monitor.scenario_metrics_history)
            total_time = sum(m['metrics']['time'] for m in monitor.scenario_metrics_history)
            total_cost = sum(m['metrics']['cost'] for m in monitor.scenario_metrics_history)

            col_cum1, col_cum2, col_cum3, col_cum4 = st.columns(4)
            with col_cum1:
                st.metric("Total Tokens In", total_tokens_in)
            with col_cum2:
                st.metric("Total Tokens Out", total_tokens_out)
            with col_cum3:
                st.metric("Total Time", f"{total_time:.1f}s")
            with col_cum4:
                st.metric("Total Cost", f"${total_cost:.4f}")

            # Per-scenario table
            st.subheader("Per-Scenario Breakdown")
            scenario_data = []
            for entry in monitor.scenario_metrics_history:
                m = entry['metrics']
                scenario_data.append({
                    'Level': entry['scenario_level'],
                    'Tokens In': m['tokens_in'],
                    'Tokens Out': m['tokens_out'],
                    'Time (s)': f"{m['time']:.1f}",
                    'Cost ($)': f"${m['cost']:.4f}"
                })
            st.table(scenario_data)
        else:
            st.info("No profiling data yet. Run a scenario to see metrics.")
# $$$


def display_home_page():
    """Display the home page with ChatGPT-like interface"""

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        display_logo()

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Advanced settings toggle
        if st.button("⚙️ Advanced Settings", width='stretch'):
            st.session_state.show_advanced_settings = not st.session_state.show_advanced_settings
            st.rerun()

        # Show advanced settings if toggled
        if st.session_state.show_advanced_settings:
            with st.expander("🔧 Advanced Configuration", expanded=True):
                config = create_advanced_settings()
                st.session_state.temp_config = config

        st.markdown("<br>", unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader(
            "📎 Upload Custom Dynamics (Python .py, MATLAB .m, or Simulink .slx)",
            type=["py", "m", "slx"],
            key="home_file_upload",
            help="Upload your system dynamics file"
        )

        # Show uploaded file content
        if uploaded_file is not None:
            with st.expander(f"📄 View: {uploaded_file.name}", expanded=False):
                if uploaded_file.name.endswith('.slx'):
                    st.info("📊 Simulink model file (.slx) - Binary format, cannot display content")
                else:
                    file_content = uploaded_file.getvalue().decode('utf-8')
                    st.code(file_content, language='python' if uploaded_file.name.endswith('.py') else 'matlab')

        # MATLAB file inputs
        matlab_func_name = None
        simulink_model_name = None
        num_states = None

        if uploaded_file:
            if uploaded_file.name.endswith('.m'):
                matlab_func_name = st.text_input("MATLAB Function Name", "dynamics", key="home_matlab_func")
                num_states = st.number_input("Number of States", 1, 20, 4, step=1, key="home_num_states")
            elif uploaded_file.name.endswith('.slx'):
                simulink_model_name = st.text_input(
                    "Simulink Model Name (without .slx)",
                    uploaded_file.name.replace('.slx', ''),
                    key="home_simulink_model"
                )
                num_states = st.number_input("Number of States", 1, 20, 4, step=1, key="home_num_states_slx")
                st.info("ℹ️ For Simulink models, specify the model name and number of states")

        # Chat input
        user_input = st.chat_input(
            "Describe your control objective and upload dynamics file...",
            key="home_chat_input"
        )

        # Process user input
        if user_input:
            if not uploaded_file:
                st.warning("⚠️ Please upload a dynamics file before submitting.")
            else:
                st.session_state.pending_objective = user_input
                st.session_state.pending_file = uploaded_file

                with st.spinner("🤔 Processing your objective with AI..."):
                    refined_objective = process_objective(user_input)

                    new_session_id = st.session_state.session_manager.create_session(
                        st.session_state.user_id,
                        title=user_input[:50] + ("..." if len(user_input) > 50 else "")
                    )

                    st.session_state.current_session_id = new_session_id

                    file_extension = uploaded_file.name.split('.')[-1]
                    custom_dynamics_path = st.session_state.session_manager.save_custom_dynamics(
                        st.session_state.user_id,
                        new_session_id,
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )

                    st.session_state.chat_history = [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": refined_objective}
                    ]
                    st.session_state.control_objective = refined_objective
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.custom_dynamics_path = custom_dynamics_path
                    if file_extension == "py":
                        st.session_state.file_type = "Python (.py)"
                    elif file_extension == "m":
                        st.session_state.file_type = "MATLAB/Octave (.m)"
                    elif file_extension == "slx":
                        st.session_state.file_type = "Simulink (.slx)"

                    if file_extension == "m":
                        st.session_state.matlab_func_name = matlab_func_name
                        st.session_state.num_states = num_states
                    elif file_extension == "slx":
                        st.session_state.simulink_model_name = simulink_model_name
                        st.session_state.num_states = num_states

                    save_current_session()

                    st.session_state.monitor = DesignMonitor()
                    st.session_state.test_mode = False
                    st.session_state.manual_gains = {}
                    st.session_state.optimal_gains = {}
                    st.session_state.ga_results = {}  # Reset GA results

                    st.session_state.design_auto_started = True
                    st.session_state.page = 'project'
                    st.rerun()


def display_project_page():
    """Display the project page with design results"""

    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []

    st.markdown("")
    with st.container():
        col1, col2 = st.columns([1, 20])
        with col1:
            st.markdown("👤")
        with col2:
            st.markdown("**Your Request:**")
            if st.session_state.chat_history:
                st.markdown(
                    f'<div class="speech-bubble-user">{st.session_state.chat_history[0]["content"]}</div>',
                    unsafe_allow_html=True
                )

            if 'uploaded_file_name' in st.session_state:
                with st.expander(f"📎 Uploaded: `{st.session_state.uploaded_file_name}`"):
                    dynamics_path = st.session_state.get('custom_dynamics_path')
                    if dynamics_path and os.path.exists(dynamics_path):
                        if dynamics_path.endswith('.slx'):
                            st.info("📊 Simulink model file (.slx) - Binary format")
                        else:
                            with open(dynamics_path, 'r') as f:
                                file_content = f.read()
                                st.code(file_content, language='python' if dynamics_path.endswith('.py') else 'matlab')

    with st.container():
        col1, col2 = st.columns([1, 20])
        with col1:
            st.markdown("🕵️")
        with col2:
            st.markdown("**Refined Control Objective:**")
            if len(st.session_state.chat_history) > 1:
                if 'objective_streamed' not in st.session_state:
                    st.session_state.objective_streamed = False

                refined_text = st.session_state.chat_history[1]['content']

                if not st.session_state.objective_streamed:
                    placeholder = st.empty()
                    displayed_text = ""
                    for char in refined_text:
                        displayed_text += char
                        placeholder.markdown(
                            f'<div class="speech-bubble-assistant">{displayed_text}</div>',
                            unsafe_allow_html=True
                        )
                        time.sleep(0.01)
                    st.session_state.objective_streamed = True
                else:
                    st.markdown(
                        f'<div class="speech-bubble-assistant">{refined_text}</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("")

    # Auto-trigger design process
    if st.session_state.get('design_auto_started', False) and not st.session_state.monitor.is_running:
        st.session_state.design_auto_started = False
        config = build_config_from_session()
        st.session_state.saved_config = config
        st.session_state.scenarios = config.get('custom_scenarios', [])
        save_current_session()

        st.session_state.test_mode = False
        st.session_state.manual_gains = {}
        st.session_state.optimal_gains = {}

        st.session_state.monitor = DesignMonitor()
        st.session_state.design_thread = threading.Thread(
            target=run_design_with_monitoring,
            args=(config, st.session_state.monitor),
            daemon=True
        )
        st.session_state.design_thread.start()

        # NEW: Start GA optimization if enabled
        if config.get('enable_ga', False) and config.get('ga_config'):
            st.session_state.ga_results = {'status': 'running'}
            st.session_state.ga_thread = threading.Thread(
                target=run_ga_optimization,
                args=(config, st.session_state.ga_results),
                daemon=True
            )
            st.session_state.ga_thread.start()

        st.rerun()

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start_button = st.button("🚀 Start Design Process", type="primary", width='stretch')
    with col_btn2:
        stop_button = st.button("🛑 Stop Process", type="secondary", width='stretch')
    with col_btn3:
        # NEW: Show GA status
        if st.session_state.ga_results.get('status') == 'running':
            st.info("🧬 GA Running...")
        elif st.session_state.ga_results.get('status') == 'complete':
            st.success("🧬 GA Complete")
        elif st.session_state.ga_results.get('status') == 'error':
            st.error("🧬 GA Error")

    # Status indicator
    if st.session_state.monitor.is_running:
        st.markdown('<p class="status-running">🔄 Design process is running...</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-complete">⏸️ Design process is idle</p>', unsafe_allow_html=True)

    # Tabs for different views
    tabs = st.tabs(["📈 Progress", "📊 Metrics", "🎚️ Gains", "⏱️ Time Response",
                    "🕵️ LLM Agents", "🧬 GA Results", "⚙️ Config", "📋 Summary"])

    with tabs[0]:  # Progress
        display_progress_feed()

    with tabs[1]:  # Metrics
        display_metrics_plots()

    with tabs[2]:  # Gains
        display_gains_plot()

    with tabs[3]:  # Time Response
        display_time_response()

    with tabs[4]:  # LLM Responses
        display_llm_responses()

    with tabs[5]:  # NEW: GA Results
        display_ga_results()

    with tabs[6]:  # Current Config
        if 'saved_config' in st.session_state:
            st.json(st.session_state.saved_config)

    with tabs[7]:  # Summary
        display_current_metrics()

    # Handle start button
    if start_button and not st.session_state.monitor.is_running:
        config = build_config_from_session()
        st.session_state.saved_config = config
        st.session_state.scenarios = config.get('custom_scenarios', [])
        save_current_session()

        st.session_state.test_mode = False
        st.session_state.manual_gains = {}
        st.session_state.optimal_gains = {}

        st.session_state.monitor = DesignMonitor()
        st.session_state.design_thread = threading.Thread(
            target=run_design_with_monitoring,
            args=(config, st.session_state.monitor),
            daemon=True
        )
        st.session_state.design_thread.start()

        # NEW: Start GA if enabled
        if config.get('enable_ga', False) and config.get('ga_config'):
            st.session_state.ga_results = {'status': 'running'}
            st.session_state.ga_thread = threading.Thread(
                target=run_ga_optimization,
                args=(config, st.session_state.ga_results),
                daemon=True
            )
            st.session_state.ga_thread.start()

        st.rerun()

    # Handle stop button
    if stop_button and st.session_state.monitor.is_running:
        st.session_state.monitor.is_running = False
        st.warning("Design process stop requested...")
        st.rerun()

    # Auto-refresh while running
    if st.session_state.monitor.is_running or st.session_state.ga_results.get('status') == 'running':
        save_current_session()
        time.sleep(1)
        st.rerun()


def display_progress_feed():
    """Display real-time progress feed"""
    progress_container = st.container()

    if st.session_state.monitor.progress_history:
        for update in st.session_state.monitor.progress_history[-10:]:
            with progress_container:
                st.text(f"[{update['timestamp']}] {update['message']}")


def display_metrics_plots():
    """Display performance plots with GA comparison"""
    if st.session_state.monitor.state_history:
        # Extract performance data from agentic workflow
        mse_values = []
        settling_times = []
        overshoots = []
        iterations = []

        global_step = 0
        last_metrics = None

        for entry in st.session_state.monitor.state_history:
            state = entry['state']
            if state.get('results') and state['results'].get('metrics'):
                metrics = state['results']['metrics']
                metrics_tuple = tuple((k, v) for k, v in sorted(metrics.items()))

                if metrics_tuple == last_metrics:
                    continue

                last_metrics = metrics_tuple
                global_step += 1
                iterations.append(global_step)
                mse_values.append(np.nan if not np.isfinite(metrics.get('mse', np.inf)) else metrics.get('mse'))
                settling_time = metrics.get('settling_time', np.inf)
                settling_times.append(np.nan if not np.isfinite(settling_time) else settling_time)
                overshoots.append(metrics.get('overshoot', 0))

        if iterations:
            target_metrics = st.session_state.saved_config.get('target_metrics', {})
            target_mse = target_metrics.get('mse', 0.15)
            target_settling = target_metrics.get('settling_time', 3.5)
            target_overshoot = target_metrics.get('overshoot', 0.0)

            # Create 2x2 subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['MSE Progress', 'Settling Time', 'Overshoot', 'Performance Summary']
            )

            # MSE plot
            fig.add_trace(
                go.Scatter(x=iterations, y=mse_values, mode='lines+markers',
                           name='Agentic MSE', line=dict(color='rgb(255, 99, 132)'), showlegend=True),
                row=1, col=1
            )
            fig.add_hline(y=target_mse, line_dash="dash", line_color="red",
                          annotation_text=f"Target: {target_mse:.3f}", row=1, col=1)

            # NEW: Add GA MSE reference if available
            if st.session_state.ga_results.get('status') == 'complete':
                ga_mse = st.session_state.ga_results['final_metrics'].get('mse', float('inf'))
                if np.isfinite(ga_mse):
                    fig.add_hline(y=ga_mse, line_dash="dot", line_color="rgb(75, 192, 192)",
                                  annotation_text=f"GA: {ga_mse:.3f}", row=1, col=1)

            # Settling time plot
            fig.add_trace(
                go.Scatter(x=iterations, y=settling_times, mode='lines+markers',
                           name='Agentic Settling', line=dict(color='rgb(255, 99, 132)'), showlegend=True),
                row=1, col=2
            )
            fig.add_hline(y=target_settling, line_dash="dash", line_color="red",
                          annotation_text=f"Target: {target_settling:.2f}s", row=1, col=2)

            # NEW: Add GA settling time reference
            if st.session_state.ga_results.get('status') == 'complete':
                ga_settling = st.session_state.ga_results['final_metrics'].get('settling_time', float('inf'))
                if np.isfinite(ga_settling):
                    fig.add_hline(y=ga_settling, line_dash="dot", line_color="rgb(75, 192, 192)",
                                  annotation_text=f"GA: {ga_settling:.2f}s", row=1, col=2)

            # Overshoot plot
            fig.add_trace(
                go.Scatter(x=iterations, y=overshoots, mode='lines+markers',
                           name='Agentic Overshoot', line=dict(color='rgb(255, 159, 64)'), showlegend=True),
                row=2, col=1
            )
            fig.add_hline(y=target_overshoot, line_dash="dash", line_color="red",
                          annotation_text=f"Target: {target_overshoot:.1f}%", row=2, col=1)

            # NEW: Add GA overshoot reference
            if st.session_state.ga_results.get('status') == 'complete':
                ga_overshoot = st.session_state.ga_results['final_metrics'].get('overshoot', 0)
                fig.add_hline(y=ga_overshoot, line_dash="dot", line_color="rgb(75, 192, 192)",
                              annotation_text=f"GA: {ga_overshoot:.1f}%", row=2, col=1)

            # Performance summary (grouped bar chart)
            if mse_values and settling_times and overshoots:
                categories = ['MSE', 'Settling Time', 'Overshoot']
                agentic_values = [mse_values[-1], settling_times[-1], overshoots[-1]]
                target_values = [target_mse, target_settling, target_overshoot]

                fig.add_trace(
                    go.Bar(x=categories, y=agentic_values, name='Agentic',
                           marker_color='rgb(255, 99, 132)'),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Bar(x=categories, y=target_values, name='Target',
                           marker_color='red'),
                    row=2, col=2
                )

                # NEW: Add GA bars if available
                if st.session_state.ga_results.get('status') == 'complete':
                    ga_values = [
                        st.session_state.ga_results['final_metrics'].get('mse', 0),
                        st.session_state.ga_results['final_metrics'].get('settling_time', 0),
                        st.session_state.ga_results['final_metrics'].get('overshoot', 0)
                    ]
                    fig.add_trace(
                        go.Bar(x=categories, y=ga_values, name='GA',
                               marker_color='rgb(75, 192, 192)'),
                        row=2, col=2
                    )

            fig.update_layout(height=600, showlegend=True, barmode='group')
            fig.update_xaxes(title_text="Global Step", row=1, col=1)
            fig.update_xaxes(title_text="Global Step", row=1, col=2)
            fig.update_xaxes(title_text="Global Step", row=2, col=1)
            fig.update_yaxes(title_text="MSE", row=1, col=1)
            fig.update_yaxes(title_text="Time (s)", row=1, col=2)
            fig.update_yaxes(title_text="Percent (%)", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=2)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                    "toImageButtonOptions": {"format": "svg", "filename": "performance"}
                }
            )
    else:
        st.info("No metrics data available yet. Start the design process to see progress.")


def display_gains_plot():
    """Display parameter/gains history"""
    if st.session_state.monitor.state_history:
        params_history = {}
        global_step = 0
        last_metrics = None

        for entry in st.session_state.monitor.state_history:
            state = entry['state']
            if state.get('results') and state['results'].get('metrics'):
                metrics = state['results']['metrics']
                metrics_tuple = tuple((k, v) for k, v in sorted(metrics.items()))

                if metrics_tuple == last_metrics:
                    continue

                last_metrics = metrics_tuple
                global_step += 1

                if state.get('current_params'):
                    for param_name, param_value in state['current_params'].items():
                        if param_name != 'reasoning':
                            if param_name not in params_history:
                                params_history[param_name] = {'x': [], 'y': []}
                            params_history[param_name]['x'].append(global_step)
                            params_history[param_name]['y'].append(param_value)

        if params_history:
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for idx, param_name in enumerate(params_history.keys()):
                data = params_history[param_name]
                fig.add_trace(
                    go.Scatter(x=data['x'], y=data['y'],
                               mode='lines+markers', name=f"Agentic {param_name}",
                               line=dict(color=colors[idx % len(colors)]))
                )

            # NEW: Add GA final values as horizontal lines
            if st.session_state.ga_results.get('status') == 'complete':
                ga_params = st.session_state.ga_results.get('best_params', {})
                for idx, (param_name, param_value) in enumerate(ga_params.items()):
                    fig.add_hline(
                        y=param_value,
                        line_dash="dot",
                        line_color=colors[idx % len(colors)],
                        annotation_text=f"GA {param_name}: {param_value:.2f}"
                    )

            fig.update_layout(
                title="Parameters History (Agentic vs GA)",
                xaxis_title="Global Step",
                yaxis_title="Parameter Value",
                height=500,
                showlegend=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {"format": "svg", "filename": "gains_history"}
                }
            )


def display_ga_results():
    """Display GA optimization results"""
    if not st.session_state.ga_results:
        st.info("No GA results available. Enable GA in Advanced Settings to run genetic algorithm optimization.")
        return

    status = st.session_state.ga_results.get('status')

    if status == 'running':
        st.info("🧬 GA optimization is currently running... Please wait.")

    elif status == 'complete':
        st.success("✅ GA optimization completed successfully!")

        # Display best parameters
        st.subheader("Optimal Parameters")
        best_params = st.session_state.ga_results.get('best_params', {})
        cols = st.columns(len(best_params))
        for idx, (param, value) in enumerate(best_params.items()):
            with cols[idx]:
                st.metric(param, f"{value:.4f}")

        # Display final metrics
        st.subheader("Performance Metrics")
        final_metrics = st.session_state.ga_results.get('final_metrics', {})
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("MSE", f"{final_metrics.get('mse', 0):.4f}")
        with metric_cols[1]:
            st.metric("Settling Time", f"{final_metrics.get('settling_time', 0):.2f}s")
        with metric_cols[2]:
            st.metric("Overshoot", f"{final_metrics.get('overshoot', 0):.2f}%")
        with metric_cols[3]:
            stable = final_metrics.get('stable', False)
            st.metric("Stable", "✓" if stable else "✗")

        # Plot optimization history
        st.subheader("Optimization History")
        history = st.session_state.ga_results.get('history', {})
        if history and 'generation' in history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history['generation'],
                y=history['best_fitness'],
                mode='lines+markers',
                name='Best Cost',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=history['generation'],
                y=history['mean_fitness'],
                mode='lines',
                name='Mean Cost',
                line=dict(color='red', width=1.5, dash='dash')
            ))
            fig.update_layout(
                title="GA Cost Evolution",
                xaxis_title="Generation",
                yaxis_title="Cost",
                height=400
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True, "displaylogo": False}
            )

        # Display comparison with agentic if available
        if st.session_state.monitor.state_history:
            st.subheader("Comparison: GA vs Agentic")

            # Get final agentic metrics
            final_state = st.session_state.monitor.state_history[-1]['state']
            if final_state.get('results') and final_state['results'].get('metrics'):
                agentic_metrics = final_state['results']['metrics']

                comparison_data = {
                    'Metric': ['MSE', 'Settling Time', 'Overshoot'],
                    'GA': [
                        final_metrics.get('mse', 0),
                        final_metrics.get('settling_time', 0),
                        final_metrics.get('overshoot', 0)
                    ],
                    'Agentic': [
                        agentic_metrics.get('mse', 0),
                        agentic_metrics.get('settling_time', 0),
                        agentic_metrics.get('overshoot', 0)
                    ]
                }

                import pandas as pd
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, width='stretch')

                # Bar chart comparison
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=comparison_data['Metric'],
                    y=comparison_data['GA'],
                    name='GA',
                    marker_color='rgb(75, 192, 192)'
                ))
                fig_comp.add_trace(go.Bar(
                    x=comparison_data['Metric'],
                    y=comparison_data['Agentic'],
                    name='Agentic',
                    marker_color='rgb(255, 99, 132)'
                ))
                fig_comp.update_layout(
                    title="Performance Comparison",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(
                    fig_comp,
                    use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False}
                )

    elif status == 'error':
        st.error("❌ GA optimization failed!")
        error_msg = st.session_state.ga_results.get('error', 'Unknown error')
        st.error(f"Error: {error_msg}")

        if st.session_state.ga_results.get('traceback'):
            with st.expander("Show Traceback"):
                st.code(st.session_state.ga_results['traceback'])


def main():
    """Main Streamlit application with page routing"""

    # Determine which page to show
    if st.session_state.page == 'home':
        display_session_sidebar_home()
        display_home_page()
    else:  # project page
        display_session_sidebar_project()
        display_project_page()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #6c757d;'>
            🎛️ AI-Powered Control System Design Studio | Agentic + GA Optimization | Powered by Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()