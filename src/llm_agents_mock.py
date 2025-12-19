import json
import random
from typing import Dict

random.seed(42)


class LLMBaseAgent:
    def __init__(self, model="", seed=None, temperature: float = 0.0, monitor=None, **kwargs):
        # Accept any extra kwargs (like max_tries) without error
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.monitor = monitor
        self.agent_name = self.__class__.__name__


class LLMActor(LLMBaseAgent):
    def generate_parameters(self, buffer, controller_type, iter_number, max_iter, system, max_retries=3):
        param_ranges = getattr(buffer, "param_ranges", {})

        if not param_ranges:  # safety fallback
            param_ranges = {"K1": [0.01, 100.0], "K2": [0.01, 100.0], "K3": [0.01, 100.0], "K4": [0.01, 100.0]}

        params = {}
        for name, (lo, hi) in param_ranges.items():
            params[name] = round(random.uniform(lo, hi), 4)

        params["reasoning"] = f"Mock random gains – iter {iter_number+1}/{max_iter}"
        return params


class LLMCritic(LLMBaseAgent):
    def analyze_results(self, *args, **kwargs):
        return json.dumps({
            "result_analysis": "Mock critic: Performance acceptable.",
            "strategy": "EXPLORE",
            "suggested_improvements": ["Try small adjustments", "Watch for overshoot"]
        })


class LLMTerminator(LLMBaseAgent):
    def judge_termination(self, buffer, current_metrics, target_metrics, max_iterations,
                           controller_type, system_description, num_iter):
        # Randomly succeed near the end so the loop can finish
        if num_iter >= max_iterations - 1 or random.random() < 0.25:
            return {"decision": "TERMINATE_SUCCESS", "reasoning": "Mock success"}, ""
        return {"decision": "CONTINUE", "reasoning": "Mock continue"}, ""


class LLMJuror(LLMBaseAgent):
    def __init__(self, *args, max_tries=0, **kwargs):
        super().__init__(*args, **kwargs)  # now accepts max_tries silently
        self.max_tries = max_tries

    def decide(self, state: Dict) -> Dict:
        # 15% chance to trigger range change (tests that code path)
        if random.random() < 0.15:
            old = state['buffer'].param_ranges
            new = {k: [round(v[0]*0.8, 3), round(v[1]*1.2, 3)] for k, v in old.items()}
            return {
                "decision": "RECONSIDER_RANGE",
                "new_range": new,
                "reasoning": "Mock juror: widening search space."
            }
        return {
            "decision": "EXPLORE_FURTHER",
            "new_range": None,
            "reasoning": "Mock juror: current range is sufficient."
        }