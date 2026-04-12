import math
from .models import SREAction, SREObservation

class AutoSREEnv:
    def __init__(self):
        self.reset()

    def reset(self, task_id="cpu_spike"):
        # Explicitly support the 3 tasks required by the platform
        self.task_id = task_id
        
        # Initial metrics: Set them so the initial reward is never exactly 1.0
        self.metrics = {
            "frontend": {"cpu": 25.0, "ram": 35.0},
            "auth_api": {"cpu": 25.0, "ram": 35.0},
            "database": {"cpu": 30.0, "ram": 40.0}
        }
        self.steps = 0
        return self._obs(f"Environment Reset for task: {task_id}")

    def step(self, action: SREAction):
        self.steps += 1
        
        # 1. Apply Action
        if action.command == "restart":
            self.metrics[action.service] = {"cpu": 15.0, "ram": 25.0}
        elif action.command == "scale_up":
            self.metrics[action.service]["cpu"] *= 0.6 # Reduced the load
            
        # 2. Task-Specific Logic
        if self.task_id == "cpu_spike":
            self.metrics["auth_api"]["cpu"] += 15.0
        elif self.task_id == "mem_leak":
            self.metrics["auth_api"]["ram"] += 20.0 
        elif self.task_id == "cascading":
            self.metrics["database"]["cpu"] += 10.0
            if self.metrics["database"]["cpu"] > 60:
                self.metrics["frontend"]["cpu"] += 25.0

        # 3. Calculate Reward (STRICTLY between 0 and 1, never 0.0 or 1.0)
        avg_cpu = sum(m["cpu"] for m in self.metrics.values()) / 3
        # Logic: 1.0 - load. Clamp to [0.01, 0.99] so validator never sees edge values.
        raw_reward = 1.0 - (avg_cpu / 100.0)
        reward = max(0.01, min(0.99, raw_reward))
        reward = round(reward, 2)
        
        # 4. Check Done Condition
        done = any(m["cpu"] >= 100 or m["ram"] >= 100 for m in self.metrics.values())
        if done:
            reward = 0.01  # Crashed state: minimum valid score, strictly > 0.0
        
        return self._obs("Metrics updated"), reward, done

    def _obs(self, msg):
        return SREObservation(
            state=self.metrics, 
            logs=msg
        )