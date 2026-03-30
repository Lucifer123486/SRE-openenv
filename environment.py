import uuid
import random
from models import SREAction, SREObservation

class AutoSREEnv:
    def __init__(self):
        self.reset()

    def reset(self, task_id="cpu_spike"):
        self.task_id = task_id
        self.metrics = {
            "frontend": {"cpu": 20.0, "ram": 30.0},
            "auth_api": {"cpu": 20.0, "ram": 30.0},
            "database": {"cpu": 15.0, "ram": 25.0}
        }
        self.steps = 0
        return self._obs(f"Environment Reset for task: {task_id}")

    def step(self, action: SREAction):
        self.steps += 1
        
        # 1. Apply Action
        if action.command == "restart":
            self.metrics[action.service] = {"cpu": 10.0, "ram": 20.0}
        elif action.command == "scale_up":
            self.metrics[action.service]["cpu"] *= 0.5 # Halve the load
            
        # 2. Task-Specific Logic (Stable version)
        if self.task_id == "cpu_spike":
            self.metrics["auth_api"]["cpu"] += 15.0
            
        elif self.task_id == "mem_leak":
            self.metrics["auth_api"]["ram"] += 20.0 # Fast leak!
            
        elif self.task_id == "cascading":
            # Hard Task: Database slows down, which kills the Frontend
            self.metrics["database"]["cpu"] += 10.0
            if self.metrics["database"]["cpu"] > 70:
                self.metrics["frontend"]["cpu"] += 30.0 # Cascading effect

        # 3. Calculate Reward
        avg_cpu = sum(m["cpu"] for m in self.metrics.values()) / 3
        reward = 1.0 - (avg_cpu / 100.0)
        
        # 4. Check Done Condition
        done = any(m["cpu"] >= 100 or m["ram"] >= 100 for m in self.metrics.values())
        if done: reward = -5.0 # Penalty for crash
        
        return self._obs("Metrics updated"), reward, done

    def _obs(self, msg):
        return SREObservation(logs=msg, metrics=self.metrics)