# 🛡️ Auto-SRE: A Reinforcement Learning Environment for Site Reliability

Auto-SRE is a high-fidelity **Reinforcement Learning (RL) Environment** designed to train AI agents in automated system recovery. Unlike static environments, Auto-SRE simulates a live microservices cluster with **Stochastic Chaos**, **Cascading Failures**, and **Resource Contention**.

Built for the **OpenEnv Hackathon**, this project provides a standardized interface for agents to observe system telemetry and execute corrective actions.

---

## 🚀 Live Environment
The environment is hosted and running on Hugging Face Spaces:
🔗 **[Live API Endpoint](https://mayur123486-auto-sre-env.hf.space)**

---

## 🧠 Environment Specification

### 1. Observation Space (The "Eyes")
The agent receives a real-time snapshot of the cluster state via a JSON object:
- **Metrics**: CPU and RAM utilization for `frontend`, `auth_api`, and `database`.
- **Logs**: Unstructured system strings (e.g., *"⚠️ Alert: Network jitter detected"*).
- **Vitality Index**: An overall system health score from 0 to 100%.

### 2. Action Space (The "Hands")
The agent can interact with the environment using discrete actions:
- `RESTART`: Resets a service to a clean baseline state.
- `SCALE_UP`: Increases resources to mitigate high CPU loads.
- `NO_OP`: Observes the system without intervention.

### 3. Reward Function
The reward signal is calculated based on **System Stability**:
$$Reward = 1.0 - (\text{Average Cluster Stress} / 100)$$
* **Crash Penalty**: A massive penalty (-5.0) is applied if any service hits 100% utilization.

---

## 🌋 Chaos Simulation Modes
Auto-SRE isn't just a counter; it's a "Game of Survival":
* **CPU Spike**: Simulates a sudden surge in user traffic.
* **Memory Leak**: A non-linear RAM climb that requires a full reboot to clear.
* **Cascading Failure (Hard)**: Database latency triggers a "retry-storm" in the Frontend, simulating real-world microservice dependency chains.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)
- **Deployment**: Docker on Hugging Face Spaces
- **Contracts**: Pydantic for Type-Safe RL schemas
- **Standards**: Compliant with the OpenEnv Specification

---

## 📖 How to Interact (API Guide)

### Reset Environment
```bash
POST /reset?task_id=cascading
```

### Take a Step
```json
POST /step
{
  "service": "auth_api",
  "command": "scale_up"
}
```
