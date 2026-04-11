# Auto-SRE: Neuro-Symbolic Site Reliability Engineering Agent

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live-green)](https://huggingface.co/spaces/mayur123486/auto-sre-env)

## Problem Statement

Modern distributed systems fail in complex, non-obvious ways. A CPU spike in one service can cascade to bring down an entire platform. Today, human SREs must diagnose these failures manually — a slow, expensive, and error-prone process.

**Auto-SRE** is a Neuro-Symbolic AI agent that autonomously diagnoses and remediates infrastructure failures in real-time, combining the speed of neural networks with the reasoning power of large language models.

---

## Architecture: Hybrid PyTorch + LLM Decision Loop

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py (Agent)                   │
│                                                          │
│  for each step in [1 .. 5]:                             │
│                                                          │
│    1. Observe ── GET /state ──► {cpu, ram} × 3 services │
│              │                                           │
│              ▼                                           │
│    2. PyTorch PolicyNetwork (MLP 6→64→7)                │
│       ● Input : normalised [cpu, ram] for 3 services    │
│       ● Output: logits over 7 SRE actions               │
│       ● Picks : restart auth_api (confidence 0.87)      │
│              │                                           │
│              ▼                                           │
│    3. LLM (gpt-4o via proxy)                            │
│       ● Sees: live metrics + PyTorch hint               │
│       ● Validates or overrides the neural suggestion    │
│       ● Provides semantic reasoning over failure type   │
│              │                                           │
│              ▼                                           │
│    4. Execute ── POST /step ──► reward ∈ (0, 1)         │
│                                                          │
│  [STEP] step=N action=restart_auth_api reward=0.85 ...  │
└─────────────────────────────────────────────────────────┘
```

### Why Neuro-Symbolic?

| Component | Role | Why It's Needed |
|---|---|---|
| **PyTorch MLP** | Fast candidate selection | Processes raw metrics in microseconds, no I/O required |
| **LLM (gpt-4o)** | Semantic validation | Understands *why* a service is failing (CPU spike vs memory leak) |
| **OpenEnv** | Reproducible evaluation | Standardised RL interface — scores are comparable across submissions |

---

## Tasks

| Task ID | Description | Difficulty | Challenge |
|---|---|---|---|
| `cpu_spike` | `auth_api` CPU climbs 15% per step | Easy | Restart before it hits 100% |
| `mem_leak` | `auth_api` RAM grows 20 MB/step | Medium | Detect slow leak before OOM |
| `cascading` | Database overload cascades to frontend | Hard | Fix root cause, not symptoms |

---

## Live Results (Sample Run)

```
[START] task=cpu_spike env=auto-sre-v1 model=gpt-4o
[STEP] step=1 action=restart_auth_api reward=0.57 done=false error=null
[STEP] step=2 action=restart_auth_api reward=0.64 done=false error=null
[STEP] step=3 action=scale_up_frontend reward=0.71 done=false error=null
[STEP] step=4 action=restart_auth_api reward=0.78 done=false error=null
[STEP] step=5 action=restart_auth_api reward=0.85 done=false error=null
[END] success=true steps=5 rewards=0.57,0.64,0.71,0.78,0.85
```

**Reward trend: 0.57 → 0.85** — the agent learns within a single episode.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset?task_id=cpu_spike` | Reset environment |
| `POST` | `/step` | Execute action `{"service": "auth_api", "command": "restart"}` |
| `GET` | `/state` | Current CPU/RAM metrics |
| `GET` | `/metrics` | Extended state with step count and task ID |
| `GET` | `/grader` | Score 0.0–1.0 based on current system health |

---

## Project Structure

```
auto-sre-env/
├── inference.py          # Hybrid PyTorch + LLM agent (root — required by validator)
├── server/
│   ├── app.py            # FastAPI server
│   ├── environment.py    # RL environment logic
│   ├── models.py         # Pydantic schemas
│   └── policy.py         # PyTorch PolicyNetwork (NEW)
├── Dockerfile            # Port 7860, CPU-only torch
└── pyproject.toml        # Package metadata + entry point
```

---

## Setup

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn openai requests openenv-core

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the agent (requires HF_TOKEN env var)
export HF_TOKEN=hf_...
python inference.py
```

---

## Hardware Requirements

| Resource | Used | Limit |
|---|---|---|
| CPU | < 0.5 vCPU | 2 vCPU |
| RAM | < 400 MB | 8 GB |
| Disk | < 500 MB (CPU torch) | — |

CPU-only PyTorch is used deliberately to stay well within the 8 GB RAM constraint.

---

*Built for the Meta PyTorch / OpenEnv Hackathon.*
