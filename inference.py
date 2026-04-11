import os
import sys
import requests
from openai import OpenAI

# ── Environment Variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Configurable via env so local testing can point to localhost
ENV_URL   = os.getenv("ENV_URL", "https://mayur123486-auto-sre-env.hf.space")
BENCHMARK = "auto-sre-v1"
TASKS     = ["cpu_spike", "mem_leak", "cascading"]
MAX_STEPS = 5
SERVICES  = ["frontend", "auth_api", "database"]

# ── LLM Decision Logic ───────────────────────────────────────────────────────
def get_llm_action(client, task, step, metrics, last_reward):
    """
    Ask the LLM to inspect real live metrics and choose the best
    corrective action. Returns a raw string like 'restart auth_api'.
    """
    state_lines = "\n".join(
        f"  {svc}: cpu={m['cpu']:.1f}%  ram={m['ram']:.1f}%"
        for svc, m in metrics.items()
    )

    prompt = (
        f"You are an AI SRE agent. Current system state at step {step}/{MAX_STEPS}:\n"
        f"{state_lines}\n\n"
        f"Active task : {task}\n"
        f"Last reward : {last_reward:.2f}  (higher = healthier, target > 0.80)\n\n"
        "Pick ONE action to improve system health:\n"
        "  restart <service>   — resets CPU/RAM (best when cpu > 70%)\n"
        "  scale_up <service>  — cuts load 40%  (best when cpu 50-70%)\n"
        "  no_op               — do nothing     (when all services are healthy)\n\n"
        "Services: frontend, auth_api, database\n"
        "Reply with EXACTLY one action, e.g.:  restart auth_api"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert SRE AI. Reply with only the action string."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=20,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip().lower()


def parse_action(raw):
    """
    Turn the LLM's text into (service, command).
    Falls back gracefully if the LLM hallucinates a weird response.
    """
    raw = raw.replace(",", "").replace("\n", "")

    # Check for command + service
    for cmd in ["restart", "scale_up"]:
        for svc in SERVICES:
            if cmd in raw and svc in raw:
                return svc, cmd

    # no_op
    if "no_op" in raw or "no op" in raw or "noop" in raw:
        return "frontend", "no_op"

    # Partial match — at least a service name
    for svc in SERVICES:
        if svc in raw:
            return svc, "restart"

    # Safe fallback
    return "frontend", "restart"


# ── Main Inference Loop ──────────────────────────────────────────────────────
def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task in TASKS:
        print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        rewards_list = []
        last_reward  = 0.50

        try:
            # Reset the environment for this specific task
            requests.post(f"{ENV_URL}/reset", params={"task_id": task}, timeout=15)

            for step in range(1, MAX_STEPS + 1):
                # ── 1. Observe current state ──────────────────────────────
                state_res = requests.get(f"{ENV_URL}/state", timeout=10)
                metrics   = state_res.json().get("metrics", {})

                # ── 2. LLM decides which action to take ───────────────────
                raw_action          = get_llm_action(client, task, step, metrics, last_reward)
                service, command    = parse_action(raw_action)

                # Build a clean, comma-free action label for the output
                action_label = f"{command}_{service}"

                # ── 3. Execute the action on the environment ──────────────
                step_res  = requests.post(
                    f"{ENV_URL}/step",
                    json={"service": service, "command": command},
                    timeout=10,
                )
                data      = step_res.json()

                # ── 4. Record reward ──────────────────────────────────────
                reward      = float(data.get("reward", 0.50))
                reward      = round(max(0.01, min(0.99, reward)), 2)
                last_reward = reward
                done_bool   = bool(data.get("done", False))
                done_str    = "true" if done_bool else "false"

                rewards_list.append(f"{reward:.2f}")

                print(
                    f"[STEP] step={step} action={action_label} "
                    f"reward={reward:.2f} done={done_str} error=null",
                    flush=True,
                )

                if done_bool:
                    break

            print(
                f"[END] success=true steps={len(rewards_list)} "
                f"rewards={','.join(rewards_list)}",
                flush=True,
            )

        except Exception as e:
            print(f"Error in task '{task}': {e}", file=sys.stderr)
            fallback = rewards_list if rewards_list else ["0.50"]
            print(
                f"[END] success=false steps={len(fallback)} "
                f"rewards={','.join(fallback)}",
                flush=True,
            )


if __name__ == "__main__":
    run_inference()