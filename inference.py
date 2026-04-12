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

# Configurable for local testing; production points to HF Space
ENV_URL   = os.getenv("ENV_URL", "https://mayur123486-auto-sre-env.hf.space")
BENCHMARK = "auto-sre-v1"
TASKS     = ["cpu_spike", "mem_leak", "cascading"]
MAX_STEPS = 5
SERVICES  = ["frontend", "auth_api", "database"]

# ── PyTorch Policy (graceful fallback if torch not installed) ────────────────
try:
    from server.policy import PolicyNetwork, get_policy_suggestion, TORCH_AVAILABLE
    _policy_net = PolicyNetwork() if TORCH_AVAILABLE else None
except Exception:
    _policy_net           = None
    TORCH_AVAILABLE       = False
    get_policy_suggestion = None     # type: ignore


# ── LLM Decision ─────────────────────────────────────────────────────────────
def get_llm_action(client, task, step, metrics, last_reward, torch_hint=None):
    """
    Build a structured prompt from live env metrics and an optional
    PyTorch policy hint, then call the LLM proxy.
    Returns the raw LLM string (e.g. 'restart auth_api').
    """
    state_lines = "\n".join(
        f"  {svc}: cpu={m['cpu']:.1f}%  ram={m['ram']:.1f}%"
        for svc, m in metrics.items()
    )

    hint_str = ""
    if torch_hint:
        svc, cmd, conf = torch_hint
        hint_str = (
            f"\n\nPyTorch PolicyNetwork suggests: {cmd} {svc} "
            f"(confidence {conf:.2f}). Agree or override based on the metrics."
        )

    prompt = (
        f"You are an AI SRE agent. Step {step}/{MAX_STEPS}, task: {task}\n"
        f"Live system state:\n{state_lines}\n"
        f"Last reward: {last_reward:.2f}  (target > 0.80)"
        f"{hint_str}\n\n"
        "Choose ONE action:\n"
        "  restart <service>   — resets CPU/RAM  (best when cpu > 70%)\n"
        "  scale_up <service>  — cuts load 40%   (best when cpu 50–70%)\n"
        "  no_op               — do nothing      (when all services are healthy)\n"
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


def parse_action(raw: str):
    """
    Convert a raw LLM / policy string into a (service, command) tuple.
    Fails gracefully — always returns a valid pair.
    """
    raw = raw.replace(",", "").replace("\n", "").strip()

    for cmd in ["restart", "scale_up"]:
        for svc in SERVICES:
            if cmd in raw and svc in raw:
                return svc, cmd

    if any(x in raw for x in ["no_op", "no op", "noop"]):
        return "frontend", "no_op"

    for svc in SERVICES:
        if svc in raw:
            return svc, "restart"

    return "auth_api", "restart"   # safe default — auth_api is most-often stressed


# ── Main Agent Loop ──────────────────────────────────────────────────────────
def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task in TASKS:
        # Phase 2 required: [START] tag
        print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        rewards_list = []
        last_reward  = 0.50

        try:
            # Reset environment to the correct task
            requests.post(
                f"{ENV_URL}/reset",
                params={"task_id": task},
                timeout=15,
            )

            for step in range(1, MAX_STEPS + 1):

                # ── 1. Observe live system state ──────────────────────────
                state_res = requests.get(f"{ENV_URL}/state", timeout=10)
                metrics   = state_res.json().get("metrics", {})

                # ── 2. PyTorch policy candidate action ────────────────────
                torch_hint = None
                if _policy_net is not None and get_policy_suggestion is not None:
                    try:
                        torch_hint = get_policy_suggestion(_policy_net, metrics)
                    except Exception:
                        pass   # silently skip if forward pass fails

                # ── 3. LLM validates / overrides PyTorch suggestion ───────
                raw_action       = get_llm_action(
                    client, task, step, metrics, last_reward, torch_hint
                )
                service, command = parse_action(raw_action)
                action_label     = f"{command}_{service}"   # e.g. restart_auth_api

                # ── 4. Execute action on the environment ──────────────────
                step_res = requests.post(
                    f"{ENV_URL}/step",
                    json={"service": service, "command": command},
                    timeout=10,
                )
                data = step_res.json()

                # ── 5. Record reward (strictly 2 d.p., range 0.01–0.99) ──
                reward = round(max(0.01, min(0.99, float(data.get("reward", 0.50)))), 2)
                # Explicit safety guards — catch any floating-point edge that slips through
                if reward <= 0.0:
                    reward = 0.01
                if reward >= 1.0:
                    reward = 0.99
                last_reward = reward
                done_bool   = bool(data.get("done", False))
                done_str    = "true" if done_bool else "false"

                rewards_list.append(f"{reward:.2f}")

                # Phase 2 required: [STEP] tag
                print(
                    f"[STEP] step={step} action={action_label} "
                    f"reward={reward:.2f} done={done_str} error=null",
                    flush=True,
                )

                if done_bool:
                    break

            # Phase 2 required: [END] tag
            print(
                f"[END] success=true steps={len(rewards_list)} "
                f"rewards={','.join(rewards_list)}",
                flush=True,
            )

        except Exception as e:
            # Always emit [END] — validator must never hang
            print(f"Error in task '{task}': {e}", file=sys.stderr)
            fallback = rewards_list if rewards_list else ["0.50"]
            print(
                f"[END] success=false steps={len(fallback)} "
                f"rewards={','.join(fallback)}",
                flush=True,
            )


if __name__ == "__main__":
    run_inference()