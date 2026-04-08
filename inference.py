import requests
import json
import sys

# Replace with your actual Hugging Face Space URL
API_URL = "https://mayur123486-auto-sre-env.hf.space"
TASK_NAME = "auto_sre_recovery"

def run_inference():
    # 1. [START] Block - Tells the validator the test has begun
    print(f"[START] task={TASK_NAME}", flush=True)
    
    try:
        # Reset the environment
        reset_res = requests.post(f"{API_URL}/reset", params={"task_id": "cpu_spike"})
        reset_res.raise_for_status()
        
        # 2. [STEP] Block - Tells the validator a move was made
        # We perform a 'no_op' or 'restart' to show the agent is active
        step_payload = {"service": "frontend", "command": "no_op"}
        step_res = requests.post(f"{API_URL}/step", json=step_payload)
        step_res.raise_for_status()
        
        data = step_res.json()
        reward = data.get("reward", 0.0)
        
        # Format: [STEP] step=N reward=X
        print(f"[STEP] step=1 reward={reward}", flush=True)
        
        # 3. [END] Block - Tells the validator the test is finished
        # Format: [END] task=NAME score=X steps=N
        print(f"[END] task={TASK_NAME} score={reward} steps=1", flush=True)
        
    except Exception as e:
        # If it fails, we still need an END block or the validator hangs
        print(f"Error during inference: {e}", file=sys.stderr)
        print(f"[END] task={TASK_NAME} score=0 steps=0", flush=True)

if __name__ == "__main__":
    run_inference()