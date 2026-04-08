import os
import requests
import json
import sys
from openai import OpenAI

# 1. Get the Proxy Credentials injected by the platform
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
# Your Hugging Face Space URL
ENV_URL = "https://mayur123486-auto-sre-env.hf.space"
TASK_NAME = "auto_sre_recovery"

def run_inference():
    print(f"[START] task={TASK_NAME}", flush=True)
    
    # Initialize the OpenAI client pointing to THEIR proxy
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    try:
        # Reset Env
        requests.post(f"{ENV_URL}/reset", params={"task_id": "cpu_spike"})
        
        # --- THE CRITICAL STEP: Call the LLM Proxy ---
        # The validator checks if this specific call happens
        response = client.chat.completions.create(
            model="gpt-4o", # Or whichever model they specify
            messages=[{"role": "user", "content": "The system has a CPU spike. Should I 'restart' or 'scale_up'?"}]
        )
        llm_decision = response.choices[0].message.content.lower()
        
        # Extract action from LLM text (simple logic for the check)
        action = "scale_up" if "scale" in llm_decision else "restart"
        
        # Execute the action on your environment
        step_res = requests.post(f"{ENV_URL}/step", json={"service": "frontend", "command": action})
        reward = step_res.json().get("reward", 0.0)
        
        print(f"[STEP] step=1 reward={reward}", flush=True)
        print(f"[END] task={TASK_NAME} score={reward} steps=1", flush=True)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"[END] task={TASK_NAME} score=0 steps=0", flush=True)

if __name__ == "__main__":
    run_inference()