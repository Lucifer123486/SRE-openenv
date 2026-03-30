import requests
import json

# Replace this with your actual Hugging Face Space URL
API_URL = "https://mayur123486-auto-sre-env.hf.space"

def run_inference():
    print(f"Connecting to environment at {API_URL}...")
    
    # 1. Reset the environment
    try:
        reset_res = requests.post(f"{API_URL}/reset", params={"task_id": "cpu_spike"})
        reset_res.raise_for_status()
        print("✅ Reset Successful")
        
        # 2. Take a test step (No-Op)
        step_payload = {
            "service": "frontend",
            "command": "no_op"
        }
        step_res = requests.post(f"{API_URL}/step", json=step_payload)
        step_res.raise_for_status()
        
        observation = step_res.json()
        print("✅ Step Successful. Observation received:")
        print(json.dumps(observation, indent=2))
        
        return True
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        return False

if __name__ == "__main__":
    run_inference()