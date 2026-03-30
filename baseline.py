import requests
import json

# The URL where your FastAPI is running
BASE_URL = "http://127.0.0.1:8000"

def run_baseline():
    print("🚀 Starting Baseline Run...")
    
    # 1. Reset the environment
    response = requests.post(f"{BASE_URL}/reset")
    obs = response.json()
    print(f"Initial State: {obs['metrics']}")

    # 2. Simple Logic (This simulates an "Agent")
    # In a real submission, you'd use the OpenAI API here.
    # For now, let's use 'Heuristic' logic to prove the API works.
    for i in range(1, 6):
        print(f"\n--- Step {i} ---")
        
        # Agent Logic: If CPU > 50, Restart. Else, No-Op.
        cpu_val = obs['metrics']['auth_api']['cpu']
        if cpu_cpu > 50:
            action = {"service": "auth_api", "command": "restart"}
        else:
            action = {"service": "auth_api", "command": "no_op"}
        
        print(f"Agent chooses: {action['command']}")
        
        # 3. Send Action to Environment
        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        step_data = step_resp.json()
        obs = step_data['observation']
        
        print(f"New CPU: {obs['metrics']['auth_api']['cpu']} | Reward: {step_data['reward']}")
        
        if step_data['done']:
            print("System Crashed!")
            break

    # 4. Final Grading
    grader_resp = requests.get(f"{BASE_URL}/grader")
    print(f"\n✅ Final Grader Score: {grader_resp.json()['score']}")

if __name__ == "__main__":
    run_baseline()