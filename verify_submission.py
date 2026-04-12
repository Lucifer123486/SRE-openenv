import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import subprocess
import time
import requests

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

class TestPhase2Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the server in the background
        cls.server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--port", "7860", "--host", "127.0.0.1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give the server time to start
        time.sleep(3)
        
    @classmethod
    def tearDownClass(cls):
        cls.server_proc.terminate()
        cls.server_proc.wait()

    @patch('openai.resources.chat.completions.Completions.create')
    def test_inference_output_format(self, mock_create):
        # 1. Setup Mock OpenAI Response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "restart"
        mock_create.return_value = mock_response

        # 2. Setup Environment Variables for local test
        os.environ["HF_TOKEN"] = "test_token"
        os.environ["API_BASE_URL"] = "http://localhost:8000/v1" 
        os.environ["MODEL_NAME"] = "gpt-4o-test"
        
        # Move import here so it picks up the environment variables
        from inference import run_inference
        
        # We need to temporarily modify ENV_URL in inference.py to localhost
        # Instead of editing the file, we can patch the string if we import it
        
        # sys.path.append(os.getcwd())
        
        # Patch the ENV_URL to point to our local server
        with patch('inference.ENV_URL', 'http://127.0.0.1:7860'):
            # Capture stdout
            import io
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                run_inference()
            except Exception as outer_e:
                print(f"Outer Exception: {outer_e}")
            finally:
                sys.stdout = sys.__stdout__
            
            output = captured_output.getvalue()
            print("--- Captured Output ---")
            print(output)
            print("-----------------------")

            # 3. Verify Requirements
            
            # Point 3 & 4: Regex check and Formatting
            lines = output.strip().split('\n')
            
            # Check individual tasks
            print(f"Full Logs:\n{output}")
            for task in ["cpu_spike", "mem_leak", "cascading"]:
                task_lines = [l for l in lines if f"task={task}" in l or "success=" in l]
                
                # [START]
                start_line = [l for l in lines if l.startswith("[START]") and f"task={task}" in l]
                self.assertTrue(len(start_line) > 0, f"Missing [START] for {task}")
                self.assertIn("env=auto-sre-v1", start_line[0])
                self.assertIn("model=gpt-4o-test", start_line[0])
                
                # [STEP]
                step_line = [l for l in lines if l.startswith("[STEP]") and "step=1" in l]
                # Note: inference.py prints [STEP] without task filter, but in sequence
                # We expect at least one [STEP] line with reward=X.XX
                for sl in step_line:
                    import re
                    # Look for reward=X.XX
                    self.assertTrue(re.search(r"reward=\d+\.\d{2}", sl), f"Reward format invalid in: {sl}")
                    self.assertTrue(re.search(r"done=(true|false)", sl), f"Done format invalid in: {sl}")

                # [END]
                # We expect an [END] block following the steps
                # Validating formatting of rewards list in [END]
                end_line = [l for l in lines if l.startswith("[END]") and "success=true" in l]
                self.assertTrue(len(end_line) > 0, "Missing [END] block")
                self.assertTrue(re.search(r"rewards=\d+\.\d{2}", end_line[0]), "End reward format invalid")

if __name__ == "__main__":
    unittest.main()
