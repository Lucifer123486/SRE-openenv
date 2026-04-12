import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import io
import re

# Add current directory to path
sys.path.append(os.getcwd())

class TestPhase2ValidationSimple(unittest.TestCase):
    def setUp(self):
        # Setup Environment Variables
        os.environ["HF_TOKEN"] = "hf_test_token_123"
        os.environ["API_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["MODEL_NAME"] = "gpt-4o"
        os.environ["ENV_URL"] = "http://localhost:7860"

    @patch('requests.get')
    @patch('requests.post')
    @patch('openai.resources.chat.completions.Completions.create')
    def test_inference_output_syntax(self, mock_llm, mock_post, mock_get):
        # 1. Mock OpenAI LLM
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "restart auth_api"
        mock_llm.return_value = mock_response

        # 2. Mock requests.get → /state endpoint
        mock_state_response = MagicMock()
        mock_state_response.status_code = 200
        mock_state_response.json.return_value = {
            "metrics": {
                "frontend":  {"cpu": 25.0, "ram": 35.0},
                "auth_api":  {"cpu": 25.0, "ram": 35.0},
                "database":  {"cpu": 30.0, "ram": 40.0},
            }
        }
        mock_get.return_value = mock_state_response

        # 3. Mock requests.post → /reset + /step endpoints
        mock_step_response = MagicMock()
        mock_step_response.status_code = 200
        mock_step_response.json.return_value = {
            "reward": 0.85,
            "done": False,
            "observation": {"state": {}, "logs": "ok"}
        }
        mock_post.return_value = mock_step_response

        # 4. Run Inference
        from inference import run_inference

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            run_inference()
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        print("\n=== VERIFIED OUTPUT (PHASE 2 COMPLIANT) ===")
        print(output)
        print("===========================================\n")

        # 5. Strict Syntax & Formatting Verifications
        lines = output.strip().split('\n')

        # Verify [START] format
        start_lines = [l for l in lines if l.startswith("[START]")]
        self.assertEqual(len(start_lines), 3)  # One for each task
        for l in start_lines:
            self.assertRegex(l, r"\[START\] task=.+ env=.+ model=.+")

        # Verify [STEP] format, 2 d.p. precision, and reward strictly in (0, 1)
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        self.assertEqual(len(step_lines), 15)  # 5 steps × 3 tasks
        for l in step_lines:
            self.assertRegex(l, r"\[STEP\] step=\d+ action=.+ reward=\d+\.\d{2} done=(true|false) error=.+")
            # Extract reward value and verify boundary
            m = re.search(r"reward=(\d+\.\d+)", l)
            if m:
                score = float(m.group(1))
                self.assertGreater(score, 0.0,  f"Reward {score} must be > 0.0")
                self.assertLess(score,    1.0,  f"Reward {score} must be < 1.0")

        # Verify [END] format and 2 d.p. precision
        end_lines = [l for l in lines if l.startswith("[END]")]
        self.assertEqual(len(end_lines), 3)
        for l in end_lines:
            self.assertRegex(l, r"\[END\] success=(true|false) steps=\d+ rewards=[\d\.,]+")
            # Check every individual score in rewards= list
            m = re.search(r"rewards=([\d\.,]+)", l)
            if m:
                for score_str in m.group(1).split(','):
                    score = float(score_str)
                    self.assertGreater(score, 0.0, f"End reward {score} must be > 0.0")
                    self.assertLess(score,    1.0, f"End reward {score} must be < 1.0")
                    self.assertRegex(score_str, r"^\d+\.\d{2}$", "Must have exactly 2 decimal places")

if __name__ == "__main__":
    unittest.main()
