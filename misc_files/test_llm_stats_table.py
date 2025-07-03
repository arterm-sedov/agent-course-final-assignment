import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import GaiaAgent

class MockGaiaAgent(GaiaAgent):
    def __init__(self):
        # Do not call super().__init__ to avoid real LLM setup
        self.total_questions = 10
        self.llm_tracking = {
            'huggingface': {
                'successes': 3,
                'failures': 2,
                'threshold_passes': 2,
                'finalist_wins': 1,
                'low_score_submissions': 1,
                'total_attempts': 5
            },
            'groq': {
                'successes': 2,
                'failures': 3,
                'threshold_passes': 1,
                'finalist_wins': 1,
                'low_score_submissions': 2,
                'total_attempts': 5
            },
            'openrouter': {
                'successes': 4,
                'failures': 1,
                'threshold_passes': 3,
                'finalist_wins': 2,
                'low_score_submissions': 0,
                'total_attempts': 5
            },
            'gemini': {
                'successes': 1,
                'failures': 4,
                'threshold_passes': 1,
                'finalist_wins': 0,
                'low_score_submissions': 3,
                'total_attempts': 5
            },
        }
        self.LLM_CONFIG = {
            'huggingface': {'name': 'HuggingFace', 'models': [{'repo_id': 'Qwen/Qwen2.5-Coder-32B-Instruct'}]},
            'groq': {'name': 'Groq', 'models': [{'model': 'qwen-qwq-32b'}]},
            'openrouter': {'name': 'OpenRouter', 'models': [{'model': 'mistralai/mistral-small-3.2-24b-instruct:free'}]},
            'gemini': {'name': 'Google Gemini', 'models': [{'model': 'gemini-2.5-pro'}]},
        }
        self.active_model_config = {
            'huggingface': {'repo_id': 'Qwen/Qwen2.5-Coder-32B-Instruct'},
            'groq': {'model': 'qwen-qwq-32b'},
            'openrouter': {'model': 'mistralai/mistral-small-3.2-24b-instruct:free'},
            'gemini': {'model': 'gemini-2.5-pro'},
        }

    def get_llm_stats(self):
        # Use the real method from GaiaAgent
        return GaiaAgent.get_llm_stats(self)

    def print_llm_stats_table(self):
        # Use the real method from GaiaAgent
        return GaiaAgent.print_llm_stats_table(self)

if __name__ == "__main__":
    print("Testing LLM statistics table with variable-length provider/model names:\n")
    agent = MockGaiaAgent()
    agent.print_llm_stats_table() 