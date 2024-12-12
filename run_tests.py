#!/usr/bin/env python3

import subprocess
import time
from datetime import datetime, timedelta
import sys
from typing import List, Dict

class TestRunner:
    def __init__(self, test_rounds: int = 3):
        self.test_rounds = test_rounds
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=10)
        self.tests = self._define_tests()
        
    def _define_tests(self) -> List[Dict]:
        """Define all test configurations."""
        return [
            {
                "name": "Control_4Random",
                "description": "Control test with 4 random agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--random-agents", "4",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "control_4random"]
            },
            {
                "name": "Control_4Informed",
                "description": "Control test with 4 informed agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--informed-agents", "4",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "control_4informed"]
            },
            {
                "name": "Informed_vs_Random_1v3",
                "description": "One informed agent versus three random agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--informed-agents", "1",
                           "--random-agents", "3",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "informed_v_random_1v3"]
            },
            {
                "name": "Informed_vs_Random_2v2",
                "description": "Two informed agents versus two random agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--informed-agents", "2",
                           "--random-agents", "2",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "informed_v_random_2v2"]
            },
            {
                "name": "Adaptive_vs_Informed_1v3",
                "description": "One adaptive agent versus three informed agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--adaptive-agents", "1",
                           "--informed-agents", "3",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "adaptive_v_informed_1v3"]
            },
            {
                "name": "Adaptive_vs_Random_1v3",
                "description": "One adaptive agent versus three random agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--adaptive-agents", "1",
                           "--random-agents", "3",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "adaptive_v_random_1v3"]
            },
            {
                "name": "LLM_vs_Random_1v3",
                "description": "One LLM agent versus three random agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--llm-agents", "1",
                           "--random-agents", "3",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "llm_v_random_1v3"]
            },
            {
                "name": "LLM_vs_Informed_1v3",
                "description": "One LLM agent versus three informed agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--llm-agents", "1",
                           "--informed-agents", "3",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "llm_v_informed_1v3"]
            },
            {
                "name": "Mixed_OneEach",
                "description": "One of each agent type competing",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--random-agents", "1",
                           "--informed-agents", "1",
                           "--adaptive-agents", "1",
                           "--llm-agents", "1",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "mixed_one_each"]
            },
            {
                "name": "Mixed_TwoEach",
                "description": "Two of each agent type competing",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--random-agents", "2",
                           "--informed-agents", "2",
                           "--adaptive-agents", "2",
                           "--llm-agents", "2",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "mixed_two_each"]
            },
            {
                "name": "Adaptive_Learning_Marathon",
                "description": "Extended game series with adaptive agents",
                "command": ["python", "simulate_game.py", "--mode", "sim",
                           "--adaptive-agents", "1",
                           "--llm-agents", "3",
                           "--rounds", str(self.test_rounds),
                           "--plt-suffix", "adaptive_learning_marathon"]
            }
        ]
    
    def print_progress(self, current_test: Dict, start_time: datetime):
        """Print progress information."""
        now = datetime.now()
        elapsed = now - start_time
        if now - self.last_update >= self.update_interval:
            self.last_update = now
            print(f"\nProgress Update ({now.strftime('%H:%M:%S')}):")
            print(f"Currently running: {current_test['name']}")
            print(f"Time elapsed: {elapsed}")
            sys.stdout.flush()
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        total_start_time = datetime.now()
        print(f"\nStarting test suite at {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running each test with {self.test_rounds} rounds")
        print("-" * 50)
        
        all_tests_passed = True
        
        for i, test in enumerate(self.tests, 1):
            test_start_time = datetime.now()
            print(f"\nTest {i}/{len(self.tests)}: {test['name']}")
            print(f"Description: {test['description']}")
            print(f"Started at: {test_start_time.strftime('%H:%M:%S')}")
            print(f"Command: {' '.join(test['command'])}")
            sys.stdout.flush()
            
            process = subprocess.Popen(
                test['command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Store output
            stdout_lines = []
            stderr_lines = []
            
            while process.poll() is None:
                # Read output without blocking
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print(stdout_line.rstrip())
                    stdout_lines.append(stdout_line)
                
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print("ERROR:", stderr_line.rstrip(), file=sys.stderr)
                    stderr_lines.append(stderr_line)
                
                self.print_progress(test, test_start_time)
                time.sleep(0.1)
            
            # Get any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                print(stdout.rstrip())
                stdout_lines.append(stdout)
            if stderr:
                print("ERROR:", stderr.rstrip(), file=sys.stderr)
                stderr_lines.append(stderr)
            
            test_end_time = datetime.now()
            test_duration = test_end_time - test_start_time
            
            # Check if test passed
            if process.returncode != 0:
                all_tests_passed = False
                print(f"\nTest {test['name']} FAILED!")
                print(f"Return code: {process.returncode}")
                if stderr_lines:
                    print("\nError output:")
                    for line in stderr_lines:
                        print(line.rstrip())
            else:
                print(f"\nTest {test['name']} PASSED!")
            
            print(f"Completed in: {test_duration}")
            sys.stdout.flush()
        
        total_duration = datetime.now() - total_start_time
        print("\n" + "=" * 50)
        print(f"All tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {total_duration}")
        if all_tests_passed:
            print("ALL TESTS PASSED!")
        else:
            print("SOME TESTS FAILED!")
            sys.exit(1)
        print("=" * 50)

if __name__ == "__main__":
    # For initial testing, use 10 rounds
    runner = TestRunner(test_rounds=2)
    runner.run_all_tests()
