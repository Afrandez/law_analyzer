import subprocess
import sys

def run_pytest():
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "-v", "--disable-warnings"], capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        if result.returncode != 0:
            print(f"Tests failed with exit code {result.returncode}")
        else:
            print("All tests passed successfully.")
    except Exception as e:
        print(f"Failed to run tests: {e}")

if __name__ == "__main__":
    run_pytest()
