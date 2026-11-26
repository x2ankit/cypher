import os
import subprocess
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmarks():
    """
    Executes all validations against the provided synthetic fixtures
    using pytest, and generates a table showing expected vs actual.
    """
    logger.info("Running CYPHER Benchmark Suite...")
    
    # We use pytest with JSON report to parse the results easily
    # Requires pytest-json-report
    try:
        import pytest_jsonreport
    except ImportError as e:
        logger.error("pytest-json-report is not installed. Please install it in your environment: pip install pytest-json-report")
        raise ImportError("pytest-json-report is required to run the benchmark suite.") from e
        
    result = subprocess.run(
        ["pytest", "tests/", "--json-report", "--json-report-file=.benchmark_report.json"],
        capture_output=True,
        text=True
    )
    
    if not os.path.exists(".benchmark_report.json"):
        logger.error("Failed to generate benchmark report.")
        print(result.stdout)
        print(result.stderr)
        return
        
    with open(".benchmark_report.json", "r") as f:
        report = json.load(f)
        
    tests = report.get("tests", [])
    
    print("\n" + "="*80)
    print(f"{'TEST NAME':<55} | {'EXPECTED':<10} | {'ACTUAL':<10}")
    print("="*80)
    
    passed_count = 0
    failed_count = 0
    
    for test in tests:
        name = test.get("nodeid", "").split("::")[-1]
        outcome = test.get("outcome", "")
        
        # In a test suite, Expected is usually PASS (i.e. test passes)
        expected = "PASS"
        actual = "PASS" if outcome == "passed" else "FAIL"
        
        if actual == "PASS":
            passed_count += 1
        else:
            failed_count += 1
            
        print(f"{name:<55} | {expected:<10} | {actual:<10}")
        
    print("="*80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("="*80)

if __name__ == "__main__":
    run_benchmarks()
