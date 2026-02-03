#!/usr/bin/env python
import subprocess
import sys
import os

os.chdir(r"c:\Users\szlavik.mi\OneDrive - MVM Informatika Zrt\Dokumentumok\privat\tozsde_webapp")

result = subprocess.Popen(
    [sys.executable, "-m", "pytest", 
     "tests/test_performance_analytics*.py",
     "--cov=app.reporting.performance_analytics",
     "--cov-report=term-missing",
     "-q"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

stdout, _ = result.communicate()
print(stdout.decode('utf-8', errors='replace'))
print(f"\nReturn code: {result.returncode}")
