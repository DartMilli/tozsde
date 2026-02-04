#!/usr/bin/env python
"""Quick coverage summary script for major modules."""

import subprocess
import sys

modules = [
    ('Technical Indicators', 'app/indicators/technical.py', ['tests/test_indicators.py', 'tests/test_technical_comprehensive.py']),
    ('Data Manager', 'app/data_access/data_manager.py', ['tests/test_data_manager.py', 'tests/test_data_manager_edge_cases.py']),
    ('Health Check', 'app/infrastructure/health_check.py', ['tests/test_health_check.py']),
]

print("=" * 70)
print("SPRINT 11+ COVERAGE SUMMARY".center(70))
print("=" * 70)

results = {}
for name, module_path, test_files in modules:
    cmd = ['python', '-m', 'pytest'] + test_files + [
        f'--cov={module_path.replace("/", ".").replace(".py", "")}',
        '--cov-report=term-missing', '-q'
    ]
    
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, _ = result.communicate()
    output = output.decode('utf-8', errors='ignore')
    
    # Parse coverage percentage
    if 'TOTAL' in output or 'Cover' in output:
        lines = output.split('\n')
        for line in lines:
            if '%' in line and ('Cover' in line or 'TOTAL' in line):
                parts = line.split()
                for i, part in enumerate(parts):
                    if '%' in part:
                        coverage = int(part.rstrip('%'))
                        results[name] = coverage
                        break
    
    print(f"\n{name:.<50}", end='')
    if name in results:
        cov = results[name]
        status = "OK" if cov >= 80 else "WARN" if cov >= 70 else "LOW"
        print(f" {cov:>3}% {status}")
    else:
        print(" N/A FAIL")

print("\n" + "=" * 70)
if results:
    avg_coverage = sum(results.values()) / len(results)
    print(f"AVERAGE COVERAGE: {avg_coverage:.1f}%")
print("=" * 70)
