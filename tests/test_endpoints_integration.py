# AdminDashboard Endpoint Testing Script
# Run this after starting Flask app: python run_dev.py

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_endpoint(name, url, method='GET', data=None):
    """Test a single endpoint and display results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"Method: {method}")
    print(f"{'='*60}")
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            try:
                json_data = response.json()
                print("\nResponse (formatted):")
                print(json.dumps(json_data, indent=2)[:1000])  # First 1000 chars
                if len(str(json_data)) > 1000:
                    print("... (truncated)")
            except:
                print("\nResponse (text):")
                print(response.text[:500])
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Error: {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Flask app not running?")
        print("   Start it with: python run_dev.py")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def main():
    """Test all AdminDashboard endpoints."""
    print("="*60)
    print("AdminDashboard Endpoint Testing")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. Health Check
    test_endpoint(
        "Health Check",
        f"{BASE_URL}/admin/health"
    )
    
    # 2. Performance Summary (30 days)
    test_endpoint(
        "Performance Summary (30 days)",
        f"{BASE_URL}/admin/performance/summary?days=30"
    )
    
    # 3. Performance Summary (90 days)
    test_endpoint(
        "Performance Summary (90 days)",
        f"{BASE_URL}/admin/performance/summary?days=90"
    )
    
    # 4. Detailed Performance
    test_endpoint(
        "Detailed Performance (90 days)",
        f"{BASE_URL}/admin/performance/detailed?days=90"
    )
    
    # 5. Chart Data
    test_endpoint(
        "Chart Data (180 days)",
        f"{BASE_URL}/admin/performance/chart-data?days=180"
    )
    
    # 6. Error Summary
    test_endpoint(
        "Error Summary",
        f"{BASE_URL}/admin/errors/summary"
    )
    
    # 7. Recent Errors
    test_endpoint(
        "Recent Errors (limit 50)",
        f"{BASE_URL}/admin/errors/recent?limit=50"
    )
    
    # 8. Critical Errors Only
    test_endpoint(
        "Critical Errors",
        f"{BASE_URL}/admin/errors/critical"
    )
    
    # 9. Error Export (CSV)
    test_endpoint(
        "Export Errors (CSV)",
        f"{BASE_URL}/admin/errors/export",
        method='POST',
        data={"format": "csv", "severity": "ERROR"}
    )
    
    # 10. Capital Status
    test_endpoint(
        "Capital Status",
        f"{BASE_URL}/admin/capital/status"
    )
    
    # 11. Capital History
    test_endpoint(
        "Capital History (30 days)",
        f"{BASE_URL}/admin/capital/history?days=30"
    )
    
    # 12. Capital Allocation
    test_endpoint(
        "Capital Allocation",
        f"{BASE_URL}/admin/capital/allocation"
    )
    
    # 13. Capital Projection
    test_endpoint(
        "Capital Projection",
        f"{BASE_URL}/admin/capital/projection"
    )
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
