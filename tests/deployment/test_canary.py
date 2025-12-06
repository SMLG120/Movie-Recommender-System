#!/usr/bin/env python3
"""Test script for canary deployment - validates traffic split and telemetry"""

""" How to run?
    Start your API first """
# python src/app.py

""" In another terminal, run the tests"""
# python tests/deployment/test_canary.py

"""Optional: customize request count or URL """
# python tests/deployment/test_canary.py 200 http://localhost:8080

import json
import sys
from pathlib import Path
import requests


def test_health_endpoint(base_url="http://localhost:8080"):
    """Test 1: Verify health endpoint shows canary status"""
    print("\n" + "="*70)
    print("TEST 1: Health Endpoint")
    print("="*70)
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        health = resp.json()
        
        print(f"Status: {health.get('status')}")
        print(f"Canary Loaded: {health.get('canary_loaded')}")
        print(f"Canary Percentage: {health.get('canary_percentage')}%")
        
        if resp.status_code == 200:
            print("Health check PASSED")
            return True
        else:
            print(f"Health check FAILED (status {resp.status_code})")
            return False
    except Exception as e:
        print(f"Health check FAILED: {e}")
        return False


def test_traffic_distribution(num_requests=100, base_url="http://localhost:8080"):
    """Test 2: Verify traffic splits according to canary percentage"""
    print("\n" + "="*70)
    print("TEST 2: Traffic Distribution")
    print("="*70)
    
    # Get current canary percentage from health endpoint
    try:
        health = requests.get(f"{base_url}/health", timeout=5).json()
        target_pct = float(health.get("canary_percentage", 0.0))
        print(f"Target canary percentage: {target_pct:.1f}%")
    except Exception as e:
        print(f"Failed to get canary config: {e}")
        return False
    
    print(f"Sending {num_requests} requests to {base_url}/recommend\n")
    
    v1_count = 0
    v2_count = 0
    errors = 0
    
    for i in range(num_requests):
        try:
            # Use different user_ids to avoid caching effects
            resp = requests.get(
                f"{base_url}/recommend",
                params={"user_id": i + 1, "top_n": 5},
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                version = data.get("model_version", "unknown")
                
                if version == "v1":
                    v1_count += 1
                elif version == "v2":
                    v2_count += 1
            else:
                errors += 1
                
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{num_requests} requests")
                
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Request {i + 1} error: {e}")
    
    print("\n" + "="*70)
    print("TRAFFIC DISTRIBUTION RESULTS")
    print("="*70)
    
    total = v1_count + v2_count
    if total == 0:
        print("No successful requests")
        return False
    
    print(f"Model v1: {v1_count} requests ({v1_count/total*100:.1f}%)")
    print(f"Model v2: {v2_count} requests ({v2_count/total*100:.1f}%)")
    print(f"Errors:   {errors} requests ({errors/num_requests*100:.1f}%)")
    
    if target_pct == 0.0:
        # No canary expected - all traffic should go to v1
        if v2_count == 0 and errors < num_requests * 0.1:
            print("Traffic distribution PASSED (no canary active)")
            return True
        else:
            print("Expected 0% canary but saw v2 traffic")
            return False
    else:
        # Expect some v2 traffic and approximate split
        observed_pct = v2_count / total * 100 if total > 0 else 0.0
        print(f"Target: {target_pct:.1f}%, Observed: {observed_pct:.1f}%")
        
        # Allow ±15 percentage points tolerance for randomness
        if v2_count > 0 and abs(observed_pct - target_pct) <= 15 and errors < num_requests * 0.1:
            print("Traffic distribution PASSED (canary split within tolerance)")
            return True
        else:
            print("Traffic distribution FAILED (split outside expected range)")
            return False


def test_telemetry_logging(base_url="http://localhost:8080"):
    """Test 3: Verify telemetry logs contain model_version"""
    print("\n" + "="*70)
    print("TEST 3: Telemetry Logging")
    print("="*70)
    
    repo_root = Path(__file__).resolve().parents[2]
    log_file = repo_root / "evaluation" / "Online" / "logs" / "online_metrics.json"
    
    # Send a test request
    try:
        resp = requests.get(
            f"{base_url}/recommend",
            params={"user_id": 9999, "top_n": 5},
            timeout=5
        )
        if resp.status_code != 200:
            print(f"Test request failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Test request failed: {e}")
        return False
    
    # Check log file
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return False
    
    try:
        with log_file.open() as f:
            data = json.load(f)
        
        recs = data.get("recommendations", [])
        if not recs:
            print("No recommendations logged")
            return False
        
        last_rec = recs[-1]
        if "model_version" in last_rec:
            print(f"Found model_version in log: {last_rec['model_version']}")
            print(f"  User: {last_rec.get('user_id')}")
            print(f"  Items: {len(last_rec.get('items', []))} recommended")
            print("Telemetry logging PASSED")
            return True
        else:
            print("model_version field missing in telemetry")
            return False
            
    except Exception as e:
        print(f"Failed to read telemetry: {e}")
        return False


def run_all_tests(num_requests=100, base_url="http://localhost:8080"):
    """Run all canary tests"""
    print("\n" + "="*70)
    print("CANARY DEPLOYMENT TEST SUITE")
    print("="*70)
    print(f"Target: {base_url}")
    
    results = []
    
    # Test 1: Health
    results.append(("Health Endpoint", test_health_endpoint(base_url)))
    
    # Test 2: Traffic distribution
    results.append(("Traffic Distribution", test_traffic_distribution(num_requests, base_url)))
    
    # Test 3: Telemetry
    results.append(("Telemetry Logging", test_telemetry_logging(base_url)))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
    
    success = run_all_tests(num_requests, base_url)
    sys.exit(0 if success else 1)
