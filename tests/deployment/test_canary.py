#!/usr/bin/env python3
"""Test script to send requests to canary deployment"""

import requests
import json
import sys

def run_experiment(num_requests=50, base_url="http://localhost:5000"):
    """Send multiple requests and analyze traffic distribution"""
    
    print(f"\n{'='*70}")
    print(f"CANARY DEPLOYMENT EXPERIMENT")
    print(f"{'='*70}")
    print(f"Sending {num_requests} requests to {base_url}/predict\n")
    
    v1_count = 0
    v2_count = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(
                f'{base_url}/predict',
                json={"features": [1.0, 2.0, 3.0]},
                timeout=5
            )
            result = response.json()
            version = result.get('model_version', 'unknown')
            
            if version == 'v1':
                v1_count += 1
            elif version == 'v2':
                v2_count += 1
                
            if (i + 1) % 10 == 0:
                print(f"✓ Progress: {i + 1}/{num_requests} requests completed")
                
        except Exception as e:
            print(f"✗ Error on request {i + 1}: {e}")
    
    print(f"\n{'='*70}")
    print("TRAFFIC DISTRIBUTION RESULTS")
    print(f"{'='*70}")
    print(f"Model v1: {v1_count} requests ({v1_count/num_requests*100:.1f}%)")
    print(f"Model v2: {v2_count} requests ({v2_count/num_requests*100:.1f}%)")
    
    # Get detailed metrics
    try:
        metrics_response = requests.get(f'{base_url}/metrics')
        metrics = metrics_response.json()
        
        print(f"\n{'='*70}")
        print("DETAILED METRICS")
        print(f"{'='*70}")
        print(f"Canary Setting: {metrics['canary_setting']}")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"\nModel v1:")
        print(f"  - Requests: {metrics['models']['v1']['requests']}")
        print(f"  - Traffic: {metrics['models']['v1']['traffic_percentage']}%")
        print(f"  - Avg Latency: {metrics['models']['v1']['avg_latency_ms']} ms")
        print(f"\nModel v2:")
        print(f"  - Requests: {metrics['models']['v2']['requests']}")
        print(f"  - Traffic: {metrics['models']['v2']['traffic_percentage']}%")
        print(f"  - Avg Latency: {metrics['models']['v2']['avg_latency_ms']} ms")
        
        print(f"\n{'='*70}")
        
    except Exception as e:
        print(f"Could not fetch detailed metrics: {e}")

if __name__ == '__main__':
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_experiment(num_requests)
