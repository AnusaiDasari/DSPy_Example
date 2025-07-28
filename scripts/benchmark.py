"""
Benchmark script to evaluate system performance
Useful for measuring improvements and production readiness
"""

import json
import time
import sys
sys.path.append('..')
from src.modules.support_system import SmartSupportSystem
from src.traditional.manual_prompting import TraditionalSupportSystem
import dspy
from dotenv import load_dotenv

def run_benchmark():
    """Run comprehensive benchmark comparing approaches."""
    
    print("🏁 DSPy Customer Support System Benchmark")
    print("=" * 50)
    
    # Setup
    load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    
    # Load test data
    try:
        with open('data/tickets/sample_tickets.json', 'r') as f:
            test_tickets = json.load(f)
    except FileNotFoundError:
        print("❌ Sample tickets not found. Please ensure data/tickets/sample_tickets.json exists.")
        return
    
    print(f"📋 Testing with {len(test_tickets)} tickets\n")
    
    # Initialize systems
    traditional_system = TraditionalSupportSystem()
    dspy_system = SmartSupportSystem()
    
    # Benchmark Traditional Approach
    print("⏱️ Benchmarking Traditional Manual Prompting...")
    traditional_results = []
    traditional_start = time.time()
    
    for ticket in test_tickets:
        start = time.time()
        try:
            classification = traditional_system.classify_ticket_manual(
                ticket['subject'], ticket['message']
            )
            response = traditional_system.generate_response_manual(
                ticket['subject'], ticket['message'],
                classification.get('category', 'Unknown'),
                classification.get('priority', 'Medium')
            )
            processing_time = time.time() - start
            
            traditional_results.append({
                'ticket_id': ticket['id'],
                'success': True,
                'processing_time': processing_time,
                'classification': classification,
                'response_length': len(response) if isinstance(response, str) else 0
            })
        except Exception as e:
            traditional_results.append({
                'ticket_id': ticket['id'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start
            })
    
    traditional_total_time = time.time() - traditional_start
    
    # Benchmark DSPy Approach
    print("⏱️ Benchmarking DSPy Declarative Programming...")
    dspy_results = []
    dspy_start = time.time()
    
    for ticket in test_tickets:
        start = time.time()
        try:
            result = dspy_system(
                subject=ticket['subject'],
                message=ticket['message']
            )
            processing_time = time.time() - start
            
            dspy_results.append({
                'ticket_id': ticket['id'],
                'success': True,
                'processing_time': processing_time,
                'category': result.category,
                'priority': result.priority,
                'quality_score': float(result.quality_score),
                'response_length': len(result.response),
                'escalation_needed': result.escalation_needed
            })
        except Exception as e:
            dspy_results.append({
                'ticket_id': ticket['id'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start
            })
    
    dspy_total_time = time.time() - dspy_start
    
    # Calculate metrics
    traditional_success_rate = sum(1 for r in traditional_results if r['success']) / len(traditional_results)
    dspy_success_rate = sum(1 for r in dspy_results if r['success']) / len(dspy_results)
    
    traditional_avg_time = sum(r['processing_time'] for r in traditional_results) / len(traditional_results)
    dspy_avg_time = sum(r['processing_time'] for r in dspy_results) / len(dspy_results)
    
    dspy_avg_quality = sum(r.get('quality_score', 0) for r in dspy_results if r['success']) / sum(1 for r in dspy_results if r['success'])
    
    # Display results
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 50)
    
    print("🔴 Traditional Manual Prompting:")
    print(f"  Success Rate: {traditional_success_rate:.1%}")
    print(f"  Average Processing Time: {traditional_avg_time:.2f}s")
    print(f"  Total Time: {traditional_total_time:.2f}s")
    print(f"  Issues: Manual parsing errors, inconsistent output")
    
    print("\n🟢 DSPy Declarative Programming:")
    print(f"  Success Rate: {dspy_success_rate:.1%}")
    print(f"  Average Processing Time: {dspy_avg_time:.2f}s")
    print(f"  Total Time: {dspy_total_time:.2f}s")
    print(f"  Average Quality Score: {dspy_avg_quality:.2f}")
    print(f"  Benefits: Structured output, quality assessment, optimizable")
    
    # Performance comparison
    speed_improvement = ((traditional_avg_time - dspy_avg_time) / traditional_avg_time) * 100
    reliability_improvement = ((dspy_success_rate - traditional_success_rate) / traditional_success_rate) * 100
    
    print(f"\n🏆 DSPy ADVANTAGES:")
    print(f"  Reliability Improvement: {reliability_improvement:.1f}%")
    print(f"  Speed Improvement: {speed_improvement:.1f}%")
    print(f"  Quality Assessment: Built-in (Traditional: None)")
    print(f"  Optimization: Automatic (Traditional: Manual)")
    print(f"  Maintainability: High (Traditional: Low)")
    
    # Save detailed results
    benchmark_results = {
        'timestamp': time.time(),
        'traditional': {
            'results': traditional_results,
            'success_rate': traditional_success_rate,
            'avg_processing_time': traditional_avg_time,
            'total_time': traditional_total_time
        },
        'dspy': {
            'results': dspy_results,
            'success_rate': dspy_success_rate,
            'avg_processing_time': dspy_avg_time,
            'total_time': dspy_total_time,
            'avg_quality_score': dspy_avg_quality
        },
        'improvements': {
            'reliability': reliability_improvement,
            'speed': speed_improvement
        }
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to benchmark_results.json")
    return benchmark_results

if __name__ == "__main__":
    run_benchmark()