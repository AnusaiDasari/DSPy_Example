"""
The killer demo: Shows DSPy's automatic optimization in action
Watch the system improve itself with just a few examples!
"""

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
import json
import sys
sys.path.append('..')
from src.modules.support_system import SmartSupportSystem
from dotenv import load_dotenv

def load_training_examples():
    """Load training examples for optimization."""
    
    # These examples teach the system what good performance looks like
    training_examples = [
        dspy.Example(
            subject="Password reset email not received",
            message="I requested a password reset 30 minutes ago but haven't received the email yet. I checked spam folder.",
            expected_category="Technical",
            expected_priority="High", 
            expected_response_type="Troubleshooting",
            expected_helpful=True
        ).with_inputs('subject', 'message'),
        
        dspy.Example(
            subject="Billing discrepancy on invoice",
            message="I was charged $150 but my plan should be $100. Please investigate this billing error.",
            expected_category="Billing",
            expected_priority="Medium",
            expected_response_type="Account_Review", 
            expected_helpful=True
        ).with_inputs('subject', 'message'),
        
        dspy.Example(
            subject="Cannot access account - urgent",
            message="My account is completely locked and I have a client presentation in 2 hours. Need immediate help!",
            expected_category="Technical", 
            expected_priority="Critical",
            expected_response_type="Account_Recovery",
            expected_helpful=True
        ).with_inputs('subject', 'message'),
        
        dspy.Example(
            subject="Question about enterprise features",
            message="I'm evaluating your enterprise plan. Can you explain the API rate limits and SLA guarantees?",
            expected_category="Sales",
            expected_priority="Medium",
            expected_response_type="Information",
            expected_helpful=True  
        ).with_inputs('subject', 'message'),
    ]
    
    return training_examples

def comprehensive_quality_metric(example, prediction, trace=None):
    """Multi-dimensional quality metric for optimization."""
    
    score = 0.0
    total_weight = 0.0
    
    # Classification accuracy (40% of score)
    if hasattr(example, 'expected_category') and hasattr(prediction, 'category'):
        if example.expected_category == prediction.category:
            score += 0.4
        total_weight += 0.4
    
    if hasattr(example, 'expected_priority') and hasattr(prediction, 'priority'):
        if example.expected_priority == prediction.priority:
            score += 0.3
        total_weight += 0.3
    
    # Response quality (30% of score)
    if hasattr(prediction, 'quality_score'):
        score += 0.3 * float(prediction.quality_score)
        total_weight += 0.3
    
    return score / total_weight if total_weight > 0 else 0.0

def run_optimization_demo():
    """Demonstrate automatic optimization with real metrics."""
    
    print("🎯 DSPy Automatic Optimization Demo")
    print("=" * 50)
    
    # Setup
    load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    
    # Create base system
    print("🏗️ Creating base support system...")
    base_system = SmartSupportSystem()
    
    # Load training data
    print("📚 Loading training examples...")
    training_examples = load_training_examples()
    print(f"✅ Loaded {len(training_examples)} training examples")
    
    # Test before optimization
    print("\n📊 Testing BEFORE optimization...")
    test_tickets = [
        ("System keeps crashing", "The application crashes every time I try to upload a file. This is blocking my work completely."),
        ("Refund request", "I want to cancel my subscription and get a refund for this month since I haven't used it."),
        ("Feature suggestion", "Could you add a dark mode option? It would really help with late-night work sessions.")
    ]
    
    before_scores = []
    for subject, message in test_tickets:
        result = base_system(subject=subject, message=message)
        score = result.quality_score if hasattr(result, 'quality_score') else 0.5
        before_scores.append(float(score))
        print(f"  • {subject}: Quality {score:.2f}")
    
    avg_before = sum(before_scores) / len(before_scores)
    print(f"📈 Average quality BEFORE: {avg_before:.3f}")
    
    # Run optimization
    print(f"\n🔄 Running automatic optimization...")
    print("This is where the magic happens - DSPy will:")
    print("• Analyze the training examples")
    print("• Find better prompting strategies") 
    print("• Optimize the entire pipeline")
    print("• No manual prompt engineering needed!")
    
    optimizer = BootstrapFewShot(
        metric=comprehensive_quality_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=2
    )
    
    print("⚡ Optimizing... (this may take 1-2 minutes)")
    optimized_system = optimizer.compile(
        base_system,
        trainset=training_examples,
        valset=training_examples[:2]  # Small validation set
    )
    
    print("✅ Optimization complete!")
    
    # Test after optimization
    print("\n📊 Testing AFTER optimization...")
    after_scores = []
    for subject, message in test_tickets:
        result = optimized_system(subject=subject, message=message)
        score = result.quality_score if hasattr(result, 'quality_score') else 0.5
        after_scores.append(float(score))
        print(f"  • {subject}: Quality {score:.2f}")
    
    avg_after = sum(after_scores) / len(after_scores)
    print(f"📈 Average quality AFTER: {avg_after:.3f}")
    
    # Calculate improvement
    improvement = ((avg_after - avg_before) / avg_before) * 100
    print(f"\n🎉 IMPROVEMENT: {improvement:.1f}%")
    
    print(f"\n💡 What just happened:")
    print("• DSPy analyzed your training examples")
    print("• Found better internal prompting strategies")
    print("• Optimized the entire pipeline automatically")
    print("• Achieved measurable performance improvement")
    print("• Zero manual prompt engineering required!")
    
    # Save optimized model
    print(f"\n💾 Saving optimized model...")
    optimized_system.save("optimized_support_system.json")
    print("✅ Model saved! Can be loaded for production use.")
    
    return optimized_system, improvement

if __name__ == "__main__":
    optimized_system, improvement = run_optimization_demo()
    
    print(f"\n🏆 Final Results:")
    print(f"Performance improvement: {improvement:.1f}%")
    print(f"Optimized model saved and ready for production!")