"""
Complete demonstration of the DSPy customer support system
This is the main showcase file that demonstrates everything
"""

import dspy
import json
import sys
sys.path.append('..')
from src.modules.support_system import SmartSupportSystem
from src.traditional.manual_prompting import TraditionalSupportSystem
from dotenv import load_dotenv
import time

def load_test_tickets():
    """Load test tickets for demonstration."""
    try:
        with open('data/tickets/sample_tickets.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback test tickets
        return [
            {
                "id": "DEMO1",
                "subject": "Can't log into my account",
                "message": "I've been trying to log in for the past hour but keep getting 'invalid credentials' error. I know my password is correct. This is preventing me from accessing important client files.",
                "customer_email": "urgent.user@company.com"
            },
            {
                "id": "DEMO2", 
                "subject": "Billing question",
                "message": "I noticed an extra charge on my card this month. Can someone help me understand what this is for?",
                "customer_email": "billing.question@email.com"
            }
        ]

def compare_approaches():
    """Direct comparison between traditional and DSPy approaches."""
    
    print("🥊 TRADITIONAL vs DSPY: HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    
    # Setup both systems
    load_dotenv()
    
    # Traditional system
    traditional = TraditionalSupportSystem()
    
    # DSPy system  
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    dspy_system = SmartSupportSystem()
    
    # Test ticket
    test_ticket = {
        "subject": "Account locked - need urgent help",
        "message": "My account got locked after I mistyped my password a few times. I have a client meeting in 30 minutes and need to access my presentation files immediately. Please help!"
    }
    
    print(f"🎫 Test Ticket:")
    print(f"Subject: {test_ticket['subject']}")
    print(f"Message: {test_ticket['message'][:100]}...\n")
    
    # Traditional approach
    print("❌ TRADITIONAL MANUAL PROMPTING")
    print("-" * 40)
    start_time = time.time()
    
    trad_classification = traditional.classify_ticket_manual(
        test_ticket['subject'], 
        test_ticket['message']
    )
    trad_response = traditional.generate_response_manual(
        test_ticket['subject'],
        test_ticket['message'], 
        trad_classification.get('category', 'Unknown'),
        trad_classification.get('priority', 'Medium')
    )
    
    trad_time = time.time() - start_time
    
    print(f"Classification: {trad_classification}")
    print(f"Response: {trad_response[:150]}...")
    print(f"⏱️ Time: {trad_time:.2f} seconds")
    print(f"❌ Issues: Manual prompts, error-prone parsing, hard to optimize\n")
    
    # DSPy approach
    print("✅ DSPY DECLARATIVE PROGRAMMING")
    print("-" * 40)
    start_time = time.time()
    
    dspy_result = dspy_system(
        subject=test_ticket['subject'],
        message=test_ticket['message']
    )
    
    dspy_time = time.time() - start_time
    
    print(f"Category: {dspy_result.category}")
    print(f"Priority: {dspy_result.priority}")
    print(f"Response Type: {dspy_result.response_type}")
    print(f"Quality Score: {dspy_result.quality_score:.2f}")
    print(f"Response: {dspy_result.response[:150]}...")
    print(f"⏱️ Time: {dspy_time:.2f} seconds")
    print(f"✅ Benefits: Declarative, optimizable, maintainable, robust\n")
    
    # Summary
    print("🏆 WINNER: DSPy")
    print("Reasons:")
    print("• More structured and reliable output")
    print("• Built-in quality assessment")
    print("• Automatic optimization capability")
    print("• Easier to maintain and extend")
    print("• Production-ready architecture")

def full_system_demo():
    """Demonstrate the complete DSPy system capabilities."""
    
    print("\n🚀 COMPLETE DSPY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Setup
    load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    
    # Initialize system
    support_system = SmartSupportSystem()
    
    # Load test tickets
    test_tickets = load_test_tickets()
    
    print(f"📋 Processing {len(test_tickets)} customer tickets...\n")
    
    results = []
    for i, ticket in enumerate(test_tickets, 1):
        print(f"🎫 Ticket {i}: {ticket['id']}")
        print(f"Subject: {ticket['subject']}")
        print(f"From: {ticket['customer_email']}")
        
        # Process with DSPy
        result = support_system(
            subject=ticket['subject'],
            message=ticket['message']
        )
        
        # Display results
        print(f"📊 Analysis:")
        print(f"  Category: {result.category}")
        print(f"  Priority: {result.priority}")
        print(f"  Response Type: {result.response_type}")
        print(f"  Quality Score: {result.quality_score:.2f}")
        print(f"  Escalation Needed: {result.escalation_needed}")
        
        print(f"💬 Generated Response:")
        print(f"  {result.response}")
        
        if result.improvement_suggestions:
            print(f"💡 Suggestions: {result.improvement_suggestions}")
        
        print(f"🧠 Reasoning: {result.classification_reasoning}")
        print("-" * 50)
        
        results.append({
            'ticket_id': ticket['id'],
            'category': result.category,
            'priority': result.priority,
            'quality_score': float(result.quality_score),
            'escalation_needed': result.escalation_needed
        })
    
    # Summary statistics
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    high_priority = sum(1 for r in results if r['priority'] in ['Critical', 'High'])
    escalations = sum(1 for r in results if r['escalation_needed'])
    
    print(f"\n📈 SYSTEM PERFORMANCE SUMMARY")
    print(f"Average Quality Score: {avg_quality:.2f}")
    print(f"High Priority Tickets: {high_priority}/{len(results)}")
    print(f"Escalations Required: {escalations}/{len(results)}")
    print(f"Processing Success Rate: 100%")

def main():
    """Run the complete demonstration."""
    
    print("🎯 DSPy Customer Support System")
    print("The definitive example of why DSPy beats traditional prompting")
    print("=" * 70)
    
    try:
        # Part 1: Direct comparison
        compare_approaches()
        
        # Part 2: Full system demo
        full_system_demo()
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print(f"Key Takeaways:")
        print(f"• DSPy eliminates manual prompt engineering")
        print(f"• Systematic approach beats trial-and-error")
        print(f"• Built-in optimization and quality assessment")
        print(f"• Production-ready and maintainable")
        print(f"• Perfect for complex, multi-step applications")
        
        print(f"\n📚 Next Steps:")
        print(f"• Run 'python examples/optimization_demo.py' to see automatic improvement")
        print(f"• Check 'scripts/benchmark.py' for performance evaluation")
        print(f"• Explore the production API in 'production/api.py'")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print(f"Make sure you have:")
        print(f"• Installed requirements: pip install -r requirements.txt")
        print(f"• Set up .env file with your OpenAI API key")

if __name__ == "__main__":
    main()