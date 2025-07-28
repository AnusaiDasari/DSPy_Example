"""
DSPy-powered customer support system - elegant and optimizable
This is how it should be done!
"""

import dspy
from typing import Literal, List, Optional
import json

# Define clear signatures for each component
class TicketClassifier(dspy.Signature):
    """Classify customer support tickets by category, priority and response type."""
    
    subject: str = dspy.InputField(desc="Ticket subject line")
    message: str = dspy.InputField(desc="Customer message content")
    
    category: Literal['Technical', 'Billing', 'Sales', 'Feature_Request'] = dspy.OutputField(
        desc="Primary category of the issue"
    )
    priority: Literal['Critical', 'High', 'Medium', 'Low'] = dspy.OutputField(
        desc="Urgency level based on business impact"
    )
    response_type: Literal['Troubleshooting', 'Account_Review', 'Information', 'Product_Feedback', 'Account_Recovery'] = dspy.OutputField(
        desc="Type of response needed"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of classification")

class KnowledgeRetriever(dspy.Signature):
    """Find relevant solution information from knowledge base."""
    
    category: str = dspy.InputField()
    issue_description: str = dspy.InputField()
    
    relevant_solution: str = dspy.OutputField(desc="Most relevant solution steps")
    escalation_needed: bool = dspy.OutputField(desc="Whether human escalation is required")

class ResponseGenerator(dspy.Signature):
    """Generate helpful customer support responses."""
    
    customer_message: str = dspy.InputField()
    issue_category: str = dspy.InputField() 
    priority_level: str = dspy.InputField()
    solution_info: str = dspy.InputField()
    escalation_needed: bool = dspy.InputField()
    
    response: str = dspy.OutputField(desc="Professional, helpful customer response")
    confidence: float = dspy.OutputField(desc="Confidence in response quality 0.0-1.0")

class QualityEvaluator(dspy.Signature):
    """Evaluate response quality and suggest improvements."""
    
    original_message: str = dspy.InputField()
    generated_response: str = dspy.InputField()
    
    quality_score: float = dspy.OutputField(desc="Quality rating 0.0-1.0")
    is_helpful: bool = dspy.OutputField(desc="Whether response addresses the issue")
    is_professional: bool = dspy.OutputField(desc="Whether tone is appropriate")
    suggestions: str = dspy.OutputField(desc="Improvement suggestions if needed")

class SmartSupportSystem(dspy.Module):
    """Complete customer support system using DSPy modules."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize all components
        self.classifier = dspy.ChainOfThought(TicketClassifier)
        self.knowledge_retriever = dspy.ChainOfThought(KnowledgeRetriever)
        self.response_generator = dspy.ChainOfThought(ResponseGenerator)
        self.quality_evaluator = dspy.ChainOfThought(QualityEvaluator)
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load solution knowledge base."""
        try:
            with open('data/knowledge_base/solutions.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback knowledge base
            return [
                {
                    "category": "Technical",
                    "topic": "General",
                    "solution": "Please try restarting the application and clearing your browser cache. If the issue persists, our technical team will investigate further.",
                    "escalation_required": False
                }
            ]
    
    def forward(self, subject: str, message: str):
        """Process a customer support ticket end-to-end."""
        
        # Step 1: Classify the ticket
        classification = self.classifier(
            subject=subject,
            message=message
        )
        
        # Step 2: Retrieve relevant knowledge
        knowledge = self.knowledge_retriever(
            category=classification.category,
            issue_description=f"{subject}\n{message}"
        )
        
        # Step 3: Generate response
        response = self.response_generator(
            customer_message=message,
            issue_category=classification.category,
            priority_level=classification.priority,
            solution_info=knowledge.relevant_solution,
            escalation_needed=knowledge.escalation_needed
        )
        
        # Step 4: Evaluate quality
        quality = self.quality_evaluator(
            original_message=message,
            generated_response=response.response
        )
        
        return dspy.Prediction(
            # Classification results
            category=classification.category,
            priority=classification.priority,
            response_type=classification.response_type,
            classification_reasoning=classification.reasoning,
            
            # Knowledge retrieval
            solution_info=knowledge.relevant_solution,
            escalation_needed=knowledge.escalation_needed,
            
            # Generated response
            response=response.response,
            response_confidence=response.confidence,
            
            # Quality assessment
            quality_score=quality.quality_score,
            is_helpful=quality.is_helpful,
            is_professional=quality.is_professional,
            improvement_suggestions=quality.suggestions
        )

def demonstrate_dspy_advantages():
    """Show how DSPy solves the prompting problems elegantly."""
    
    print("✅ DSPY DECLARATIVE PROGRAMMING APPROACH")
    print("=" * 50)
    print("Advantages of DSPy:")
    print("• Declare what you want, not how to prompt")
    print("• Automatic optimization with examples")
    print("• Composable modules that work together") 
    print("• Type safety and clear contracts")
    print("• Systematic performance improvement")
    print("• Production-ready and maintainable\n")
    
    # Setup DSPy
    from dotenv import load_dotenv
    load_dotenv()
    
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    
    # Initialize system
    support_system = SmartSupportSystem()
    
    # Test the same ticket as traditional approach
    subject = "Account locked after failed login attempts"
    message = "My account got locked and I can't access my work files. This is urgent!"
    
    print(f"🎫 Test Ticket:")
    print(f"Subject: {subject}")
    print(f"Message: {message}\n")
    
    # Process with DSPy
    print("🔄 Processing with DSPy...")
    result = support_system(subject=subject, message=message)
    
    print("📊 Results:")
    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")
    print(f"Response Type: {result.response_type}")
    print(f"Escalation Needed: {result.escalation_needed}")
    print(f"Quality Score: {result.quality_score:.2f}")
    
    print(f"\n💬 Generated Response:")
    print(result.response)
    
    print(f"\n🧠 System Reasoning:")
    print(result.classification_reasoning)
    
    if result.improvement_suggestions:
        print(f"\n💡 Improvement Suggestions:")
        print(result.improvement_suggestions)
    
    print(f"\n🎉 DSPy Benefits Demonstrated:")
    print("• Clean, declarative code")
    print("• Systematic multi-step processing")
    print("• Built-in quality evaluation") 
    print("• Ready for automatic optimization")
    print("• Much easier to maintain and extend")

if __name__ == "__main__":
    demonstrate_dspy_advantages()