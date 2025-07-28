"""
Traditional manual prompting approach - shows the pain points
This is what developers have to do without DSPy
"""

import openai
import json
import os
from dotenv import load_dotenv

class TraditionalSupportSystem:
    """Manual prompting approach - brittle and hard to optimize"""
    
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI()
    
    def classify_ticket_manual(self, subject, message):
        """Manual prompt for ticket classification - hard to optimize"""
        
        # Manually crafted prompt (this took hours to get right!)
        prompt = f"""You are a customer support classifier. Please analyze this ticket carefully.

Subject: {subject}
Message: {message}

Classify this ticket by determining:
1. Category (Technical, Billing, Sales, Feature_Request)  
2. Priority (Critical, High, Medium, Low)
3. Response Type (Troubleshooting, Account_Review, Information, Product_Feedback, Account_Recovery)

Be very careful and accurate. Consider urgency keywords and business impact.

Respond in this exact format:
Category: [category]
Priority: [priority] 
Response Type: [response_type]
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Manual parsing (error-prone!)
            result = response.choices[0].message.content
            lines = result.strip().split('\n')
            
            classification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    classification[key.strip().lower().replace(' ', '_')] = value.strip()
            
            return classification
            
        except Exception as e:
            return {"error": f"Classification failed: {e}"}
    
    def generate_response_manual(self, subject, message, category, priority):
        """Manual prompt for response generation - also hard to optimize"""
        
        # Another manually crafted prompt
        prompt = f"""You are a helpful customer support agent. Generate a professional response.

Customer Issue:
Subject: {subject}
Message: {message}
Category: {category}
Priority: {priority}

Write a helpful, empathetic response that:
1. Acknowledges the issue
2. Provides helpful information or next steps
3. Shows you understand the urgency level
4. Maintains a professional but friendly tone

Response:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Response generation failed: {e}"

def demonstrate_traditional_problems():
    """Show why traditional prompting is problematic"""
    
    print("❌ TRADITIONAL MANUAL PROMPTING APPROACH")
    print("=" * 50)
    print("Problems with this approach:")
    print("• Manual prompt crafting takes hours")
    print("• Hard to optimize systematically") 
    print("• Brittle - breaks with small changes")
    print("• Error-prone parsing")
    print("• No way to improve automatically")
    print("• Doesn't scale to complex workflows\n")
    
    # Demo the system
    system = TraditionalSupportSystem()
    
    # Test ticket
    subject = "Account locked after failed login attempts"
    message = "My account got locked and I can't access my work files. This is urgent!"
    
    print(f"🎫 Test Ticket:")
    print(f"Subject: {subject}")
    print(f"Message: {message}\n")
    
    # Classification
    print("🔍 Manual Classification:")
    classification = system.classify_ticket_manual(subject, message)
    print(json.dumps(classification, indent=2))
    
    # Response generation  
    print("\n💬 Manual Response Generation:")
    response = system.generate_response_manual(
        subject, message, 
        classification.get('category', 'Unknown'),
        classification.get('priority', 'Medium')
    )
    print(response)
    
    print(f"\n💔 Issues with this approach:")
    print("• Took hours to craft these prompts")
    print("• No systematic way to improve them")
    print("• Will break if requirements change")
    print("• Manual error handling everywhere")

if __name__ == "__main__":
    demonstrate_traditional_problems()