"""
Production-ready FastAPI server with the optimized DSPy system
Shows how to deploy DSPy applications in production
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
import dspy
import sys
sys.path.append('..')
from src.modules.support_system import SmartSupportSystem
from dotenv import load_dotenv
import uvicorn
import logging
from typing import Optional, List
import asyncio
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Initialize the optimized system
support_system = SmartSupportSystem()

# Try to load optimized model if available
try:
    support_system.load("optimized_support_system.json")
    logger.info("✅ Loaded optimized DSPy model")
except:
    logger.info("ℹ️ Using base DSPy model (run optimization_demo.py to create optimized version)")

# FastAPI app
app = FastAPI(
    title="DSPy Customer Support API",
    description="Production customer support system powered by DSPy",
    version="1.0.0"
)

# Request models
class SupportTicket(BaseModel):
    subject: str
    message: str
    customer_email: EmailStr
    priority_override: Optional[str] = None

class BatchTickets(BaseModel):
    tickets: List[SupportTicket]

class FeedbackRequest(BaseModel):
    ticket_id: str
    response_quality: float  # 0.0 to 1.0
    was_helpful: bool
    comments: Optional[str] = None

# Response models
class TicketResponse(BaseModel):
    ticket_id: str
    category: str
    priority: str
    response_type: str
    generated_response: str
    quality_score: float
    escalation_needed: bool
    processing_time_ms: int
    timestamp: datetime

# Global ticket counter
ticket_counter = 0

@app.post("/process-ticket", response_model=TicketResponse)
async def process_support_ticket(ticket: SupportTicket):
    """Process a single customer support ticket."""
    global ticket_counter
    ticket_counter += 1
    ticket_id = f"API{ticket_counter:06d}"
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Process with DSPy system
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: support_system(
                subject=ticket.subject,
                message=ticket.message
            )
        )
        
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Override priority if requested
        final_priority = ticket.priority_override or result.priority
        
        response = TicketResponse(
            ticket_id=ticket_id,
            category=result.category,
            priority=final_priority,
            response_type=result.response_type,
            generated_response=result.response,
            quality_score=float(result.quality_score),
            escalation_needed=result.escalation_needed,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Processed ticket {ticket_id}: {result.category}/{final_priority}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing ticket {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process-batch")
async def process_ticket_batch(batch: BatchTickets):
    """Process multiple tickets in batch for efficiency."""
    
    if len(batch.tickets) > 50:
        raise HTTPException(status_code=400, detail="Batch size limited to 50 tickets")
    
    start_time = asyncio.get_event_loop().time()
    
    # Process tickets concurrently
    semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
    
    async def process_single(ticket: SupportTicket):
        async with semaphore:
            try:
                return await process_support_ticket(ticket)
            except Exception as e:
                return {"error": str(e), "ticket": ticket.dict()}
    
    tasks = [process_single(ticket) for ticket in batch.tickets]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
    
    # Separate successful and failed results
    successful = [r for r in results if not isinstance(r, Exception) and "error" not in r]
    failed = [r for r in results if isinstance(r, Exception) or "error" in r]
    
    return {
        "total_tickets": len(batch.tickets),
        "successful": len(successful),
        "failed": len(failed),
        "results": successful,
        "errors": failed,
        "total_processing_time_ms": processing_time
    }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for continuous improvement."""
    
    # In production, this would be stored in a database for retraining
    logger.info(f"Feedback for {feedback.ticket_id}: quality={feedback.response_quality}, helpful={feedback.was_helpful}")
    
    # TODO: Add to training data for next optimization cycle
    
    return {
        "message": "Feedback received",
        "ticket_id": feedback.ticket_id,
        "status": "recorded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "system": "DSPy Customer Support",
        "model_loaded": True,
        "tickets_processed": ticket_counter,
        "timestamp": datetime.now()
    }

@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics."""
    return {
        "tickets_processed_today": ticket_counter,
        "system_version": "1.0.0",
        "dspy_optimized": True,
        "average_processing_time_ms": 2500,  # Estimated
        "supported_categories": ["Technical", "Billing", "Sales", "Feature_Request"],
        "supported_priorities": ["Critical", "High", "Medium", "Low"]
    }

@app.get("/")
async def root():
    """API root with information."""
    return {
        "message": "DSPy Customer Support API",
        "description": "Production customer support powered by optimized DSPy models",
        "version": "1.0.0",
        "endpoints": {
            "process_ticket": "POST /process-ticket - Process single ticket",
            "process_batch": "POST /process-batch - Process multiple tickets", 
            "feedback": "POST /feedback - Submit response feedback",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - System metrics"
        },
        "docs": "/docs",
        "advantages": [
            "Automatic classification and prioritization",
            "Intelligent response generation", 
            "Built-in quality assessment",
            "Optimized with DSPy for better performance",
            "Scalable batch processing"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting DSPy Customer Support API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎯 Example: curl -X POST http://localhost:8000/process-ticket \\")
    print('   -H "Content-Type: application/json" \\')
    print('   -d \'{"subject":"Help needed","message":"I need assistance","customer_email":"user@example.com"}\'')
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)