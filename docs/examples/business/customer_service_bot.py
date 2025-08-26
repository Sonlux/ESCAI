"""
Example: Customer Service Bot with ESCAI Monitoring
Description: Demonstrates monitoring a customer service chatbot that handles inquiries,
            escalates complex issues, and tracks customer satisfaction
Framework: LangChain
Complexity: Intermediate
"""

import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage

from escai_framework import monitor_agent
from escai_framework.metrics import CustomMetric


class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class CustomerTicket:
    ticket_id: str
    customer_id: str
    issue_type: str
    description: str
    priority: TicketPriority
    status: TicketStatus
    created_at: float
    resolved_at: Optional[float] = None
    satisfaction_score: Optional[int] = None


class KnowledgeBaseTool(BaseTool):
    """Tool for searching the knowledge base."""
    
    name = "knowledge_base_search"
    description = "Search the knowledge base for solutions to customer issues"
    
    def __init__(self):
        super().__init__()
        # Simulated knowledge base
        self.knowledge_base = {
            "password_reset": {
                "solution": "Guide customer to reset password via email link",
                "steps": [
                    "Go to login page",
                    "Click 'Forgot Password'",
                    "Enter email address",
                    "Check email for reset link"
                ],
                "estimated_time": 5
            },
            "billing_inquiry": {
                "solution": "Check billing history and explain charges",
                "steps": [
                    "Verify customer identity",
                    "Access billing system",
                    "Review recent charges",
                    "Explain billing details"
                ],
                "estimated_time": 10
            },
            "technical_issue": {
                "solution": "Troubleshoot technical problems step by step",
                "steps": [
                    "Identify the specific issue",
                    "Check system status",
                    "Guide through troubleshooting steps",
                    "Escalate if unresolved"
                ],
                "estimated_time": 15
            }
        }
    
    def _run(self, query: str) -> str:
        """Search knowledge base for relevant information."""
        query_lower = query.lower()
        
        for issue_type, info in self.knowledge_base.items():
            if any(keyword in query_lower for keyword in issue_type.split('_')):
                return f"Found solution for {issue_type}:\n" \
                       f"Solution: {info['solution']}\n" \
                       f"Steps: {', '.join(info['steps'])}\n" \
                       f"Estimated time: {info['estimated_time']} minutes"
        
        return "No specific solution found in knowledge base. Consider escalating to human agent."


class TicketManagementTool(BaseTool):
    """Tool for managing customer tickets."""
    
    name = "ticket_management"
    description = "Create, update, and manage customer support tickets"
    
    def __init__(self):
        super().__init__()
        self.tickets: Dict[str, CustomerTicket] = {}
        self.ticket_counter = 1
    
    def _run(self, action: str, **kwargs) -> str:
        """Perform ticket management actions."""
        if action == "create":
            return self._create_ticket(**kwargs)
        elif action == "update":
            return self._update_ticket(**kwargs)
        elif action == "get":
            return self._get_ticket(**kwargs)
        elif action == "escalate":
            return self._escalate_ticket(**kwargs)
        else:
            return f"Unknown action: {action}"
    
    def _create_ticket(self, customer_id: str, issue_type: str, 
                      description: str, priority: str = "medium") -> str:
        """Create a new customer ticket."""
        ticket_id = f"TICKET-{self.ticket_counter:04d}"
        self.ticket_counter += 1
        
        ticket = CustomerTicket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            issue_type=issue_type,
            description=description,
            priority=TicketPriority(priority),
            status=TicketStatus.OPEN,
            created_at=time.time()
        )
        
        self.tickets[ticket_id] = ticket
        return f"Created ticket {ticket_id} for customer {customer_id}"
    
    def _update_ticket(self, ticket_id: str, status: str = None, 
                      satisfaction_score: int = None) -> str:
        """Update an existing ticket."""
        if ticket_id not in self.tickets:
            return f"Ticket {ticket_id} not found"
        
        ticket = self.tickets[ticket_id]
        
        if status:
            ticket.status = TicketStatus(status)
            if status == "resolved":
                ticket.resolved_at = time.time()
        
        if satisfaction_score:
            ticket.satisfaction_score = satisfaction_score
        
        return f"Updated ticket {ticket_id}"
    
    def _get_ticket(self, ticket_id: str) -> str:
        """Get ticket information."""
        if ticket_id not in self.tickets:
            return f"Ticket {ticket_id} not found"
        
        ticket = self.tickets[ticket_id]
        return f"Ticket {ticket_id}: {ticket.issue_type} - {ticket.status.value}"
    
    def _escalate_ticket(self, ticket_id: str, reason: str) -> str:
        """Escalate ticket to human agent."""
        if ticket_id not in self.tickets:
            return f"Ticket {ticket_id} not found"
        
        ticket = self.tickets[ticket_id]
        ticket.status = TicketStatus.ESCALATED
        
        return f"Escalated ticket {ticket_id} to human agent. Reason: {reason}"


class CustomerSatisfactionTool(BaseTool):
    """Tool for collecting customer satisfaction feedback."""
    
    name = "satisfaction_survey"
    description = "Collect customer satisfaction feedback after resolving issues"
    
    def _run(self, ticket_id: str, ask_feedback: bool = True) -> str:
        """Collect customer satisfaction feedback."""
        if ask_feedback:
            # Simulate customer feedback (in real implementation, this would be interactive)
            import random
            satisfaction_score = random.randint(1, 5)  # 1-5 scale
            
            feedback_messages = {
                1: "Very dissatisfied - issue not resolved properly",
                2: "Dissatisfied - took too long to resolve",
                3: "Neutral - issue resolved but could be better",
                4: "Satisfied - good service and quick resolution",
                5: "Very satisfied - excellent service and support"
            }
            
            return f"Customer satisfaction score: {satisfaction_score}/5 - {feedback_messages[satisfaction_score]}"
        
        return "Satisfaction survey sent to customer"


class CustomerServiceBot:
    """Customer service chatbot with ESCAI monitoring."""
    
    def __init__(self, openai_api_key: str):
        """Initialize the customer service bot."""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,  # Low temperature for consistent responses
            openai_api_key=openai_api_key
        )
        
        # Initialize tools
        self.knowledge_base = KnowledgeBaseTool()
        self.ticket_manager = TicketManagementTool()
        self.satisfaction_tool = CustomerSatisfactionTool()
        
        self.tools = [
            self.knowledge_base,
            self.ticket_manager,
            self.satisfaction_tool
        ]
        
        # Create conversation memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,  # Remember last 10 exchanges
            return_messages=True
        )
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer service agent. Your goal is to:
            1. Understand customer issues clearly
            2. Search the knowledge base for solutions
            3. Create tickets for tracking issues
            4. Provide step-by-step guidance
            5. Escalate complex issues when necessary
            6. Collect satisfaction feedback after resolution
            
            Always be polite, professional, and empathetic. If you cannot resolve an issue,
            escalate it to a human agent with a clear explanation.
            
            Available tools:
            - knowledge_base_search: Search for solutions
            - ticket_management: Create and manage tickets
            - satisfaction_survey: Collect feedback
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=10
        )
    
    def handle_customer_inquiry(self, customer_id: str, inquiry: str) -> Dict:
        """Handle a customer inquiry with monitoring."""
        
        # Define custom metrics for customer service
        resolution_time_metric = CustomMetric(
            name="resolution_time",
            description="Time taken to resolve customer issue",
            extractor=lambda context: context.get("resolution_time_seconds", 0)
        )
        
        escalation_rate_metric = CustomMetric(
            name="escalation_rate",
            description="Rate of issues escalated to human agents",
            extractor=lambda context: 1 if context.get("escalated", False) else 0
        )
        
        satisfaction_metric = CustomMetric(
            name="customer_satisfaction",
            description="Customer satisfaction score (1-5)",
            extractor=lambda context: context.get("satisfaction_score", 0)
        )
        
        # Configure monitoring
        config = {
            "capture_reasoning": True,
            "monitor_tools": True,
            "track_tool_selection": True,
            "capture_tool_outputs": True,
            "monitor_memory": True,
            "track_conversation_flow": True,
            "custom_metrics": [
                resolution_time_metric,
                escalation_rate_metric,
                satisfaction_metric
            ],
            "alert_thresholds": {
                "low_confidence": 0.3,
                "high_resolution_time": 300,  # 5 minutes
                "high_escalation_rate": 0.3   # 30%
            }
        }
        
        start_time = time.time()
        
        with monitor_agent(
            agent_id=f"customer-service-bot-{customer_id}",
            framework="langchain",
            config=config
        ) as session:
            
            try:
                # Process the customer inquiry
                response = self.agent_executor.invoke({
                    "input": f"Customer {customer_id} says: {inquiry}"
                })
                
                resolution_time = time.time() - start_time
                
                # Extract information from the response
                escalated = "escalat" in response["output"].lower()
                
                # Simulate satisfaction score collection
                if not escalated and "resolved" in response["output"].lower():
                    satisfaction_response = self.satisfaction_tool._run("", ask_feedback=True)
                    satisfaction_score = int(satisfaction_response.split("score: ")[1].split("/")[0])
                else:
                    satisfaction_score = 0
                
                # Update session context with metrics
                session.update_context({
                    "resolution_time_seconds": resolution_time,
                    "escalated": escalated,
                    "satisfaction_score": satisfaction_score,
                    "customer_id": customer_id,
                    "inquiry_type": self._classify_inquiry(inquiry)
                })
                
                # Get monitoring insights
                current_state = session.get_current_epistemic_state()
                tool_analysis = session.get_tool_usage_analysis()
                
                return {
                    "response": response["output"],
                    "resolution_time": resolution_time,
                    "escalated": escalated,
                    "satisfaction_score": satisfaction_score,
                    "confidence": current_state.confidence_level,
                    "tools_used": [step.tool for step in response.get("intermediate_steps", [])],
                    "tool_analysis": tool_analysis,
                    "reasoning_steps": len(response.get("intermediate_steps", [])),
                    "memory_items": len(self.memory.chat_memory.messages)
                }
                
            except Exception as e:
                # Handle errors gracefully
                session.update_context({
                    "error": str(e),
                    "resolution_time_seconds": time.time() - start_time,
                    "escalated": True,  # Errors should be escalated
                    "satisfaction_score": 1  # Low satisfaction for errors
                })
                
                return {
                    "response": f"I apologize, but I encountered an error while processing your request. "
                              f"I'm escalating this to a human agent who will assist you shortly. "
                              f"Error reference: {str(e)[:50]}...",
                    "resolution_time": time.time() - start_time,
                    "escalated": True,
                    "satisfaction_score": 1,
                    "error": str(e)
                }
    
    def _classify_inquiry(self, inquiry: str) -> str:
        """Classify the type of customer inquiry."""
        inquiry_lower = inquiry.lower()
        
        if any(word in inquiry_lower for word in ["password", "login", "access"]):
            return "authentication"
        elif any(word in inquiry_lower for word in ["bill", "charge", "payment", "invoice"]):
            return "billing"
        elif any(word in inquiry_lower for word in ["bug", "error", "not working", "broken"]):
            return "technical"
        elif any(word in inquiry_lower for word in ["cancel", "refund", "return"]):
            return "cancellation"
        elif any(word in inquiry_lower for word in ["how to", "help", "guide", "tutorial"]):
            return "how_to"
        else:
            return "general"
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for the customer service bot."""
        tickets = self.ticket_manager.tickets
        
        if not tickets:
            return {"message": "No tickets processed yet"}
        
        total_tickets = len(tickets)
        resolved_tickets = sum(1 for t in tickets.values() if t.status == TicketStatus.RESOLVED)
        escalated_tickets = sum(1 for t in tickets.values() if t.status == TicketStatus.ESCALATED)
        
        satisfaction_scores = [t.satisfaction_score for t in tickets.values() 
                             if t.satisfaction_score is not None]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
        
        resolution_times = []
        for ticket in tickets.values():
            if ticket.resolved_at:
                resolution_times.append(ticket.resolved_at - ticket.created_at)
        
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        return {
            "total_tickets": total_tickets,
            "resolved_tickets": resolved_tickets,
            "escalated_tickets": escalated_tickets,
            "resolution_rate": resolved_tickets / total_tickets if total_tickets > 0 else 0,
            "escalation_rate": escalated_tickets / total_tickets if total_tickets > 0 else 0,
            "average_satisfaction": avg_satisfaction,
            "average_resolution_time_seconds": avg_resolution_time,
            "satisfaction_distribution": {
                score: sum(1 for s in satisfaction_scores if s == score)
                for score in range(1, 6)
            }
        }


def simulate_customer_interactions(bot: CustomerServiceBot, num_interactions: int = 5):
    """Simulate multiple customer interactions."""
    
    # Sample customer inquiries
    sample_inquiries = [
        ("CUST001", "I forgot my password and can't log into my account"),
        ("CUST002", "I was charged twice for my subscription this month"),
        ("CUST003", "The mobile app keeps crashing when I try to upload files"),
        ("CUST004", "How do I cancel my subscription?"),
        ("CUST005", "I need help setting up two-factor authentication"),
        ("CUST006", "My data export has been stuck at 50% for hours"),
        ("CUST007", "Can you explain the charges on my latest invoice?"),
        ("CUST008", "The website is very slow and sometimes doesn't load"),
    ]
    
    results = []
    
    print("=== Customer Service Bot Simulation ===\n")
    
    for i in range(min(num_interactions, len(sample_inquiries))):
        customer_id, inquiry = sample_inquiries[i]
        
        print(f"--- Interaction {i+1}: Customer {customer_id} ---")
        print(f"Inquiry: {inquiry}")
        
        # Process the inquiry
        result = bot.handle_customer_inquiry(customer_id, inquiry)
        
        print(f"Response: {result['response'][:200]}...")
        print(f"Resolution Time: {result['resolution_time']:.2f}s")
        print(f"Escalated: {result['escalated']}")
        print(f"Satisfaction: {result['satisfaction_score']}/5")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Tools Used: {', '.join(result.get('tools_used', []))}")
        print()
        
        results.append(result)
        
        # Brief pause between interactions
        time.sleep(1)
    
    return results


def main():
    """Main function to run the customer service bot example."""
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Initialize the customer service bot
    print("Initializing Customer Service Bot...")
    bot = CustomerServiceBot(openai_api_key)
    
    # Simulate customer interactions
    results = simulate_customer_interactions(bot, num_interactions=5)
    
    # Print performance summary
    print("=== Performance Summary ===")
    summary = bot.get_performance_summary()
    
    if "message" not in summary:
        print(f"Total Tickets: {summary['total_tickets']}")
        print(f"Resolution Rate: {summary['resolution_rate']:.1%}")
        print(f"Escalation Rate: {summary['escalation_rate']:.1%}")
        print(f"Average Satisfaction: {summary['average_satisfaction']:.1f}/5")
        print(f"Average Resolution Time: {summary['average_resolution_time_seconds']:.1f}s")
        
        print("\nSatisfaction Distribution:")
        for score, count in summary['satisfaction_distribution'].items():
            if count > 0:
                print(f"  {score} stars: {count} customers")
    else:
        print(summary['message'])
    
    # Overall analysis
    total_resolution_time = sum(r['resolution_time'] for r in results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    escalation_rate = sum(1 for r in results if r['escalated']) / len(results)
    
    print(f"\n=== Overall Analysis ===")
    print(f"Total Processing Time: {total_resolution_time:.2f}s")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Escalation Rate: {escalation_rate:.1%}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if escalation_rate > 0.3:
        print("- High escalation rate detected. Consider expanding knowledge base.")
    if avg_confidence < 0.7:
        print("- Low average confidence. Consider improving agent training.")
    if total_resolution_time / len(results) > 30:
        print("- High average resolution time. Consider optimizing workflows.")
    
    print("\nCustomer service bot simulation completed!")


if __name__ == "__main__":
    main()