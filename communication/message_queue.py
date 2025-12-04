"""
Message Queue for Inter-Agent Communication
============================================

EDUCATIONAL PURPOSE - AGENT COMMUNICATION CONCEPTS:

1. SPEECH ACT THEORY (Austin & Searle):
   - Locutionary act: The act of saying something
   - Illocutionary act: The act performed IN saying (inform, request, query)
   - Perlocutionary act: The effect on the hearer

2. FIPA-ACL PERFORMATIVES:
   - INFORM (tell): Share a belief with another agent
   - ACHIEVE (request): Ask another agent to achieve a goal
   - QUERY_REF (ask): Request information from another agent

3. COMMUNICATION AS ACTION:
   When an agent sends a message, it changes the mental state of the receiver.
   This is a key concept in MAS - communication is not just data transfer,
   it's an action that modifies the world (specifically, agent beliefs).

4. BELIEF SOURCE TRACKING:
   Messages carry sender information, allowing the receiver to track
   the provenance of beliefs (e.g., belief[source(radiologist)]).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from queue import Queue
from datetime import datetime
import threading
import uuid


class Performative(Enum):
    """
    FIPA-ACL / KQML Performatives for Agent Communication
    
    EDUCATIONAL NOTE:
    Performatives represent the "illocutionary force" of a message -
    the intent behind the communication. Different performatives
    require different responses from the receiving agent.
    
    Mapping to KQML/FIPA-ACL:
    - INFORM → tell: "I believe X is true"
    - ACHIEVE → request/achieve: "Please make X true" 
    - QUERY_REF → askOne/query-ref: "What is the value of X?"
    - CFP → call-for-proposal: "Who can do task X?"
    - PROPOSE → propose: "I can do X with conditions Y"
    - FAILURE → failure: "I cannot do X because Y"
    """
    
    # Assertive performatives - sharing information
    INFORM = "inform"      # Tell another agent about a belief
    
    # Directive performatives - requesting action
    ACHIEVE = "achieve"    # Request goal achievement
    QUERY_REF = "query-ref"  # Request information
    
    # Contract Net performatives (for coordination)
    CFP = "cfp"            # Call for proposals
    PROPOSE = "propose"    # Submit a proposal
    ACCEPT = "accept"      # Accept a proposal
    REJECT = "reject"      # Reject a proposal
    
    # Status performatives
    FAILURE = "failure"    # Report failure to achieve goal
    CONFIRM = "confirm"    # Confirm receipt/completion


@dataclass
class Message:
    """
    Agent Communication Message
    
    EDUCATIONAL NOTE:
    This structure follows the FIPA-ACL message format, which includes:
    - Sender/Receiver: The communicating agents
    - Performative: The type of speech act
    - Content: The propositional content
    - Reply-with/In-reply-to: Conversation threading
    
    Example Jason/AgentSpeak equivalent:
        .send(oncologist, tell, nodule_features(n001, 0.85, 15))
        
    Becomes:
        Message(
            sender="radiologist",
            receiver="oncologist", 
            performative=Performative.INFORM,
            content={"predicate": "nodule_features", "args": ["n001", 0.85, 15]}
        )
    """
    
    sender: str
    receiver: str
    performative: Performative
    content: Any
    
    # Optional fields for conversation management
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reply_to: Optional[str] = None  # ID of message being replied to
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Metadata for belief tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Human-readable message representation"""
        return (
            f"[{self.timestamp.strftime('%H:%M:%S')}] "
            f"{self.sender} → {self.receiver} "
            f"({self.performative.value}): {self.content}"
        )
    
    def to_belief_source(self) -> str:
        """
        Generate source annotation for belief tracking.
        
        EDUCATIONAL NOTE:
        In Jason/AgentSpeak, beliefs acquired via communication
        are annotated with their source:
            +nodule_features(n001, 0.85)[source(radiologist)]
        
        This allows agents to reason about belief provenance.
        """
        return f"source({self.sender})"


class MessageBroker:
    """
    In-Memory Message Broker for Agent Communication
    
    EDUCATIONAL PURPOSE:
    This implements a simple message-passing infrastructure for MAS.
    In production systems, this might be replaced by:
    - JADE's messaging infrastructure
    - XMPP servers (used by SPADE)
    - Message queues (RabbitMQ, etc.)
    
    Key Features:
    1. Per-agent message queues (mailboxes)
    2. Thread-safe message delivery
    3. Message trace logging for debugging/analysis
    
    COMMUNICATION AS ACTION:
    When send() is called, it's an ACTION that changes the world state
    by adding a message to another agent's mailbox. The receiving agent's
    beliefs will change when it processes this message.
    """
    
    def __init__(self):
        """Initialize the message broker with empty mailboxes."""
        self._mailboxes: Dict[str, Queue] = {}
        self._message_trace: List[Message] = []
        self._lock = threading.Lock()
        
    def register_agent(self, agent_name: str) -> None:
        """
        Register an agent with the broker, creating its mailbox.
        
        Args:
            agent_name: Unique identifier for the agent
        """
        with self._lock:
            if agent_name not in self._mailboxes:
                self._mailboxes[agent_name] = Queue()
                
    def send(self, message: Message) -> bool:
        """
        Send a message to another agent.
        
        EDUCATIONAL NOTE - COMMUNICATION AS ACTION:
        This method represents a speech act. When called:
        1. The sender performs an illocutionary act (e.g., informing)
        2. The message is placed in the receiver's mailbox
        3. When processed, it will change the receiver's mental state
        
        Args:
            message: The message to send
            
        Returns:
            True if message was delivered, False if receiver unknown
        """
        with self._lock:
            # Log the message for analysis
            self._message_trace.append(message)
            
            # Check if receiver exists
            if message.receiver not in self._mailboxes:
                # Auto-register unknown receivers (for flexibility)
                self._mailboxes[message.receiver] = Queue()
            
            # Deliver message to receiver's mailbox
            self._mailboxes[message.receiver].put(message)
            
            return True
    
    def receive(self, agent_name: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message from the agent's mailbox.
        
        EDUCATIONAL NOTE:
        This is where "communication as action" completes its effect.
        The receiving agent will typically:
        1. Parse the message content
        2. Update beliefs (if INFORM)
        3. Add goals (if ACHIEVE)
        4. Query and respond (if QUERY_REF)
        
        Args:
            agent_name: The receiving agent's name
            timeout: How long to wait for a message (None = non-blocking)
            
        Returns:
            The received message, or None if mailbox is empty
        """
        if agent_name not in self._mailboxes:
            return None
            
        try:
            if timeout is None:
                # Non-blocking receive
                if self._mailboxes[agent_name].empty():
                    return None
                return self._mailboxes[agent_name].get_nowait()
            else:
                # Blocking receive with timeout
                return self._mailboxes[agent_name].get(timeout=timeout)
        except:
            return None
    
    def has_messages(self, agent_name: str) -> bool:
        """Check if an agent has pending messages."""
        if agent_name not in self._mailboxes:
            return False
        return not self._mailboxes[agent_name].empty()
    
    def get_message_trace(self) -> List[Message]:
        """
        Get the complete message trace for analysis.
        
        EDUCATIONAL PURPOSE:
        The message trace allows us to:
        1. Visualize agent interactions
        2. Debug communication patterns
        3. Analyze system behavior
        4. Demonstrate multi-agent coordination
        """
        return self._message_trace.copy()
    
    def get_agent_messages(self, agent_name: str) -> List[Message]:
        """Get all messages sent to a specific agent."""
        return [m for m in self._message_trace if m.receiver == agent_name]
    
    def print_trace(self) -> None:
        """Print a formatted message trace for debugging."""
        print("\n" + "="*60)
        print("MESSAGE TRACE - Agent Communication Log")
        print("="*60)
        for msg in self._message_trace:
            print(msg)
        print("="*60 + "\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about agent communication.
        
        Returns:
            Dict with message counts by agent and performative
        """
        stats = {
            "total_messages": len(self._message_trace),
            "by_sender": {},
            "by_receiver": {},
            "by_performative": {}
        }
        
        for msg in self._message_trace:
            # Count by sender
            stats["by_sender"][msg.sender] = stats["by_sender"].get(msg.sender, 0) + 1
            # Count by receiver
            stats["by_receiver"][msg.receiver] = stats["by_receiver"].get(msg.receiver, 0) + 1
            # Count by performative
            perf = msg.performative.value
            stats["by_performative"][perf] = stats["by_performative"].get(perf, 0) + 1
            
        return stats


# Convenience function for creating messages
def create_message(
    sender: str,
    receiver: str,
    performative: Performative,
    content: Any,
    reply_to: Optional[str] = None
) -> Message:
    """
    Factory function for creating messages.
    
    Example:
        msg = create_message(
            sender="radiologist",
            receiver="oncologist",
            performative=Performative.INFORM,
            content={"type": "nodule_features", "size": 15.0, "probability": 0.85}
        )
    """
    return Message(
        sender=sender,
        receiver=receiver,
        performative=performative,
        content=content,
        reply_to=reply_to
    )


# Example usage and demonstration
if __name__ == "__main__":
    print("=== Message Queue Demo ===\n")
    
    # Create broker
    broker = MessageBroker()
    
    # Register agents
    broker.register_agent("radiologist")
    broker.register_agent("pathologist")
    broker.register_agent("oncologist")
    
    # Simulate main sending requests to specialists
    broker.send(Message(
        sender="main",
        receiver="radiologist",
        performative=Performative.ACHIEVE,
        content={"goal": "analyze_image", "nodule_id": "n001", "path": "data/nodule.png"}
    ))
    
    broker.send(Message(
        sender="main",
        receiver="pathologist",
        performative=Performative.ACHIEVE,
        content={"goal": "analyze_report", "nodule_id": "n001", "text": "CT shows 15mm nodule..."}
    ))
    
    # Simulate agents sending results to oncologist
    broker.send(Message(
        sender="radiologist",
        receiver="oncologist",
        performative=Performative.INFORM,
        content={"predicate": "nodule_features", "nodule_id": "n001", "probability": 0.85, "size": 15.2}
    ))
    
    broker.send(Message(
        sender="pathologist",
        receiver="oncologist",
        performative=Performative.INFORM,
        content={"predicate": "pathology_findings", "nodule_id": "n001", "margin": "spiculated", "texture": "solid"}
    ))
    
    # Print trace
    broker.print_trace()
    
    # Print statistics
    print("Statistics:", broker.get_statistics())
