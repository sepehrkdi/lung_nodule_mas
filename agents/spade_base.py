"""
SPADE-BDI Base Agent Module
===========================

EDUCATIONAL PURPOSE - PROPER BDI IMPLEMENTATION:

This module provides base classes for SPADE-BDI agents.
SPADE (Smart Python Agent Development Environment) is a
multi-agent platform based on XMPP for communication.

SPADE-BDI extends SPADE with:
- AgentSpeak(L) interpreter for plan execution
- Proper belief-desire-intention cycle
- Internal actions callable from AgentSpeak plans

KEY CONCEPTS:

1. XMPP Communication:
   - Agents have JIDs (Jabber IDs): agent@xmppserver.com
   - Messages are XML stanzas
   - Supports presence, pub/sub, and more

2. AgentSpeak Plans:
   - Loaded from .asl files
   - Triggered by beliefs or goals
   - Call Python functions as internal actions

3. Internal Actions:
   - Python functions decorated with @agent.action
   - Called from AgentSpeak as .action_name(args)
   - Bridge between symbolic plans and subsymbolic processing

SPADE-BDI ARCHITECTURE:
    
    ┌──────────────────────────────────────────┐
    │              SPADE-BDI Agent             │
    ├──────────────────────────────────────────┤
    │  AgentSpeak Interpreter                  │
    │  ┌────────────────────────────────────┐  │
    │  │ .asl Plans                         │  │
    │  │ +!goal : context <- actions.       │  │
    │  └────────────────────────────────────┘  │
    ├──────────────────────────────────────────┤
    │  Belief Base                             │
    │  ┌────────────────────────────────────┐  │
    │  │ belief(arg1, arg2)[source(agent)]  │  │
    │  └────────────────────────────────────┘  │
    ├──────────────────────────────────────────┤
    │  Internal Actions (Python)               │
    │  ┌────────────────────────────────────┐  │
    │  │ @agent.action                      │  │
    │  │ def classify_image(self, img): ... │  │
    │  └────────────────────────────────────┘  │
    ├──────────────────────────────────────────┤
    │  XMPP Communication Layer                │
    │  ┌────────────────────────────────────┐  │
    │  │ .send() / message handler          │  │
    │  └────────────────────────────────────┘  │
    └──────────────────────────────────────────┘
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class XMPPConfig:
    """
    XMPP server configuration for SPADE agents.
    
    EDUCATIONAL NOTE:
    SPADE uses XMPP (Extensible Messaging and Presence Protocol)
    for agent communication. You need an XMPP server running.
    
    Options:
    1. Prosody (recommended): lightweight, easy to configure
    2. ejabberd: enterprise-grade
    3. OpenFire: GUI-based configuration
    
    For development, you can also use public XMPP servers,
    but a local server is recommended for testing.
    """
    server: str = "localhost"
    port: int = 5222
    domain: str = "localhost"
    
    # Default credentials (for demo)
    # In production, use proper authentication
    password: str = "secret"
    
    def get_jid(self, agent_name: str) -> str:
        """Get full JID for an agent."""
        return f"{agent_name}@{self.domain}"


# Default configuration
DEFAULT_XMPP_CONFIG = XMPPConfig()


# =============================================================================
# Belief Representation (compatible with SPADE-BDI)
# =============================================================================

@dataclass
class Belief:
    """
    Represents a belief in the agent's belief base.
    
    SPADE-BDI uses Prolog-like terms for beliefs.
    Format: predicate(arg1, arg2, ...)[annotations]
    
    Example:
        nodule_size(nodule_001, 15.5)[source(pathologist)]
    """
    functor: str  # Predicate name
    args: tuple   # Arguments
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        base = f"{self.functor}({args_str})"
        if self.annotations:
            annot_str = ", ".join(f"{k}({v})" for k, v in self.annotations.items())
            return f"{base}[{annot_str}]"
        return base
    
    def to_asl(self) -> str:
        """Convert to AgentSpeak literal format."""
        return str(self)


# =============================================================================
# Abstract Base for Medical Agents
# =============================================================================

class MedicalAgentBase(ABC):
    """
    Abstract base class for medical domain agents.
    
    This provides a common interface that works with or without
    SPADE-BDI, allowing for graceful fallback when XMPP is unavailable.
    
    Subclasses implement domain-specific logic:
    - RadiologistAgent: Image classification
    - PathologistAgent: NLP extraction
    - OncologistAgent: Prolog reasoning
    """
    
    def __init__(self, name: str, asl_file: Optional[str] = None):
        self.name = name
        self.asl_file = asl_file
        self.beliefs: List[Belief] = []
        self.internal_actions: Dict[str, Callable] = {}
        self._running = False
        
        # Register internal actions from subclass
        self._register_actions()
    
    @abstractmethod
    def _register_actions(self) -> None:
        """Register internal actions that can be called from ASL plans."""
        pass
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an analysis request."""
        pass
    
    def add_belief(self, belief: Belief) -> None:
        """Add a belief to the belief base."""
        self.beliefs.append(belief)
        logger.debug(f"[{self.name}] Added belief: {belief}")
    
    def get_beliefs(self, functor: Optional[str] = None) -> List[Belief]:
        """Get beliefs, optionally filtered by functor."""
        if functor is None:
            return self.beliefs.copy()
        return [b for b in self.beliefs if b.functor == functor]
    
    def clear_beliefs(self) -> None:
        """Clear all beliefs."""
        self.beliefs.clear()


# =============================================================================
# SPADE-BDI Agent Wrapper
# =============================================================================

def create_spade_bdi_agent(
    agent_class: type,
    name: str,
    xmpp_config: XMPPConfig,
    asl_file: str,
    **kwargs
):
    """
    Factory function to create a SPADE-BDI agent.
    
    This wraps our domain agent in SPADE-BDI infrastructure.
    
    Args:
        agent_class: The domain agent class (RadiologistAgent, etc.)
        name: Agent name (used for JID)
        xmpp_config: XMPP server configuration
        asl_file: Path to AgentSpeak plan file
        **kwargs: Additional arguments for domain agent
        
    Returns:
        Configured SPADE-BDI agent instance
    """
    try:
        from spade_bdi.bdi import BDIAgent
        
        class SPADEMedicalAgent(BDIAgent):
            """
            SPADE-BDI wrapper for medical domain agents.
            
            This class bridges SPADE-BDI's AgentSpeak interpreter
            with our domain-specific Python implementations.
            """
            
            def __init__(self, jid: str, password: str, asl_path: str, **agent_kwargs):
                super().__init__(jid, password, asl_path)
                
                # Create domain agent instance
                self.domain_agent = agent_class(name=name, **agent_kwargs)
                
                # Register internal actions
                self._register_internal_actions()
            
            def _register_internal_actions(self):
                """
                Register Python functions as SPADE-BDI internal actions.
                
                EDUCATIONAL NOTE:
                Internal actions are the bridge between AgentSpeak plans
                and Python code. They allow plans to call arbitrary Python
                functions for tasks like ML inference or API calls.
                
                In AgentSpeak, called as: .action_name(arg1, arg2)
                """
                for action_name, action_func in self.domain_agent.internal_actions.items():
                    # SPADE-BDI uses add_custom_action
                    self.add_custom_action(action_name, action_func)
            
            async def setup(self):
                """Called when agent starts."""
                await super().setup()
                logger.info(f"[{name}] SPADE-BDI agent started")
            
            def add_belief_from_domain(self, belief: Belief):
                """Add a belief from the domain agent."""
                # SPADE-BDI uses different belief format
                self.bdi.add_belief(belief.to_asl())
        
        # Create agent instance
        jid = xmpp_config.get_jid(name)
        agent = SPADEMedicalAgent(
            jid=jid,
            password=xmpp_config.password,
            asl_path=asl_file,
            **kwargs
        )
        
        return agent
        
    except ImportError:
        logger.warning("SPADE-BDI not available, using fallback mode")
        return agent_class(name=name, asl_file=asl_file, **kwargs)


# =============================================================================
# Standalone Mode (No XMPP)
# =============================================================================

class StandaloneAgentRunner:
    """
    Runs agents without XMPP server using in-memory communication.
    
    EDUCATIONAL NOTE:
    This provides a fallback mode for development and testing
    when an XMPP server is not available. It simulates SPADE-BDI
    behavior using async Python.
    
    The key difference from full SPADE-BDI:
    - No XMPP: Messages are passed directly via Python queues
    - Simplified BDI cycle: Plans are executed as Python coroutines
    - Same interface: Code works with or without XMPP
    """
    
    def __init__(self):
        self.agents: Dict[str, MedicalAgentBase] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self._running = False
    
    def register_agent(self, agent: MedicalAgentBase) -> None:
        """Register an agent with the runner."""
        self.agents[agent.name] = agent
        self.message_queues[agent.name] = asyncio.Queue()
        logger.info(f"Registered agent: {agent.name}")
    
    async def send_message(
        self,
        sender: str,
        receiver: str,
        performative: str,
        content: Dict[str, Any]
    ) -> None:
        """
        Send a message between agents.
        
        EDUCATIONAL NOTE:
        This simulates FIPA-ACL message passing.
        In full SPADE, this would be an XMPP stanza.
        """
        if receiver not in self.message_queues:
            logger.error(f"Unknown receiver: {receiver}")
            return
        
        message = {
            "sender": sender,
            "receiver": receiver,
            "performative": performative,
            "content": content
        }
        
        await self.message_queues[receiver].put(message)
        logger.debug(f"Message: {sender} -> {receiver} ({performative})")
    
    async def receive_message(self, agent_name: str, timeout: float = 1.0) -> Optional[Dict]:
        """Receive a message for an agent."""
        try:
            return await asyncio.wait_for(
                self.message_queues[agent_name].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    async def run_agent_cycle(self, agent: MedicalAgentBase) -> None:
        """
        Run the BDI reasoning cycle for an agent.
        
        EDUCATIONAL NOTE:
        This is a simplified BDI cycle:
        1. Perceive: Check for messages
        2. Update beliefs: Process message content
        3. Deliberate: Select goal based on message performative
        4. Execute: Run appropriate handler
        """
        while self._running:
            message = await self.receive_message(agent.name)
            
            if message:
                performative = message.get("performative", "")
                content = message.get("content", {})
                
                if performative == "achieve":
                    # Goal request
                    result = await agent.process_request(content)
                    
                    # Send response
                    await self.send_message(
                        sender=agent.name,
                        receiver=message["sender"],
                        performative="inform",
                        content={"result": result}
                    )
                
                elif performative == "inform":
                    # Belief update
                    if "belief" in content:
                        belief_data = content["belief"]
                        belief = Belief(
                            functor=belief_data.get("functor", "info"),
                            args=tuple(belief_data.get("args", [])),
                            annotations={"source": message["sender"]}
                        )
                        agent.add_belief(belief)
            
            await asyncio.sleep(0.01)  # Prevent busy loop
    
    async def start(self) -> None:
        """Start all registered agents."""
        self._running = True
        tasks = [
            self.run_agent_cycle(agent)
            for agent in self.agents.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop(self) -> None:
        """Stop all agents."""
        self._running = False


# =============================================================================
# Utility Functions
# =============================================================================

def load_asl_file(path: str) -> str:
    """Load AgentSpeak file content."""
    with open(path, 'r') as f:
        return f.read()


def get_asl_path(agent_name: str) -> str:
    """Get the default ASL file path for an agent."""
    base_dir = Path(__file__).parent.parent / "asl"
    return str(base_dir / f"{agent_name}.asl")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'XMPPConfig',
    'DEFAULT_XMPP_CONFIG',
    'Belief',
    'MedicalAgentBase',
    'create_spade_bdi_agent',
    'StandaloneAgentRunner',
    'load_asl_file',
    'get_asl_path',
]
