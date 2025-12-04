"""
Base BDI Agent Implementation
=============================

EDUCATIONAL PURPOSE - BDI ARCHITECTURE (Bratman's Theory):

The BDI (Belief-Desire-Intention) model is based on Michael Bratman's
theory of practical reasoning. It provides a cognitive architecture
for intelligent agents with three key mental attitudes:

1. BELIEFS: Information the agent has about the world
   - Can be incomplete or incorrect
   - Updated through perception and communication
   - Example: believes(nodule_size(n001, 15))

2. DESIRES (Goals): Objectives the agent would like to achieve
   - Not all desires can be pursued simultaneously
   - Example: desires(diagnose(patient001))

3. INTENTIONS: Desires the agent has COMMITTED to pursuing
   - Filter for selecting among competing options
   - Persist over time (but can be dropped)
   - Example: intends(analyze_image(n001))

AGENTSPEAK/JASON CONCEPTS IMPLEMENTED:
- Triggering Events: +belief (addition), -belief (removal), +!goal
- Plans: event : context <- body
- Actions: Internal (.send) and external (custom)
- Belief Annotations: source(agent_name) for provenance tracking

PRACTICAL REASONING CYCLE:
1. Perceive environment → Update beliefs
2. Check triggered events → Activate applicable plans
3. Select intention → Commit to plan execution
4. Execute plan step
5. Repeat
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from datetime import datetime
import re
import os

from communication.message_queue import Message, MessageBroker, Performative


@dataclass
class Belief:
    """
    Represents an agent belief with source tracking.
    
    EDUCATIONAL NOTE:
    In BDI systems, beliefs are not just facts - they carry metadata:
    - Source: Where did this belief come from? (self, percept, communication)
    - Timestamp: When was this belief acquired?
    - Confidence: How certain is the agent? (optional)
    
    In Jason/AgentSpeak notation:
        nodule_size(n001, 15)[source(radiologist), timestamp(123456)]
    
    Attributes:
        key: The belief predicate (e.g., "nodule_size")
        value: The belief content (e.g., {"id": "n001", "size": 15})
        source: Origin of the belief (e.g., "radiologist", "percept", "self")
        timestamp: When the belief was added
    """
    key: str
    value: Any
    source: str = "self"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"{self.key}({self.value})[source({self.source})]"
    
    def matches(self, pattern: str) -> bool:
        """Check if belief key matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        # Simple pattern matching with _
        regex = pattern.replace("_", ".*")
        return bool(re.match(f"^{regex}$", self.key))


class BeliefBase:
    """
    Agent's Belief Base - A database of current beliefs.
    
    EDUCATIONAL PURPOSE:
    The belief base is the agent's "view of the world". Key features:
    
    1. CLOSED WORLD ASSUMPTION (CWA):
       If a belief is not in the base, it's assumed false.
       This is the default in Prolog and AgentSpeak.
       
    2. SOURCE TRACKING:
       Each belief knows where it came from, enabling:
       - Trust reasoning (do I trust this source?)
       - Belief revision (update based on more reliable sources)
       
    3. EVENT GENERATION:
       Adding/removing beliefs generates events that can trigger plans:
       - +belief: Belief addition event
       - -belief: Belief deletion event
    """
    
    def __init__(self):
        """Initialize an empty belief base."""
        self._beliefs: Dict[str, Belief] = {}
        self._event_queue: List[Tuple[str, Belief]] = []  # (+/-belief_key, Belief)
        
    def add(self, key: str, value: Any, source: str = "self") -> None:
        """
        Add a belief to the base.
        
        EDUCATIONAL NOTE:
        Adding a belief generates a +key event that can trigger plans.
        In Jason: +nodule_size(N, S)[source(radiologist)]
        
        Args:
            key: Belief predicate name
            value: Belief content
            source: Where this belief came from
        """
        belief = Belief(key=key, value=value, source=source)
        self._beliefs[key] = belief
        self._event_queue.append((f"+{key}", belief))
        
    def remove(self, key: str) -> Optional[Belief]:
        """
        Remove a belief from the base.
        
        EDUCATIONAL NOTE:
        Removing a belief generates a -key event.
        In Jason: -nodule_size(N, S)
        
        Args:
            key: Belief predicate to remove
            
        Returns:
            The removed belief, or None if not found
        """
        if key in self._beliefs:
            belief = self._beliefs.pop(key)
            self._event_queue.append((f"-{key}", belief))
            return belief
        return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get a belief value by key."""
        if key in self._beliefs:
            return self._beliefs[key].value
        return None
    
    def get_belief(self, key: str) -> Optional[Belief]:
        """Get the full Belief object by key."""
        return self._beliefs.get(key)
    
    def has(self, key: str) -> bool:
        """Check if a belief exists (test goal in AgentSpeak: ?belief)."""
        return key in self._beliefs
    
    def query(self, pattern: str) -> List[Belief]:
        """
        Query beliefs matching a pattern.
        
        EDUCATIONAL NOTE:
        This is like Prolog's pattern matching / unification.
        
        Args:
            pattern: Pattern to match (supports * for any)
            
        Returns:
            List of matching beliefs
        """
        return [b for b in self._beliefs.values() if b.matches(pattern)]
    
    def get_by_source(self, source: str) -> List[Belief]:
        """Get all beliefs from a specific source."""
        return [b for b in self._beliefs.values() if b.source == source]
    
    def pop_events(self) -> List[Tuple[str, Belief]]:
        """
        Pop all pending events from the event queue.
        
        EDUCATIONAL NOTE:
        Events are processed in the agent's reasoning cycle
        to trigger applicable plans.
        """
        events = self._event_queue.copy()
        self._event_queue.clear()
        return events
    
    def all(self) -> List[Belief]:
        """Get all beliefs."""
        return list(self._beliefs.values())
    
    def __len__(self) -> int:
        return len(self._beliefs)
    
    def __str__(self) -> str:
        return "\n".join(str(b) for b in self._beliefs.values())


class EventType(Enum):
    """Types of events that can trigger plans."""
    BELIEF_ADDITION = "+"      # +belief
    BELIEF_DELETION = "-"      # -belief
    GOAL_ADDITION = "+!"       # +!goal (achievement goal)
    GOAL_DELETION = "-!"       # -!goal
    TEST_GOAL = "?"            # ?belief (test if true)


@dataclass
class Plan:
    """
    BDI Plan - A recipe for achieving goals or responding to events.
    
    EDUCATIONAL PURPOSE - PLAN STRUCTURE:
    In AgentSpeak/Jason, plans have the form:
        triggering_event : context <- body.
    
    - Triggering Event: What activates this plan (+belief, +!goal, etc.)
    - Context: Conditions that must be true (beliefs to check)
    - Body: Sequence of actions/subgoals to execute
    
    Example in Jason:
        +!classify(Nodule) : has_image(Nodule) & ready(classifier)
            <- load_image(Nodule);
               run_classifier(Result);
               .send(oncologist, tell, result(Nodule, Result)).
    
    Attributes:
        name: Plan identifier
        trigger: Event that activates this plan (e.g., "+!analyze_image")
        context: Callable that checks if plan is applicable
        body: Callable that executes the plan
    """
    name: str
    trigger: str  # e.g., "+!analyze_image", "+nodule_features"
    context: Callable[['BDIAgent'], bool]  # Context condition
    body: Callable[['BDIAgent', Any], None]  # Plan body
    
    def is_applicable(self, event: str, agent: 'BDIAgent') -> bool:
        """
        Check if this plan is applicable for the given event.
        
        EDUCATIONAL NOTE:
        A plan is applicable if:
        1. Its trigger matches the event
        2. Its context is satisfied (beliefs check)
        """
        # Check trigger match (simple prefix matching)
        if not event.startswith(self.trigger.split("(")[0]):
            return False
        # Check context
        return self.context(agent)


@dataclass 
class Goal:
    """
    Agent Goal (Desire/Intention)
    
    EDUCATIONAL NOTE:
    In BDI, goals can be:
    - Achievement goals (+!g): Achieve state g, then done
    - Maintenance goals: Keep condition true
    - Test goals (?g): Check if g is believed
    
    Attributes:
        name: Goal predicate
        args: Goal arguments
        priority: For goal selection
        status: pending, active, achieved, failed
    """
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    status: str = "pending"
    
    def __str__(self) -> str:
        args_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"!{self.name}({args_str})"


class BDIAgent(ABC):
    """
    Abstract Base BDI Agent
    
    EDUCATIONAL PURPOSE - AGENT ARCHITECTURE:
    
    This class implements a BDI agent with the following components:
    
    1. MENTAL STATE:
       - Belief Base: What the agent believes
       - Goal Set: What the agent wants to achieve
       - Plan Library: How to achieve goals
       
    2. REASONING CYCLE:
       The agent repeatedly executes:
       a) Perceive: Check for new messages
       b) Update Beliefs: Process incoming information
       c) Generate Options: Find applicable plans
       d) Deliberate: Select what to do
       e) Execute: Perform the selected action
       
    3. COMMUNICATION:
       - Uses speech acts (performatives)
       - Messages become beliefs with source tracking
       
    IMPLEMENTING A CONCRETE AGENT:
    Subclass this and implement:
    - setup_plans(): Define the agent's plans
    - Additional custom actions as needed
    """
    
    def __init__(self, name: str, broker: MessageBroker):
        """
        Initialize the BDI agent.
        
        Args:
            name: Unique agent identifier
            broker: Message broker for communication
        """
        self.name = name
        self.broker = broker
        
        # Mental state
        self.beliefs = BeliefBase()
        self.goals: List[Goal] = []
        self.plans: List[Plan] = []
        
        # Intentions (currently executing plans)
        self._current_intention: Optional[Tuple[Plan, Any]] = None
        
        # Register with broker
        self.broker.register_agent(name)
        
        # Setup agent-specific plans
        self.setup_plans()
        
        # Add initial belief that agent is ready
        self.beliefs.add("ready", True, source="self")
        
    @abstractmethod
    def setup_plans(self) -> None:
        """
        Define the agent's plans.
        
        EDUCATIONAL NOTE:
        Each agent type defines its own plans:
        - Radiologist: Plans for image analysis
        - Pathologist: Plans for NLP extraction
        - Oncologist: Plans for Prolog reasoning
        
        Override this method to add plans to self.plans
        """
        pass
    
    def add_plan(self, plan: Plan) -> None:
        """Add a plan to the plan library."""
        self.plans.append(plan)
        
    def add_goal(self, name: str, **args) -> None:
        """
        Add an achievement goal.
        
        EDUCATIONAL NOTE:
        In Jason: !goal_name(args)
        This adds a goal that will trigger +!goal_name plans.
        """
        goal = Goal(name=name, args=args)
        self.goals.append(goal)
        # Generate goal addition event
        self.beliefs._event_queue.append((f"+!{name}", goal))
        
    def send(self, receiver: str, performative: Performative, content: Any) -> None:
        """
        Send a message to another agent.
        
        EDUCATIONAL NOTE:
        In Jason: .send(receiver, performative, content)
        
        Examples:
            .send(oncologist, tell, nodule_features(n001, 0.85))
            .send(pathologist, achieve, analyze_report(r001))
        """
        message = Message(
            sender=self.name,
            receiver=receiver,
            performative=performative,
            content=content
        )
        self.broker.send(message)
        
    def _process_messages(self) -> None:
        """
        Process incoming messages and update beliefs.
        
        EDUCATIONAL NOTE - COMMUNICATION AS ACTION:
        When a message is received, it changes the agent's mental state:
        - INFORM → Add belief with source annotation
        - ACHIEVE → Add goal to goal set
        - QUERY_REF → Generate query response
        """
        while self.broker.has_messages(self.name):
            message = self.broker.receive(self.name)
            if message is None:
                break
                
            if message.performative == Performative.INFORM:
                # Add content as belief with source
                if isinstance(message.content, dict):
                    predicate = message.content.get("predicate", "info")
                    self.beliefs.add(
                        key=predicate,
                        value=message.content,
                        source=message.sender
                    )
                else:
                    self.beliefs.add(
                        key="message",
                        value=message.content,
                        source=message.sender
                    )
                    
            elif message.performative == Performative.ACHIEVE:
                # Add goal to achieve
                if isinstance(message.content, dict):
                    goal_name = message.content.get("goal", "unknown")
                    self.add_goal(goal_name, **message.content)
                    
            elif message.performative == Performative.QUERY_REF:
                # Handle query (subclasses can override)
                self._handle_query(message)
    
    def _handle_query(self, message: Message) -> None:
        """Handle a query message. Override in subclasses."""
        pass
    
    def _select_plan(self, event: str, event_data: Any) -> Optional[Plan]:
        """
        Select an applicable plan for the given event.
        
        EDUCATIONAL NOTE:
        Plan selection involves:
        1. Finding all plans with matching triggers
        2. Filtering by context (are conditions met?)
        3. Selecting one (first match, priority, etc.)
        
        This is a simplified version - Jason has more sophisticated selection.
        """
        for plan in self.plans:
            if plan.is_applicable(event, self):
                return plan
        return None
    
    def run_cycle(self) -> bool:
        """
        Execute one iteration of the BDI reasoning cycle.
        
        EDUCATIONAL PURPOSE - PRACTICAL REASONING:
        This is the core of the BDI architecture:
        
        1. BELIEF REVISION:
           - Process incoming messages
           - Update beliefs based on perception
           
        2. OPTION GENERATION:
           - Check for events (belief/goal changes)
           - Find applicable plans
           
        3. DELIBERATION:
           - Select which intention to pursue
           
        4. EXECUTION:
           - Execute one step of the selected plan
           
        Returns:
            True if any action was taken, False if idle
        """
        action_taken = False
        
        # 1. Process incoming messages (belief revision)
        self._process_messages()
        
        # 2. Get pending events
        events = self.beliefs.pop_events()
        
        # 3. For each event, find and execute applicable plan
        for event_type, event_data in events:
            plan = self._select_plan(event_type, event_data)
            if plan:
                # 4. Execute plan body
                try:
                    plan.body(self, event_data)
                    action_taken = True
                except Exception as e:
                    print(f"[{self.name}] Plan {plan.name} failed: {e}")
                    
        return action_taken
    
    def run(self, max_cycles: int = 100) -> None:
        """
        Run the agent for multiple cycles.
        
        Args:
            max_cycles: Maximum number of reasoning cycles
        """
        for _ in range(max_cycles):
            if not self.run_cycle():
                # No action taken, could break or continue
                pass


class ASLParser:
    """
    Simple AgentSpeak Language Parser
    
    EDUCATIONAL PURPOSE:
    This parser handles a simplified subset of AgentSpeak syntax:
    
    Beliefs:    belief_name(arg1, arg2).
    Goals:      !goal_name(arg1).
    Plans:      +!goal(X) : context(X) <- action1; action2.
    
    Full Jason/AgentSpeak is more complex, but this captures
    the essential concepts for educational purposes.
    """
    
    @staticmethod
    def parse_file(filepath: str) -> Dict[str, Any]:
        """
        Parse an ASL file and return beliefs, goals, and plans.
        
        Args:
            filepath: Path to .asl file
            
        Returns:
            Dict with 'beliefs', 'goals', 'plans' keys
        """
        result = {
            "beliefs": [],
            "goals": [],
            "plans": [],
            "raw_plans": []
        }
        
        if not os.path.exists(filepath):
            return result
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Parse beliefs (lines ending with . that aren't plans)
        belief_pattern = r'^([a-z]\w*)\(([^)]*)\)\s*\.'
        for match in re.finditer(belief_pattern, content, re.MULTILINE):
            result["beliefs"].append({
                "predicate": match.group(1),
                "args": match.group(2)
            })
            
        # Parse initial goals (lines starting with !)
        goal_pattern = r'^!\s*(\w+)\s*(?:\(([^)]*)\))?\s*\.'
        for match in re.finditer(goal_pattern, content, re.MULTILINE):
            result["goals"].append({
                "name": match.group(1),
                "args": match.group(2) or ""
            })
            
        # Parse plans (trigger : context <- body)
        plan_pattern = r'([+\-][!?]?\w+(?:\([^)]*\))?)\s*:\s*([^<]*?)\s*<-\s*([^.]+)\.'
        for match in re.finditer(plan_pattern, content, re.MULTILINE | re.DOTALL):
            result["raw_plans"].append({
                "trigger": match.group(1).strip(),
                "context": match.group(2).strip(),
                "body": match.group(3).strip()
            })
            
        return result


# Example concrete agent for testing
class TestAgent(BDIAgent):
    """Simple test agent for demonstration."""
    
    def setup_plans(self):
        # Plan: When we get a greeting, respond
        self.add_plan(Plan(
            name="respond_to_greeting",
            trigger="+greeting",
            context=lambda agent: True,
            body=lambda agent, data: print(f"[{agent.name}] Received greeting: {data}")
        ))


if __name__ == "__main__":
    print("=== BDI Agent Demo ===\n")
    
    # Create broker and agents
    broker = MessageBroker()
    agent1 = TestAgent("agent1", broker)
    agent2 = TestAgent("agent2", broker)
    
    # Agent1 sends greeting to Agent2
    agent1.send("agent2", Performative.INFORM, {"predicate": "greeting", "message": "Hello!"})
    
    # Run cycles
    agent2.run_cycle()
    
    # Check beliefs
    print(f"\nAgent2 beliefs:\n{agent2.beliefs}")
