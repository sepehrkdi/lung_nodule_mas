"""
Radiologist Agent
=================

EDUCATIONAL PURPOSE - BDI AGENT FOR COMPUTER VISION:

This agent implements the Radiologist role in the Multi-Agent System.
It uses Computer Vision (DenseNet121) to analyze nodule images.

BDI COMPONENTS (Bratman's Model):
- BELIEFS: Current knowledge about nodule images and classifications
- DESIRES: Goals to analyze images and provide assessments
- INTENTIONS: Active plans being executed

AGENT COMMUNICATION:
- Receives ACHIEVE requests from main orchestrator
- Sends INFORM messages with findings to Oncologist
- Uses belief annotations with [source(radiologist)]

CV ROLE:
The Radiologist is the "eyes" of the system, extracting visual
features from CT images that complement the textual analysis
from the Pathologist.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import json

# Import base agent components
from agents.base_agent import BDIAgent, Belief, BeliefBase, Plan, ASLParser
from communication.message_queue import Message, Performative, MessageBroker

# Import CV model
from models.classifier import NoduleClassifier


class RadiologistAgent(BDIAgent):
    """
    BDI Agent for image-based nodule analysis.
    
    EDUCATIONAL PURPOSE:
    
    This agent demonstrates:
    1. BDI Architecture - Beliefs, Desires, Intentions cycle
    2. Computer Vision integration with DenseNet121
    3. Agent communication via message passing
    4. Plan execution from AgentSpeak-like specifications
    
    The Radiologist analyzes CT images and extracts:
    - Malignancy probability
    - Visual features (texture, shape)
    - Size estimation
    
    Architecture:
        [Image] → [DenseNet121] → [Features] → [Beliefs] → [Message to Oncologist]
    """
    
    def __init__(self, message_broker: MessageBroker, asl_file: Optional[str] = None):
        """
        Initialize the Radiologist agent.
        
        Args:
            message_broker: Shared message broker for agent communication
            asl_file: Path to AgentSpeak plan file (optional)
        """
        super().__init__("radiologist", message_broker)
        
        # Load plans from ASL file
        if asl_file:
            self._load_asl_plans(asl_file)
        
        # Initialize CV classifier
        self.classifier: Optional[NoduleClassifier] = None
        self._init_classifier()
        
        # Add initial beliefs
        self._init_beliefs()
        
        print(f"[{self.name}] Agent initialized")
    
    def _init_classifier(self) -> None:
        """Initialize the DenseNet classifier."""
        try:
            self.classifier = NoduleClassifier()
            self.add_belief(Belief("model", "loaded", "DenseNet121"))
            print(f"[{self.name}] DenseNet121 classifier loaded")
        except Exception as e:
            self.add_belief(Belief("model", "failed", str(e)))
            print(f"[{self.name}] Classifier loading failed: {e}")
    
    def _init_beliefs(self) -> None:
        """Add initial beliefs about agent capabilities."""
        # Threshold for suspicious classification
        self.add_belief(Belief("threshold", "suspicious", 0.5))
        
        # Input specifications
        self.add_belief(Belief("input", "size", 224))
        self.add_belief(Belief("input", "channels", 3))
        
        # Model architecture
        self.add_belief(Belief("architecture", "name", "DenseNet121"))
        self.add_belief(Belief("architecture", "pretrained", True))
    
    def _load_asl_plans(self, asl_file: str) -> None:
        """
        Load plans from AgentSpeak file.
        
        EDUCATIONAL NOTE:
        In a full Jason implementation, plans would be parsed
        and executed by the interpreter. Here we use them
        as documentation and map to Python methods.
        """
        try:
            parser = ASLParser()
            raw_plans = parser.parse_file(asl_file)
            
            for plan_data in raw_plans:
                plan = Plan(
                    trigger=plan_data["trigger"],
                    context=plan_data.get("context", "true"),
                    body=plan_data.get("body", [])
                )
                self.add_plan(plan)
            
            print(f"[{self.name}] Loaded {len(raw_plans)} plans from {asl_file}")
            
        except Exception as e:
            print(f"[{self.name}] Could not load ASL file: {e}")
    
    def reason(self) -> None:
        """
        BDI Reasoning Cycle.
        
        EDUCATIONAL PURPOSE - BDI CYCLE:
        
        1. PERCEIVE: Check message queue for new messages
        2. UPDATE BELIEFS: Incorporate new information
        3. DELIBERATE: Select which goals to pursue
        4. PLAN: Find applicable plans for goals
        5. EXECUTE: Run plan actions
        
        This cycle runs continuously while the agent is active.
        """
        # Check for incoming messages
        message = self.broker.receive(self.name)
        
        if message:
            self._handle_message(message)
        
        # Process any pending intentions
        self._process_intentions()
    
    def _handle_message(self, message: Message) -> None:
        """
        Handle incoming message.
        
        EDUCATIONAL NOTE - SPEECH ACTS:
        Different performatives require different responses:
        - ACHIEVE: Request to achieve a goal (analyze image)
        - QUERY_REF: Request for information
        - INFORM: Update beliefs with new information
        """
        print(f"[{self.name}] Received {message.performative.value} from {message.sender}")
        
        if message.performative == Performative.ACHIEVE:
            self._handle_achieve(message)
        
        elif message.performative == Performative.QUERY_REF:
            self._handle_query(message)
        
        elif message.performative == Performative.INFORM:
            self._handle_inform(message)
    
    def _handle_achieve(self, message: Message) -> None:
        """
        Handle ACHIEVE request (goal to accomplish).
        
        Expected content format:
        {
            "goal": "analyze_image",
            "nodule_id": "nodule_001",
            "image_path": "path/to/image.png",  # Optional
            "image_data": np.ndarray,           # Optional
            "features": {...}                   # Optional pre-extracted features
        }
        """
        content = message.content
        goal = content.get("goal", "")
        
        if goal == "analyze_image":
            nodule_id = content.get("nodule_id", "unknown")
            
            # Try to get image data
            image_data = content.get("image_data")
            image_path = content.get("image_path")
            features = content.get("features", {})
            
            # Perform analysis
            result = self.analyze_nodule(
                nodule_id=nodule_id,
                image=image_data,
                image_path=image_path,
                features=features
            )
            
            # Send result back
            self._send_result(message.sender, nodule_id, result)
        
        else:
            print(f"[{self.name}] Unknown goal: {goal}")
    
    def _handle_query(self, message: Message) -> None:
        """Handle QUERY_REF (information request)."""
        content = message.content
        query_type = content.get("query", "")
        
        if query_type == "nodule_assessment":
            nodule_id = content.get("nodule_id")
            beliefs = self.get_beliefs(nodule_id)
            
            response = {
                "nodule_id": nodule_id,
                "beliefs": [b.to_dict() for b in beliefs]
            }
            
            self.broker.send(Message(
                sender=self.name,
                receiver=message.sender,
                performative=Performative.INFORM,
                content=response
            ))
    
    def _handle_inform(self, message: Message) -> None:
        """Handle INFORM (belief update)."""
        content = message.content
        
        # Update beliefs based on information
        if "beliefs" in content:
            for belief_data in content["beliefs"]:
                belief = Belief(
                    predicate=belief_data.get("predicate", "info"),
                    attribute=belief_data.get("attribute", ""),
                    value=belief_data.get("value", ""),
                    source=message.sender
                )
                self.add_belief(belief)
    
    def analyze_nodule(
        self,
        nodule_id: str,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a nodule image.
        
        EDUCATIONAL PURPOSE - CV PIPELINE:
        
        1. Load image (from array or file)
        2. Preprocess for DenseNet (resize, normalize)
        3. Extract features and classify
        4. Update beliefs with findings
        5. Return structured result
        
        Args:
            nodule_id: Unique identifier for the nodule
            image: Numpy array of image data (optional)
            image_path: Path to image file (optional)
            features: Pre-extracted features (optional)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"[{self.name}] Analyzing nodule: {nodule_id}")
        
        result = {
            "nodule_id": nodule_id,
            "agent": self.name,
            "status": "success",
            "findings": {}
        }
        
        # Try to get or create image
        if image is None and image_path:
            image = self._load_image(image_path)
        
        if image is None:
            # Generate synthetic image if features provided
            if features and "size_mm" in features:
                image = self._generate_synthetic_image(features)
            else:
                # Create default image
                image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        
        # Classify using DenseNet
        if self.classifier:
            try:
                classification = self.classifier.classify(image)
                
                # Add findings
                result["findings"]["malignancy_probability"] = classification["malignancy_probability"]
                result["findings"]["predicted_class"] = classification["predicted_class"]
                result["findings"]["estimated_size_mm"] = classification["estimated_size_mm"]
                result["findings"]["confidence"] = classification["confidence"]
                result["findings"]["image_stats"] = classification.get("image_stats", {})
                
                # Update beliefs
                self.add_belief(Belief(
                    "classification", "malignancy_prob",
                    classification["malignancy_probability"],
                    source=self.name
                ))
                self.add_belief(Belief(
                    "classification", "predicted_class",
                    classification["predicted_class"],
                    source=self.name
                ))
                self.add_belief(Belief(
                    "classification", "nodule_id",
                    nodule_id,
                    source=self.name
                ))
                
            except Exception as e:
                result["status"] = "partial"
                result["error"] = str(e)
                print(f"[{self.name}] Classification error: {e}")
        
        else:
            result["status"] = "fallback"
            result["findings"]["note"] = "Classifier not available, using features only"
            
            # Use provided features
            if features:
                result["findings"]["size_mm"] = features.get("size_mm", 10)
                result["findings"]["texture"] = features.get("texture", "unknown")
        
        # Add source annotation
        result["source"] = self.name
        
        return result
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file."""
        try:
            from PIL import Image
            img = Image.open(image_path)
            return np.array(img)
        except Exception as e:
            print(f"[{self.name}] Could not load image: {e}")
            return None
    
    def _generate_synthetic_image(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Generate synthetic nodule image from features.
        
        EDUCATIONAL NOTE:
        This is for demonstration when real images aren't available.
        Real systems would use actual CT data.
        """
        size = int(features.get("size_mm", 10) * 3)  # Scale factor
        size = max(32, min(size, 128))  # Clamp
        
        # Create base image
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Add circular nodule
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 3
        
        # Distance from center
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Create gradient nodule
        mask = dist <= radius
        image[mask] = 180
        
        # Add some texture based on features
        texture = features.get("texture", "solid")
        if texture == "ground_glass":
            noise = np.random.randint(0, 50, (size, size), dtype=np.uint8)
            image = np.clip(image.astype(int) + noise - 25, 0, 255).astype(np.uint8)
        elif texture == "part_solid":
            noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
            image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add spiculation if present
        if features.get("spiculation") == "present":
            # Add radiating lines
            for angle in range(0, 360, 45):
                rad = np.radians(angle)
                for r in range(radius, size//2):
                    xx = int(center + r * np.cos(rad))
                    yy = int(center + r * np.sin(rad))
                    if 0 <= xx < size and 0 <= yy < size:
                        image[yy, xx] = 150
        
        return image
    
    def _send_result(self, recipient: str, nodule_id: str, result: Dict[str, Any]) -> None:
        """Send analysis result to recipient."""
        message = Message(
            sender=self.name,
            receiver=recipient,
            performative=Performative.INFORM,
            content={
                "type": "analysis_result",
                "nodule_id": nodule_id,
                "result": result
            }
        )
        self.broker.send(message)
        print(f"[{self.name}] Sent results for {nodule_id} to {recipient}")
    
    def _process_intentions(self) -> None:
        """Process any active intentions (plans being executed)."""
        while self.intentions:
            plan = self.intentions.pop(0)
            self._execute_plan(plan)
    
    def _execute_plan(self, plan: Plan) -> None:
        """
        Execute a plan.
        
        EDUCATIONAL NOTE:
        In Jason, plan bodies are executed step by step.
        Here we map plan bodies to Python method calls.
        """
        for action in plan.body:
            if action.startswith("!"):
                # Achievement goal - add to intentions
                print(f"[{self.name}] Sub-goal: {action}")
            elif action.startswith("?"):
                # Test goal - query beliefs
                print(f"[{self.name}] Query: {action}")
            else:
                # Action
                print(f"[{self.name}] Action: {action}")
    
    def get_beliefs(self, nodule_id: Optional[str] = None) -> List[Belief]:
        """Get beliefs, optionally filtered by nodule_id."""
        if nodule_id is None:
            return list(self.beliefs.beliefs)
        
        return [
            b for b in self.beliefs.beliefs
            if str(nodule_id) in str(b.value) or str(nodule_id) in str(b.attribute)
        ]


def create_radiologist_agent(
    broker: MessageBroker,
    asl_path: Optional[str] = None
) -> RadiologistAgent:
    """
    Factory function to create a Radiologist agent.
    
    Args:
        broker: Message broker for communication
        asl_path: Path to ASL file with plans
        
    Returns:
        Configured RadiologistAgent instance
    """
    # Default ASL path
    if asl_path is None:
        asl_path = str(Path(__file__).parent.parent / "asl" / "radiologist.asl")
    
    return RadiologistAgent(broker, asl_path)


if __name__ == "__main__":
    # Demo usage
    print("=== Radiologist Agent Demo ===\n")
    
    # Create message broker
    broker = MessageBroker()
    
    # Create agent
    agent = RadiologistAgent(broker)
    
    # Simulate analysis request
    test_features = {
        "size_mm": 15,
        "texture": "solid",
        "spiculation": "present"
    }
    
    result = agent.analyze_nodule(
        nodule_id="demo_001",
        features=test_features
    )
    
    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2, default=str))
    
    print("\nAgent Beliefs:")
    for belief in agent.beliefs.beliefs:
        print(f"  {belief}")
