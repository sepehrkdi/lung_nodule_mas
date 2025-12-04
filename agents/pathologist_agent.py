"""
Pathologist Agent
=================

EDUCATIONAL PURPOSE - BDI AGENT FOR NLP:

This agent implements the Pathologist role in the Multi-Agent System.
It uses Natural Language Processing to analyze radiology reports.

NLP TECHNIQUES DEMONSTRATED:
- Tokenization: Breaking text into words/sentences
- Part-of-Speech Tagging: Identifying word types
- Named Entity Recognition: Finding medical concepts
- Pattern Matching: Extracting structured information

BDI COMPONENTS:
- BELIEFS: Extracted information from reports
- DESIRES: Goals to analyze text and extract features
- INTENTIONS: Active NLP processing plans

AGENT COMMUNICATION:
- Receives ACHIEVE requests from main orchestrator
- Sends INFORM messages with findings to Oncologist
- Uses belief annotations with [source(pathologist)]
"""

import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

# Import base agent components
from agents.base_agent import BDIAgent, Belief, BeliefBase, Plan, ASLParser
from communication.message_queue import Message, Performative, MessageBroker

# Import NLP extractor
from nlp.extractor import MedicalNLPExtractor, ExtractionResult


class PathologistAgent(BDIAgent):
    """
    BDI Agent for NLP-based report analysis.
    
    EDUCATIONAL PURPOSE:
    
    This agent demonstrates:
    1. BDI Architecture - Beliefs, Desires, Intentions
    2. NLP pipeline integration with scispaCy
    3. Information extraction from medical text
    4. Agent communication for multi-agent reasoning
    
    The Pathologist extracts from radiology reports:
    - Nodule size measurements
    - Texture descriptions (solid, ground-glass)
    - Margin characteristics (spiculated, well-defined)
    - Malignancy assessment from impression
    
    Architecture:
        [Report Text] → [NLP Pipeline] → [Extracted Info] → [Beliefs] → [Message]
    """
    
    def __init__(self, message_broker: MessageBroker, asl_file: Optional[str] = None):
        """
        Initialize the Pathologist agent.
        
        Args:
            message_broker: Shared message broker for agent communication
            asl_file: Path to AgentSpeak plan file (optional)
        """
        super().__init__("pathologist", message_broker)
        
        # Load plans from ASL file
        if asl_file:
            self._load_asl_plans(asl_file)
        
        # Initialize NLP extractor
        self.extractor: Optional[MedicalNLPExtractor] = None
        self._init_nlp()
        
        # Add initial beliefs
        self._init_beliefs()
        
        print(f"[{self.name}] Agent initialized")
    
    def _init_nlp(self) -> None:
        """Initialize the NLP extractor."""
        try:
            self.extractor = MedicalNLPExtractor()
            self.add_belief(Belief("nlp", "status", "loaded"))
            
            if self.extractor.nlp:
                model_name = self.extractor.nlp.meta.get("name", "unknown")
                self.add_belief(Belief("nlp", "model", model_name))
            else:
                self.add_belief(Belief("nlp", "model", "regex_only"))
            
            print(f"[{self.name}] NLP extractor initialized")
            
        except Exception as e:
            self.add_belief(Belief("nlp", "status", "failed"))
            self.add_belief(Belief("nlp", "error", str(e)))
            print(f"[{self.name}] NLP initialization failed: {e}")
    
    def _init_beliefs(self) -> None:
        """Add initial beliefs about agent capabilities."""
        # NLP capabilities
        self.add_belief(Belief("capability", "tokenization", True))
        self.add_belief(Belief("capability", "ner", True))
        self.add_belief(Belief("capability", "pattern_matching", True))
        
        # Supported entity types
        self.add_belief(Belief("entity_types", "supported", 
                               ["size", "texture", "margin", "location", "malignancy"]))
    
    def _load_asl_plans(self, asl_file: str) -> None:
        """Load plans from AgentSpeak file."""
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
        
        EDUCATIONAL PURPOSE - NLP IN BDI:
        
        The Pathologist's reasoning cycle focuses on:
        1. PERCEIVE: Check for new report analysis requests
        2. UPDATE BELIEFS: Add extracted information
        3. DELIBERATE: Determine what to extract
        4. PLAN: Select appropriate NLP plans
        5. EXECUTE: Run extraction and update beliefs
        """
        # Check for incoming messages
        message = self.broker.receive(self.name)
        
        if message:
            self._handle_message(message)
        
        # Process any pending intentions
        self._process_intentions()
    
    def _handle_message(self, message: Message) -> None:
        """Handle incoming message based on performative."""
        print(f"[{self.name}] Received {message.performative.value} from {message.sender}")
        
        if message.performative == Performative.ACHIEVE:
            self._handle_achieve(message)
        
        elif message.performative == Performative.QUERY_REF:
            self._handle_query(message)
        
        elif message.performative == Performative.INFORM:
            self._handle_inform(message)
    
    def _handle_achieve(self, message: Message) -> None:
        """
        Handle ACHIEVE request.
        
        Expected content format:
        {
            "goal": "analyze_report",
            "nodule_id": "nodule_001",
            "report_text": "CT findings show...",  # Optional
            "features": {...}                      # Optional features to convert
        }
        """
        content = message.content
        goal = content.get("goal", "")
        
        if goal == "analyze_report":
            nodule_id = content.get("nodule_id", "unknown")
            report_text = content.get("report_text")
            features = content.get("features", {})
            
            # Generate report from features if not provided
            if not report_text and features:
                report_text = self._generate_report_from_features(features)
            
            if report_text:
                result = self.analyze_report(nodule_id, report_text)
            else:
                result = {
                    "nodule_id": nodule_id,
                    "agent": self.name,
                    "status": "error",
                    "error": "No report text or features provided"
                }
            
            # Send result back
            self._send_result(message.sender, nodule_id, result)
        
        else:
            print(f"[{self.name}] Unknown goal: {goal}")
    
    def _handle_query(self, message: Message) -> None:
        """Handle QUERY_REF (information request)."""
        content = message.content
        query_type = content.get("query", "")
        
        if query_type == "report_findings":
            nodule_id = content.get("nodule_id")
            beliefs = self.get_beliefs_for_nodule(nodule_id)
            
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
        
        if "beliefs" in content:
            for belief_data in content["beliefs"]:
                belief = Belief(
                    predicate=belief_data.get("predicate", "info"),
                    attribute=belief_data.get("attribute", ""),
                    value=belief_data.get("value", ""),
                    source=message.sender
                )
                self.add_belief(belief)
    
    def analyze_report(self, nodule_id: str, report_text: str) -> Dict[str, Any]:
        """
        Analyze a radiology report using NLP.
        
        EDUCATIONAL PURPOSE - NLP PIPELINE:
        
        1. Tokenization: Split text into tokens
        2. Entity extraction: Find medical entities
        3. Pattern matching: Extract measurements
        4. Keyword detection: Identify key features
        5. Belief update: Store extracted information
        
        Args:
            nodule_id: Unique identifier for the nodule
            report_text: Raw radiology report text
            
        Returns:
            Dictionary with extraction results
        """
        print(f"[{self.name}] Analyzing report for: {nodule_id}")
        
        result = {
            "nodule_id": nodule_id,
            "agent": self.name,
            "status": "success",
            "findings": {}
        }
        
        # Run NLP extraction
        if self.extractor:
            try:
                extraction = self.extractor.extract(report_text)
                
                # Add findings from extraction
                result["findings"]["size_mm"] = extraction.size_mm
                result["findings"]["texture"] = extraction.texture
                result["findings"]["margin"] = extraction.margin
                result["findings"]["spiculation"] = extraction.spiculation
                result["findings"]["lobulation"] = extraction.lobulation
                result["findings"]["calcification"] = extraction.calcification
                result["findings"]["location"] = extraction.location
                result["findings"]["malignancy_assessment"] = extraction.malignancy_assessment
                result["findings"]["lung_rads_category"] = extraction.lung_rads_category
                
                # Add entities
                result["findings"]["entities"] = [
                    {"text": e.text, "label": e.label}
                    for e in extraction.entities
                ]
                
                # Update beliefs
                self._update_beliefs_from_extraction(nodule_id, extraction)
                
                # Add linguistic analysis
                if self.extractor.nlp:
                    result["findings"]["tokens"] = self.extractor.tokenize(report_text)[:20]
                    result["findings"]["noun_phrases"] = self.extractor.get_noun_phrases(report_text)[:10]
                
            except Exception as e:
                result["status"] = "partial"
                result["error"] = str(e)
                print(f"[{self.name}] Extraction error: {e}")
        
        else:
            result["status"] = "fallback"
            result["findings"] = self._regex_extraction(report_text)
        
        # Add source annotation
        result["source"] = self.name
        
        return result
    
    def _update_beliefs_from_extraction(
        self, 
        nodule_id: str, 
        extraction: ExtractionResult
    ) -> None:
        """Update agent beliefs from NLP extraction."""
        # Size belief
        if extraction.size_mm:
            self.add_belief(Belief(
                "extraction", "size_mm", extraction.size_mm,
                source=self.name
            ))
        
        # Texture belief
        if extraction.texture:
            self.add_belief(Belief(
                "extraction", "texture", extraction.texture,
                source=self.name
            ))
        
        # Margin belief
        if extraction.margin:
            self.add_belief(Belief(
                "extraction", "margin", extraction.margin,
                source=self.name
            ))
        
        # Spiculation belief
        if extraction.spiculation:
            self.add_belief(Belief(
                "extraction", "spiculation", extraction.spiculation,
                source=self.name
            ))
        
        # Location belief
        if extraction.location:
            self.add_belief(Belief(
                "extraction", "location", extraction.location,
                source=self.name
            ))
        
        # Malignancy assessment belief
        if extraction.malignancy_assessment:
            self.add_belief(Belief(
                "extraction", "malignancy_text", extraction.malignancy_assessment,
                source=self.name
            ))
        
        # Lung-RADS belief
        if extraction.lung_rads_category:
            self.add_belief(Belief(
                "extraction", "lung_rads", extraction.lung_rads_category,
                source=self.name
            ))
        
        # Nodule ID tracking
        self.add_belief(Belief(
            "processed", "nodule", nodule_id,
            source=self.name
        ))
    
    def _regex_extraction(self, text: str) -> Dict[str, Any]:
        """
        Fallback regex-based extraction when NLP is unavailable.
        
        EDUCATIONAL NOTE:
        Regular expressions provide a reliable baseline for
        structured extraction even without ML models.
        """
        findings = {}
        text_lower = text.lower()
        
        # Extract size
        size_match = re.search(r'(\d+\.?\d*)\s*(mm|cm)', text_lower)
        if size_match:
            value = float(size_match.group(1))
            unit = size_match.group(2)
            if 'cm' in unit:
                value *= 10
            findings["size_mm"] = value
        
        # Extract texture
        if 'ground-glass' in text_lower or 'ground glass' in text_lower:
            findings["texture"] = "ground_glass"
        elif 'part-solid' in text_lower or 'part solid' in text_lower:
            findings["texture"] = "part_solid"
        elif 'solid' in text_lower:
            findings["texture"] = "solid"
        
        # Extract margins
        if 'spiculated' in text_lower or 'spiculation' in text_lower:
            findings["margin"] = "spiculated"
            findings["spiculation"] = "present"
        elif 'well-defined' in text_lower or 'well defined' in text_lower:
            findings["margin"] = "well_defined"
        elif 'lobulated' in text_lower:
            findings["margin"] = "lobulated"
        
        # Extract malignancy assessment
        if 'highly suspicious' in text_lower:
            findings["malignancy_assessment"] = "highly_suspicious"
        elif 'suspicious' in text_lower:
            findings["malignancy_assessment"] = "moderately_suspicious"
        elif 'benign' in text_lower:
            findings["malignancy_assessment"] = "benign"
        
        return findings
    
    def _generate_report_from_features(self, features: Dict[str, Any]) -> str:
        """
        Generate a synthetic report from structured features.
        
        This is used when we have LIDC features but no report text.
        """
        from data.report_generator import ReportGenerator
        
        try:
            generator = ReportGenerator()
            return generator.generate(features)
        except Exception as e:
            print(f"[{self.name}] Report generation failed: {e}")
            return self._simple_report_from_features(features)
    
    def _simple_report_from_features(self, features: Dict[str, Any]) -> str:
        """Simple report generation fallback."""
        size = features.get("size_mm", features.get("diameter", 10))
        texture = features.get("texture", "solid")
        margin = features.get("margin", "well-defined")
        location = features.get("location", "right lung")
        malignancy = features.get("malignancy", 3)
        
        # Convert malignancy to text
        mal_text = {
            1: "highly unlikely to be malignant",
            2: "probably benign",
            3: "indeterminate for malignancy",
            4: "suspicious for malignancy",
            5: "highly suspicious for malignancy"
        }.get(malignancy, "indeterminate")
        
        return f"""
        CHEST CT - PULMONARY NODULE EVALUATION
        
        FINDINGS:
        A {size:.1f} mm {texture} pulmonary nodule is identified in the {location}.
        The nodule has {margin} margins.
        
        IMPRESSION:
        Pulmonary nodule noted. Features are {mal_text}.
        
        RECOMMENDATION:
        Recommend follow-up imaging based on Lung-RADS guidelines.
        """
    
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
        """Process active intentions."""
        while self.intentions:
            plan = self.intentions.pop(0)
            self._execute_plan(plan)
    
    def _execute_plan(self, plan: Plan) -> None:
        """Execute a plan."""
        for action in plan.body:
            if action.startswith("!"):
                print(f"[{self.name}] Sub-goal: {action}")
            elif action.startswith("?"):
                print(f"[{self.name}] Query: {action}")
            else:
                print(f"[{self.name}] Action: {action}")
    
    def get_beliefs_for_nodule(self, nodule_id: str) -> List[Belief]:
        """Get beliefs related to a specific nodule."""
        # This is a simplified filter
        return [
            b for b in self.beliefs.beliefs
            if str(nodule_id) in str(b.value) or b.predicate == "extraction"
        ]


def create_pathologist_agent(
    broker: MessageBroker,
    asl_path: Optional[str] = None
) -> PathologistAgent:
    """
    Factory function to create a Pathologist agent.
    
    Args:
        broker: Message broker for communication
        asl_path: Path to ASL file with plans
        
    Returns:
        Configured PathologistAgent instance
    """
    if asl_path is None:
        asl_path = str(Path(__file__).parent.parent / "asl" / "pathologist.asl")
    
    return PathologistAgent(broker, asl_path)


if __name__ == "__main__":
    # Demo usage
    print("=== Pathologist Agent Demo ===\n")
    
    # Create message broker
    broker = MessageBroker()
    
    # Create agent
    agent = PathologistAgent(broker)
    
    # Sample report
    sample_report = """
    CHEST CT - PULMONARY NODULE EVALUATION
    
    FINDINGS:
    A 18.5 mm solid pulmonary nodule is identified in the right upper lobe.
    The nodule demonstrates marked spiculation with poorly defined margins.
    No internal calcification is identified.
    
    IMPRESSION:
    Large solid nodule in the right upper lobe with spiculated margins.
    Features are highly suspicious for malignancy.
    Lung-RADS Category: 4B.
    
    RECOMMENDATION:
    PET-CT recommended. Consider CT-guided biopsy for tissue diagnosis.
    """
    
    result = agent.analyze_report("demo_001", sample_report)
    
    print("\nExtraction Result:")
    print(json.dumps(result, indent=2, default=str))
    
    print("\nAgent Beliefs:")
    for belief in agent.beliefs.beliefs:
        print(f"  {belief}")
