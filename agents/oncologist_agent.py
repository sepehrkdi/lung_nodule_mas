"""
Oncologist Agent
================

EDUCATIONAL PURPOSE - BDI AGENT WITH PROLOG REASONING:

This agent implements the Oncologist role in the Multi-Agent System.
It uses Prolog (via PySwip) for symbolic AI reasoning.

SYMBOLIC AI CONCEPTS DEMONSTRATED:
- First-Order Logic (FOL): Predicates, variables, quantifiers
- Unification: Pattern matching with logical variables
- Backtracking: Exploring the search space
- Resolution: Deriving new facts from rules
- Horn Clauses: Rule representation

PROLOG INTEGRATION:
The agent queries a Prolog knowledge base containing:
- Lung-RADS classification rules
- TNM staging criteria
- Clinical recommendations

BDI COMPONENTS:
- BELIEFS: Combined findings from other agents + reasoning results
- DESIRES: Goals to synthesize findings and generate recommendations
- INTENTIONS: Active reasoning plans

AGENT ROLE:
The Oncologist is the "reasoning brain" that combines visual (Radiologist)
and textual (Pathologist) information to make clinical decisions.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

# Import base agent components
from agents.base_agent import BDIAgent, Belief, BeliefBase, Plan, ASLParser
from communication.message_queue import Message, Performative, MessageBroker


class PrologReasoner:
    """
    Prolog reasoning engine using PySwip.
    
    EDUCATIONAL PURPOSE - PROLOG IN PYTHON:
    
    PySwip provides a bridge between Python and SWI-Prolog.
    This allows us to:
    1. Load Prolog knowledge bases
    2. Assert new facts dynamically
    3. Query for solutions using unification
    4. Use backtracking to find all solutions
    
    Key Prolog Concepts:
    - Facts: ground truths (e.g., solid(nodule1).)
    - Rules: implications (e.g., suspicious(X) :- size(X, S), S > 15.)
    - Queries: questions to be solved (e.g., ?- lung_rads(nodule1, Cat).)
    """
    
    def __init__(self, kb_path: Optional[str] = None):
        """
        Initialize the Prolog reasoner.
        
        Args:
            kb_path: Path to Prolog knowledge base file
        """
        self.prolog = None
        self.kb_loaded = False
        
        self._init_prolog()
        
        if kb_path:
            self.load_knowledge_base(kb_path)
    
    def _init_prolog(self) -> None:
        """Initialize PySwip Prolog engine."""
        try:
            from pyswip import Prolog
            self.prolog = Prolog()
            print("[PrologReasoner] SWI-Prolog initialized")
        except ImportError:
            print("[PrologReasoner] PySwip not installed. Using fallback rules.")
        except Exception as e:
            print(f"[PrologReasoner] Prolog initialization failed: {e}")
    
    def load_knowledge_base(self, kb_path: str) -> bool:
        """
        Load a Prolog knowledge base.
        
        EDUCATIONAL NOTE:
        The KB contains rules like:
            lung_rads_category(Size, Texture, Margin, '4B') :-
                Size > 15, Texture = solid, Margin = spiculated.
        """
        if not self.prolog:
            return False
        
        try:
            # Consult the Prolog file
            self.prolog.consult(kb_path)
            self.kb_loaded = True
            print(f"[PrologReasoner] Loaded KB: {kb_path}")
            return True
        except Exception as e:
            print(f"[PrologReasoner] Failed to load KB: {e}")
            return False
    
    def assert_fact(self, predicate: str, *args) -> bool:
        """
        Assert a new fact into the Prolog database.
        
        EDUCATIONAL NOTE - DYNAMIC PREDICATES:
        Prolog allows adding facts at runtime using assertz/1.
        This lets us add nodule information for reasoning.
        
        Example:
            assert_fact("size", "nodule1", 15)
            -> assertz(size(nodule1, 15))
        """
        if not self.prolog:
            return False
        
        try:
            # Format arguments
            formatted_args = ', '.join(
                f"'{a}'" if isinstance(a, str) else str(a)
                for a in args
            )
            fact = f"assertz({predicate}({formatted_args}))"
            list(self.prolog.query(fact))
            return True
        except Exception as e:
            print(f"[PrologReasoner] Assert failed: {e}")
            return False
    
    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """
        Query the Prolog knowledge base.
        
        EDUCATIONAL NOTE - PROLOG QUERIES:
        
        A query asks Prolog to find variable bindings that satisfy
        a goal. Prolog uses:
        - Unification: Match terms with variables
        - Backtracking: Try alternative solutions
        - Resolution: Apply rules to derive answers
        
        Example:
            query("lung_rads_category(15, solid, spiculated, Cat)")
            -> [{'Cat': '4B'}]
        
        Args:
            query_str: Prolog query (without '?-')
            
        Returns:
            List of solution dictionaries
        """
        if not self.prolog:
            return []
        
        try:
            results = list(self.prolog.query(query_str))
            return results
        except Exception as e:
            print(f"[PrologReasoner] Query failed: {e}")
            return []
    
    def query_one(self, query_str: str) -> Optional[Dict[str, Any]]:
        """Get first solution to a query."""
        results = self.query(query_str)
        return results[0] if results else None
    
    def query_all(self, predicate: str, *args) -> List[Any]:
        """
        Use findall/3 to collect all solutions.
        
        EDUCATIONAL NOTE:
        findall(X, Goal, List) collects all values of X
        that satisfy Goal into List.
        
        Example:
            query_all("recommendation", "'4B'", "Rec")
            -> Uses: findall(Rec, recommendation('4B', Rec), List)
        """
        if not self.prolog:
            return []
        
        try:
            # Build the goal
            formatted_args = ', '.join(
                f"'{a}'" if isinstance(a, str) and not a.startswith("_") else str(a)
                for a in args
            )
            goal = f"{predicate}({formatted_args})"
            
            # Find the variable to collect
            var = next((a for a in args if isinstance(a, str) and a[0].isupper()), "X")
            
            query = f"findall({var}, {goal}, List)"
            results = self.query(query)
            
            if results and 'List' in results[0]:
                return results[0]['List']
            return []
        except Exception as e:
            print(f"[PrologReasoner] findall failed: {e}")
            return []


class FallbackReasoner:
    """
    Fallback reasoning when Prolog is unavailable.
    
    EDUCATIONAL NOTE:
    This implements the same logic as the Prolog KB
    but in Python. It demonstrates how rule-based
    systems can be implemented procedurally.
    """
    
    def get_lung_rads_category(
        self, 
        size_mm: float, 
        texture: str, 
        margin: Optional[str] = None
    ) -> str:
        """
        Determine Lung-RADS category.
        
        Lung-RADS Categories:
        1: Negative (no nodules)
        2: Benign appearance or behavior
        3: Probably benign (short-term follow-up)
        4A: Suspicious (requires follow-up)
        4B: Very suspicious (additional workup)
        4X: Highly suspicious features
        """
        # Normalize texture
        if texture in ['ground_glass', 'ggo']:
            texture = 'ground-glass'
        
        is_solid = texture == 'solid'
        is_gg = texture in ['ground-glass', 'ground_glass']
        is_part_solid = texture in ['part-solid', 'part_solid', 'subsolid']
        is_spiculated = margin in ['spiculated', 'spiculation']
        
        # Category 2: Benign
        if size_mm < 6 and is_solid:
            return '2'
        
        # Category 3: Probably benign
        if 6 <= size_mm < 8 and is_solid:
            return '3'
        if size_mm < 20 and is_gg:
            return '3'
        
        # Category 4A: Suspicious
        if 8 <= size_mm < 15 and is_solid:
            return '4A'
        if 6 <= size_mm < 8 and is_part_solid:
            return '4A'
        if size_mm >= 20 and is_gg:
            return '4A'
        
        # Category 4B: Very suspicious
        if size_mm >= 15 and is_solid:
            return '4B'
        if size_mm >= 8 and is_part_solid:
            return '4B'
        
        # Category 4X: Additional suspicious features
        if is_spiculated and size_mm >= 8:
            return '4X'
        
        return '3'  # Default to probably benign
    
    def get_t_stage(self, size_mm: float) -> str:
        """
        Determine TNM T-stage based on size.
        
        TNM T-Stages (simplified):
        T1a: ≤1 cm
        T1b: >1-2 cm
        T1c: >2-3 cm
        T2a: >3-4 cm
        T2b: >4-5 cm
        T3: >5-7 cm
        T4: >7 cm
        """
        if size_mm <= 10:
            return 'T1a'
        elif size_mm <= 20:
            return 'T1b'
        elif size_mm <= 30:
            return 'T1c'
        elif size_mm <= 40:
            return 'T2a'
        elif size_mm <= 50:
            return 'T2b'
        elif size_mm <= 70:
            return 'T3'
        else:
            return 'T4'
    
    def get_recommendation(self, lung_rads: str) -> str:
        """Get clinical recommendation based on Lung-RADS category."""
        recommendations = {
            '1': 'Continue annual screening.',
            '2': 'Continue annual screening.',
            '3': 'Follow-up CT in 6 months.',
            '4A': 'Follow-up CT in 3 months or PET-CT.',
            '4B': 'PET-CT and possible biopsy recommended.',
            '4X': 'Immediate evaluation with PET-CT and tissue sampling.'
        }
        return recommendations.get(lung_rads, 'Clinical correlation recommended.')
    
    def get_malignancy_probability(
        self,
        size_mm: float,
        texture: str,
        margin: Optional[str] = None,
        cv_prob: float = 0.5,
        nlp_assessment: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Calculate combined malignancy probability.
        
        Uses weighted combination of:
        - Size-based probability
        - Texture-based adjustment
        - Margin-based adjustment
        - CV model probability
        - NLP assessment
        """
        # Base probability from size
        if size_mm < 6:
            size_prob = 0.01
        elif size_mm < 8:
            size_prob = 0.02
        elif size_mm < 15:
            size_prob = 0.15
        elif size_mm < 20:
            size_prob = 0.30
        else:
            size_prob = 0.50
        
        # Texture adjustment
        texture_factor = 1.0
        if texture in ['solid']:
            texture_factor = 1.5
        elif texture in ['part-solid', 'part_solid']:
            texture_factor = 2.0
        elif texture in ['ground-glass', 'ground_glass']:
            texture_factor = 0.5
        
        # Margin adjustment
        margin_factor = 1.0
        if margin in ['spiculated', 'spiculation']:
            margin_factor = 2.0
        elif margin in ['lobulated']:
            margin_factor = 1.5
        elif margin in ['well-defined', 'well_defined']:
            margin_factor = 0.7
        
        # NLP assessment adjustment
        nlp_factor = 1.0
        if nlp_assessment:
            if 'highly_suspicious' in nlp_assessment:
                nlp_factor = 1.8
            elif 'suspicious' in nlp_assessment or 'moderately' in nlp_assessment:
                nlp_factor = 1.3
            elif 'benign' in nlp_assessment:
                nlp_factor = 0.4
        
        # Combine probabilities
        # Weighted: 40% size-based, 30% CV, 30% feature-adjusted
        rule_based = size_prob * texture_factor * margin_factor * nlp_factor
        rule_based = min(0.95, max(0.01, rule_based))
        
        combined = 0.4 * rule_based + 0.3 * cv_prob + 0.3 * (rule_based * 0.5 + cv_prob * 0.5)
        combined = min(0.99, max(0.01, combined))
        
        # Determine confidence based on agreement
        agreement = 1.0 - abs(rule_based - cv_prob)
        confidence = "high" if agreement > 0.7 else "medium" if agreement > 0.4 else "low"
        
        return combined, confidence


class OncologistAgent(BDIAgent):
    """
    BDI Agent for Prolog-based reasoning.
    
    EDUCATIONAL PURPOSE:
    
    This agent demonstrates:
    1. BDI Architecture with symbolic reasoning
    2. Prolog integration for First-Order Logic
    3. Multi-source information synthesis
    4. Rule-based clinical decision support
    
    The Oncologist synthesizes findings and applies:
    - Lung-RADS classification rules
    - TNM staging criteria
    - Clinical recommendations
    
    Architecture:
        [Radiologist Findings] + [Pathologist Findings]
                    ↓
            [Prolog KB Query]
                    ↓
            [Clinical Decision]
    """
    
    def __init__(
        self, 
        message_broker: MessageBroker, 
        kb_path: Optional[str] = None,
        asl_file: Optional[str] = None
    ):
        """
        Initialize the Oncologist agent.
        
        Args:
            message_broker: Shared message broker
            kb_path: Path to Prolog knowledge base
            asl_file: Path to AgentSpeak plan file
        """
        super().__init__("oncologist", message_broker)
        
        # Load plans from ASL file
        if asl_file:
            self._load_asl_plans(asl_file)
        
        # Initialize Prolog reasoner
        self.reasoner: Optional[PrologReasoner] = None
        self.fallback = FallbackReasoner()
        self._init_reasoner(kb_path)
        
        # Storage for agent findings
        self.pending_findings: Dict[str, Dict[str, Any]] = {}
        
        # Add initial beliefs
        self._init_beliefs()
        
        print(f"[{self.name}] Agent initialized")
    
    def _init_reasoner(self, kb_path: Optional[str] = None) -> None:
        """Initialize Prolog reasoner with knowledge base."""
        if kb_path is None:
            kb_path = str(Path(__file__).parent.parent / "knowledge" / "lung_rads.pl")
        
        try:
            self.reasoner = PrologReasoner(kb_path)
            
            if self.reasoner.kb_loaded:
                self.add_belief(Belief("prolog", "status", "loaded"))
                self.add_belief(Belief("prolog", "kb", kb_path))
            else:
                self.add_belief(Belief("prolog", "status", "fallback"))
            
        except Exception as e:
            self.add_belief(Belief("prolog", "status", "error"))
            self.add_belief(Belief("prolog", "error", str(e)))
            print(f"[{self.name}] Prolog init failed, using fallback: {e}")
    
    def _init_beliefs(self) -> None:
        """Add initial beliefs."""
        self.add_belief(Belief("role", "type", "synthesizer"))
        self.add_belief(Belief("requires", "agents", ["radiologist", "pathologist"]))
        self.add_belief(Belief("reasoning", "type", "symbolic"))
    
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
        """BDI Reasoning Cycle."""
        # Check for incoming messages
        message = self.broker.receive(self.name)
        
        if message:
            self._handle_message(message)
        
        # Process pending intentions
        self._process_intentions()
    
    def _handle_message(self, message: Message) -> None:
        """Handle incoming message."""
        print(f"[{self.name}] Received {message.performative.value} from {message.sender}")
        
        if message.performative == Performative.ACHIEVE:
            self._handle_achieve(message)
        
        elif message.performative == Performative.INFORM:
            self._handle_inform(message)
        
        elif message.performative == Performative.QUERY_REF:
            self._handle_query(message)
    
    def _handle_achieve(self, message: Message) -> None:
        """
        Handle ACHIEVE request.
        
        Expected content:
        {
            "goal": "assess_nodule",
            "nodule_id": "nodule_001",
            "radiologist_findings": {...},  # Optional
            "pathologist_findings": {...}   # Optional
        }
        """
        content = message.content
        goal = content.get("goal", "")
        
        if goal == "assess_nodule":
            nodule_id = content.get("nodule_id", "unknown")
            rad_findings = content.get("radiologist_findings", {})
            path_findings = content.get("pathologist_findings", {})
            
            # Perform assessment
            result = self.assess_nodule(
                nodule_id=nodule_id,
                radiologist_findings=rad_findings,
                pathologist_findings=path_findings
            )
            
            # Send result back
            self._send_result(message.sender, nodule_id, result)
        
        else:
            print(f"[{self.name}] Unknown goal: {goal}")
    
    def _handle_inform(self, message: Message) -> None:
        """Handle INFORM (findings from other agents)."""
        content = message.content
        
        if content.get("type") == "analysis_result":
            nodule_id = content.get("nodule_id")
            result = content.get("result", {})
            
            # Store findings
            if nodule_id not in self.pending_findings:
                self.pending_findings[nodule_id] = {}
            
            self.pending_findings[nodule_id][message.sender] = result
            
            # Add belief about received findings
            self.add_belief(Belief(
                "received", message.sender, nodule_id,
                source=message.sender
            ))
            
            # Check if we have all findings
            self._check_completeness(nodule_id)
    
    def _handle_query(self, message: Message) -> None:
        """Handle QUERY_REF (information request)."""
        content = message.content
        query_type = content.get("query", "")
        
        if query_type == "assessment":
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
    
    def _check_completeness(self, nodule_id: str) -> None:
        """Check if all agent findings are received."""
        findings = self.pending_findings.get(nodule_id, {})
        
        if "radiologist" in findings and "pathologist" in findings:
            print(f"[{self.name}] All findings received for {nodule_id}, assessing...")
            
            result = self.assess_nodule(
                nodule_id=nodule_id,
                radiologist_findings=findings["radiologist"],
                pathologist_findings=findings["pathologist"]
            )
            
            # Broadcast result
            self.broker.send(Message(
                sender=self.name,
                receiver="main",
                performative=Performative.INFORM,
                content={
                    "type": "final_assessment",
                    "nodule_id": nodule_id,
                    "result": result
                }
            ))
    
    def assess_nodule(
        self,
        nodule_id: str,
        radiologist_findings: Dict[str, Any],
        pathologist_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess a nodule by combining findings and applying reasoning.
        
        EDUCATIONAL PURPOSE - SYMBOLIC REASONING:
        
        This method demonstrates:
        1. Information fusion from multiple sources
        2. Prolog-based rule application
        3. Confidence-weighted decision making
        4. Clinical recommendation generation
        
        Args:
            nodule_id: Unique nodule identifier
            radiologist_findings: CV-based analysis results
            pathologist_findings: NLP-based extraction results
            
        Returns:
            Comprehensive assessment dictionary
        """
        print(f"[{self.name}] Assessing nodule: {nodule_id}")
        
        result = {
            "nodule_id": nodule_id,
            "agent": self.name,
            "status": "success",
            "assessment": {}
        }
        
        # Extract and merge findings
        merged = self._merge_findings(radiologist_findings, pathologist_findings)
        result["merged_findings"] = merged
        
        # Get key features
        size_mm = merged.get("size_mm", 10)
        texture = merged.get("texture", "solid")
        margin = merged.get("margin")
        cv_prob = merged.get("cv_probability", 0.5)
        nlp_assessment = merged.get("nlp_assessment")
        
        # Apply Prolog reasoning (or fallback)
        if self.reasoner and self.reasoner.prolog:
            result["assessment"] = self._prolog_assessment(
                size_mm, texture, margin
            )
            result["reasoning"] = "prolog"
        else:
            result["assessment"] = self._fallback_assessment(
                size_mm, texture, margin, cv_prob, nlp_assessment
            )
            result["reasoning"] = "fallback"
        
        # Calculate combined malignancy probability
        prob, confidence = self.fallback.get_malignancy_probability(
            size_mm, texture, margin, cv_prob, nlp_assessment
        )
        result["assessment"]["malignancy_probability"] = prob
        result["assessment"]["confidence"] = confidence
        
        # Update beliefs
        self._update_beliefs_from_assessment(nodule_id, result["assessment"])
        
        # Add source annotation
        result["source"] = self.name
        
        return result
    
    def _merge_findings(
        self,
        rad_findings: Dict[str, Any],
        path_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge findings from Radiologist and Pathologist.
        
        EDUCATIONAL NOTE - INFORMATION FUSION:
        
        When sources disagree, we need a resolution strategy:
        1. Confidence-based: Trust higher confidence source
        2. Source-based: Trust domain expert
        3. Averaging: Combine numerical values
        
        Here we prioritize:
        - Pathologist for textual features (texture, margin)
        - Radiologist for visual features (size from image)
        """
        merged = {}
        
        # Extract nested findings
        rad = rad_findings.get("findings", rad_findings)
        path = path_findings.get("findings", path_findings)
        
        # Size: prefer NLP extraction, fallback to CV estimate
        merged["size_mm"] = path.get("size_mm") or rad.get("estimated_size_mm", 10)
        
        # Texture: trust NLP extraction
        merged["texture"] = path.get("texture") or "solid"
        
        # Margin: from NLP
        merged["margin"] = path.get("margin")
        merged["spiculation"] = path.get("spiculation")
        
        # Malignancy: keep both for comparison
        merged["cv_probability"] = rad.get("malignancy_probability", 0.5)
        merged["cv_class"] = rad.get("predicted_class", 3)
        merged["nlp_assessment"] = path.get("malignancy_assessment")
        
        # Location
        merged["location"] = path.get("location")
        
        # Calcification
        merged["calcification"] = path.get("calcification")
        
        return merged
    
    def _prolog_assessment(
        self,
        size_mm: float,
        texture: str,
        margin: Optional[str]
    ) -> Dict[str, Any]:
        """
        Apply Prolog rules for assessment.
        
        EDUCATIONAL NOTE - PROLOG QUERIES:
        
        We query the KB with specific facts:
        - lung_rads_category(Size, Texture, Margin, Category)
        - tnm_t_stage(Size, Stage)
        - recommendation(Category, Rec)
        """
        assessment = {}
        
        # Assert current facts
        self.reasoner.assert_fact("current_size", size_mm)
        self.reasoner.assert_fact("current_texture", texture)
        if margin:
            self.reasoner.assert_fact("current_margin", margin)
        
        # Query Lung-RADS category
        texture_prolog = self._to_prolog_texture(texture)
        margin_prolog = self._to_prolog_margin(margin)
        
        query = f"lung_rads_category({size_mm}, {texture_prolog}, {margin_prolog}, Cat)"
        result = self.reasoner.query_one(query)
        
        if result:
            assessment["lung_rads"] = result.get("Cat", "3")
        else:
            # Fallback
            assessment["lung_rads"] = self.fallback.get_lung_rads_category(
                size_mm, texture, margin
            )
        
        # Query TNM stage
        query = f"tnm_t_stage({size_mm}, Stage)"
        result = self.reasoner.query_one(query)
        
        if result:
            assessment["t_stage"] = result.get("Stage", "T1b")
        else:
            assessment["t_stage"] = self.fallback.get_t_stage(size_mm)
        
        # Query recommendation
        cat = assessment["lung_rads"]
        query = f"recommendation('{cat}', Rec)"
        result = self.reasoner.query_one(query)
        
        if result:
            assessment["recommendation"] = result.get("Rec", "")
        else:
            assessment["recommendation"] = self.fallback.get_recommendation(cat)
        
        return assessment
    
    def _fallback_assessment(
        self,
        size_mm: float,
        texture: str,
        margin: Optional[str],
        cv_prob: float,
        nlp_assessment: Optional[str]
    ) -> Dict[str, Any]:
        """Apply fallback Python rules."""
        return {
            "lung_rads": self.fallback.get_lung_rads_category(size_mm, texture, margin),
            "t_stage": self.fallback.get_t_stage(size_mm),
            "recommendation": self.fallback.get_recommendation(
                self.fallback.get_lung_rads_category(size_mm, texture, margin)
            )
        }
    
    def _to_prolog_texture(self, texture: Optional[str]) -> str:
        """Convert texture to Prolog atom."""
        if not texture:
            return "solid"
        
        mapping = {
            "solid": "solid",
            "ground_glass": "ground_glass",
            "ground-glass": "ground_glass",
            "ggo": "ground_glass",
            "part_solid": "part_solid",
            "part-solid": "part_solid",
            "subsolid": "part_solid"
        }
        return mapping.get(texture.lower(), "solid")
    
    def _to_prolog_margin(self, margin: Optional[str]) -> str:
        """Convert margin to Prolog atom."""
        if not margin:
            return "undefined"
        
        mapping = {
            "spiculated": "spiculated",
            "well_defined": "well_defined",
            "well-defined": "well_defined",
            "poorly_defined": "poorly_defined",
            "lobulated": "lobulated"
        }
        return mapping.get(margin.lower(), "undefined")
    
    def _update_beliefs_from_assessment(
        self,
        nodule_id: str,
        assessment: Dict[str, Any]
    ) -> None:
        """Update agent beliefs from assessment."""
        # Lung-RADS
        if "lung_rads" in assessment:
            self.add_belief(Belief(
                "assessment", "lung_rads", assessment["lung_rads"],
                source=self.name
            ))
        
        # T-stage
        if "t_stage" in assessment:
            self.add_belief(Belief(
                "assessment", "t_stage", assessment["t_stage"],
                source=self.name
            ))
        
        # Malignancy probability
        if "malignancy_probability" in assessment:
            self.add_belief(Belief(
                "assessment", "malignancy_prob", assessment["malignancy_probability"],
                source=self.name
            ))
        
        # Recommendation
        if "recommendation" in assessment:
            self.add_belief(Belief(
                "assessment", "recommendation", assessment["recommendation"],
                source=self.name
            ))
        
        # Track processed nodule
        self.add_belief(Belief(
            "processed", "nodule", nodule_id,
            source=self.name
        ))
    
    def _send_result(self, recipient: str, nodule_id: str, result: Dict[str, Any]) -> None:
        """Send assessment result."""
        message = Message(
            sender=self.name,
            receiver=recipient,
            performative=Performative.INFORM,
            content={
                "type": "final_assessment",
                "nodule_id": nodule_id,
                "result": result
            }
        )
        self.broker.send(message)
        print(f"[{self.name}] Sent assessment for {nodule_id} to {recipient}")
    
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
            elif action.startswith("prolog_query"):
                print(f"[{self.name}] Prolog query: {action}")
            else:
                print(f"[{self.name}] Action: {action}")
    
    def get_beliefs_for_nodule(self, nodule_id: str) -> List[Belief]:
        """Get beliefs related to a specific nodule."""
        return [
            b for b in self.beliefs.beliefs
            if str(nodule_id) in str(b.value) or b.predicate == "assessment"
        ]


def create_oncologist_agent(
    broker: MessageBroker,
    kb_path: Optional[str] = None,
    asl_path: Optional[str] = None
) -> OncologistAgent:
    """
    Factory function to create an Oncologist agent.
    
    Args:
        broker: Message broker for communication
        kb_path: Path to Prolog knowledge base
        asl_path: Path to ASL file with plans
        
    Returns:
        Configured OncologistAgent instance
    """
    if asl_path is None:
        asl_path = str(Path(__file__).parent.parent / "asl" / "oncologist.asl")
    
    if kb_path is None:
        kb_path = str(Path(__file__).parent.parent / "knowledge" / "lung_rads.pl")
    
    return OncologistAgent(broker, kb_path, asl_path)


if __name__ == "__main__":
    # Demo usage
    print("=== Oncologist Agent Demo ===\n")
    
    # Create message broker
    broker = MessageBroker()
    
    # Create agent
    agent = OncologistAgent(broker)
    
    # Simulate findings from other agents
    rad_findings = {
        "findings": {
            "malignancy_probability": 0.7,
            "predicted_class": 4,
            "estimated_size_mm": 18.5
        }
    }
    
    path_findings = {
        "findings": {
            "size_mm": 18.5,
            "texture": "solid",
            "margin": "spiculated",
            "malignancy_assessment": "highly_suspicious"
        }
    }
    
    result = agent.assess_nodule(
        nodule_id="demo_001",
        radiologist_findings=rad_findings,
        pathologist_findings=path_findings
    )
    
    print("\nAssessment Result:")
    print(json.dumps(result, indent=2, default=str))
    
    print("\nAgent Beliefs:")
    for belief in agent.beliefs.beliefs:
        print(f"  {belief}")
