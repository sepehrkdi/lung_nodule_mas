"""
SPADE-BDI Pathologist Agent
============================

EDUCATIONAL PURPOSE - BDI AGENT WITH NLP:

This module implements the Pathologist agent using SPADE-BDI.
The agent uses scispaCy for medical NLP, with the NLP code
called as internal actions from AgentSpeak plans.

NLP PIPELINE:
1. Named Entity Recognition (medical terms)
2. Entity classification (anatomy, finding, etc.)
3. Sentiment/severity analysis
4. Report-based malignancy assessment
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from agents.spade_base import MedicalAgentBase, Belief, get_asl_path

logger = logging.getLogger(__name__)


class PathologistAgent(MedicalAgentBase):
    """
    SPADE-BDI Pathologist Agent with NLP capabilities.
    
    This agent:
    1. Receives report analysis requests
    2. Runs NLP entity extraction (internal action)
    3. Analyzes clinical context (internal action)
    4. Assesses malignancy probability from text
    5. Sends findings to Oncologist
    
    The NLP code is encapsulated in internal actions.
    """
    
    # Pattern dictionaries for medical NLP
    MALIGNANCY_PATTERNS = {
        "high": [
            r"malignan[ct]", r"adenocarcinoma", r"carcinoma",
            r"metastas[ie]s", r"invasive", r"poorly\s+differentiated",
            r"high[\s-]grade", r"necrosis", r"rapid\s+growth",
            r"suspicious", r"highly\s+suggestive"
        ],
        "moderate": [
            r"atypical", r"indeterminate", r"borderline",
            r"part[- ]solid", r"ground[- ]glass", r"increasing",
            r"moderate", r"follow[\s-]up\s+recommended", r"persistent"
        ],
        "low": [
            r"benign", r"granuloma", r"calcifi",
            r"stable", r"unchanged", r"hamartoma",
            r"resolved", r"artifact", r"lymph\s+node"
        ]
    }
    
    ANATOMY_PATTERNS = {
        "location": [
            r"upper\s+lobe", r"lower\s+lobe", r"middle\s+lobe",
            r"right\s+lung", r"left\s+lung", r"hilum",
            r"pleura", r"mediastin", r"subpleural",
            r"peribronchovascular", r"apex", r"base"
        ],
        "structure": [
            r"nodule", r"mass", r"lesion", r"opacity",
            r"consolidation", r"effusion", r"infiltrate",
            r"cavity", r"airway", r"vessel"
        ]
    }
    
    def __init__(self, name: str = "pathologist", asl_file: Optional[str] = None):
        """
        Initialize the Pathologist agent.
        
        Args:
            name: Agent name
            asl_file: Path to AgentSpeak plans file
        """
        if asl_file is None:
            asl_file = get_asl_path("pathologist")
        
        super().__init__(name=name, asl_file=asl_file)
        
        # NLP components - lazy loaded
        self._nlp = None
        self._nlp_loaded = False
        
        logger.info(f"[{self.name}] Agent created")
    
    def _register_actions(self) -> None:
        """Register internal actions callable from AgentSpeak."""
        self.internal_actions = {
            "load_nlp": self._action_load_nlp,
            "extract_entities": self._action_extract_entities,
            "analyze_context": self._action_analyze_context,
            "assess_malignancy": self._action_assess_malignancy,
        }
    
    # =========================================================================
    # Internal Actions (called from AgentSpeak)
    # =========================================================================
    
    def _action_load_nlp(self) -> bool:
        """
        Internal action: Load the spaCy NLP model (required).
        
        Called from ASL: .load_nlp
        
        Returns:
            True if successful
        """
        import spacy
        self._nlp = spacy.load("en_core_sci_sm")
        logger.info(f"[{self.name}] Loaded scispaCy model")
        
        self._nlp_loaded = True
        
        # Update beliefs
        self.add_belief(Belief("nlp_loaded", (True,)))
        self.add_belief(Belief("nlp_model", ("en_core_sci_sm",)))
        
        return True
    
    def _action_extract_entities(
        self,
        text: str,
        nodule_id: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """
        Internal action: Extract medical entities from text.
        
        Called from ASL: .extract_entities(Text, NoduleId, Entities)
        
        Args:
            text: Clinical report text
            nodule_id: Nodule identifier
            
        Returns:
            List of extracted entities with types
        """
        if not self._nlp_loaded:
            self._action_load_nlp()
        
        entities = []
        
        # Use spaCy if available
        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Always add rule-based extraction
        rule_entities = self._extract_entities_rule_based(text)
        
        # Merge and deduplicate
        seen = set()
        merged = []
        for e in entities + rule_entities:
            key = (e["text"].lower(), e["label"])
            if key not in seen:
                seen.add(key)
                merged.append(e)
        
        # Add belief about entities
        self.add_belief(Belief(
            "extracted_entities",
            (nodule_id, len(merged)),
            annotations={"source": "self"}
        ))
        
        logger.info(
            f"[{self.name}] Extracted {len(merged)} entities from report for {nodule_id}"
        )
        
        return merged
    
    def _action_analyze_context(
        self,
        text: str,
        nodule_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Internal action: Analyze clinical context of report.
        
        Called from ASL: .analyze_context(Text, NoduleId, Context)
        
        Args:
            text: Clinical report text
            nodule_id: Nodule identifier
            
        Returns:
            Context analysis dictionary
        """
        text_lower = text.lower()
        
        # Find anatomical mentions
        location = "unspecified"
        for pattern in self.ANATOMY_PATTERNS["location"]:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group()
                break
        
        # Detect structure mentions
        structures = []
        for pattern in self.ANATOMY_PATTERNS["structure"]:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                if match:
                    structures.append(match.group())
        
        # Check for temporal context
        is_followup = bool(re.search(
            r"follow[\s-]?up|previous|prior|compared|interval",
            text_lower
        ))
        
        # Check for recommendations
        has_biopsy_rec = bool(re.search(
            r"biopsy|tissue\s+sampling|pathologic|histologic",
            text_lower
        ))
        
        has_followup_rec = bool(re.search(
            r"recommend.{0,20}follow|suggest.{0,20}ct|repeat.{0,20}imaging",
            text_lower
        ))
        
        context = {
            "location": location,
            "structures": structures,
            "is_followup": is_followup,
            "biopsy_recommended": has_biopsy_rec,
            "followup_recommended": has_followup_rec,
            "report_length": len(text.split())
        }
        
        # Add belief
        self.add_belief(Belief(
            "clinical_context",
            (nodule_id, location, is_followup),
            annotations={"source": "self"}
        ))
        
        logger.info(f"[{self.name}] Analyzed context for {nodule_id}: {location}")
        
        return context
    
    def _action_assess_malignancy(
        self,
        text: str,
        nodule_id: str = "unknown"
    ) -> Tuple[float, str]:
        """
        Internal action: Assess malignancy probability from report text.
        
        Called from ASL: .assess_malignancy(Text, NoduleId, Probability, Risk)
        
        EDUCATIONAL NOTE:
        This demonstrates how NLP sentiment/pattern analysis can
        contribute to medical decision making alongside imaging.
        
        Args:
            text: Clinical report text
            nodule_id: Nodule identifier
            
        Returns:
            Tuple of (probability, risk_level)
        """
        text_lower = text.lower()
        
        # Count pattern matches by category
        high_count = sum(
            1 for p in self.MALIGNANCY_PATTERNS["high"]
            if re.search(p, text_lower)
        )
        mod_count = sum(
            1 for p in self.MALIGNANCY_PATTERNS["moderate"]
            if re.search(p, text_lower)
        )
        low_count = sum(
            1 for p in self.MALIGNANCY_PATTERNS["low"]
            if re.search(p, text_lower)
        )
        
        # Calculate weighted score
        total = high_count * 3 + mod_count * 2 + low_count * 1
        high_score = high_count * 3
        
        if total == 0:
            probability = 0.5
            risk = "indeterminate"
        else:
            probability = (high_score + mod_count) / (total + 1)
            
            if probability >= 0.6:
                risk = "high"
            elif probability >= 0.35:
                risk = "moderate"
            else:
                risk = "low"
        
        # Adjust for specific indicators
        if high_count >= 2:
            probability = min(probability + 0.1, 0.95)
            risk = "high"
        
        if low_count >= 3 and high_count == 0:
            probability = max(probability - 0.15, 0.05)
            risk = "low"
        
        # Add belief
        self.add_belief(Belief(
            "text_assessment",
            (nodule_id, round(probability, 3), risk),
            annotations={"source": "self"}
        ))
        
        logger.info(
            f"[{self.name}] Assessed {nodule_id}: "
            f"prob={probability:.3f}, risk={risk}"
        )
        
        return (probability, risk)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _extract_entities_rule_based(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns."""
        entities = []
        text_lower = text.lower()
        
        # Extract malignancy indicators
        for risk, patterns in self.MALIGNANCY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    entities.append({
                        "text": match.group(),
                        "label": f"MALIGNANCY_{risk.upper()}",
                        "start": match.start(),
                        "end": match.end()
                    })
        
        # Extract anatomy mentions
        for category, patterns in self.ANATOMY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    entities.append({
                        "text": match.group(),
                        "label": f"ANATOMY_{category.upper()}",
                        "start": match.start(),
                        "end": match.end()
                    })
        
        # Extract measurements
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(mm|cm|millimeter)", text_lower):
            entities.append({
                "text": match.group(),
                "label": "MEASUREMENT",
                "start": match.start(),
                "end": match.end()
            })
        
        return entities
    
    def _generate_report_from_features(self, features: Dict[str, Any]) -> str:
        """Generate synthetic report from feature dictionary."""
        size = features.get("size_mm", 10)
        texture = features.get("texture", "solid")
        malignancy = features.get("malignancy", 3)
        
        # Size description
        if size < 6:
            size_desc = f"small {size}mm"
        elif size < 15:
            size_desc = f"moderate-sized {size}mm"
        else:
            size_desc = f"large {size}mm"
        
        # Texture description
        texture_map = {
            "ground_glass": "ground glass opacity",
            "ground-glass": "ground glass opacity",
            "part_solid": "part-solid nodule",
            "solid": "solid nodule",
            "calcified": "calcified nodule"
        }
        texture_desc = texture_map.get(texture, "nodule")
        
        # Risk description based on malignancy score
        if malignancy >= 4:
            risk_desc = "findings are suspicious for malignancy"
            rec = "Biopsy is recommended for tissue diagnosis."
        elif malignancy >= 3:
            risk_desc = "findings are indeterminate"
            rec = "Short-term follow-up CT recommended."
        else:
            risk_desc = "findings suggest benign etiology"
            rec = "Routine follow-up appropriate."
        
        report = (
            f"FINDINGS: A {size_desc} {texture_desc} is identified in the lung. "
            f"The {risk_desc}. "
            f"IMPRESSION: {rec}"
        )
        
        return report
    
    # =========================================================================
    # Main Processing Interface
    # =========================================================================
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an analysis request.
        
        Args:
            request: Dictionary with 'nodule_id' and 'report' or 'features'
            
        Returns:
            Analysis results dictionary
        """
        nodule_id = request.get("nodule_id", "unknown")
        report_text = request.get("report")
        
        # Generate report from features if not provided
        if not report_text:
            features = request.get("features", {})
            report_text = self._generate_report_from_features(features)
        
        logger.info(f"[{self.name}] Processing request for {nodule_id}")
        
        # Extract entities
        entities = self._action_extract_entities(report_text, nodule_id)
        
        # Analyze context
        context = self._action_analyze_context(report_text, nodule_id)
        
        # Assess malignancy
        probability, risk = self._action_assess_malignancy(report_text, nodule_id)
        
        result = {
            "nodule_id": nodule_id,
            "agent": self.name,
            "status": "success",
            "findings": {
                "text_malignancy_probability": probability,
                "risk_level": risk,
                "entities_found": len(entities),
                "key_entities": [e["text"] for e in entities[:5]],
                "location": context.get("location", "unspecified"),
                "biopsy_recommended": context.get("biopsy_recommended", False),
                "followup_recommended": context.get("followup_recommended", False)
            }
        }
        
        return result


# =============================================================================
# SPADE-BDI Integration
# =============================================================================

def create_spade_pathologist(xmpp_config=None):
    """
    Create a SPADE-BDI Pathologist agent.
    
    Args:
        xmpp_config: XMPP server configuration
        
    Returns:
        SPADE-BDI agent instance or standalone agent
    """
    from agents.spade_base import create_spade_bdi_agent, DEFAULT_XMPP_CONFIG
    
    if xmpp_config is None:
        xmpp_config = DEFAULT_XMPP_CONFIG
    
    asl_file = get_asl_path("pathologist")
    
    return create_spade_bdi_agent(
        agent_class=PathologistAgent,
        name="pathologist",
        xmpp_config=xmpp_config,
        asl_file=asl_file
    )


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Pathologist Agent Test ===\n")
    
    # Create agent
    agent = PathologistAgent()
    
    # Test report
    test_report = """
    FINDINGS: A 15mm part-solid nodule is identified in the right upper lobe.
    The lesion demonstrates peripheral ground glass halo with central solid 
    component. Compared to prior CT from 6 months ago, the nodule has increased
    in size from 10mm. No calcification is present.
    
    IMPRESSION: Suspicious pulmonary nodule with interval growth. 
    Findings are concerning for malignancy. 
    Recommend tissue sampling for pathologic diagnosis.
    """
    
    test_request = {
        "nodule_id": "test_001",
        "report": test_report
    }
    
    async def test():
        result = await agent.process_request(test_request)
        print("\nResult:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\nBeliefs:")
        for belief in agent.beliefs:
            print(f"  {belief}")
    
    asyncio.run(test())
