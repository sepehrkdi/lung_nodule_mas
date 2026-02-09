"""
SPADE-BDI Oncologist Agent
===========================

EDUCATIONAL PURPOSE - BDI AGENT WITH SYMBOLIC REASONING:

This module implements the Oncologist agent using SPADE-BDI.
The agent uses Prolog for symbolic reasoning (Lung-RADS rules),
with the logic queries called as internal actions from AgentSpeak plans.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────┐
    │           OncologistAgent (SPADE-BDI)           │
    ├─────────────────────────────────────────────────┤
    │  AgentSpeak Plans (oncologist.asl)              │
    │  ┌─────────────────────────────────────────┐    │
    │  │ +!decide(Id) : findings_ready(Id)       │    │
    │  │   <- .query_lung_rads(...);             │    │
    │  │      .compute_consensus(...);           │    │
    │  └─────────────────────────────────────────┘    │
    ├─────────────────────────────────────────────────┤
    │  Internal Actions (Python)                      │
    │  ┌─────────────────────────────────────────┐    │
    │  │ @action                                 │    │
    │  │ def query_lung_rads(self, ...):         │    │
    │  │     return self.prolog.query(...)       │    │
    │  └─────────────────────────────────────────┘    │
    ├─────────────────────────────────────────────────┤
    │  Prolog Engine (PySwip)                         │
    │  ┌─────────────────────────────────────────┐    │
    │  │ lung_rads_category(Size, Texture, Cat)  │    │
    │  │ management(Category, Recommendation)    │    │
    │  └─────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────┘

MULTI-AGENT CONSENSUS:
The Oncologist handles multiple findings from multiple agents
and implements weighted voting for final classification.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import Counter

from agents.spade_base import MedicalAgentBase, Belief, get_asl_path

logger = logging.getLogger(__name__)


class OncologistAgent(MedicalAgentBase):
    """
    SPADE-BDI Oncologist Agent with Prolog reasoning.
    
    This agent:
    1. Collects findings from Radiologists and Pathologists
    2. Runs Prolog queries for Lung-RADS classification
    3. Computes weighted consensus from multiple agents
    4. Generates final treatment recommendations
    
    EDUCATIONAL VALUE:
    - Integration of symbolic (Prolog) and subsymbolic (ML) AI
    - Multi-agent consensus with disagreement handling
    - Clinical decision support reasoning
    """
    
    # Agent weights for consensus voting
    AGENT_WEIGHTS = {
        "radiologist": 1.0,
        "pathologist": 0.8,
        "oncologist": 1.2
    }
    
    # Lung-RADS fallback rules (used if Prolog unavailable)
    LUNG_RADS_RULES = {
        # (size_min, size_max, texture): (category, management)
        (0, 6, "solid"): ("2", "annual_ct"),
        (6, 8, "solid"): ("3", "ct_6_months"),
        (8, 15, "solid"): ("4A", "ct_3_months_or_pet"),
        (15, 999, "solid"): ("4B", "tissue_sampling"),
        
        (0, 6, "ground_glass"): ("2", "annual_ct"),
        (6, 999, "ground_glass"): ("3", "ct_6_months"),
        
        (0, 6, "part_solid"): ("2", "annual_ct"),
        (6, 8, "part_solid"): ("3", "ct_6_months"),
        (8, 999, "part_solid"): ("4A", "ct_3_months_or_pet"),
    }
    
    def __init__(self, name: str = "oncologist", asl_file: Optional[str] = None):
        """
        Initialize the Oncologist agent.
        
        Args:
            name: Agent name
            asl_file: Path to AgentSpeak plans file
        """
        if asl_file is None:
            asl_file = get_asl_path("oncologist")
        
        super().__init__(name=name, asl_file=asl_file)
        
        # Prolog engine - lazy loaded
        self._prolog = None
        self._prolog_loaded = False
        
        # Findings storage for consensus
        self._findings: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"[{self.name}] Agent created")
    
    def _register_actions(self) -> None:
        """Register internal actions callable from AgentSpeak."""
        self.internal_actions = {
            "load_prolog": self._action_load_prolog,
            "query_lung_rads": self._action_query_lung_rads,
            "get_management": self._action_get_management,
            "compute_consensus": self._action_compute_consensus,
            "add_finding": self._action_add_finding,
        }
    
    # =========================================================================
    # Internal Actions (called from AgentSpeak)
    # =========================================================================
    
    def _action_load_prolog(self) -> bool:
        """
        Internal action: Load the Prolog knowledge base.
        
        Called from ASL: .load_prolog
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from knowledge_base.prolog_kb import LungRADSKnowledgeBase
            
            logger.info(f"[{self.name}] Loading Prolog knowledge base...")
            self._prolog = LungRADSKnowledgeBase()
            self._prolog_loaded = True
            
            self.add_belief(Belief("prolog_loaded", (True,)))
            logger.info(f"[{self.name}] Prolog KB loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to load Prolog: {e}")
            logger.info(f"[{self.name}] Using fallback rule-based reasoning")
            self.add_belief(Belief("prolog_error", (str(e),)))
            self._prolog_loaded = True  # Use fallback
            return True
    
    def _action_query_lung_rads(
        self,
        size_mm: float,
        texture: str,
        nodule_id: str = "unknown"
    ) -> Tuple[str, str]:
        """
        Internal action: Query Lung-RADS category from Prolog.
        
        Called from ASL: .query_lung_rads(Size, Texture, NoduleId, Category, Mgmt)
        
        EDUCATIONAL NOTE:
        This demonstrates First-Order Logic reasoning via Prolog.
        The knowledge base encodes medical guidelines as logical rules.
        
        Args:
            size_mm: Nodule size in millimeters
            texture: Nodule texture (solid, ground_glass, part_solid)
            nodule_id: Nodule identifier
            
        Returns:
            Tuple of (category, management)
        """
        if not self._prolog_loaded:
            self._action_load_prolog()
        
        # Normalize texture to expected format
        texture = str(texture).replace("-", "_").lower().strip()
        
        # Map common texture variations
        if texture in ["ground_glass", "groundglass", "ggo"]:
            texture = "ground_glass"
        elif texture in ["part_solid", "partsolid", "mixed"]:
            texture = "part_solid"
        elif texture in ["solid", "dense"]:
            texture = "solid"
        
        # Try Prolog query first
        if self._prolog is not None:
            try:
                result = self._prolog.query_lung_rads(
                    size=float(size_mm),
                    texture=texture
                )
                category = result.get("category", "2")
                management = result.get("management", "annual_ct")
                
                logger.info(
                    f"[{self.name}] Prolog query: size={size_mm}, texture={texture} "
                    f"-> category={category}, mgmt={management}"
                )
                
                self.add_belief(Belief(
                    "lung_rads",
                    (nodule_id, category, management),
                    annotations={"source": "prolog"}
                ))
                
                return (category, management)
                
            except Exception as e:
                logger.warning(f"[{self.name}] Prolog query failed: {e}")
        
        # Fallback to rule-based
        category, management = self._fallback_lung_rads(size_mm, texture)
        
        self.add_belief(Belief(
            "lung_rads",
            (nodule_id, category, management),
            annotations={"source": "fallback"}
        ))
        
        return (category, management)
    
    def _action_get_management(self, category: str) -> str:
        """
        Internal action: Get management recommendation for category.
        
        Called from ASL: .get_management(Category, Recommendation)
        
        Args:
            category: Lung-RADS category
            
        Returns:
            Management recommendation string
        """
        management_map = {
            "1": "routine_annual_screening",
            "2": "annual_ct_screening",
            "3": "ct_6_months",
            "4A": "ct_3_months_or_pet",
            "4B": "tissue_sampling",
            "4X": "immediate_tissue_sampling",
        }
        
        return management_map.get(category, "follow_clinical_judgment")
    
    def _action_add_finding(
        self,
        nodule_id: str,
        agent_name: str,
        finding: Dict[str, Any]
    ) -> bool:
        """
        Internal action: Add finding from another agent.
        
        Called from ASL: .add_finding(NoduleId, AgentName, Finding)
        
        Args:
            nodule_id: Nodule identifier
            agent_name: Name of agent providing finding
            finding: Finding dictionary
            
        Returns:
            True on success
        """
        if nodule_id not in self._findings:
            self._findings[nodule_id] = []
        
        self._findings[nodule_id].append({
            "agent": agent_name,
            "data": finding,
            "weight": self.AGENT_WEIGHTS.get(
                agent_name.split("_")[0].lower(), 1.0
            )
        })
        
        logger.info(f"[{self.name}] Added finding from {agent_name} for {nodule_id}")
        
        # Update belief about findings
        count = len(self._findings[nodule_id])
        self.add_belief(Belief("findings_count", (nodule_id, count)))
        
        return True
    
    def _action_compute_consensus(
        self,
        nodule_id: str
    ) -> Tuple[float, str, float]:
        """
        Internal action: Compute weighted consensus from all findings.
        
        Called from ASL: .compute_consensus(NoduleId, Probability, Risk, Confidence)
        
        EDUCATIONAL NOTE:
        This implements multi-agent consensus with weighted voting.
        Different agent types have different expertise weights.
        
        Args:
            nodule_id: Nodule identifier
            
        Returns:
            Tuple of (probability, risk_level, confidence)
        """
        findings = self._findings.get(nodule_id, [])
        
        if not findings:
            logger.warning(f"[{self.name}] No findings for {nodule_id}")
            return (0.5, "indeterminate", 0.0)
        
        # Collect probabilities with weights
        weighted_probs = []
        risk_votes = []
        
        for f in findings:
            weight = f["weight"]
            data = f["data"]
            
            # Get probability
            prob = data.get("malignancy_probability")
            if prob is None:
                prob = data.get("probability")
            if prob is not None:
                weighted_probs.append((float(prob), weight))
            
            # Get risk level
            risk = data.get("risk_level") or data.get("risk")
            if risk:
                risk_votes.append((risk, weight))
        
        # Compute weighted average probability
        if weighted_probs:
            total_weight = sum(w for _, w in weighted_probs)
            avg_prob = sum(p * w for p, w in weighted_probs) / total_weight
        else:
            avg_prob = 0.5
        
        # Compute weighted risk consensus
        if risk_votes:
            risk_counter = Counter()
            for risk, weight in risk_votes:
                risk_counter[risk.lower()] += weight
            consensus_risk = risk_counter.most_common(1)[0][0]
        else:
            if avg_prob >= 0.6:
                consensus_risk = "high"
            elif avg_prob >= 0.35:
                consensus_risk = "moderate"
            else:
                consensus_risk = "low"
        
        # Calculate confidence based on agreement
        if len(findings) >= 2:
            probs_only = [p for p, _ in weighted_probs]
            if probs_only:
                variance = sum((p - avg_prob)**2 for p in probs_only) / len(probs_only)
                confidence = 1.0 - min(variance * 5, 1.0)
            else:
                confidence = 0.5
        else:
            confidence = 0.6
        
        logger.info(
            f"[{self.name}] Consensus for {nodule_id}: "
            f"prob={avg_prob:.3f}, risk={consensus_risk}, conf={confidence:.3f}"
        )
        
        self.add_belief(Belief(
            "consensus",
            (nodule_id, round(avg_prob, 3), consensus_risk, round(confidence, 3)),
            annotations={"source": "self", "agent_count": len(findings)}
        ))
        
        return (avg_prob, consensus_risk, confidence)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _fallback_lung_rads(
        self,
        size_mm: float,
        texture: str
    ) -> Tuple[str, str]:
        """Fallback Lung-RADS classification without Prolog."""
        # Handle None size_mm
        if size_mm is None:
            size_mm = 10.0  # Default to intermediate size
            
        texture = texture.replace("-", "_").lower()
        
        for (min_s, max_s, tex), (cat, mgmt) in self.LUNG_RADS_RULES.items():
            if min_s <= size_mm < max_s and tex == texture:
                return (cat, mgmt)
        
        # Default fallback
        if size_mm < 6:
            return ("2", "annual_ct")
        elif size_mm < 8:
            return ("3", "ct_6_months")
        elif size_mm < 15:
            return ("4A", "ct_3_months_or_pet")
        else:
            return ("4B", "tissue_sampling")
    
    def _generate_recommendation(
        self,
        nodule_id: str,
        category: str,
        management: str,
        probability: float,
        confidence: float
    ) -> str:
        """Generate clinical recommendation text."""
        cat_desc = {
            "1": "No nodules detected",
            "2": "Benign appearance",
            "3": "Probably benign",
            "4A": "Suspicious",
            "4B": "Very suspicious",
            "4X": "Additional concerning features"
        }
        
        mgmt_desc = {
            "annual_ct": "Annual low-dose CT screening recommended",
            "ct_6_months": "Follow-up CT in 6 months recommended",
            "ct_3_months_or_pet": "Follow-up CT in 3 months or PET-CT recommended",
            "tissue_sampling": "Tissue sampling for pathologic diagnosis recommended",
            "immediate_tissue_sampling": "Immediate tissue sampling strongly recommended"
        }
        
        desc = cat_desc.get(category, "Requires evaluation")
        mgmt = mgmt_desc.get(management, "Clinical correlation recommended")
        
        recommendation = (
            f"LUNG-RADS CATEGORY {category}: {desc}. "
            f"Malignancy probability: {probability:.1%}. "
            f"RECOMMENDATION: {mgmt}. "
            f"(Confidence: {confidence:.1%})"
        )
        
        return recommendation
    
    # =========================================================================
    # Main Processing Interface
    # =========================================================================
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a classification request with all collected findings.
        
        Args:
            request: Dictionary with 'nodule_id', 'features', and optionally
                    'radiologist_findings' and 'pathologist_findings'
            
        Returns:
            Final classification and recommendation
        """
        nodule_id = request.get("nodule_id", "unknown")
        features = request.get("features", {})
        
        logger.info(f"[{self.name}] Processing request for {nodule_id}")
        
        # Add any provided findings
        if "radiologist_findings" in request:
            for i, f in enumerate(request["radiologist_findings"]):
                self._action_add_finding(
                    nodule_id,
                    f"radiologist_{i+1}",
                    f
                )
        
        if "pathologist_findings" in request:
            for i, f in enumerate(request["pathologist_findings"]):
                self._action_add_finding(
                    nodule_id,
                    f"pathologist_{i+1}",
                    f
                )
        
        # Get size and texture for Lung-RADS
        size_mm = features.get("size_mm", 10)
        texture = features.get("texture", "solid")
        
        # Check radiologist findings for size/texture
        rad_findings = request.get("radiologist_findings", [])
        if rad_findings:
            f = rad_findings[0]
            size_mm = f.get("estimated_size_mm", size_mm)
            texture = f.get("texture", texture)
        
        # Query Lung-RADS
        category, management = self._action_query_lung_rads(
            size_mm, texture, nodule_id
        )
        
        # Compute consensus from all findings
        probability, risk, confidence = self._action_compute_consensus(nodule_id)
        
        # Adjust category if consensus differs significantly
        if probability >= 0.7 and category in ["2", "3"]:
            category = "4A"
            management = "ct_3_months_or_pet"
        elif probability < 0.2 and category in ["4A", "4B"]:
            category = "3"
            management = "ct_6_months"
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            nodule_id, category, management, probability, confidence
        )
        
        result = {
            "nodule_id": nodule_id,
            "agent": self.name,
            "status": "success",
            "classification": {
                "lung_rads_category": category,
                "malignancy_probability": round(probability, 3),
                "risk_level": risk,
                "confidence": round(confidence, 3)
            },
            "recommendation": {
                "management": management,
                "description": recommendation
            },
            "sources": {
                "agent_count": len(self._findings.get(nodule_id, [])),
                "method": "prolog" if self._prolog else "rule_based"
            }
        }
        
        # Clear findings for this nodule
        if nodule_id in self._findings:
            del self._findings[nodule_id]
        
        return result


# =============================================================================
# SPADE-BDI Integration
# =============================================================================

def create_spade_oncologist(xmpp_config=None):
    """
    Create a SPADE-BDI Oncologist agent.
    
    Args:
        xmpp_config: XMPP server configuration
        
    Returns:
        SPADE-BDI agent instance or standalone agent
    """
    from agents.spade_base import create_spade_bdi_agent, DEFAULT_XMPP_CONFIG
    
    if xmpp_config is None:
        xmpp_config = DEFAULT_XMPP_CONFIG
    
    asl_file = get_asl_path("oncologist")
    
    return create_spade_bdi_agent(
        agent_class=OncologistAgent,
        name="oncologist",
        xmpp_config=xmpp_config,
        asl_file=asl_file
    )


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Oncologist Agent Test ===\n")
    
    # Create agent
    agent = OncologistAgent()
    
    # Simulate findings from other agents
    radiologist_findings = [
        {
            "malignancy_probability": 0.65,
            "predicted_class": 4,
            "estimated_size_mm": 12,
            "texture": "part_solid"
        },
        {
            "malignancy_probability": 0.72,
            "predicted_class": 4,
            "estimated_size_mm": 13,
            "texture": "part_solid"
        }
    ]
    
    pathologist_findings = [
        {
            "text_malignancy_probability": 0.58,
            "risk_level": "moderate",
            "biopsy_recommended": True
        }
    ]
    
    test_request = {
        "nodule_id": "test_001",
        "features": {
            "size_mm": 12,
            "texture": "part_solid",
            "malignancy": 4
        },
        "radiologist_findings": radiologist_findings,
        "pathologist_findings": pathologist_findings
    }
    
    async def test():
        result = await agent.process_request(test_request)
        print("\nResult:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, str) and len(v) > 60:
                        print(f"    {k}: {v[:60]}...")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\nBeliefs:")
        for belief in agent.beliefs:
            print(f"  {belief}")
    
    asyncio.run(test())
