"""
Multi-Agent Orchestrator - Extended Architecture
================================================

EDUCATIONAL PURPOSE - 5-AGENT MULTI-AGENT SYSTEM:

This orchestrator coordinates 5 specialized agents:
- 3 Radiologists: DenseNet121, ResNet50, Rule-based
- 2 Pathologists: Regex-based, spaCy NER

ARCHITECTURE DIAGRAM:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      ORCHESTRATOR                               │
    │                   (Coordination Layer)                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
    │   │DenseNet  │  │ ResNet50 │  │Rule-Based│   ← Radiologists    │
    │   │  W=1.0   │  │  W=1.0   │  │  W=0.7   │                     │
    │   └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
    │        │             │             │                            │
    │        └─────────────┴─────────────┘                           │
    │                      │                                          │
    │   ┌──────────────────┴───────────────────┐                     │
    │   │         PROLOG CONSENSUS             │                     │
    │   │   (Weighted Voting + Conflict Res)   │                     │
    │   └──────────────────┬───────────────────┘                     │
    │                      │                                          │
    │        ┌─────────────┴─────────────┐                           │
    │        │             │             │                            │
    │   ┌────┴─────┐  ┌────┴─────┐                                   │
    │   │  Regex   │  │  spaCy   │         ← Pathologists            │
    │   │  W=0.8   │  │  W=0.9   │                                   │
    │   └──────────┘  └──────────┘                                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

CONSENSUS MECHANISM:
- Weighted voting in Prolog (multi_agent_consensus.pl)
- Automatic disagreement detection
- Confidence interval computation
- Final decision with justification
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import all specialized agents
from agents.radiologist_variants import (
    RadiologistDenseNet,
    RadiologistResNet,
    RadiologistRuleBased,
    create_all_radiologists,
)
from agents.pathologist_variants import (
    PathologistRegex,
    PathologistSpacy,
    create_all_pathologists,
)

# Import Prolog consensus engine
try:
    from knowledge.prolog_engine import PrologEngine
    PROLOG_AVAILABLE = True
except ImportError:
    PROLOG_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentFinding:
    """Result from a single agent."""
    agent_name: str
    agent_type: str  # "radiologist" or "pathologist"
    approach: str    # e.g., "densenet", "resnet", "regex", "spacy"
    weight: float
    probability: float
    predicted_class: int
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_prolog_fact(self) -> str:
        """Convert to Prolog assertion."""
        return (
            f"agent_finding({self.agent_name}, {self.agent_type}, "
            f"{self.approach}, {self.weight}, {self.probability:.4f}, "
            f"{self.predicted_class})"
        )


@dataclass  
class ConsensusResult:
    """Combined result from multi-agent consensus."""
    nodule_id: str
    final_probability: float
    final_class: int
    confidence: float
    radiologist_findings: List[AgentFinding]
    pathologist_findings: List[AgentFinding]
    agreement_level: str  # "unanimous", "majority", "split"
    disagreement_agents: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    prolog_reasoning: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PROLOG CONSENSUS ENGINE
# =============================================================================

class PrologConsensusEngine:
    """
    Prolog-based consensus engine for multi-agent decision making.
    
    Uses multi_agent_consensus.pl for:
    - Weighted voting
    - Disagreement detection
    - Confidence calculation
    - Final decision with justification
    """
    
    def __init__(self, kb_path: Optional[str] = None):
        self.prolog = None
        self.kb_path = kb_path
        
        if PROLOG_AVAILABLE:
            try:
                self.prolog = PrologEngine()
                if kb_path:
                    self.prolog.load_knowledge_base(kb_path)
                logger.info("Prolog consensus engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Prolog: {e}")
                self.prolog = None
    
    def compute_consensus(
        self,
        nodule_id: str,
        radiologist_findings: List[AgentFinding],
        pathologist_findings: List[AgentFinding]
    ) -> Dict[str, Any]:
        """Compute consensus using Prolog weighted voting."""
        
        if self.prolog is None:
            return self._fallback_consensus(
                nodule_id, radiologist_findings, pathologist_findings
            )
        
        try:
            # Clear previous findings
            self.prolog.query("retractall(agent_finding(_, _, _, _, _, _))")
            
            # Assert all findings
            for finding in radiologist_findings + pathologist_findings:
                self.prolog.assertz(finding.to_prolog_fact())
            
            # Query for consensus
            result = list(self.prolog.query(
                f"weighted_consensus({nodule_id}, Prob, Class, Conf)"
            ))
            
            if result:
                return {
                    "method": "prolog_weighted_voting",
                    "probability": float(result[0]["Prob"]),
                    "class": int(result[0]["Class"]),
                    "confidence": float(result[0]["Conf"])
                }
            else:
                return self._fallback_consensus(
                    nodule_id, radiologist_findings, pathologist_findings
                )
                
        except Exception as e:
            logger.warning(f"Prolog consensus failed: {e}")
            return self._fallback_consensus(
                nodule_id, radiologist_findings, pathologist_findings
            )
    
    def _fallback_consensus(
        self,
        nodule_id: str,
        radiologist_findings: List[AgentFinding],
        pathologist_findings: List[AgentFinding]
    ) -> Dict[str, Any]:
        """Python fallback for weighted consensus."""
        all_findings = radiologist_findings + pathologist_findings
        
        if not all_findings:
            return {
                "method": "fallback_no_findings",
                "probability": 0.5,
                "class": 3,
                "confidence": 0.0
            }
        
        # Weighted average
        total_weight = sum(f.weight for f in all_findings)
        weighted_prob = sum(
            f.probability * f.weight for f in all_findings
        ) / total_weight
        
        # Confidence from agreement
        probabilities = [f.probability for f in all_findings]
        variance = np.var(probabilities)
        confidence = max(0, 1 - variance * 4)  # Higher variance = lower confidence
        
        # Determine class
        if weighted_prob < 0.2:
            pred_class = 1
        elif weighted_prob < 0.4:
            pred_class = 2
        elif weighted_prob < 0.6:
            pred_class = 3
        elif weighted_prob < 0.8:
            pred_class = 4
        else:
            pred_class = 5
        
        return {
            "method": "weighted_average_fallback",
            "probability": weighted_prob,
            "class": pred_class,
            "confidence": confidence
        }


# =============================================================================
# MULTI-AGENT ORCHESTRATOR
# =============================================================================

class MultiAgentOrchestrator:
    """
    Orchestrates 5 specialized agents for lung nodule classification.
    
    EDUCATIONAL DEMONSTRATION:
    - Parallel agent execution
    - Prolog-based consensus mechanism
    - Disagreement detection and resolution
    - Confidence-weighted decision making
    """
    
    def __init__(
        self,
        consensus_kb_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the orchestrator with all agents.
        
        Args:
            consensus_kb_path: Path to multi_agent_consensus.pl
            use_gpu: Whether to use GPU for CNN agents
        """
        self.use_gpu = use_gpu
        
        # Initialize radiologists
        logger.info("Initializing radiologist agents...")
        self.radiologists = self._create_radiologists()
        
        # Initialize pathologists
        logger.info("Initializing pathologist agents...")
        self.pathologists = self._create_pathologists()
        
        # Initialize Prolog consensus engine
        self.consensus_engine = PrologConsensusEngine(consensus_kb_path)
        
        # Statistics
        self.stats = {
            "cases_processed": 0,
            "unanimous_decisions": 0,
            "majority_decisions": 0,
            "split_decisions": 0
        }
        
        logger.info(
            f"Orchestrator initialized with {len(self.radiologists)} radiologists "
            f"and {len(self.pathologists)} pathologists"
        )
    
    def _create_radiologists(self) -> List:
        """Create all three radiologist agents."""
        return [
            RadiologistDenseNet("radiologist_densenet"),
            RadiologistResNet("radiologist_resnet"),
            RadiologistRuleBased("radiologist_rulebased"),
        ]
    
    def _create_pathologists(self) -> List:
        """Create both pathologist agents."""
        return [
            PathologistRegex("pathologist_regex"),
            PathologistSpacy("pathologist_spacy"),
        ]
    
    async def analyze_case(
        self,
        nodule_id: str,
        image_path: Optional[str] = None,
        image_array: Optional[np.ndarray] = None,
        report: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        Analyze a single nodule case with all 5 agents.
        
        Args:
            nodule_id: Unique identifier for the nodule
            image_path: Path to CT/X-ray image
            image_array: Pre-loaded image array
            report: Radiology report text
            features: Pre-extracted nodule features
            
        Returns:
            ConsensusResult with combined decision
        """
        logger.info(f"Analyzing case {nodule_id} with 5 agents...")
        
        # Prepare requests
        radiologist_request = {
            "nodule_id": nodule_id,
            "image_path": image_path,
            "image_array": image_array,
            "features": features or {},
        }
        
        pathologist_request = {
            "nodule_id": nodule_id,
            "report": report,
            "features": features or {},
        }
        
        # Run all agents in parallel
        radiologist_tasks = [
            agent.process_request(radiologist_request)
            for agent in self.radiologists
        ]
        
        pathologist_tasks = [
            agent.process_request(pathologist_request)
            for agent in self.pathologists
        ]
        
        # Gather all results
        all_results = await asyncio.gather(
            *radiologist_tasks, *pathologist_tasks,
            return_exceptions=True
        )
        
        # Split results
        radiologist_results = all_results[:len(self.radiologists)]
        pathologist_results = all_results[len(self.radiologists):]
        
        # Convert to AgentFinding objects
        radiologist_findings = self._process_results(
            radiologist_results, self.radiologists, "radiologist"
        )
        pathologist_findings = self._process_results(
            pathologist_results, self.pathologists, "pathologist"
        )
        
        # Compute consensus
        consensus = self.consensus_engine.compute_consensus(
            nodule_id, radiologist_findings, pathologist_findings
        )
        
        # Determine agreement level
        agreement_level, disagreement_agents = self._check_agreement(
            radiologist_findings + pathologist_findings
        )
        
        # Update statistics
        self._update_stats(agreement_level)
        
        # Create final result
        result = ConsensusResult(
            nodule_id=nodule_id,
            final_probability=consensus["probability"],
            final_class=consensus["class"],
            confidence=consensus["confidence"],
            radiologist_findings=radiologist_findings,
            pathologist_findings=pathologist_findings,
            agreement_level=agreement_level,
            disagreement_agents=disagreement_agents,
            prolog_reasoning=consensus
        )
        
        logger.info(
            f"Case {nodule_id}: class={result.final_class}, "
            f"prob={result.final_probability:.3f}, "
            f"confidence={result.confidence:.3f}, "
            f"agreement={agreement_level}"
        )
        
        return result
    
    def _process_results(
        self,
        results: List,
        agents: List,
        agent_type: str
    ) -> List[AgentFinding]:
        """Convert raw results to AgentFinding objects."""
        findings = []
        
        for i, result in enumerate(results):
            agent = agents[i]
            
            if isinstance(result, Exception):
                logger.warning(f"Agent {agent.name} failed: {result}")
                continue
            
            if result.get("status") != "success":
                logger.warning(f"Agent {agent.name} returned error")
                continue
            
            agent_findings = result.get("findings", {})
            
            # Extract probability (different keys for different agents)
            prob = (
                agent_findings.get("malignancy_probability") or
                agent_findings.get("text_malignancy_probability") or
                0.5
            )
            
            findings.append(AgentFinding(
                agent_name=agent.name,
                agent_type=agent_type,
                approach=agent.APPROACH,
                weight=agent.WEIGHT,
                probability=prob,
                predicted_class=agent_findings.get("predicted_class", 3),
                details=agent_findings
            ))
        
        return findings
    
    def _check_agreement(
        self,
        findings: List[AgentFinding]
    ) -> Tuple[str, List[str]]:
        """Check agreement level among agents."""
        if not findings:
            return "no_findings", []
        
        classes = [f.predicted_class for f in findings]
        
        # Check for unanimity (all same class)
        if len(set(classes)) == 1:
            return "unanimous", []
        
        # Check for majority (at least 3 of 5 agree)
        from collections import Counter
        class_counts = Counter(classes)
        most_common_class, most_common_count = class_counts.most_common(1)[0]
        
        if most_common_count >= 3:
            disagreement_agents = [
                f.agent_name for f in findings
                if f.predicted_class != most_common_class
            ]
            return "majority", disagreement_agents
        
        # Otherwise split
        return "split", [f.agent_name for f in findings]
    
    def _update_stats(self, agreement_level: str):
        """Update processing statistics."""
        self.stats["cases_processed"] += 1
        if agreement_level == "unanimous":
            self.stats["unanimous_decisions"] += 1
        elif agreement_level == "majority":
            self.stats["majority_decisions"] += 1
        else:
            self.stats["split_decisions"] += 1
    
    async def analyze_batch(
        self,
        cases: List[Dict[str, Any]]
    ) -> List[ConsensusResult]:
        """
        Analyze multiple cases.
        
        Args:
            cases: List of case dictionaries with nodule_id, image_path, report, etc.
            
        Returns:
            List of ConsensusResult objects
        """
        results = []
        
        for i, case in enumerate(cases):
            logger.info(f"Processing case {i+1}/{len(cases)}")
            
            result = await self.analyze_case(
                nodule_id=case.get("nodule_id", f"case_{i}"),
                image_path=case.get("image_path"),
                image_array=case.get("image_array"),
                report=case.get("report"),
                features=case.get("features")
            )
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = self.stats["cases_processed"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "unanimous_rate": self.stats["unanimous_decisions"] / total,
            "majority_rate": self.stats["majority_decisions"] / total,
            "split_rate": self.stats["split_decisions"] / total,
        }
    
    def export_results(
        self,
        results: List[ConsensusResult],
        output_path: str
    ):
        """Export results to JSON file."""
        export_data = []
        
        for result in results:
            export_data.append({
                "nodule_id": result.nodule_id,
                "final_probability": result.final_probability,
                "final_class": result.final_class,
                "confidence": result.confidence,
                "agreement_level": result.agreement_level,
                "disagreement_agents": result.disagreement_agents,
                "timestamp": result.timestamp,
                "radiologist_findings": [
                    {
                        "agent": f.agent_name,
                        "approach": f.approach,
                        "weight": f.weight,
                        "probability": f.probability,
                        "class": f.predicted_class
                    }
                    for f in result.radiologist_findings
                ],
                "pathologist_findings": [
                    {
                        "agent": f.agent_name,
                        "approach": f.approach,
                        "weight": f.weight,
                        "probability": f.probability,
                        "class": f.predicted_class
                    }
                    for f in result.pathologist_findings
                ],
                "prolog_reasoning": result.prolog_reasoning
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(results)} results to {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Test the multi-agent orchestrator."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Multi-Agent Orchestrator - 5-Agent Architecture Test")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Test case with features and report
    test_case = {
        "nodule_id": "test_001",
        "features": {
            "size_mm": 12,
            "texture": "spiculated",
            "location": "right upper lobe",
            "malignancy": 4
        },
        "report": """
        FINDINGS: A 12mm spiculated nodule is identified in the right upper lobe.
        The lesion has irregular margins and no calcification.
        
        IMPRESSION: Suspicious pulmonary nodule. Highly concerning for malignancy.
        Recommend PET-CT and biopsy for further evaluation.
        """
    }
    
    # Analyze
    result = await orchestrator.analyze_case(**test_case)
    
    # Print results
    print(f"\nCase: {result.nodule_id}")
    print(f"Final Class: {result.final_class}")
    print(f"Final Probability: {result.final_probability:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Agreement Level: {result.agreement_level}")
    
    print("\n--- Radiologist Findings ---")
    for f in result.radiologist_findings:
        print(f"  {f.agent_name}: class={f.predicted_class}, prob={f.probability:.3f}")
    
    print("\n--- Pathologist Findings ---")
    for f in result.pathologist_findings:
        print(f"  {f.agent_name}: class={f.predicted_class}, prob={f.probability:.3f}")
    
    print("\n--- Statistics ---")
    stats = orchestrator.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
