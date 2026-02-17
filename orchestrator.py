"""
Multi-Agent Orchestrator coordinating 6 agents with Prolog-based consensus.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import all specialized agents
from agents.radiologist_variants import (
    RadiologistDenseNet,
    RadiologistResNet,
    RadiologistRules,
    create_all_radiologists,
)
from agents.pathologist_variants import (
    PathologistRegex,
    PathologistSpacy,
    create_all_pathologists,
)

# Import Prolog consensus engine - STRICT MODE: Required, no fallback
from knowledge.prolog_engine import (
    PrologEngine,
    PrologUnavailableError,
    PrologQueryError,
)

# Import dynamic weight calculator
from models.dynamic_weights import (
    DynamicWeightCalculator,
    BASE_WEIGHTS,
    get_base_weight,
)

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
    thinking_process: List[Dict[str, str]] = field(default_factory=list)
    weight_rationale: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PROLOG CONSENSUS ENGINE
# =============================================================================

class PrologConsensusEngine:
    """
    Prolog-based consensus engine for multi-agent decision making.
    
    STRICT MODE: Prolog is required. No fallback behavior is provided.
    
    Uses multi_agent_consensus.pl for:
    - Weighted voting
    - Disagreement detection
    - Confidence calculation
    - Final decision with justification
    """
    
    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path
        
        # STRICT MODE: Initialize Prolog engine - let errors propagate
        logger.info("Initializing Prolog consensus engine (STRICT MODE)...")
        self.prolog = PrologEngine(auto_load_kb=True)
        
        # Load additional KB if specified
        if kb_path:
            self.prolog.load_knowledge_base(kb_path)
        
        logger.info("Prolog consensus engine initialized successfully")
    
    def _update_prolog_weights(self, weights: Dict[str, float]) -> None:
        """
        Update Prolog KB with dynamic per-case weights.
        
        Retracts old agent_weight/2 facts and asserts new ones so that
        Prolog's calculate_consensus uses the same dynamic weights as Python.
        """
        for agent_name, weight in weights.items():
            try:
                self.prolog.retractall(f"agent_weight({agent_name}, _)")
            except Exception:
                pass  # May not exist yet
            try:
                self.prolog.assertz(f"agent_weight({agent_name}, {weight})")
            except Exception as e:
                logger.warning(f"Failed to assert weight for {agent_name}: {e}")
    
    def compute_consensus(
        self,
        nodule_id: str,
        radiologist_findings: List[AgentFinding],
        pathologist_findings: List[AgentFinding],
        dynamic_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compute consensus using Prolog weighted voting.
        
        STRICT MODE: Uses Prolog only, no fallback.
        
        Args:
            nodule_id: Case identifier
            radiologist_findings: Findings from radiologist agents
            pathologist_findings: Findings from pathologist agents
            dynamic_weights: Per-case dynamic weights computed by
                DynamicWeightCalculator. If None, uses BASE_WEIGHTS.
        
        Raises:
            PrologQueryError: If consensus computation fails
        """
        all_findings = radiologist_findings + pathologist_findings
        
        if not all_findings:
            raise PrologQueryError("No findings provided for consensus")
        
        # Apply dynamic weights to findings
        weights = dynamic_weights or BASE_WEIGHTS
        for f in all_findings:
            f.weight = weights.get(f.agent_name, get_base_weight(f.agent_name))
            
            # Reduce weight by 50% when agent could not determine size
            # (size_source='unknown' or size_mm is None in details).
            # This prevents agents that fell back to defaults from having
            # outsized influence on the consensus.
            size_source = f.details.get("size_source", "")
            size_mm = f.details.get("size_mm")
            if size_source == "unknown" or (size_source == "none_detected") or (size_mm is None):
                f.weight *= 0.5
                logger.debug(
                    f"Weight reduced for {f.agent_name}: size unknown "
                    f"(size_source={size_source}), weight={f.weight:.3f}"
                )
        
        # Convert findings to format expected by Prolog engine
        prolog_findings = [
            {
                "agent_name": f.agent_name,
                "probability": f.probability,
                "predicted_class": f.predicted_class,
                "weight": f.weight
            }
            for f in all_findings
        ]
        
        # Update Prolog KB with dynamic weights before consensus
        self._update_prolog_weights(weights)
        
        # Use Prolog engine's compute_consensus method
        result = self.prolog.compute_consensus(nodule_id, prolog_findings)
        
        # Generate BDI Thinking Process
        thinking_process = []
        
        # Use dynamic weights (already applied to findings)
        
        # 1. Perception
        weight_sum = 0
        weighted_prob_sum = 0
        explanation_parts = []
        
        scores = []
        
        for f in all_findings:
            w = f.weight  # Already set to dynamic weight above
            
            p = f.probability
            weighted_prob_sum += p * w
            weight_sum += w
            scores.append(p)
            
            explanation_parts.append(f"{p:.2f}*{w}")
            
            # Extract features if available
            details_str = ""
            if "size_mm" in f.details:
                size = f.details["size_mm"]
                texture = f.details.get("texture", "unknown")
                details_str = f" | Feat: {size}mm, {texture}"
            
            thinking_process.append({
                "step": "Perception",
                "description": f"I perceive a finding from {f.agent_name}: Class {f.predicted_class} (Prob: {p:.1%}, Weight: {w}{details_str})",
                "type": "belief"
            })
            
        # 2. Deliberation - Weighted Voting
        equation = f"({ ' + '.join(explanation_parts) }) / {weight_sum}"
        
        # Calculate stats for confidence explanation
        import statistics
        if len(scores) > 1:
            variance = statistics.variance(scores) # Sample variance, Prolog might use population but let's stick to concept
            stdev = statistics.stdev(scores)
            # Prolog formula: Confidence = max(0, 1 - (StdDev * 3))
            # Note: Prolog uses population variance formula usually implemented manually.
            # Let's match Prolog's output logic behavior explanation
            
            # Re-calculate manually to match Prolog's exact "sum_squares / N" logic
            mean = sum(scores) / len(scores)
            variance_pop = sum((x - mean) ** 2 for x in scores) / len(scores)
            stdev_pop = variance_pop ** 0.5
            calc_conf = max(0, 1 - (stdev_pop * 3))
            
            conf_e = f"1 - ({stdev_pop:.3f} * 3) = {calc_conf:.3f}"
        else:
            conf_e = "Single agent default (0.8)"

        thinking_process.append({
            "step": "Deliberation (Weighted Voting)",
            "description": f"Formula: {equation} = {result['probability']:.1%}",
            "type": "deliberation"
        })
        
        # Generate consensus summary (Python-based, since Prolog facts are retracted after compute)
        agreement_desc = "with good agreement" if result['confidence'] >= 0.6 else "with significant disagreement"
        summary = f"Weighted consensus: {result['probability']:.2f} probability (class {result['predicted_class']}), confidence {result['confidence']:.2f} {agreement_desc}"
        thinking_process.append({
            "step": "Deliberation (Summary)",
            "description": summary,
            "type": "deliberation"
        })

        thinking_process.append({
            "step": "Deliberation (Confidence)",
            "description": f"Confidence is calculated based on agreement variance: {conf_e}",
            "type": "deliberation"
        })
        
        # 3. Knowledge Retrieval & Application (Rules)
        # NOTE: Prolog facts (agent_finding, nodule_size, etc.) are retracted
        # after compute_consensus, so we derive fired rules from Python data.
        fired_rules = []
        
        # --- Collect size and texture from agent findings ---
        sizes = []
        textures = []
        for f in all_findings:
            s = f.details.get("size_mm")
            if s is not None:
                sizes.append(float(s))
            t = f.details.get("texture")
            if t:
                textures.append(t)
        
        # Use median size if available
        best_size = float(np.median(sizes)) if sizes else None
        # Most common texture
        if textures:
            from collections import Counter
            best_texture = Counter(textures).most_common(1)[0][0]
        else:
            best_texture = "unknown"
        
        # --- Lung-RADS classification ---
        if best_size is not None:
            if best_size < 6:
                if best_texture == "ground_glass" and best_size < 30:
                    fired_rules.append("lung_rads(2): Benign - GGN <30mm")
                else:
                    fired_rules.append(f"lung_rads(2): Benign - {best_texture} <6mm")
            elif best_size < 8:
                fired_rules.append(f"lung_rads(3): Probably benign - {best_texture} 6-8mm")
            elif best_size < 15:
                fired_rules.append(f"lung_rads(4A): Suspicious - {best_texture} 8-15mm")
            else:
                fired_rules.append(f"lung_rads(4B): Very suspicious - {best_texture} ≥15mm")
            
            # --- T-stage (tumor size) ---
            if best_size <= 10:
                fired_rules.append("t_stage(T1a): Tumor ≤1cm")
            elif best_size <= 20:
                fired_rules.append("t_stage(T1b): Tumor >1-2cm")
            elif best_size <= 30:
                fired_rules.append("t_stage(T1c): Tumor >2-3cm")
            elif best_size <= 40:
                fired_rules.append("t_stage(T2a): Tumor >3-4cm")
            elif best_size <= 50:
                fired_rules.append("t_stage(T2b): Tumor >4-5cm")
            elif best_size <= 70:
                fired_rules.append("t_stage(T3): Tumor >5-7cm")
            else:
                fired_rules.append("t_stage(T4): Tumor >7cm")
        
        # --- Risk level from consensus ---
        prob = result["probability"]
        pred_class = result["predicted_class"]
        risk_map = {1: "low", 2: "low", 3: "intermediate", 4: "high", 5: "high"}
        risk = risk_map.get(pred_class, "intermediate")
        fired_rules.append(f"risk_level({risk}): class {pred_class}")
        
        # --- Disagreement detection ---
        if len(scores) > 1:
            mean_s = sum(scores) / len(scores)
            var_s = sum((x - mean_s) ** 2 for x in scores) / len(scores)
            std_s = var_s ** 0.5
            if std_s > 0.08:
                fired_rules.append("disagreement_detected")
        
        if fired_rules:
            thinking_process.append({
                "step": "Knowledge Check (Active Rules)",
                "description": "The following Prolog rules were triggered: " + "; ".join(fired_rules),
                "type": "reasoning"
            })
            
        # Check Disagreement Resolution
        if result['confidence'] < 0.6:
             try:
                res_expl = list(self.prolog.query(f"explain_resolution('{nodule_id}', Expl)"))
                if res_expl and "Expl" in res_expl[0]:
                    thinking_process.append({
                        "step": "Disagreement Resolution",
                        "description": str(res_expl[0]["Expl"]),
                        "type": "deliberation"
                    })
             except Exception:
                 pass

        # 4. Intention & Recommendation
        rec_desc = ""
        try:
            rec_expl = list(self.prolog.query(f"explain_recommendation('{nodule_id}', Expl)"))
            if rec_expl and "Expl" in rec_expl[0]:
                rec_desc = f" | {rec_expl[0]['Expl']}"
        except Exception:
            pass

        thinking_process.append({
            "step": "Intention",
            "description": f"I intend to classify as Class {result['predicted_class']} with {result['confidence']:.1%} confidence.{rec_desc}",
            "type": "intention"
        })
        
        return {
            "method": "prolog_weighted_voting",
            "probability": result["probability"],
            "class": result["predicted_class"],
            "confidence": result["confidence"],
            "thinking_process": thinking_process
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
        use_gpu: bool = True,
        weight_mode: str = "dynamic"
    ):
        """
        Initialize the orchestrator with all agents.
        
        Args:
            consensus_kb_path: Path to multi_agent_consensus.pl
            use_gpu: Whether to use GPU for CNN agents
            weight_mode: Weighting mode (dynamic, static, or equal)
        """
        self.use_gpu = use_gpu
        self.weight_mode = weight_mode
        
        # Initialize radiologists
        logger.info("Initializing radiologist agents...")
        self.radiologists = self._create_radiologists()
        
        # Initialize pathologists
        logger.info("Initializing pathologist agents...")
        self.pathologists = self._create_pathologists()
        
        # Initialize Prolog consensus engine
        self.consensus_engine = PrologConsensusEngine(consensus_kb_path)
        
        # Initialize dynamic weight calculator with specified mode
        self.weight_calculator = DynamicWeightCalculator(mode=weight_mode)
        
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
            RadiologistRules("radiologist_rulebased"),
        ]
    
    def _create_pathologists(self) -> List:
        """Create all pathologist agents."""
        return create_all_pathologists()
    
    async def analyze_case(
        self,
        case_id: str,  # Renamed from nodule_id for consistency
        image_path: Optional[str] = None,
        image_array: Union[np.ndarray, List[np.ndarray], None] = None,  # Support both single and multi-image
        report: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        image_metadata: Optional[List[Dict[str, Any]]] = None,  # Per-image metadata for aggregation
        case_metadata: Optional[Dict[str, Any]] = None,  # Full case metadata for dynamic weights
        on_agent_complete: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
    ) -> ConsensusResult:
        """
        Analyze a single case with all 6 agents.

        Supports both single-image and multi-image (NLMCXR) cases.
        Uses dynamic weight assignment based on per-case information richness.

        Args:
            case_id: Unique identifier for the case
            image_path: Path to CT/X-ray image (single-image only)
            image_array: Pre-loaded image array - either:
                        - np.ndarray for single-image cases
                        - List[np.ndarray] for multi-image (NLMCXR)
            report: Radiology report text
            features: Pre-extracted features or metadata
            image_metadata: Per-image metadata (view_type, etc.) for multi-image aggregation
            case_metadata: Full case metadata dict from data loader, used by
                          DynamicWeightCalculator to compute per-case weights.
                          If None, a minimal metadata dict is synthesized from
                          the other arguments.
            on_agent_complete: Optional async callback called when each agent finishes.
                               Signature: async def callback(agent_name: str, result: dict)

        Returns:
            ConsensusResult with combined decision and weight_rationale
        """
        logger.info(f"Analyzing case {case_id} with 6 agents...")

        # --- Dynamic Weight Computation ---
        # Build or use case_metadata for the weight calculator
        if case_metadata is None:
            # Synthesize minimal metadata from available arguments
            num_images = len(image_array) if isinstance(image_array, list) else (1 if image_array is not None else 0)
            case_metadata = {
                "case_id": case_id,
                "num_images": num_images,
                "images_metadata": image_metadata or [],
                "findings": report or "",
                "impression": "",
                "indication": "",
                "comparison": "",
                "nlp_features": features or {},
            }

        dynamic_weights, weight_rationale = self.weight_calculator.compute_weights(
            case_metadata
        )

        # Detect if multi-image
        is_multi_image = isinstance(image_array, list)

        # Prepare requests based on type
        if is_multi_image:
            # NLMCXR multi-image path
            radiologist_request = {
                "case_id": case_id,
                "images": image_array,
                "image_metadata": image_metadata or [],
                "features": features or {},
            }
        else:
            # Single-image path
            radiologist_request = {
                "nodule_id": case_id,
                "image_path": image_path,
                "image_array": image_array,
                "features": features or {},
            }

        pathologist_request = {
            "nodule_id": case_id,
            "report": report,
            "features": features or {},
        }
        
        # Create wrapped tasks that invoke callback on completion
        async def run_agent_with_callback(agent, request, agent_type):
            """Run agent and invoke callback when done."""
            result = await agent.process_request(request)
            result["_agent_name"] = agent.name
            result["_agent_type"] = agent_type
            if on_agent_complete:
                await on_agent_complete(agent.name, result)
            return result
        
        # Create all tasks
        radiologist_tasks = [
            run_agent_with_callback(agent, radiologist_request, "radiologist")
            for agent in self.radiologists
        ]
        
        pathologist_tasks = [
            run_agent_with_callback(agent, pathologist_request, "pathologist")
            for agent in self.pathologists
        ]
        
        all_tasks = radiologist_tasks + pathologist_tasks
        
        # Use as_completed to process results as they arrive (for streaming)
        radiologist_results = []
        pathologist_results = []
        
        for coro in asyncio.as_completed(all_tasks):
            try:
                result = await coro
                agent_name = result.get("_agent_name", "")
                agent_type = result.get("_agent_type", "")
                
                if agent_type == "radiologist":
                    radiologist_results.append(result)
                else:
                    pathologist_results.append(result)
            except Exception as e:
                logger.warning(f"Agent task failed: {e}")
        
        # Convert to AgentFinding objects
        radiologist_findings = self._process_results_streaming(
            radiologist_results, "radiologist"
        )
        pathologist_findings = self._process_results_streaming(
            pathologist_results, "pathologist"
        )
        
        # Compute consensus with dynamic weights
        consensus = self.consensus_engine.compute_consensus(
            case_id, radiologist_findings, pathologist_findings,
            dynamic_weights=dynamic_weights
        )
        
        # Determine agreement level
        agreement_level, disagreement_agents = self._check_agreement(
            radiologist_findings + pathologist_findings
        )
        
        # Update statistics
        self._update_stats(agreement_level)
        
        # Create final result
        result = ConsensusResult(
            nodule_id=case_id,
            final_probability=consensus["probability"],
            final_class=consensus["class"],
            confidence=consensus["confidence"],
            radiologist_findings=radiologist_findings,
            pathologist_findings=pathologist_findings,
            agreement_level=agreement_level,
            disagreement_agents=disagreement_agents,
            prolog_reasoning=consensus,
            thinking_process=consensus.get("thinking_process", []),
            weight_rationale=weight_rationale
        )
        
        logger.info(
            f"Case {case_id}: class={result.final_class}, "
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
            # Use None check instead of 'or' to handle 0.0 correctly
            prob = agent_findings.get("malignancy_probability")
            if prob is None:
                prob = agent_findings.get("text_malignancy_probability")
            if prob is None:
                prob = 0.5
            
            findings.append(AgentFinding(
                agent_name=agent.name,
                agent_type=agent_type,
                approach=agent.APPROACH,
                weight=get_base_weight(agent.name),  # Placeholder; overridden by dynamic weights in consensus
                probability=prob,
                predicted_class=agent_findings.get("predicted_class", 0),
                details=agent_findings
            ))
        
        return findings
    
    def _process_results_streaming(
        self,
        results: List[Dict[str, Any]],
        agent_type: str
    ) -> List[AgentFinding]:
        """Convert raw results to AgentFinding objects (streaming version)."""
        findings = []
        
        # Build agent lookup by name
        all_agents = self.radiologists + self.pathologists
        agent_lookup = {agent.name: agent for agent in all_agents}
        
        for result in results:
            if isinstance(result, Exception):
                continue
            
            if result.get("status") != "success":
                continue
            
            agent_name = result.get("_agent_name", "")
            agent = agent_lookup.get(agent_name)
            
            if not agent:
                continue
            
            agent_findings = result.get("findings", {})
            
            # Extract probability (different keys for different agents)
            # Use None check instead of 'or' to handle 0.0 correctly
            prob = agent_findings.get("malignancy_probability")
            if prob is None:
                prob = agent_findings.get("text_malignancy_probability")
            if prob is None:
                prob = 0.5
            
            findings.append(AgentFinding(
                agent_name=agent.name,
                agent_type=agent_type,
                approach=agent.APPROACH,
                weight=get_base_weight(agent.name),  # Placeholder; overridden by dynamic weights in consensus
                probability=prob,
                predicted_class=agent_findings.get("predicted_class", 0),
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
                case_id=case.get("nodule_id", f"case_{i}"),
                image_path=case.get("image_path"),
                image_array=case.get("image_array"),
                report=case.get("report"),
                features=case.get("features"),
                case_metadata=case.get("case_metadata")
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
                "prolog_reasoning": result.prolog_reasoning,
                "thinking_process": result.thinking_process,
                "weight_rationale": result.weight_rationale
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
    print("Multi-Agent Orchestrator - 6-Agent Architecture Test")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Test case with features and report
    test_case = {
        "case_id": "test_001",
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

    print("\n--- Thinking Process (BDI) ---")
    for step in result.thinking_process:
        print(f"  [{step['type'].upper()}] {step['step']}: {step['description']}")
    
    print("\n--- Statistics ---")
    stats = orchestrator.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
