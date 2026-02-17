"""
Dynamic weight assignment for multi-agent consensus.

Computes per-case weights based on information richness:
- Radiology richness: image count, PA view presence, quality
- Pathology richness: text length, entity count, section completeness

BASE_WEIGHTS defined here is the single source of truth for agent weights.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


import json
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================
LEARNED_WEIGHTS_FILE = Path("data/learned_weights.json")
LEARNING_RATE = 0.01
MIN_WEIGHT = 0.2
MAX_WEIGHT = 3.0


# =============================================================================
# SINGLE SOURCE OF TRUTH: BASE WEIGHTS
# =============================================================================

BASE_WEIGHTS: Dict[str, float] = {
    # Radiologists
    "radiologist_densenet": 1.0,   # CNN — strong visual classifier
    "radiologist_resnet": 1.0,     # CNN — strong visual classifier
    "radiologist_rulebased": 0.7,  # Heuristic — simpler but interpretable

    # Pathologists
    "pathologist_regex": 0.8,      # Regex — brittle but fast
    "pathologist_spacy": 0.9,      # NLP/NER — more robust
    "pathologist_context": 0.9,    # Negation/uncertainty — high clinical value
}

DEFAULT_WEIGHT = 0.5  # Fallback for unknown agents


def get_base_weight(agent_name: str) -> float:
    """Get the base weight for an agent (single source of truth)."""
    return BASE_WEIGHTS.get(agent_name, DEFAULT_WEIGHT)


# =============================================================================
# RICHNESS SCORES
# =============================================================================

@dataclass
class RichnessScores:
    """
    Per-case information richness scores for each modality.
    
    These scores capture how much usable information the current case
    provides for radiology (image) and pathology (text) analysis.
    """
    radiology_richness: float = 0.5   # 0 = no/poor images, 1 = many high-quality views
    pathology_richness: float = 0.5   # 0 = sparse/empty report, 1 = detailed report

    # Sub-component breakdown (for transparency/auditability)
    radiology_components: Dict[str, float] = field(default_factory=dict)
    pathology_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "radiology_richness": round(self.radiology_richness, 4),
            "pathology_richness": round(self.pathology_richness, 4),
            "radiology_components": {
                k: round(v, 4) for k, v in self.radiology_components.items()
            },
            "pathology_components": {
                k: round(v, 4) for k, v in self.pathology_components.items()
            },
        }


# =============================================================================
# WEIGHTING MODES (for ablation studies)
# =============================================================================

class WeightingMode:
    """Weighting modes for ablation studies."""
    DYNAMIC = "dynamic"      # Per-case richness-based scaling (default)
    STATIC = "static"        # Fixed BASE_WEIGHTS only
    EQUAL = "equal"          # All weights = 1.0


# =============================================================================
# DYNAMIC WEIGHT CALCULATOR
# =============================================================================

class DynamicWeightCalculator:
    """
    Computes per-case dynamic weights for all agents based on
    information richness of the available data.
    
    Supports CONTINUAL LEARNING by persisting updated base weights
    to 'data/learned_weights.json'.
    
    For ablation studies, supports different weighting modes:
    - DYNAMIC: Per-case richness-based scaling (default)
    - STATIC: Fixed BASE_WEIGHTS only
    - EQUAL: All agents weight = 1.0
    
    Usage:
        calculator = DynamicWeightCalculator()
        weights, rationale = calculator.compute_weights(case_metadata)
        
        # After diagnosis is confirmed:
        calculator.update_weights(agent_findings, ground_truth=1)
        
        # For ablation studies:
        calculator = DynamicWeightCalculator(
            mode=WeightingMode.STATIC,
            use_size_penalty=False
        )
    """

    # Sub-component weights for radiology richness
    RAD_WEIGHT_NUM_IMAGES = 0.35     # More views → more information
    RAD_WEIGHT_PA_VIEW = 0.35        # PA is the most informative projection
    RAD_WEIGHT_IMAGE_QUALITY = 0.30  # Higher quality → more reliable features

    # Sub-component weights for pathology richness
    PATH_WEIGHT_REPORT_LENGTH = 0.25     # Longer reports tend to have more detail
    PATH_WEIGHT_ENTITY_COUNT = 0.30      # More extracted entities → richer content
    PATH_WEIGHT_SECTION_COMPLETENESS = 0.20  # More sections filled → better coverage
    PATH_WEIGHT_CERTAINTY = 0.25         # More affirmed mentions → higher confidence

    # Scaling bounds: weight = base × (SCALE_FLOOR + (1-SCALE_FLOOR) × richness)
    SCALE_FLOOR = 0.5   # Minimum fraction of base weight (never zero out an agent)

    def __init__(
        self,
        mode: str = WeightingMode.DYNAMIC,
        use_reliability_adjustment: bool = True,
        use_size_penalty: bool = True
    ):
        """
        Initialize weight calculator with ablation options.
        
        Args:
            mode: Weighting mode (DYNAMIC, STATIC, or EQUAL)
            use_reliability_adjustment: Enable continual learning updates
            use_size_penalty: Apply 50% penalty for unknown size source
        """
        self.mode = mode
        self.use_reliability_adjustment = use_reliability_adjustment
        self.use_size_penalty = use_size_penalty
        self._ensure_weights_file()
        
        logger.info(f"DynamicWeightCalculator initialized: mode={mode}, "
                   f"reliability_adj={use_reliability_adjustment}, "
                   f"size_penalty={use_size_penalty}")

    def _ensure_weights_file(self):
        """Load or initialize learned weights."""
        if not LEARNED_WEIGHTS_FILE.exists():
            LEARNED_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._save_weights(BASE_WEIGHTS)
        
    def _load_weights(self) -> Dict[str, float]:
        """Load current learned weights from disk."""
        try:
            with open(LEARNED_WEIGHTS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return BASE_WEIGHTS.copy()

    def _save_weights(self, weights: Dict[str, float]):
        """Save weights to disk."""
        try:
            with open(LEARNED_WEIGHTS_FILE, "w") as f:
                json.dump(weights, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")

    def compute_weights(
        self,
        case_metadata: Dict[str, Any],
        base_weights: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """
        Compute dynamic per-agent weights for a single case.

        Args:
            case_metadata: Case data dictionary. Expected keys:
                - num_images (int): Number of available images
                - images_metadata (list): Per-image dicts with 'view_type', 'shape', etc.
                - findings (str): FINDINGS section text
                - impression (str): IMPRESSION section text
                - indication (str): INDICATION section text
                - comparison (str): COMPARISON section text
                - nlp_features (dict): Output from MedicalNLPExtractor
                - ground_truth (int): Optional ground truth label
            base_weights: Override base weights (defaults to learned weights).

        Returns:
            Tuple of (weights_dict, rationale_dict):
                - weights_dict: {agent_name: dynamic_weight}
                - rationale_dict: full breakdown for auditability
        """
        # Handle different weighting modes for ablation studies
        if self.mode == WeightingMode.EQUAL:
            # All agents get equal weight of 1.0
            equal_weights = {agent: 1.0 for agent in BASE_WEIGHTS.keys()}
            rationale = {
                "mode": "equal",
                "dynamic_weights": equal_weights,
                "base_weights": equal_weights,
            }
            logger.info(f"Equal weights mode: all agents = 1.0")
            return equal_weights, rationale
        
        if self.mode == WeightingMode.STATIC:
            # Use fixed BASE_WEIGHTS without dynamic scaling
            static_weights = BASE_WEIGHTS.copy()
            rationale = {
                "mode": "static",
                "dynamic_weights": static_weights,
                "base_weights": static_weights,
            }
            logger.info(f"Static weights mode: {static_weights}")
            return static_weights, rationale
        
        # DYNAMIC mode: full richness-based scaling
        # Load current learned weights if no override provided
        if base_weights is None:
            base_weights = self._load_weights()

        # Compute richness scores
        richness = self._compute_richness(case_metadata)

        # Scale each agent's weight
        dynamic_weights = {}
        for agent_name, base_w in base_weights.items():
            agent_type = self._get_agent_type(agent_name)
            if agent_type == "radiologist":
                scale = self.SCALE_FLOOR + (1 - self.SCALE_FLOOR) * richness.radiology_richness
            elif agent_type == "pathologist":
                scale = self.SCALE_FLOOR + (1 - self.SCALE_FLOOR) * richness.pathology_richness
            else:
                scale = 1.0  # Unknown type → no scaling
            dynamic_weights[agent_name] = round(base_w * scale, 4)

        # Build rationale
        rationale = {
            "mode": "dynamic",
            **richness.to_dict(),
            "dynamic_weights": dynamic_weights,
            "base_weights": dict(base_weights),
            "scale_floor": self.SCALE_FLOOR,
        }

        logger.info(
            f"Dynamic weights computed: "
            f"rad_richness={richness.radiology_richness:.3f}, "
            f"path_richness={richness.pathology_richness:.3f}, "
            f"weights={dynamic_weights}"
        )

        return dynamic_weights, rationale

    def update_weights(self, findings: Dict[str, Any], ground_truth: int) -> Dict[str, float]:
        """
        Update agent base weights based on feedback (Continual Learning).

        Args:
            findings: Dict of agent findings (from compute_consensus) or 
                     raw list of finding dicts.
            ground_truth: The confirmed diagnosis (0 or 1).

        Returns:
            The new updated base weights.
        """
        # Skip update if reliability adjustment is disabled (ablation)
        if not self.use_reliability_adjustment:
            logger.info("Reliability adjustment disabled (ablation mode)")
            return self._load_weights()
        
        current_weights = self._load_weights()
        updated_weights = current_weights.copy()
        
        # Normalize findings input to list of dicts if needed
        # (Handling different formats the orchestrator might pass)
        agent_results = []
        if isinstance(findings, list):
            agent_results = findings
        elif isinstance(findings, dict) and "findings" in findings:
             agent_results = findings["findings"]
        else:
             # Try to parse if it's a dict mapping agent_name -> details
             for name, res in findings.items():
                 if isinstance(res, dict):
                     res['agent_name'] = name # Ensure name is present
                     agent_results.append(res)

        updates_log = []

        for finding in agent_results:
            agent_name = finding.get("agent_name") or finding.get("_agent_name")
            if not agent_name: 
                continue

            # Skip if agent not tracked in weights
            if agent_name not in updated_weights:
                continue

            # Determine correctness
            # We look at 'predicted_class' (0/1) or prob > 0.5
            pred_class = finding.get("predicted_class")
            prob = finding.get("probability")
            
            if pred_class is None and prob is not None:
                pred_class = 1 if prob >= 0.5 else 0
            
            if pred_class is None:
                continue

            # Update Rule:
            # Correct -> +LearningRate
            # Incorrect -> -LearningRate
            old_w = updated_weights[agent_name]
            
            if pred_class == ground_truth:
                new_w = min(MAX_WEIGHT, old_w + LEARNING_RATE)
                action = "reward"
            else:
                new_w = max(MIN_WEIGHT, old_w - LEARNING_RATE)
                action = "penalize"
            
            updated_weights[agent_name] = round(new_w, 4)
            updates_log.append(f"{agent_name}: {old_w} -> {new_w} ({action})")

        # Save updates
        self._save_weights(updated_weights)
        
        if updates_log:
            logger.info(f"Continual Learning Update (GT={ground_truth}): {', '.join(updates_log)}")
            
        return updated_weights

    # =========================================================================
    # RADIOLOGY RICHNESS
    # =========================================================================

    def _compute_radiology_richness(self, case_metadata: Dict[str, Any]) -> tuple:
        """
        Compute radiology information richness (0–1).
        
        Sub-components:
            1. Image count score: 0 images → 0, 1 → 0.5, 2+ → 1.0
            2. PA view score: 1.0 if PA/frontal view present, 0.3 otherwise
            3. Image quality score: average quality from std_intensity + edge_strength
        """
        components = {}

        # --- 1. Image count ---
        num_images = case_metadata.get("num_images", 0)
        if num_images == 0:
            components["image_count"] = 0.0
        elif num_images == 1:
            components["image_count"] = 0.5
        else:
            # Diminishing returns: 2→0.8, 3+→1.0
            components["image_count"] = min(1.0, 0.5 + 0.25 * num_images)

        # --- 2. PA view presence ---
        images_metadata = case_metadata.get("images_metadata", [])
        view_types = [
            img.get("view_type", "Unknown").upper()
            for img in images_metadata
        ]
        has_pa = any(vt in ("PA", "FRONTAL") for vt in view_types)
        components["pa_view"] = 1.0 if has_pa else 0.3

        # --- 3. Image quality ---
        quality_scores = []
        for img_meta in images_metadata:
            # Use shape as a proxy for resolution
            shape = img_meta.get("shape", (0, 0))
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                resolution_score = min(1.0, (shape[0] * shape[1]) / (512 * 512))
            else:
                resolution_score = 0.5
            quality_scores.append(resolution_score)

        if quality_scores:
            components["image_quality"] = sum(quality_scores) / len(quality_scores)
        else:
            components["image_quality"] = 0.0 if num_images == 0 else 0.5

        # Weighted combination
        richness = (
            self.RAD_WEIGHT_NUM_IMAGES * components["image_count"]
            + self.RAD_WEIGHT_PA_VIEW * components["pa_view"]
            + self.RAD_WEIGHT_IMAGE_QUALITY * components["image_quality"]
        )

        return min(1.0, max(0.0, richness)), components

    # =========================================================================
    # PATHOLOGY RICHNESS
    # =========================================================================

    def _compute_pathology_richness(self, case_metadata: Dict[str, Any]) -> tuple:
        """
        Compute pathology (report) information richness (0–1).
        
        Sub-components:
            1. Report length score: based on combined FINDINGS + IMPRESSION char count
            2. Entity count score: from NLP-extracted entities/measurements
            3. Section completeness: fraction of key sections that are non-empty
            4. Certainty signal: proportion of affirmed vs negated/uncertain
        """
        components = {}

        # --- 1. Report text length ---
        findings_text = case_metadata.get("findings", "") or ""
        impression_text = case_metadata.get("impression", "") or ""
        indication_text = case_metadata.get("indication", "") or ""
        combined_text = findings_text + " " + impression_text
        text_len = len(combined_text.strip())

        # Score: 0 → 0, 100 → 0.5, 300+ → 1.0
        if text_len == 0:
            components["report_length"] = 0.0
        elif text_len < 100:
            components["report_length"] = text_len / 200.0  # 0 to 0.5
        else:
            components["report_length"] = min(1.0, 0.5 + (text_len - 100) / 400.0)

        # --- 2. Entity count ---
        nlp_features = case_metadata.get("nlp_features", {}) or {}
        entities = nlp_features.get("entities", [])
        measurements = nlp_features.get("measurements", [])
        entity_count = len(entities) + len(measurements)

        # Score: 0 → 0, 1 → 0.3, 3 → 0.6, 5+ → 1.0
        if entity_count == 0:
            components["entity_count"] = 0.0
        else:
            components["entity_count"] = min(1.0, entity_count / 5.0)

        # --- 3. Section completeness ---
        key_sections = ["findings", "impression", "indication"]
        filled = sum(
            1 for s in key_sections
            if (case_metadata.get(s, "") or "").strip()
        )
        components["section_completeness"] = filled / len(key_sections)

        # --- 4. Certainty signal ---
        certainty = nlp_features.get("certainty", "affirmed")
        affirmed_count = nlp_features.get("affirmed_count", 0)
        negated_count = nlp_features.get("negated_count", 0)
        uncertain_count = nlp_features.get("uncertain_count", 0)
        total_mentions = affirmed_count + negated_count + uncertain_count

        if total_mentions > 0:
            # Higher proportion of affirmed → more useful signal
            components["certainty"] = affirmed_count / total_mentions
        elif certainty == "affirmed":
            components["certainty"] = 0.7  # Default: assume moderate confidence
        elif certainty == "negated":
            components["certainty"] = 0.3
        else:
            components["certainty"] = 0.5

        # Weighted combination
        richness = (
            self.PATH_WEIGHT_REPORT_LENGTH * components["report_length"]
            + self.PATH_WEIGHT_ENTITY_COUNT * components["entity_count"]
            + self.PATH_WEIGHT_SECTION_COMPLETENESS * components["section_completeness"]
            + self.PATH_WEIGHT_CERTAINTY * components["certainty"]
        )

        return min(1.0, max(0.0, richness)), components

    # =========================================================================
    # COMBINED
    # =========================================================================

    def _compute_richness(self, case_metadata: Dict[str, Any]) -> RichnessScores:
        """Compute both radiology and pathology richness for a case."""
        rad_richness, rad_components = self._compute_radiology_richness(case_metadata)
        path_richness, path_components = self._compute_pathology_richness(case_metadata)

        return RichnessScores(
            radiology_richness=rad_richness,
            pathology_richness=path_richness,
            radiology_components=rad_components,
            pathology_components=path_components,
        )

    @staticmethod
    def _get_agent_type(agent_name: str) -> str:
        """Determine agent type from name."""
        if "radiologist" in agent_name:
            return "radiologist"
        elif "pathologist" in agent_name:
            return "pathologist"
        return "unknown"
