"""
Architectural Claim Verification
================================

This module provides automated verification of architectural claims
based on ablation study results and statistical tests.

After all experiments are completed, this module explicitly verifies:
1. Does multi-agent consensus outperform single best agent?
2. Does dynamic weighting outperform static weighting?
3. Does Prolog outperform pure Python aggregation?
4. Does dependency parsing outperform regex-only NLP?
5. Does NegEx improve measurable performance?
6. Does the full system outperform majority baseline?

If any answer is "no," the corresponding claim must be weakened or removed.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

from config import RESULTS_DIR, ensure_directories, SIGNIFICANCE_LEVEL
from evaluation.statistical_tests import (
    TestResult, ComparisonResult, compare_models,
    bootstrap_confidence_interval, cohens_d, interpret_effect_size
)

logger = logging.getLogger(__name__)


# =============================================================================
# CLAIM DEFINITIONS
# =============================================================================

class ClaimStatus(Enum):
    """Status of a verified claim."""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    NOT_TESTED = "not_tested"


@dataclass
class VerifiedClaim:
    """Result of verifying a single architectural claim."""
    claim_id: str
    description: str
    status: ClaimStatus
    
    # Evidence
    baseline_value: float
    comparison_value: float
    metric_name: str
    difference: float
    
    # Statistical validation
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    
    # Interpretation
    recommendation: str
    evidence_strength: str  # strong, moderate, weak, none
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "description": self.description,
            "status": self.status.value,
            "baseline_value": self.baseline_value,
            "comparison_value": self.comparison_value,
            "metric_name": self.metric_name,
            "difference": self.difference,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "effect_size": self.effect_size,
            "recommendation": self.recommendation,
            "evidence_strength": self.evidence_strength
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        status_icon = {
            ClaimStatus.SUPPORTED: "✓",
            ClaimStatus.NOT_SUPPORTED: "✗",
            ClaimStatus.INCONCLUSIVE: "?",
            ClaimStatus.NOT_TESTED: "-"
        }
        return (f"[{status_icon[self.status]}] {self.claim_id}: {self.description}\n"
                f"    {self.metric_name}: {self.baseline_value:.4f} → {self.comparison_value:.4f} "
                f"(Δ={self.difference:+.4f}, p={self.p_value:.4f})\n"
                f"    Evidence: {self.evidence_strength}, Effect: d={self.effect_size:.3f}\n"
                f"    → {self.recommendation}")


# =============================================================================
# CLAIM DEFINITIONS
# =============================================================================

ARCHITECTURAL_CLAIMS = {
    "multi_agent_vs_single": {
        "description": "Multi-agent consensus outperforms single best agent",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["single_R_densenet", "single_R_resnet", "single_R_rulebased",
                                  "single_P_regex", "single_P_spacy", "single_P_context"],
        "comparison_type": "best_single",  # Compare against best single agent
        "metric": "accuracy",
        "expected_direction": "higher"  # full_system should be higher
    },
    
    "dynamic_vs_static_weights": {
        "description": "Dynamic weighting outperforms static weighting",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["static_weights"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    },
    
    "prolog_vs_python": {
        "description": "Prolog consensus outperforms pure Python aggregation",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["python_consensus"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    },
    
    "dependency_vs_regex": {
        "description": "Dependency parsing outperforms regex-only NLP",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["regex_only"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    },
    
    "negex_improvement": {
        "description": "NegEx improves measurable performance",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["no_negex"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    },
    
    "system_vs_majority": {
        "description": "Full system outperforms majority class baseline",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["majority_class"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    },
    
    "ensemble_vs_vote": {
        "description": "Weighted ensemble outperforms unweighted majority vote",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["equal_weights"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    },
    
    "disagreement_resolution": {
        "description": "Disagreement resolution improves performance",
        "baseline_ablation": "full_system",
        "comparison_ablations": ["no_disagreement_resolution"],
        "comparison_type": "single",
        "metric": "accuracy",
        "expected_direction": "higher"
    }
}


# =============================================================================
# CLAIM VERIFIER
# =============================================================================

class ClaimVerifier:
    """
    Automated verifier for architectural claims.
    
    Uses ablation study results to statistically verify or refute
    each architectural claim.
    """
    
    def __init__(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        predictions: Dict[str, List[int]] = None,
        y_true: List[int] = None
    ):
        """
        Initialize verifier.
        
        Args:
            ablation_results: Dict mapping ablation name to metrics dict
            predictions: Dict mapping ablation name to predictions list
            y_true: Ground truth labels
        """
        self.ablation_results = ablation_results
        self.predictions = predictions or {}
        self.y_true = y_true
    
    def verify_claim(
        self,
        claim_id: str,
        claim_def: Dict[str, Any]
    ) -> VerifiedClaim:
        """
        Verify a single claim based on ablation results.
        
        Args:
            claim_id: Claim identifier
            claim_def: Claim definition dictionary
            
        Returns:
            VerifiedClaim with verification results
        """
        description = claim_def["description"]
        baseline_name = claim_def["baseline_ablation"]
        comparison_names = claim_def["comparison_ablations"]
        comparison_type = claim_def["comparison_type"]
        metric = claim_def["metric"]
        expected_direction = claim_def["expected_direction"]
        
        # Get baseline metrics
        baseline_metrics = self.ablation_results.get(baseline_name, {})
        baseline_value = baseline_metrics.get(metric)
        
        if baseline_value is None:
            return VerifiedClaim(
                claim_id=claim_id,
                description=description,
                status=ClaimStatus.NOT_TESTED,
                baseline_value=0.0,
                comparison_value=0.0,
                metric_name=metric,
                difference=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                recommendation="Run baseline ablation first",
                evidence_strength="none"
            )
        
        # Get comparison value(s)
        comparison_values = []
        for comp_name in comparison_names:
            comp_metrics = self.ablation_results.get(comp_name, {})
            comp_value = comp_metrics.get(metric)
            if comp_value is not None:
                comparison_values.append((comp_name, comp_value))
        
        if not comparison_values:
            return VerifiedClaim(
                claim_id=claim_id,
                description=description,
                status=ClaimStatus.NOT_TESTED,
                baseline_value=baseline_value,
                comparison_value=0.0,
                metric_name=metric,
                difference=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                recommendation="Run comparison ablations first",
                evidence_strength="none"
            )
        
        # Determine comparison value based on type
        if comparison_type == "best_single":
            # Compare against best of the single-agent baselines
            best_comp_name, comparison_value = max(comparison_values, key=lambda x: x[1])
        elif comparison_type == "worst_single":
            # Compare against worst
            best_comp_name, comparison_value = min(comparison_values, key=lambda x: x[1])
        else:
            # Single comparison
            best_comp_name, comparison_value = comparison_values[0]
        
        # Compute difference
        # For "higher" direction: positive diff means baseline is better
        # For "lower" direction: negative diff means baseline is better
        difference = baseline_value - comparison_value
        
        # Determine if claim is supported
        if expected_direction == "higher":
            claim_supported = difference > 0
        else:
            claim_supported = difference < 0
        
        # Statistical test (if predictions available)
        p_value = 1.0
        effect_size = 0.0
        ci = (difference, difference)
        
        baseline_preds = self.predictions.get(baseline_name)
        comp_preds = self.predictions.get(best_comp_name)
        
        if baseline_preds and comp_preds and self.y_true:
            try:
                comparison_result = compare_models(
                    self.y_true,
                    comp_preds,
                    baseline_preds,
                    model_a_name=best_comp_name,
                    model_b_name=baseline_name
                )
                
                # Get p-value from tests
                for test in comparison_result.test_results:
                    if test.p_value < p_value:
                        p_value = test.p_value
                        if test.effect_size:
                            effect_size = test.effect_size
                        if test.confidence_interval:
                            ci = test.confidence_interval
            except Exception as e:
                logger.warning(f"Statistical test failed for {claim_id}: {e}")
        
        # Determine status based on significance and direction
        if p_value < SIGNIFICANCE_LEVEL:
            if claim_supported:
                status = ClaimStatus.SUPPORTED
            else:
                status = ClaimStatus.NOT_SUPPORTED
        else:
            if abs(difference) < 0.01:
                status = ClaimStatus.INCONCLUSIVE
            elif claim_supported:
                status = ClaimStatus.INCONCLUSIVE  # Trend but not significant
            else:
                status = ClaimStatus.NOT_SUPPORTED
        
        # Evidence strength
        if status == ClaimStatus.SUPPORTED:
            if p_value < 0.001:
                evidence_strength = "strong"
            elif p_value < 0.01:
                evidence_strength = "moderate"
            else:
                evidence_strength = "weak"
        else:
            evidence_strength = "none"
        
        # Generate recommendation
        if status == ClaimStatus.SUPPORTED:
            recommendation = f"Claim supported: {description}"
        elif status == ClaimStatus.NOT_SUPPORTED:
            recommendation = f"REVISE OR REMOVE: Evidence does not support '{description}'"
        else:
            recommendation = f"INCONCLUSIVE: More data needed to verify '{description}'"
        
        return VerifiedClaim(
            claim_id=claim_id,
            description=description,
            status=status,
            baseline_value=baseline_value,
            comparison_value=comparison_value,
            metric_name=metric,
            difference=difference,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            recommendation=recommendation,
            evidence_strength=evidence_strength
        )
    
    def verify_all_claims(self) -> Dict[str, VerifiedClaim]:
        """Verify all architectural claims."""
        results = {}
        for claim_id, claim_def in ARCHITECTURAL_CLAIMS.items():
            results[claim_id] = self.verify_claim(claim_id, claim_def)
        return results


# =============================================================================
# VERIFICATION REPORT
# =============================================================================

@dataclass
class VerificationReport:
    """Complete verification report for all claims."""
    verified_claims: Dict[str, VerifiedClaim]
    summary_stats: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified_claims": {k: v.to_dict() for k, v in self.verified_claims.items()},
            "summary_stats": self.summary_stats
        }
    
    def save(self, output_path: Path = None) -> Path:
        """Save report to JSON."""
        ensure_directories()
        output_path = output_path or RESULTS_DIR / "claim_verification.json"
        
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return output_path
    
    def generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = ["# Architectural Claim Verification Report", ""]
        lines.append("## Summary")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status, count in self.summary_stats.items():
            lines.append(f"| {status} | {count} |")
        lines.append("")
        
        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")
        
        for claim_id, claim in self.verified_claims.items():
            status_icon = {
                ClaimStatus.SUPPORTED: "✅",
                ClaimStatus.NOT_SUPPORTED: "❌",
                ClaimStatus.INCONCLUSIVE: "⚠️",
                ClaimStatus.NOT_TESTED: "⏸️"
            }.get(claim.status, "?")
            
            lines.append(f"### {status_icon} {claim.claim_id}")
            lines.append("")
            lines.append(f"**Claim:** {claim.description}")
            lines.append("")
            lines.append(f"| Metric | Baseline | Comparison | Difference | p-value |")
            lines.append(f"|--------|----------|------------|------------|---------|")
            lines.append(f"| {claim.metric_name} | {claim.baseline_value:.4f} | "
                        f"{claim.comparison_value:.4f} | {claim.difference:+.4f} | "
                        f"{claim.p_value:.4f} |")
            lines.append("")
            lines.append(f"**Evidence Strength:** {claim.evidence_strength}")
            lines.append("")
            lines.append(f"**Effect Size:** Cohen's d = {claim.effect_size:.3f} "
                        f"({interpret_effect_size(claim.effect_size)})")
            lines.append("")
            lines.append(f"**Recommendation:** {claim.recommendation}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Actionable items
        lines.append("## Action Items")
        lines.append("")
        
        not_supported = [c for c in self.verified_claims.values() 
                        if c.status == ClaimStatus.NOT_SUPPORTED]
        if not_supported:
            lines.append("### Claims to Revise or Remove")
            lines.append("")
            for claim in not_supported:
                lines.append(f"- **{claim.claim_id}**: {claim.description}")
            lines.append("")
        
        inconclusive = [c for c in self.verified_claims.values() 
                       if c.status == ClaimStatus.INCONCLUSIVE]
        if inconclusive:
            lines.append("### Claims Needing More Evidence")
            lines.append("")
            for claim in inconclusive:
                lines.append(f"- **{claim.claim_id}**: {claim.description}")
            lines.append("")
        
        return "\n".join(lines)


def generate_verification_report(
    ablation_results: Dict[str, Dict[str, float]],
    predictions: Dict[str, List[int]] = None,
    y_true: List[int] = None
) -> VerificationReport:
    """
    Generate a complete claim verification report.
    
    Args:
        ablation_results: Dict mapping ablation names to metrics
        predictions: Dict mapping ablation names to predictions
        y_true: Ground truth labels
        
    Returns:
        VerificationReport with all verified claims
    """
    verifier = ClaimVerifier(ablation_results, predictions, y_true)
    verified_claims = verifier.verify_all_claims()
    
    # Compute summary stats
    summary_stats = {
        "supported": 0,
        "not_supported": 0,
        "inconclusive": 0,
        "not_tested": 0
    }
    
    for claim in verified_claims.values():
        if claim.status == ClaimStatus.SUPPORTED:
            summary_stats["supported"] += 1
        elif claim.status == ClaimStatus.NOT_SUPPORTED:
            summary_stats["not_supported"] += 1
        elif claim.status == ClaimStatus.INCONCLUSIVE:
            summary_stats["inconclusive"] += 1
        else:
            summary_stats["not_tested"] += 1
    
    report = VerificationReport(
        verified_claims=verified_claims,
        summary_stats=summary_stats
    )
    
    # Log summary
    logger.info("Claim Verification Summary:")
    logger.info(f"  Supported: {summary_stats['supported']}")
    logger.info(f"  Not Supported: {summary_stats['not_supported']}")
    logger.info(f"  Inconclusive: {summary_stats['inconclusive']}")
    logger.info(f"  Not Tested: {summary_stats['not_tested']}")
    
    return report


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Run claim verification from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify architectural claims")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                       help="Directory containing ablation results")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "claim_verification.md"),
                       help="Output file for verification report")
    args = parser.parse_args()
    
    # Load ablation results
    results_path = Path(args.results_dir) / "ablation_results"
    ablation_results = {}
    
    for result_file in results_path.glob("*_results.json"):
        with open(result_file) as f:
            data = json.load(f)
            if "baseline" in data:
                ablation_results[data["baseline"]["config"]["name"]] = data["baseline"]["metrics"]
            if "ablations" in data:
                for name, ablation in data["ablations"].items():
                    ablation_results[name] = ablation["metrics"]
    
    if not ablation_results:
        logger.error("No ablation results found. Run ablation studies first.")
        return
    
    # Generate report
    report = generate_verification_report(ablation_results)
    
    # Save report
    report.save()
    
    # Save markdown
    md_path = Path(args.output)
    with open(md_path, "w") as f:
        f.write(report.generate_markdown())
    
    logger.info(f"Verification report saved to {md_path}")
    
    # Print summary
    print(report.generate_markdown())


if __name__ == "__main__":
    main()
