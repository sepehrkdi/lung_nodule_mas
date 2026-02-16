#!/usr/bin/env python3
"""
Extended Multi-Agent System - 6 Specialized Agents
==================================================

EDUCATIONAL PURPOSE:

This module demonstrates an extended Multi-Agent System with:
- 3 Radiologist agents (DenseNet, ResNet, Rule-based)
- 3 Pathologist agents (Regex, spaCy NER, Context Analyzer)
- Prolog-based weighted consensus

ARCHITECTURE:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    EXTENDED MAS ARCHITECTURE                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   RADIOLOGIST AGENTS (Image Analysis)                          │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
    │   │  DenseNet121 │  │   ResNet50   │  │  Rule-Based  │         │
    │   │   W = 1.0    │  │   W = 1.0    │  │   W = 0.7    │         │
    │   │   (Deep CNN) │  │  (Deep CNN)  │  │ (Heuristics) │         │
    │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
    │          │                 │                  │                 │
    │          └────────────────┬┴──────────────────┘                │
    │                           │                                     │
    │   PATHOLOGIST AGENTS (Report Analysis)                         │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
    │   │ Regex-Based  │  │  spaCy NER   │  │   Context    │         │
    │   │   W = 0.8    │  │   W = 0.9    │  │   W = 0.9    │         │
    │   │(Pattern Match)│  │(Statistical) │  │ (Negation)   │         │
    │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
    │          │                 │                  │                 │
    │          └────────────────┬┴──────────────────┘                │
    │                           │                                     │
    │   ┌───────────────────────┴───────────────────────┐            │
    │   │              PROLOG CONSENSUS                  │            │
    │   │   • Weighted voting (agent expertise weights)  │            │
    │   │   • Disagreement detection & resolution        │            │
    │   │   • Confidence interval computation            │            │
    │   │   • Dempster-Shafer evidence combination       │            │
    │   └───────────────────────┬───────────────────────┘            │
    │                           │                                     │
    │                    ┌──────┴──────┐                             │
    │                    │   FINAL     │                             │
    │                    │ DIAGNOSIS   │                             │
    │                    │ + LUNG-RADS │                             │
    │                    └─────────────┘                             │
    └─────────────────────────────────────────────────────────────────┘

AGENT DIVERSITY RATIONALE (Academic Justification):

1. RadiologistDenseNet vs RadiologistResNet:
   - Different CNN architectures capture different visual features
   - Ensemble of diverse architectures improves robustness
   - Demonstrates deep learning approach diversity

2. RadiologistRuleBased:
   - Interpretable baseline for comparison
   - Follows clinical guidelines (Lung-RADS)
   - Provides sanity check for ML predictions

3. PathologistRegex vs PathologistSpacy:
   - Regex: Fast, explicit patterns, fully interpretable
   - spaCy: Robust to variations, contextual understanding
   - Demonstrates symbolic vs statistical NLP tradeoff

Usage:
    python main_extended.py                    # Run with sample data
    python main_extended.py --demo             # Quick demo
    python main_extended.py --evaluate         # Full evaluation
    python main_extended.py --export results.json

References:
- Multi-Agent Systems: Wooldridge & Jennings (1995)
- BDI Architecture: Rao & Georgeff (1991)
- Lung-RADS: ACR (2022)
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Result from analyzing a single case with all 6 agents."""
    nodule_id: str
    ground_truth: Optional[int]
    final_class: int
    final_probability: float
    confidence: float
    agreement_level: str
    lung_rads: str
    recommendation: str
    processing_time: float
    
    # Individual agent findings
    radiologist_densenet: Dict[str, Any] = field(default_factory=dict)
    radiologist_resnet: Dict[str, Any] = field(default_factory=dict)
    radiologist_rulebased: Dict[str, Any] = field(default_factory=dict)
    pathologist_regex: Dict[str, Any] = field(default_factory=dict)
    pathologist_spacy: Dict[str, Any] = field(default_factory=dict)
    pathologist_context: Dict[str, Any] = field(default_factory=dict)


class ExtendedMAS:
    """
    Extended Multi-Agent System with 6 Specialized Agents.
    
    EDUCATIONAL PURPOSE:
    Demonstrates agent diversity, weighted consensus, and
    Prolog-based decision fusion in medical AI.
    """
    
    def __init__(
        self,
        data_source: str = "nlmcxr",
        verbose: bool = True,
        max_cases: Optional[int] = 100,
        start_index: int = 0,
        no_filter: bool = False,
        weight_mode: str = "dynamic",
        consensus_mode: str = "prolog",
        use_negex: bool = True,
        use_dependency_parsing: bool = True
    ):
        """
        Initialize the extended MAS.
        
        Args:
            data_source: "nlmcxr" or "sample"
            verbose: Print progress messages
            max_cases: Maximum number of cases to load (default: 100)
            start_index: Offset for loading cases (default: 0)
            no_filter: If True, disable NLP richness filtering
            weight_mode: "dynamic", "static", or "equal"
            consensus_mode: "prolog" or "python"
            use_negex: Enable NegEx negation detection
            use_dependency_parsing: Enable dependency parsing
        """
        self.verbose = verbose
        self.data_source = data_source
        self.max_cases = max_cases if max_cases is not None else 100
        self.start_index = start_index
        
        # Ablation flags
        self.no_filter = no_filter
        self.weight_mode = weight_mode
        self.consensus_mode = consensus_mode
        self.use_negex = use_negex
        self.use_dependency_parsing = use_dependency_parsing
        
        # Import orchestrator
        from orchestrator import MultiAgentOrchestrator
        
        # Initialize orchestrator with all agents
        self._log("Initializing 6-Agent Orchestrator...")
        self._log(f"  - NLP filter: {'DISABLED' if no_filter else 'enabled (score >= 3.0)'}")
        self._log(f"  - Weight mode: {weight_mode}")
        self._log(f"  - Consensus: {consensus_mode}")
        self._log(f"  - NegEx: {'enabled' if use_negex else 'DISABLED'}")
        self._log(f"  - Dependency parsing: {'enabled' if use_dependency_parsing else 'DISABLED'}")
        
        kb_path = Path(__file__).parent / "knowledge" / "multi_agent_consensus.pl"
        self.orchestrator = MultiAgentOrchestrator(
            consensus_kb_path=str(kb_path) if kb_path.exists() else None,
            weight_mode=weight_mode
        )
        
        # Load data
        self._load_data()
        
        # Results storage
        self.results: List[CaseResult] = []
        
        self._log("Extended MAS initialized with 6 specialized agents")
    
    def _log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            logger.info(f"[ExtendedMAS] {message}")
    
    def _load_data(self) -> None:
        """Load case data based on source, using NLP richness filtering."""
        self.cases = []
        
        if self.data_source == "nlmcxr":
            try:
                from data.nlmcxr_loader import NLMCXRLoader
                loader = NLMCXRLoader()
                
                # Support both filtered and unfiltered evaluation
                if self.no_filter:
                    # NO FILTER MODE: Evaluate ALL cases with valid labels
                    # This is important for evaluation integrity
                    self._log("Loading ALL labeled cases (NLP filtering DISABLED)")
                    case_ids = loader.get_all_labeled_case_ids(
                        limit=self.max_cases,
                        offset=self.start_index
                    )
                else:
                    # Use NLP richness filtering (default)
                    # Selects cases where the radiology report contains enough
                    # extractable content for pathologist agents
                    case_ids = loader.get_nlp_rich_case_ids(
                        min_score=3.0,
                        limit=self.max_cases,
                        offset=self.start_index,
                        no_filter=False
                    )
                
                count = 0
                for case_id in case_ids:
                    if count >= self.max_cases:
                        break
                    
                    try:
                        images, metadata = loader.load_case(case_id)
                        
                        # Build case dict with real report
                        findings = metadata.get("findings", "")
                        impression = metadata.get("impression", "")
                        
                        # METHODOLOGY UPDATE: Agents see ONLY findings
                        report = f"FINDINGS: {findings}"
                        
                        # Impression is used internally for GT mainly, but we keep it in metadata
                        # if needed for debugging, but 'report' passed to agents is stripped.
                        
                        ground_truth = metadata.get("ground_truth", -1)
                        
                        self.cases.append({
                            "nodule_id": case_id,
                            "features": metadata.get("nlp_features", {}),
                            "ground_truth": ground_truth,
                            "report": report,
                            "images": images,
                            "image_metadata": metadata.get("images_metadata", [])
                        })
                        count += 1
                        self._log(f"Added case {case_id} (count: {count}/{self.max_cases})")
                            
                    except Exception as e:
                        self._log(f"Failed to load case {case_id}: {e}")
                        continue
                        
                self._log(f"Loaded {len(self.cases)} NLP-rich cases from NLMCXR (offset {self.start_index})")
            except Exception as e:
                self._log(f"Failed to load NLMCXR: {e}, using sample data")
                self._create_sample_cases()
        else:
            self._create_sample_cases()
            # Apply slicing for sample cases
            start = self.start_index
            end = start + self.max_cases
            self.cases = self.cases[start:end]
            self._log(f"Selected {len(self.cases)} sample cases (offset {start})")
    
    def _create_sample_cases(self) -> None:
        """Create sample cases for testing."""
        sample_nodules = [
            {
                "nodule_id": "sample_001",
                "features": {
                    "size_mm": 5, "texture": "ground-glass",
                    "location": "left_upper_lobe"
                },
                "ground_truth": 0,  # Normal
                "report": "FINDINGS: Small ground-glass opacity in left upper lobe. Likely benign.\n\nIMPRESSION: Probably benign nodule."
            },
            {
                "nodule_id": "sample_002",
                "features": {
                    "size_mm": 12, "texture": "solid",
                    "location": "right_upper_lobe"
                },
                "ground_truth": 1,  # Abnormal
                "report": "FINDINGS: Solid nodule in right upper lobe, spiculated margins.\n\nIMPRESSION: Suspicious for malignancy."
            },
            {
                "nodule_id": "sample_003",
                "features": {
                    "size_mm": 8, "texture": "part-solid",
                    "location": "right_lower_lobe"
                },
                "ground_truth": -1,  # Indeterminate
                "report": "FINDINGS: Part-solid nodule in right lower lobe.\n\nIMPRESSION: Indeterminate, recommend follow-up."
            },
            {
                "nodule_id": "sample_004",
                "features": {
                    "size_mm": 20, "texture": "solid",
                    "location": "right_upper_lobe"
                },
                "ground_truth": 1,  # Abnormal
                "report": "FINDINGS: Large spiculated mass in right upper lobe.\n\nIMPRESSION: Highly suspicious for malignancy."
            },
            {
                "nodule_id": "sample_005",
                "features": {
                    "size_mm": 4, "texture": "solid",
                    "location": "left_lower_lobe"
                },
                "ground_truth": 0,  # Normal
                "report": "FINDINGS: Tiny calcified nodule in left lower lobe.\n\nIMPRESSION: Benign calcified granuloma."
            },
        ]
        
        self.cases = sample_nodules
        self._log(f"Created {len(self.cases)} sample cases")
    
    def _prob_to_lungrads(self, prob: float, size_mm: float = 10) -> str:
        """Convert probability and size to Lung-RADS category."""
        if prob < 0.15 or size_mm < 3:
            return "1"  # Negative
        elif prob < 0.3 or size_mm < 6:
            return "2"  # Benign
        elif prob < 0.5 or size_mm < 8:
            return "3"  # Probably benign
        elif prob < 0.7 or size_mm < 15:
            return "4A"  # Suspicious
        elif prob < 0.85:
            return "4B"  # Very suspicious
        else:
            return "4X"  # Additional features
    
    def _get_recommendation(self, lung_rads: str, prob: float) -> str:
        """Get clinical recommendation based on Lung-RADS."""
        recommendations = {
            "1": "Continue annual screening",
            "2": "Continue annual screening",
            "3": "Follow-up CT in 6 months",
            "4A": "Follow-up CT in 3 months or PET-CT",
            "4B": "Tissue sampling (biopsy) or PET-CT",
            "4X": "Additional imaging and/or tissue sampling"
        }
        return recommendations.get(lung_rads, "Clinical correlation recommended")
    
    async def process_single_case(
        self,
        case: Dict[str, Any]
    ) -> CaseResult:
        """Process a single case with all 6 agents."""
        start_time = time.time()
        
        nodule_id = case.get("nodule_id", "unknown")
        features = case.get("features", {})
        report = case.get("report", "")
        ground_truth = case.get("ground_truth")
        
        self._log(f"Processing case {nodule_id}...")
        
        # Run orchestrator
        consensus = await self.orchestrator.analyze_case(
            case_id=nodule_id,
            features=features,
            report=report,
            image_array=case.get("images"),
            image_metadata=case.get("image_metadata")
        )
        
        # Extract individual findings
        size_mm = features.get("size_mm")
        if size_mm is None:
            size_mm = 10  # Fallback only for Lung-RADS display
        lung_rads = self._prob_to_lungrads(consensus.final_probability, size_mm)
        recommendation = self._get_recommendation(lung_rads, consensus.final_probability)
        
        # Build result
        result = CaseResult(
            nodule_id=nodule_id,
            ground_truth=ground_truth,
            final_class=consensus.final_class,
            final_probability=consensus.final_probability,
            confidence=consensus.confidence,
            agreement_level=consensus.agreement_level,
            lung_rads=lung_rads,
            recommendation=recommendation,
            processing_time=time.time() - start_time
        )
        
        # Extract individual agent findings
        for finding in consensus.radiologist_findings:
            if "densenet" in finding.agent_name:
                result.radiologist_densenet = finding.details
            elif "resnet" in finding.agent_name:
                result.radiologist_resnet = finding.details
            elif "rulebased" in finding.agent_name:
                result.radiologist_rulebased = finding.details
        
        for finding in consensus.pathologist_findings:
            if "regex" in finding.agent_name:
                result.pathologist_regex = finding.details
            elif "spacy" in finding.agent_name:
                result.pathologist_spacy = finding.details
        
        self.results.append(result)
        return result
    
    async def run_evaluation(
        self,
        max_cases: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on all cases.
        
        Returns:
            Dictionary with evaluation metrics
        """
        cases_to_process = self.cases[:max_cases] if max_cases else self.cases
        
        self._log(f"Starting evaluation on {len(cases_to_process)} cases...")
        
        for i, case in enumerate(cases_to_process):
            self._log(f"Case {i+1}/{len(cases_to_process)}")
            await self.process_single_case(case)
        
        # Compute metrics
        return self._compute_metrics()

    async def run_cross_validation(
        self,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Run k-fold cross-validation evaluation.
        
        Args:
            n_folds: Number of folds (default: 5)
            
        Returns:
            Dictionary with CV metrics (mean ± std)
        """
        try:
            from evaluation.cross_validation import CrossValidationEvaluator
        except ImportError:
            self._log("Cross-validation module not available, running single evaluation")
            return await self.run_evaluation()
        
        self._log(f"Starting {n_folds}-fold cross-validation...")
        
        # Prepare data
        case_ids = [case["nodule_id"] for case in self.cases]
        y_true = [case.get("ground_truth", -1) for case in self.cases]
        
        # Filter to valid labels
        valid_indices = [i for i, y in enumerate(y_true) if y in (0, 1)]
        case_ids = [case_ids[i] for i in valid_indices]
        y_true = [y_true[i] for i in valid_indices]
        
        if len(case_ids) < n_folds:
            self._log(f"Not enough cases ({len(case_ids)}) for {n_folds}-fold CV")
            return await self.run_evaluation()
        
        # Create evaluator (stratification is built-in via StratifiedKFoldSplitter)
        cv_evaluator = CrossValidationEvaluator(
            n_folds=n_folds,
            random_state=42,
            save_results=False  # Don't auto-save during interactive runs
        )
        
        # Run CV folds
        fold_metrics = []
        for fold_idx, (train_ids, test_ids) in enumerate(
            cv_evaluator.splitter.split(case_ids, y_true)
        ):
            self._log(f"Fold {fold_idx + 1}/{n_folds}: {len(train_ids)} train, {len(test_ids)} test")
            
            # Process test cases for this fold
            test_cases = [self.cases[valid_indices[i]] for i in test_ids]
            
            # Store current results and run on fold test set
            original_results = self.results.copy()
            self.results = []
            
            for case in test_cases:
                await self.process_single_case(case)
            
            # Compute fold metrics
            fold_result = self._compute_metrics()
            fold_metrics.append(fold_result)
            
            # Restore results
            self.results = original_results
        
        # Aggregate CV results
        cv_results = {
            "n_folds": n_folds,
            "total_cases": len(case_ids),
            "fold_metrics": fold_metrics
        }
        
        # Compute mean ± std for key metrics
        if fold_metrics:
            accuracies = [f.get("accuracy", 0) for f in fold_metrics]
            cv_results["accuracy_mean"] = np.mean(accuracies)
            cv_results["accuracy_std"] = np.std(accuracies)
            cv_results["accuracy_ci95"] = (
                np.mean(accuracies) - 1.96 * np.std(accuracies) / np.sqrt(n_folds),
                np.mean(accuracies) + 1.96 * np.std(accuracies) / np.sqrt(n_folds)
            )
        
        self._log(f"CV Complete: Accuracy = {cv_results.get('accuracy_mean', 0):.3f} ± {cv_results.get('accuracy_std', 0):.3f}")
        
        return cv_results
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if not self.results:
            return {}
        
        results_with_gt = [r for r in self.results if r.ground_truth is not None]
        
        if not results_with_gt:
            return {
                "total_cases": len(self.results),
                "cases_with_ground_truth": 0,
                "note": "No ground truth available for metrics"
            }
        
        # Accuracy
        correct = sum(
            1 for r in results_with_gt
            if r.final_class == r.ground_truth
        )
        accuracy = correct / len(results_with_gt)
        
        # Mean absolute error
        mae = np.mean([
            abs(r.final_class - r.ground_truth)
            for r in results_with_gt
        ])
        
        # Agreement statistics
        agreement_counts = {
            "unanimous": 0,
            "majority": 0,
            "split": 0
        }
        for r in self.results:
            if r.agreement_level in agreement_counts:
                agreement_counts[r.agreement_level] += 1
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in self.results])
        
        # Average processing time
        avg_time = np.mean([r.processing_time for r in self.results])
        
        metrics = {
            "total_cases": len(self.results),
            "cases_with_ground_truth": len(results_with_gt),
            "accuracy": accuracy,
            "mean_absolute_error": mae,
            "average_confidence": avg_confidence,
            "average_processing_time_sec": avg_time,
            "agreement_statistics": agreement_counts,
            "agent_statistics": self.orchestrator.get_statistics()
        }
        
        return metrics
    
    def print_results_summary(self) -> None:
        """Print a summary of all results."""
        print("\n" + "=" * 70)
        print("EXTENDED MAS - 6-AGENT RESULTS SUMMARY")
        print("=" * 70)
        
        for r in self.results:
            gt_str = f"GT: {'Malignant' if r.ground_truth == 1 else 'Benign' if r.ground_truth == 0 else 'N/A'}" if r.ground_truth is not None else "GT: N/A"
            class_label = "Malignant" if r.final_class == 1 else "Benign"
            print(f"\n{r.nodule_id}: {class_label} "
                  f"(prob={r.final_probability:.3f}) | {gt_str}")
            print(f"  Lung-RADS: {r.lung_rads} | {r.recommendation}")
            print(f"  Agreement: {r.agreement_level} | Confidence: {r.confidence:.3f}")
            print(f"  Time: {r.processing_time:.2f}s")
        
        # Print metrics
        metrics = self._compute_metrics()
        
        print("\n" + "-" * 50)
        print("EVALUATION METRICS:")
        print("-" * 50)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
    
    def export_results(self, output_path: str) -> None:
        """Export results to JSON file."""
        export_data = {
            "metadata": {
                "system": "Extended MAS - 5 Specialized Agents",
                "agents": [
                    "RadiologistDenseNet", "RadiologistResNet", "RadiologistRuleBased",
                    "PathologistRegex", "PathologistSpacy"
                ],
                "data_source": self.data_source
            },
            "metrics": self._compute_metrics(),
            "results": [
                {
                    "nodule_id": r.nodule_id,
                    "ground_truth": r.ground_truth,
                    "final_class": r.final_class,
                    "final_probability": r.final_probability,
                    "confidence": r.confidence,
                    "agreement_level": r.agreement_level,
                    "lung_rads": r.lung_rads,
                    "recommendation": r.recommendation,
                    "processing_time": r.processing_time,
                    "agent_findings": {
                        "radiologist_densenet": r.radiologist_densenet,
                        "radiologist_resnet": r.radiologist_resnet,
                        "radiologist_rulebased": r.radiologist_rulebased,
                        "pathologist_regex": r.pathologist_regex,
                        "pathologist_spacy": r.pathologist_spacy
                    }
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self._log(f"Results exported to {output_path}")


async def demo_mode():
    """Quick demonstration of the 6-agent system."""
    print("\n" + "=" * 70)
    print("EXTENDED MAS DEMO - 6 Specialized Agents")
    print("=" * 70)
    
    print("\nArchitecture:")
    print("  Radiologists (3): DenseNet121, ResNet50, Rule-based")
    print("  Pathologists (3): Regex-based, spaCy NER, Context Analyzer")
    print("  Consensus: Prolog weighted voting")
    
    print("\nInitializing system...")
    
    mas = ExtendedMAS(data_source="sample", verbose=True)
    
    print("\nProcessing sample cases...")
    
    for case in mas.cases[:2]:
        result = await mas.process_single_case(case)
        
        print(f"\n--- Case: {result.nodule_id} ---")
        print(f"Final Class: {'Malignant' if result.final_class == 1 else 'Benign'} "
              f"(probability: {result.final_probability:.3f})")
        print(f"Agreement: {result.agreement_level}")
        print(f"Lung-RADS: {result.lung_rads}")
        print(f"Recommendation: {result.recommendation}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extended Multi-Agent System with 6 Specialized Agents"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run quick demo mode"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run full evaluation"
    )
    parser.add_argument(
        "--data", type=str, default="sample",
        choices=["sample", "nlmcxr"],
        help="Data source to use"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Maximum number of cases to process"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Start index for processing cases (offset)"
    )
    
    # =========================================================================
    # ABLATION STUDY FLAGS
    # =========================================================================
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Disable NLP richness filtering (evaluate ALL cases)"
    )
    parser.add_argument(
        "--weight-mode", type=str, default="dynamic",
        choices=["dynamic", "static", "equal"],
        help="Weighting mode: dynamic (learned), static (fixed), equal (uniform)"
    )
    parser.add_argument(
        "--consensus", type=str, default="prolog",
        choices=["prolog", "python"],
        help="Consensus engine: prolog (symbolic) or python (ablation baseline)"
    )
    parser.add_argument(
        "--no-negex", action="store_true",
        help="Disable NegEx negation detection in NLP"
    )
    parser.add_argument(
        "--no-dependency-parsing", action="store_true",
        help="Disable dependency parsing in NLP"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=0,
        help="Number of cross-validation folds (0 = no CV, single split)"
    )
    parser.add_argument(
        "--run-baselines", action="store_true",
        help="Run all baseline predictors for comparison"
    )
    parser.add_argument(
        "--run-ablations", action="store_true",
        help="Run full ablation study (agents, weights, symbolic, NLP)"
    )
    parser.add_argument(
        "--verify-claims", action="store_true",
        help="Run architectural claim verification matrix"
    )
    parser.add_argument(
        "--single-agent", type=str, default=None,
        choices=["R1", "R2", "R3", "P1", "P2", "P3"],
        help="Use only a single agent (for ablation comparison)"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        await demo_mode()
        return
    
    # Handle ablation study modes
    if args.run_baselines:
        await run_baselines_evaluation(args)
        return
    
    if args.run_ablations:
        await run_ablation_study(args)
        return
    
    if args.verify_claims:
        await run_claim_verification(args)
        return
    
    # Initialize system with ablation flags
    mas = ExtendedMAS(
        data_source=args.data,
        verbose=True,
        max_cases=args.max_cases if args.max_cases else 100,
        start_index=args.start_index,
        no_filter=args.no_filter,
        weight_mode=args.weight_mode,
        consensus_mode=args.consensus,
        use_negex=not args.no_negex,
        use_dependency_parsing=not args.no_dependency_parsing
    )
    
    if args.evaluate:
        # Run evaluation with optional cross-validation
        if args.cv_folds > 1:
            metrics = await mas.run_cross_validation(n_folds=args.cv_folds)
        else:
            metrics = await mas.run_evaluation(max_cases=args.max_cases)
        mas.print_results_summary()
    else:
        # Process all cases
        for case in mas.cases[:args.max_cases] if args.max_cases else mas.cases:
            await mas.process_single_case(case)
        mas.print_results_summary()
    
    # Export if requested
    if args.export:
        mas.export_results(args.export)


async def run_baselines_evaluation(args):
    """Run baseline predictor evaluation with actual agent predictions."""
    from evaluation.baselines import (
        evaluate_baselines, 
        MajorityClassBaseline, 
        RandomBaseline,
        UnweightedMajorityVote,
        StaticWeightedAverage
    )
    from data.nlmcxr_loader import NLMCXRLoader
    
    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    
    # Load data
    loader = NLMCXRLoader()
    if args.no_filter:
        case_ids = loader.get_all_labeled_case_ids(limit=args.max_cases or 100)
    else:
        case_ids = loader.get_nlp_rich_case_ids(
            min_score=3.0, limit=args.max_cases or 100
        )
    
    print(f"\nProcessing {len(case_ids)} cases...")
    
    # Initialize MAS to get real agent predictions
    mas = ExtendedMAS(
        data_source="nlmcxr",
        verbose=False,
        max_cases=args.max_cases or 100,
        no_filter=args.no_filter
    )
    
    # Process cases and collect predictions
    y_true = []
    agent_predictions = []
    
    for i, case_id in enumerate(case_ids):
        try:
            case = next((c for c in mas.cases if c.get("nodule_id") == case_id), None)
            if case is None:
                continue
                
            gt = case.get("ground_truth", -1)
            if gt not in (0, 1):
                continue
                
            # Process case with all agents
            result = await mas.process_single_case(case)
            if result is None:
                continue
            
            y_true.append(gt)
            
            # Collect agent predictions in expected format
            agent_preds = {
                "R1": result.radiologist_densenet.get("probability", 0.5),
                "R2": result.radiologist_resnet.get("probability", 0.5),
                "R3": result.radiologist_rulebased.get("probability", 0.5),
                "P1": result.pathologist_regex.get("probability", 0.5),
                "P2": result.pathologist_spacy.get("probability", 0.5),
                "P3": result.pathologist_context.get("probability", 0.5),
            }
            agent_predictions.append(agent_preds)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(case_ids)} cases")
                
        except Exception as e:
            logger.debug(f"Error processing {case_id}: {e}")
            continue
    
    if len(y_true) < 2:
        print("Error: Not enough valid cases to evaluate baselines")
        return
    
    y_true = np.array(y_true)
    print(f"\nEvaluating baselines on {len(y_true)} cases...")
    print(f"Class distribution: {sum(y_true == 0)} benign, {sum(y_true == 1)} malignant")
    
    # Evaluate all baselines
    results = evaluate_baselines(y_true, agent_predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"{'Baseline':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for name, metrics in results.items():
        # Metrics may be nested under 'binary' key
        if 'binary' in metrics:
            binary = metrics['binary']
        else:
            binary = metrics
        acc = binary.get('accuracy', 0)
        prec = binary.get('precision', 0)
        rec = binary.get('sensitivity', binary.get('recall', 0))  # sensitivity = recall
        f1 = binary.get('f1_score', binary.get('f1', 0))
        print(f"{name:<25} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")


async def run_ablation_study(args):
    """Run full ablation study."""
    try:
        from evaluation.ablation_framework import (
            AblationRunner,
            create_agent_ablations,
            create_weighting_ablations,
            create_symbolic_ablations,
            create_nlp_ablations
        )
    except ImportError as e:
        print(f"Error importing ablation framework: {e}")
        return
    
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    # Create ablation configs (functions return Dict[str, AblationConfig])
    configs = {}
    configs.update(create_agent_ablations())
    configs.update(create_weighting_ablations())
    configs.update(create_symbolic_ablations())
    configs.update(create_nlp_ablations())
    
    print(f"Total ablation configurations: {len(configs)}")
    for name, cfg in configs.items():
        print(f"  - {cfg.name}: {cfg.description[:50]}...")
    
    # Run ablations (placeholder - actual implementation requires data)
    print("\n[Note: Full ablation execution requires dataset. Use --evaluate with flags.]")


async def run_claim_verification(args):
    """Run architectural claim verification."""
    try:
        from evaluation.claim_verification import (
            ClaimVerifier, ClaimStatus, ARCHITECTURAL_CLAIMS
        )
    except ImportError as e:
        print(f"Error importing claim verification: {e}")
        return
    
    print("=" * 60)
    print("ARCHITECTURAL CLAIM VERIFICATION")
    print("=" * 60)
    
    # List claims to verify
    print("\nClaims to verify:")
    print("-" * 60)
    for claim_id, claim_def in ARCHITECTURAL_CLAIMS.items():
        print(f"\n  [{claim_id}]")
        print(f"    Description: {claim_def['description']}")
        print(f"    Metric: {claim_def['metric']}")
        print(f"    Expected: {claim_def['expected_direction']}")
        print(f"    Baseline: {claim_def['baseline_ablation']}")
        print(f"    Compare: {', '.join(claim_def['comparison_ablations'])}")
    
    # Check if we have results to verify against
    results_path = Path("results/ablation_results.json")
    if results_path.exists():
        print("\n" + "=" * 60)
        print("Verifying claims against ablation results...")
        
        with open(results_path) as f:
            ablation_results = json.load(f)
        
        verifier = ClaimVerifier(ablation_results)
        report = verifier.verify_all_claims(ARCHITECTURAL_CLAIMS)
        print(report)
    else:
        print("\n" + "-" * 60)
        print("[Note: No ablation results found. Run --run-ablations first,")
        print(" then --verify-claims to verify the claims with actual data.]")
        print("\nExample workflow:")
        print("  1. python main_extended.py --evaluate --data nlmcxr --max-cases 100")
        print("  2. python main_extended.py --run-baselines --data nlmcxr")
        print("  3. python main_extended.py --verify-claims")


if __name__ == "__main__":
    asyncio.run(main())
