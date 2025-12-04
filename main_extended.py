#!/usr/bin/env python3
"""
Extended Multi-Agent System - 5 Specialized Agents
==================================================

EDUCATIONAL PURPOSE:

This module demonstrates an extended Multi-Agent System with:
- 3 Radiologist agents (DenseNet, ResNet, Rule-based)
- 2 Pathologist agents (Regex, spaCy NER)
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
    │   ┌──────────────────────┐  ┌──────────────────────┐           │
    │   │    Regex-Based       │  │    spaCy NER + Rules │           │
    │   │     W = 0.8          │  │       W = 0.9        │           │
    │   │  (Pattern Match)     │  │   (Statistical NLP)  │           │
    │   └──────────┬───────────┘  └──────────┬───────────┘           │
    │              │                          │                       │
    │              └────────────┬─────────────┘                      │
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
    """Result from analyzing a single case with all 5 agents."""
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


class ExtendedMAS:
    """
    Extended Multi-Agent System with 5 Specialized Agents.
    
    EDUCATIONAL PURPOSE:
    Demonstrates agent diversity, weighted consensus, and
    Prolog-based decision fusion in medical AI.
    """
    
    def __init__(
        self,
        data_source: str = "openi",
        verbose: bool = True
    ):
        """
        Initialize the extended MAS.
        
        Args:
            data_source: "openi" or "lidc" or "sample"
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.data_source = data_source
        
        # Import orchestrator
        from orchestrator import MultiAgentOrchestrator
        
        # Initialize orchestrator with all agents
        self._log("Initializing 5-Agent Orchestrator...")
        
        kb_path = Path(__file__).parent / "knowledge" / "multi_agent_consensus.pl"
        self.orchestrator = MultiAgentOrchestrator(
            consensus_kb_path=str(kb_path) if kb_path.exists() else None
        )
        
        # Load data
        self._load_data()
        
        # Results storage
        self.results: List[CaseResult] = []
        
        self._log("Extended MAS initialized with 5 specialized agents")
    
    def _log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            logger.info(f"[ExtendedMAS] {message}")
    
    def _load_data(self) -> None:
        """Load nodule data based on source."""
        self.cases = []
        
        if self.data_source == "openi":
            try:
                from data.openi_loader import OpenILoader
                data_path = Path(__file__).parent / "data" / "openi_data"
                if data_path.exists():
                    loader = OpenILoader(str(data_path))
                    self.cases = loader.load_all_cases()
                    self._log(f"Loaded {len(self.cases)} cases from Open-I")
                else:
                    self._log("Open-I data not found, using sample data")
                    self._create_sample_cases()
            except Exception as e:
                self._log(f"Failed to load Open-I: {e}, using sample data")
                self._create_sample_cases()
                
        elif self.data_source == "lidc":
            try:
                from data.lidc_loader import LIDCLoader
                loader = LIDCLoader()
                nodules = loader.load_all_nodules(max_nodules=50)
                
                from data.report_generator import ReportGenerator
                report_gen = ReportGenerator()
                
                for n in nodules:
                    self.cases.append({
                        "nodule_id": n["nodule_id"],
                        "features": n["features"],
                        "ground_truth": n["malignancy"],
                        "report": report_gen.generate_report(n["features"])
                    })
                self._log(f"Loaded {len(self.cases)} cases from LIDC-IDRI")
            except Exception as e:
                self._log(f"Failed to load LIDC: {e}, using sample data")
                self._create_sample_cases()
        else:
            self._create_sample_cases()
    
    def _create_sample_cases(self) -> None:
        """Create sample cases for testing."""
        from data.report_generator import ReportGenerator
        report_gen = ReportGenerator()
        
        sample_nodules = [
            {
                "nodule_id": "sample_001",
                "features": {
                    "size_mm": 5, "texture": "ground_glass",
                    "location": "left_upper_lobe", "malignancy": 2
                },
                "ground_truth": 2
            },
            {
                "nodule_id": "sample_002",
                "features": {
                    "size_mm": 12, "texture": "solid",
                    "location": "right_upper_lobe", "malignancy": 4
                },
                "ground_truth": 4
            },
            {
                "nodule_id": "sample_003",
                "features": {
                    "size_mm": 8, "texture": "part_solid",
                    "location": "right_lower_lobe", "malignancy": 3
                },
                "ground_truth": 3
            },
            {
                "nodule_id": "sample_004",
                "features": {
                    "size_mm": 20, "texture": "spiculated",
                    "location": "right_upper_lobe", "malignancy": 5
                },
                "ground_truth": 5
            },
            {
                "nodule_id": "sample_005",
                "features": {
                    "size_mm": 4, "texture": "calcified",
                    "location": "left_lower_lobe", "malignancy": 1
                },
                "ground_truth": 1
            },
        ]
        
        for n in sample_nodules:
            n["report"] = report_gen.generate_report(n["features"])
        
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
        """Process a single case with all 5 agents."""
        start_time = time.time()
        
        nodule_id = case.get("nodule_id", "unknown")
        features = case.get("features", {})
        report = case.get("report", "")
        ground_truth = case.get("ground_truth")
        
        self._log(f"Processing case {nodule_id}...")
        
        # Run orchestrator
        consensus = await self.orchestrator.analyze_case(
            nodule_id=nodule_id,
            features=features,
            report=report
        )
        
        # Extract individual findings
        size_mm = features.get("size_mm", 10)
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
        print("EXTENDED MAS - 5-AGENT RESULTS SUMMARY")
        print("=" * 70)
        
        for r in self.results:
            gt_str = f"GT: {r.ground_truth}" if r.ground_truth else "GT: N/A"
            print(f"\n{r.nodule_id}: Class {r.final_class} "
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
    """Quick demonstration of the 5-agent system."""
    print("\n" + "=" * 70)
    print("EXTENDED MAS DEMO - 5 Specialized Agents")
    print("=" * 70)
    
    print("\nArchitecture:")
    print("  Radiologists (3): DenseNet121, ResNet50, Rule-based")
    print("  Pathologists (2): Regex-based, spaCy NER")
    print("  Consensus: Prolog weighted voting")
    
    print("\nInitializing system...")
    
    mas = ExtendedMAS(data_source="sample", verbose=True)
    
    print("\nProcessing sample cases...")
    
    for case in mas.cases[:2]:
        result = await mas.process_single_case(case)
        
        print(f"\n--- Case: {result.nodule_id} ---")
        print(f"Final Class: {result.final_class} "
              f"(probability: {result.final_probability:.3f})")
        print(f"Agreement: {result.agreement_level}")
        print(f"Lung-RADS: {result.lung_rads}")
        print(f"Recommendation: {result.recommendation}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extended Multi-Agent System with 5 Specialized Agents"
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
        choices=["sample", "openi", "lidc"],
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
    
    args = parser.parse_args()
    
    if args.demo:
        await demo_mode()
        return
    
    # Initialize system
    mas = ExtendedMAS(data_source=args.data, verbose=True)
    
    if args.evaluate:
        # Run evaluation
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


if __name__ == "__main__":
    asyncio.run(main())
