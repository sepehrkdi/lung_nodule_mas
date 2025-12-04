#!/usr/bin/env python3
"""
SPADE-BDI Multi-Agent System for Lung Nodule Classification
============================================================

EDUCATIONAL PURPOSE:

This module demonstrates a proper Multi-Agent System using SPADE-BDI,
an AgentSpeak(L) interpreter that provides genuine BDI agent semantics.

FRAMEWORK COMPLIANCE:
- SPADE-BDI: Python-based BDI framework with AgentSpeak support
- AgentSpeak(L): Proper plan execution with triggering events
- XMPP: Standard communication protocol for agent messaging

ARCHITECTURE:
    
    ┌────────────────────────────────────────────────────────────┐
    │                   SPADE-BDI Platform                       │
    ├────────────────────────────────────────────────────────────┤
    │                                                            │
    │   ┌──────────────┐    ┌──────────────┐    ┌────────────┐  │
    │   │ Radiologist1 │    │ Radiologist2 │    │Radiologist3│  │
    │   │ (DenseNet)   │    │ (DenseNet)   │    │ (DenseNet) │  │
    │   └──────┬───────┘    └──────┬───────┘    └─────┬──────┘  │
    │          │                   │                   │         │
    │   ┌──────────────┐    ┌──────────────┐          │         │
    │   │ Pathologist1 │    │ Pathologist2 │          │         │
    │   │ (scispaCy)   │    │ (scispaCy)   │          │         │
    │   └──────┬───────┘    └──────┬───────┘          │         │
    │          │                   │                   │         │
    │          └──────────┬────────┴───────────────────┘         │
    │                     ▼                                      │
    │           ┌─────────────────┐                              │
    │           │   Oncologist    │                              │
    │           │   (Prolog KB)   │                              │
    │           │ Weighted Voting │                              │
    │           └────────┬────────┘                              │
    │                    │                                       │
    │                    ▼                                       │
    │           ┌─────────────────┐                              │
    │           │ Final Consensus │                              │
    │           │ Lung-RADS + Rec │                              │
    │           └─────────────────┘                              │
    └────────────────────────────────────────────────────────────┘

AGENT TYPES:
- Multiple Radiologists: Independent image classification
- Multiple Pathologists: Independent report analysis  
- Single Oncologist: Weighted consensus with disagreement handling

Usage:
    python spade_main.py                    # Run with fallback data
    python spade_main.py --demo             # Quick demo mode
    python spade_main.py --num-radiologists 3 --num-pathologists 2
    python spade_main.py --evaluate         # Full evaluation
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing a single nodule."""
    nodule_id: str
    ground_truth: int  # Malignancy 1-5
    predicted_class: int
    malignancy_probability: float
    lung_rads: str
    recommendation: str
    processing_time: float
    confidence: float = 0.0
    agent_count: int = 0
    agent_findings: Dict[str, Any] = field(default_factory=dict)


class SPADEMedicalMAS:
    """
    SPADE-BDI Multi-Agent System for Lung Nodule Classification.
    
    EDUCATIONAL PURPOSE:
    
    This class demonstrates a proper BDI-based MAS using SPADE-BDI:
    1. Multiple agents of same type for redundancy/consensus
    2. Weighted voting for disagreement resolution
    3. AgentSpeak plans for decision logic
    4. Internal actions bridging symbolic plans with ML/NLP
    
    The system supports configurable numbers of radiologists and
    pathologists to demonstrate multi-agent consensus.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        num_radiologists: int = 3,
        num_pathologists: int = 2,
        verbose: bool = True
    ):
        """
        Initialize the SPADE-BDI Multi-Agent System.
        
        Args:
            data_path: Path to nodule data
            num_radiologists: Number of radiologist agents (default: 3)
            num_pathologists: Number of pathologist agents (default: 2)
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.num_radiologists = num_radiologists
        self.num_pathologists = num_pathologists
        
        # Import agents
        from agents.spade_radiologist import RadiologistAgent
        from agents.spade_pathologist import PathologistAgent
        from agents.spade_oncologist import OncologistAgent
        
        # Initialize agents
        self._log("Initializing SPADE-BDI agents...")
        
        # Create multiple radiologists
        self.radiologists = [
            RadiologistAgent(name=f"radiologist_{i+1}")
            for i in range(num_radiologists)
        ]
        
        # Create multiple pathologists
        self.pathologists = [
            PathologistAgent(name=f"pathologist_{i+1}")
            for i in range(num_pathologists)
        ]
        
        # Single oncologist for consensus
        self.oncologist = OncologistAgent(name="oncologist")
        
        # Initialize data loader
        from data.lidc_loader import LIDCLoader
        from data.report_generator import ReportGenerator
        
        self.loader = LIDCLoader(data_path)
        self.report_generator = ReportGenerator()
        
        # Results storage
        self.results: List[ProcessingResult] = []
        
        self._log(
            f"MAS initialized with {num_radiologists} radiologists, "
            f"{num_pathologists} pathologists"
        )
    
    def _log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            logger.info(f"[SPADE-MAS] {message}")
    
    async def _run_radiologist(
        self,
        agent,
        nodule_id: str,
        image: Any,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single radiologist analysis."""
        request = {
            "nodule_id": nodule_id,
            "image": image,
            "features": features
        }
        result = await agent.process_request(request)
        return result.get("findings", result)
    
    async def _run_pathologist(
        self,
        agent,
        nodule_id: str,
        report_text: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single pathologist analysis."""
        request = {
            "nodule_id": nodule_id,
            "report": report_text,
            "features": features
        }
        result = await agent.process_request(request)
        return result.get("findings", result)
    
    async def process_nodule_async(self, nodule_id: str) -> ProcessingResult:
        """
        Process a single nodule through all agents asynchronously.
        
        EDUCATIONAL PURPOSE - PARALLEL AGENT EXECUTION:
        
        1. Load nodule data (image + features)
        2. Generate synthetic report from features
        3. Run ALL radiologists in parallel (concurrent classification)
        4. Run ALL pathologists in parallel (concurrent NLP)
        5. Collect all findings
        6. Send to oncologist for weighted consensus
        7. Return final assessment
        
        Args:
            nodule_id: Identifier for the nodule to process
            
        Returns:
            ProcessingResult with consensus classification
        """
        start_time = time.time()
        self._log(f"Processing {nodule_id} with {self.num_radiologists}R/{self.num_pathologists}P agents...")
        
        # Load nodule data
        nodule_data = self.loader.load_nodule(nodule_id)
        
        if nodule_data is None:
            self._log(f"Could not load {nodule_id}")
            return self._empty_result(nodule_id, time.time() - start_time)
        
        features = nodule_data.get("features", {})
        image = nodule_data.get("image")
        ground_truth = nodule_data.get("malignancy", 3)
        
        # Generate report from features
        report_text = self.report_generator.generate(features)
        
        # Run ALL radiologists in parallel
        rad_tasks = [
            self._run_radiologist(agent, nodule_id, image, features)
            for agent in self.radiologists
        ]
        rad_results = await asyncio.gather(*rad_tasks)
        
        # Run ALL pathologists in parallel
        path_tasks = [
            self._run_pathologist(agent, nodule_id, report_text, features)
            for agent in self.pathologists
        ]
        path_results = await asyncio.gather(*path_tasks)
        
        # Send all findings to oncologist
        onc_request = {
            "nodule_id": nodule_id,
            "features": features,
            "radiologist_findings": rad_results,
            "pathologist_findings": path_results
        }
        onc_result = await self.oncologist.process_request(onc_request)
        
        # Build result
        processing_time = time.time() - start_time
        
        classification = onc_result.get("classification", {})
        recommendation = onc_result.get("recommendation", {})
        
        result = ProcessingResult(
            nodule_id=nodule_id,
            ground_truth=ground_truth,
            predicted_class=self._prob_to_class(
                classification.get("malignancy_probability", 0.5)
            ),
            malignancy_probability=classification.get("malignancy_probability", 0.5),
            lung_rads=classification.get("lung_rads_category", "3"),
            recommendation=recommendation.get("description", ""),
            processing_time=processing_time,
            confidence=classification.get("confidence", 0.0),
            agent_count=len(rad_results) + len(path_results) + 1,
            agent_findings={
                "radiologists": rad_results,
                "pathologists": path_results,
                "oncologist": onc_result
            }
        )
        
        self.results.append(result)
        self._log(
            f"Completed {nodule_id}: class={result.predicted_class}, "
            f"prob={result.malignancy_probability:.3f}, "
            f"confidence={result.confidence:.3f} "
            f"({processing_time:.2f}s)"
        )
        
        return result
    
    def process_nodule(self, nodule_id: str) -> ProcessingResult:
        """
        Synchronous wrapper for processing a nodule.
        
        Args:
            nodule_id: Identifier for the nodule to process
            
        Returns:
            ProcessingResult with all findings
        """
        return asyncio.run(self.process_nodule_async(nodule_id))
    
    async def process_all_async(self) -> List[ProcessingResult]:
        """
        Process all nodules asynchronously.
        
        Returns:
            List of ProcessingResult for each nodule
        """
        nodule_ids = self.loader.list_nodules()
        self._log(f"Processing {len(nodule_ids)} nodules with SPADE-BDI...")
        
        for nodule_id in nodule_ids:
            await self.process_nodule_async(nodule_id)
        
        return self.results
    
    def process_all(self) -> List[ProcessingResult]:
        """
        Process all nodules synchronously.
        
        Returns:
            List of ProcessingResult for each nodule
        """
        return asyncio.run(self.process_all_async())
    
    def _prob_to_class(self, prob: float) -> int:
        """Convert probability to malignancy class 1-5."""
        if prob < 0.2:
            return 1
        elif prob < 0.4:
            return 2
        elif prob < 0.6:
            return 3
        elif prob < 0.8:
            return 4
        else:
            return 5
    
    def _empty_result(self, nodule_id: str, processing_time: float) -> ProcessingResult:
        """Create empty result for failed processing."""
        return ProcessingResult(
            nodule_id=nodule_id,
            ground_truth=3,
            predicted_class=3,
            malignancy_probability=0.5,
            lung_rads="3",
            recommendation="Unable to process",
            processing_time=processing_time
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of processing.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        total = len(self.results)
        
        # Calculate accuracy
        correct = sum(
            1 for r in self.results 
            if r.predicted_class == r.ground_truth
        )
        
        # Binary accuracy (exclude indeterminate)
        binary_results = [
            r for r in self.results 
            if r.ground_truth != 3
        ]
        binary_correct = sum(
            1 for r in binary_results
            if (r.predicted_class <= 2 and r.ground_truth <= 2) or
               (r.predicted_class >= 4 and r.ground_truth >= 4)
        )
        
        # Lung-RADS distribution
        lung_rads_dist = {}
        for r in self.results:
            cat = r.lung_rads
            lung_rads_dist[cat] = lung_rads_dist.get(cat, 0) + 1
        
        # Average processing time
        avg_time = sum(r.processing_time for r in self.results) / total
        
        # Average confidence
        avg_conf = sum(r.confidence for r in self.results) / total
        
        return {
            "total_nodules": total,
            "five_class_accuracy": correct / total if total > 0 else 0,
            "binary_accuracy": binary_correct / len(binary_results) if binary_results else 0,
            "binary_total": len(binary_results),
            "lung_rads_distribution": lung_rads_dist,
            "average_processing_time": avg_time,
            "average_confidence": avg_conf,
            "agent_config": {
                "radiologists": self.num_radiologists,
                "pathologists": self.num_pathologists
            }
        }
    
    def print_results(self) -> None:
        """Print formatted results."""
        print("\n" + "="*70)
        print("SPADE-BDI MULTI-AGENT SYSTEM RESULTS")
        print("="*70)
        print(f"Agents: {self.num_radiologists} Radiologists, "
              f"{self.num_pathologists} Pathologists, 1 Oncologist")
        print("="*70)
        
        for result in self.results:
            print(f"\n--- {result.nodule_id} ---")
            print(f"  Ground Truth:    Malignancy {result.ground_truth}")
            print(f"  Predicted:       Malignancy {result.predicted_class}")
            print(f"  Probability:     {result.malignancy_probability:.3f}")
            print(f"  Confidence:      {result.confidence:.3f}")
            print(f"  Lung-RADS:       Category {result.lung_rads}")
            print(f"  Time:            {result.processing_time:.2f}s")
            
            match = "✓" if result.predicted_class == result.ground_truth else "✗"
            print(f"  Match:           {match}")
        
        # Summary
        summary = self.get_summary()
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total Nodules:       {summary['total_nodules']}")
        print(f"5-Class Accuracy:    {summary['five_class_accuracy']:.1%}")
        print(f"Binary Accuracy:     {summary['binary_accuracy']:.1%} ({summary['binary_total']} nodules)")
        print(f"Avg Confidence:      {summary['average_confidence']:.1%}")
        print(f"Avg Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"\nLung-RADS Distribution:")
        for cat, count in sorted(summary['lung_rads_distribution'].items()):
            print(f"  Category {cat}: {count}")


def run_demo():
    """Run a quick demonstration of the SPADE-BDI system."""
    print("="*70)
    print("SPADE-BDI LUNG NODULE MULTI-AGENT SYSTEM DEMO")
    print("="*70)
    print()
    print("This demo showcases:")
    print("1. SPADE-BDI: Proper AgentSpeak(L) interpreter")
    print("2. Multiple Agents: 3 Radiologists + 2 Pathologists + 1 Oncologist")
    print("3. Weighted Consensus: Disagreement resolution via voting")
    print("4. Internal Actions: Python ML/NLP called from AgentSpeak plans")
    print()
    
    # Create MAS with multiple agents
    mas = SPADEMedicalMAS(
        num_radiologists=3,
        num_pathologists=2,
        verbose=True
    )
    
    # Process first 3 nodules
    nodule_ids = mas.loader.list_nodules()[:3]
    
    for nodule_id in nodule_ids:
        result = mas.process_nodule(nodule_id)
        
        print(f"\n{'='*50}")
        print(f"Result for {nodule_id}:")
        print(f"  Ground Truth:    Malignancy {result.ground_truth}")
        print(f"  Prediction:      Malignancy {result.predicted_class}")
        print(f"  Probability:     {result.malignancy_probability:.2%}")
        print(f"  Confidence:      {result.confidence:.2%}")
        print(f"  Lung-RADS:       Category {result.lung_rads}")
        
        # Show individual agent votes
        print(f"\n  Agent Votes:")
        rad_findings = result.agent_findings.get("radiologists", [])
        for i, f in enumerate(rad_findings):
            prob = f.get("malignancy_probability", 0.5)
            print(f"    Radiologist_{i+1}: {prob:.2%}")
        
        path_findings = result.agent_findings.get("pathologists", [])
        for i, f in enumerate(path_findings):
            prob = f.get("text_malignancy_probability", 0.5)
            print(f"    Pathologist_{i+1}: {prob:.2%}")
    
    print(f"\n{'='*50}")
    print("DEMO COMPLETE")
    print()
    print("Run Options:")
    print("  python spade_main.py --all        # Process all nodules")
    print("  python spade_main.py --num-radiologists 5  # More agents")
    print("  python spade_main.py --evaluate   # Full evaluation metrics")


def main():
    """Main entry point for SPADE-BDI MAS."""
    parser = argparse.ArgumentParser(
        description="SPADE-BDI Multi-Agent System for Lung Nodule Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python spade_main.py                          Run with default 3R/2P agents
    python spade_main.py --demo                   Quick demonstration
    python spade_main.py --num-radiologists 5     Use 5 radiologists
    python spade_main.py --all                    Process all nodules
    python spade_main.py --evaluate               Run with full evaluation
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run quick demonstration"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Process all nodules in dataset"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=None,
        help="Path to data directory"
    )
    parser.add_argument(
        "--num-radiologists",
        type=int,
        default=3,
        help="Number of radiologist agents (default: 3)"
    )
    parser.add_argument(
        "--num-pathologists",
        type=int,
        default=2,
        help="Number of pathologist agents (default: 2)"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run with full evaluation metrics"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--nodule", 
        type=str,
        help="Process specific nodule by ID"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run demo mode
    if args.demo:
        run_demo()
        return
    
    # Initialize MAS
    mas = SPADEMedicalMAS(
        data_path=args.data,
        num_radiologists=args.num_radiologists,
        num_pathologists=args.num_pathologists,
        verbose=not args.quiet
    )
    
    # Process nodules
    if args.nodule:
        result = mas.process_nodule(args.nodule)
        print(json.dumps({
            "nodule_id": result.nodule_id,
            "ground_truth": result.ground_truth,
            "predicted_class": result.predicted_class,
            "malignancy_probability": result.malignancy_probability,
            "lung_rads": result.lung_rads,
            "confidence": result.confidence,
            "recommendation": result.recommendation
        }, indent=2))
    
    elif args.all:
        mas.process_all()
        mas.print_results()
    
    else:
        # Default: process first 5 or all if less
        nodule_ids = mas.loader.list_nodules()[:5]
        for nodule_id in nodule_ids:
            mas.process_nodule(nodule_id)
        mas.print_results()
    
    # Run evaluation if requested
    if args.evaluate:
        try:
            from evaluation.metrics import evaluate_results
            metrics = evaluate_results(mas.results)
            print("\nEVALUATION METRICS:")
            print(json.dumps(metrics, indent=2))
        except ImportError:
            print("\nNote: evaluation module not available")
    
    # Save results if requested
    if args.output:
        output_data = {
            "summary": mas.get_summary(),
            "results": [
                {
                    "nodule_id": r.nodule_id,
                    "ground_truth": r.ground_truth,
                    "predicted_class": r.predicted_class,
                    "malignancy_probability": r.malignancy_probability,
                    "lung_rads": r.lung_rads,
                    "confidence": r.confidence,
                    "recommendation": r.recommendation,
                    "processing_time": r.processing_time,
                    "agent_count": r.agent_count
                }
                for r in mas.results
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
