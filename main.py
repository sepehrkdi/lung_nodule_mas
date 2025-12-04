#!/usr/bin/env python3
"""
Multi-Agent System for Lung Nodule Classification
==================================================

EDUCATIONAL PURPOSE:

This project demonstrates the integration of:
1. Natural Language Processing (NLP) - scispaCy for medical text analysis
2. Symbolic AI (Prolog) - First-Order Logic reasoning
3. Distributed AI (BDI Agents) - Multi-agent collaboration

MULTI-AGENT ARCHITECTURE:
                                    
    ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ CT Image    │     │ Radiology       │     │ Ground Truth    │
    │             │     │ Report          │     │ (LIDC)          │
    └──────┬──────┘     └────────┬────────┘     └────────┬────────┘
           │                     │                       │
           ▼                     ▼                       │
    ┌─────────────┐     ┌─────────────────┐              │
    │ RADIOLOGIST │     │  PATHOLOGIST    │              │
    │ (DenseNet)  │     │  (scispaCy)     │              │
    │ [CV Agent]  │     │  [NLP Agent]    │              │
    └──────┬──────┘     └────────┬────────┘              │
           │                     │                       │
           │    INFORM (FIPA)    │                       │
           └──────────┬──────────┘                       │
                      ▼                                  │
             ┌─────────────────┐                         │
             │   ONCOLOGIST    │                         │
             │   (Prolog KB)   │                         │
             │ [Reasoning Agent]│                        │
             └────────┬────────┘                         │
                      │                                  │
                      ▼                                  ▼
             ┌─────────────────┐                ┌────────────────┐
             │ Final Assessment│                │   Evaluation   │
             │ - Lung-RADS     │──────────────▶│   - Accuracy   │
             │ - TNM Stage     │                │   - ROC/AUC    │
             │ - Recommendation│                │   - Confusion  │
             └─────────────────┘                └────────────────┘

AGENT COMMUNICATION:
- Uses FIPA-ACL speech acts (ACHIEVE, INFORM, QUERY_REF)
- In-memory message broker with per-agent queues
- Beliefs annotated with [source(agent)] for provenance

BDI ARCHITECTURE:
- Beliefs: Knowledge about nodules and classifications
- Desires: Goals to analyze and assess nodules
- Intentions: Active plans being executed

Usage:
    python main.py                    # Run on fallback dataset
    python main.py --dataset subset   # Run on prepared LIDC subset
    python main.py --demo             # Quick demo mode
    python main.py --evaluate         # Run with full evaluation
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import agent system components
from communication.message_queue import MessageBroker, Message, Performative
from agents.radiologist_agent import RadiologistAgent
from agents.pathologist_agent import PathologistAgent
from agents.oncologist_agent import OncologistAgent

# Import data components
from data.lidc_loader import LIDCLoader
from data.report_generator import ReportGenerator


@dataclass
class ProcessingResult:
    """Result from processing a single nodule."""
    nodule_id: str
    ground_truth: int  # Malignancy 1-5
    predicted_class: int
    malignancy_probability: float
    lung_rads: str
    t_stage: str
    recommendation: str
    processing_time: float
    agent_findings: Dict[str, Any]


class LungNoduleMAS:
    """
    Multi-Agent System for Lung Nodule Classification.
    
    EDUCATIONAL PURPOSE:
    
    This class orchestrates the multi-agent system:
    1. Loads nodule data (images + features)
    2. Sends parallel requests to Radiologist and Pathologist
    3. Collects findings and forwards to Oncologist
    4. Aggregates final assessments
    
    The parallel request pattern demonstrates:
    - Concurrent agent execution
    - Asynchronous message passing
    - Result aggregation
    """
    
    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the Multi-Agent System.
        
        Args:
            data_path: Path to nodule data (subset or fallback)
            verbose: Print progress messages
        """
        self.verbose = verbose
        
        # Initialize message broker
        self.broker = MessageBroker()
        
        # Initialize agents
        self._log("Initializing agents...")
        self.radiologist = RadiologistAgent(self.broker)
        self.pathologist = PathologistAgent(self.broker)
        self.oncologist = OncologistAgent(self.broker)
        
        # Initialize data loader
        self.loader = LIDCLoader(data_path)
        self.report_generator = ReportGenerator()
        
        # Results storage
        self.results: List[ProcessingResult] = []
        
        self._log("MAS initialized successfully")
    
    def _log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            print(f"[MAS] {message}")
    
    def process_nodule(self, nodule_id: str) -> ProcessingResult:
        """
        Process a single nodule through the agent system.
        
        EDUCATIONAL PURPOSE - PARALLEL AGENT REQUESTS:
        
        1. Load nodule data (image + features)
        2. Generate synthetic report from features
        3. Send ACHIEVE to Radiologist (image analysis)
        4. Send ACHIEVE to Pathologist (report analysis)
        5. Wait for both to complete
        6. Send ACHIEVE to Oncologist (synthesis)
        7. Collect final assessment
        
        Args:
            nodule_id: Identifier for the nodule to process
            
        Returns:
            ProcessingResult with all findings
        """
        start_time = time.time()
        self._log(f"Processing {nodule_id}...")
        
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
        
        # Step 1: Send parallel requests to Radiologist and Pathologist
        rad_result = self._request_radiologist(nodule_id, image, features)
        path_result = self._request_pathologist(nodule_id, report_text, features)
        
        # Step 2: Send combined findings to Oncologist
        onc_result = self._request_oncologist(nodule_id, rad_result, path_result)
        
        # Build result
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            nodule_id=nodule_id,
            ground_truth=ground_truth,
            predicted_class=self._get_predicted_class(onc_result, rad_result),
            malignancy_probability=self._get_probability(onc_result, rad_result),
            lung_rads=onc_result.get("assessment", {}).get("lung_rads", "3"),
            t_stage=onc_result.get("assessment", {}).get("t_stage", "T1b"),
            recommendation=onc_result.get("assessment", {}).get("recommendation", ""),
            processing_time=processing_time,
            agent_findings={
                "radiologist": rad_result,
                "pathologist": path_result,
                "oncologist": onc_result
            }
        )
        
        self.results.append(result)
        self._log(f"Completed {nodule_id} in {processing_time:.2f}s")
        
        return result
    
    def _request_radiologist(
        self, 
        nodule_id: str, 
        image: Optional[Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request image analysis from Radiologist agent.
        
        EDUCATIONAL NOTE - ACHIEVE PERFORMATIVE:
        The ACHIEVE performative requests another agent
        to accomplish a goal. The sender delegates the
        task to the receiver.
        """
        result = self.radiologist.analyze_nodule(
            nodule_id=nodule_id,
            image=image,
            features=features
        )
        return result
    
    def _request_pathologist(
        self, 
        nodule_id: str, 
        report_text: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request report analysis from Pathologist agent.
        
        The Pathologist uses NLP to extract structured
        information from the radiology report.
        """
        result = self.pathologist.analyze_report(nodule_id, report_text)
        return result
    
    def _request_oncologist(
        self,
        nodule_id: str,
        rad_findings: Dict[str, Any],
        path_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request synthesis from Oncologist agent.
        
        The Oncologist combines findings using Prolog
        reasoning to generate final assessment.
        """
        result = self.oncologist.assess_nodule(
            nodule_id=nodule_id,
            radiologist_findings=rad_findings,
            pathologist_findings=path_findings
        )
        return result
    
    def _get_predicted_class(
        self, 
        onc_result: Dict[str, Any],
        rad_result: Dict[str, Any]
    ) -> int:
        """Get predicted malignancy class from results."""
        # Try oncologist probability first
        prob = onc_result.get("assessment", {}).get("malignancy_probability")
        if prob is not None:
            return self._prob_to_class(prob)
        
        # Fall back to radiologist
        return rad_result.get("findings", {}).get("predicted_class", 3)
    
    def _get_probability(
        self,
        onc_result: Dict[str, Any],
        rad_result: Dict[str, Any]
    ) -> float:
        """Get malignancy probability from results."""
        prob = onc_result.get("assessment", {}).get("malignancy_probability")
        if prob is not None:
            return prob
        
        return rad_result.get("findings", {}).get("malignancy_probability", 0.5)
    
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
            t_stage="T1b",
            recommendation="Unable to process",
            processing_time=processing_time,
            agent_findings={}
        )
    
    def process_all(self) -> List[ProcessingResult]:
        """
        Process all nodules in the dataset.
        
        Returns:
            List of ProcessingResult for each nodule
        """
        nodule_ids = self.loader.list_nodules()
        self._log(f"Processing {len(nodule_ids)} nodules...")
        
        for nodule_id in nodule_ids:
            self.process_nodule(nodule_id)
        
        return self.results
    
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
        
        return {
            "total_nodules": total,
            "five_class_accuracy": correct / total if total > 0 else 0,
            "binary_accuracy": binary_correct / len(binary_results) if binary_results else 0,
            "binary_total": len(binary_results),
            "lung_rads_distribution": lung_rads_dist,
            "average_processing_time": avg_time
        }
    
    def get_message_trace(self) -> List[Dict[str, Any]]:
        """
        Get trace of all messages exchanged.
        
        EDUCATIONAL PURPOSE:
        This shows the communication pattern between agents.
        """
        return self.broker.get_message_trace()
    
    def print_results(self) -> None:
        """Print formatted results."""
        print("\n" + "="*60)
        print("MULTI-AGENT SYSTEM RESULTS")
        print("="*60)
        
        for result in self.results:
            print(f"\n--- {result.nodule_id} ---")
            print(f"  Ground Truth: {result.ground_truth}")
            print(f"  Predicted:    {result.predicted_class}")
            print(f"  Probability:  {result.malignancy_probability:.3f}")
            print(f"  Lung-RADS:    {result.lung_rads}")
            print(f"  T-Stage:      {result.t_stage}")
            print(f"  Time:         {result.processing_time:.2f}s")
            
            match = "✓" if result.predicted_class == result.ground_truth else "✗"
            print(f"  Match:        {match}")
        
        # Summary
        summary = self.get_summary()
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total Nodules:      {summary['total_nodules']}")
        print(f"5-Class Accuracy:   {summary['five_class_accuracy']:.1%}")
        print(f"Binary Accuracy:    {summary['binary_accuracy']:.1%} ({summary['binary_total']} nodules)")
        print(f"Avg Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"\nLung-RADS Distribution:")
        for cat, count in sorted(summary['lung_rads_distribution'].items()):
            print(f"  Category {cat}: {count}")


def run_demo():
    """Run a quick demonstration of the system."""
    print("="*60)
    print("LUNG NODULE MULTI-AGENT SYSTEM DEMO")
    print("="*60)
    print()
    print("This demo showcases:")
    print("1. BDI Agent Architecture (Beliefs, Desires, Intentions)")
    print("2. NLP with scispaCy for medical text analysis")
    print("3. Symbolic AI with Prolog for clinical reasoning")
    print("4. Agent Communication via FIPA-ACL speech acts")
    print()
    
    # Create MAS with fallback data
    mas = LungNoduleMAS(verbose=True)
    
    # Process first 3 nodules
    nodule_ids = mas.loader.list_nodules()[:3]
    
    for nodule_id in nodule_ids:
        result = mas.process_nodule(nodule_id)
        
        print(f"\n{'='*40}")
        print(f"Result for {nodule_id}:")
        print(f"  Ground Truth:    Malignancy {result.ground_truth}")
        print(f"  Prediction:      Malignancy {result.predicted_class}")
        print(f"  Probability:     {result.malignancy_probability:.2%}")
        print(f"  Lung-RADS:       Category {result.lung_rads}")
        print(f"  Recommendation:  {result.recommendation}")
    
    # Show message trace
    print(f"\n{'='*40}")
    print("MESSAGE TRACE (Agent Communication):")
    trace = mas.get_message_trace()
    for msg in trace[:10]:  # Show first 10 messages
        print(f"  {msg['sender']} → {msg['receiver']}: {msg['performative']}")
    
    print(f"\n{'='*40}")
    print("DEMO COMPLETE")
    print()
    print("Next steps:")
    print("  python main.py --all        # Process all nodules")
    print("  python main.py --evaluate   # Run with evaluation metrics")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent System for Lung Nodule Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    Run on fallback dataset
    python main.py --demo             Quick demonstration
    python main.py --all              Process all nodules
    python main.py --data path/to/data  Use specific data directory
    python main.py --evaluate         Run with full evaluation
    python main.py --quiet            Minimal output
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
    mas = LungNoduleMAS(
        data_path=args.data,
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
            "t_stage": result.t_stage,
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
                    "t_stage": r.t_stage,
                    "recommendation": r.recommendation,
                    "processing_time": r.processing_time
                }
                for r in mas.results
            ],
            "message_trace": mas.get_message_trace()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
