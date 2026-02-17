#!/usr/bin/env python3
"""
SPADE-BDI Multi-Agent System for lung nodule classification.

Usage:
    python spade_main.py                    # Run with fallback data
    python spade_main.py --demo             # Quick demo mode
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
    """Result from processing a single case."""
    nodule_id: str
    ground_truth: int  # Binary: 1=abnormal, 0=normal, -1=indeterminate
    predicted_class: int
    malignancy_probability: float
    lung_rads: str
    recommendation: str
    processing_time: float
    confidence: float = 0.0
    agent_count: int = 0
    agent_findings: Dict[str, Any] = field(default_factory=dict)


class SPADEMedicalMAS:
    """SPADE-BDI Multi-Agent System for lung nodule classification."""
    
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
        
        # Import BDI factory
        from agents.spade_base import create_spade_bdi_agent, DEFAULT_XMPP_CONFIG, get_asl_path
        
        # Import Variants
        from agents.radiologist_variants import (
            RadiologistDenseNet, RadiologistResNet, RadiologistRules
        )
        from agents.pathologist_variants import (
            PathologistRegex, PathologistSpacy, PathologistContext
        )
        from agents.spade_oncologist import OncologistAgent
        
        # Initialize agents
        self._log("Initializing 6 Specialized SPADE-BDI agents...")
        
        # XMPP Config (default)
        xmpp_config = DEFAULT_XMPP_CONFIG
        
        # Warn if custom counts requested
        if num_radiologists != 3 or num_pathologists != 3:
            self._log(
                "WARNING: Custom agent counts ignored. Using fixed specialized team: "
                "3 Radiologists (DenseNet/ResNet/Rule) and 3 Pathologists (Regex/SpaCy/Context)."
            )

        # Create 3 Radiologists
        self.radiologists = []
        
        # Rad 1: DenseNet
        self.radiologists.append(create_spade_bdi_agent(
            RadiologistDenseNet, "radiologist_densenet", 
            xmpp_config, get_asl_path("radiologist")
        ))
        
        # Rad 2: ResNet
        self.radiologists.append(create_spade_bdi_agent(
            RadiologistResNet, "radiologist_resnet", 
            xmpp_config, get_asl_path("radiologist")
        ))
        
        # Rad 3: Rules
        self.radiologists.append(create_spade_bdi_agent(
            RadiologistRules, "radiologist_rules", 
            xmpp_config, get_asl_path("radiologist")
        ))
        
        # Create 3 Pathologists
        self.pathologists = []
        
        # Path 1: Regex
        self.pathologists.append(create_spade_bdi_agent(
            PathologistRegex, "pathologist_regex", 
            xmpp_config, get_asl_path("pathologist")
        ))
        
        # Path 2: SpaCy
        self.pathologists.append(create_spade_bdi_agent(
            PathologistSpacy, "pathologist_spacy", 
            xmpp_config, get_asl_path("pathologist")
        ))
        
        # Path 3: Context
        self.pathologists.append(create_spade_bdi_agent(
            PathologistContext, "pathologist_context", 
            xmpp_config, get_asl_path("pathologist")
        ))
        
        # Single oncologist for consensus
        self.oncologist = OncologistAgent(name="oncologist")
        
        # Initialize data loader (NLMCXR with real radiology reports)
        from data.nlmcxr_loader import NLMCXRLoader
        
        self.loader = NLMCXRLoader(data_path)
        
        # Results storage
        self.results: List[ProcessingResult] = []
        
        self._log(
            f"MAS initialized with {len(self.radiologists)} radiologists, "
            f"{len(self.pathologists)} pathologists"
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
        """Process a single case through all agents asynchronously."""
        start_time = time.time()
        self._log(f"Processing {nodule_id} with {len(self.radiologists)}R/{len(self.pathologists)}P agents...")
        
        # Load case data (NLMCXR returns list of images + metadata)
        try:
            images, metadata = self.loader.load_case(nodule_id)
        except Exception as e:
            self._log(f"Could not load {nodule_id}: {e}")
            return self._empty_result(nodule_id, time.time() - start_time)
        
        if metadata is None:
            self._log(f"No metadata for {nodule_id}")
            return self._empty_result(nodule_id, time.time() - start_time)
        
        # Ground truth from NLP extraction of report text
        ground_truth = metadata.get("ground_truth", -1)
        
        # Use real radiology report text (FINDINGS ONLY to prevent data leakage)
        # IMPRESSION is the ground truth, so agents should not see it.
        findings = metadata.get("findings", "")
        # impression = metadata.get("impression", "")
        # report_text = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}".strip()
        report_text = f"FINDINGS: {findings}".strip()
        
        # Build features dict from NLP extraction and metadata
        features = metadata.get("nlp_features", {})
        features.update({
            "case_id": nodule_id,
            "num_images": len(images),
            "ground_truth": ground_truth
        })
        
        # Run ALL radiologists in parallel (pass first image or list)
        image = images[0] if len(images) == 1 else images
        rad_tasks = [
            self._run_radiologist(agent, nodule_id, image, features)
            for agent in self.radiologists
        ]
        rad_results = await asyncio.gather(*rad_tasks)
        
        # Run ALL pathologists in parallel with real report text
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
        Process all cases asynchronously.
        
        Returns:
            List of ProcessingResult for each case
        """
        case_ids = self.loader.get_case_ids()
        self._log(f"Processing {len(case_ids)} cases with SPADE-BDI...")
        
        for case_id in case_ids:
            await self.process_nodule_async(case_id)
        
        return self.results
    
    def process_all(self) -> List[ProcessingResult]:
        """
        Process all nodules synchronously.
        
        Returns:
            List of ProcessingResult for each nodule
        """
        return asyncio.run(self.process_all_async())
    
    def _prob_to_class(self, prob: float, threshold: float = 0.5) -> int:
        """Convert probability to binary class (0=benign, 1=malignant)."""
        return 1 if prob >= threshold else 0
    
    def _empty_result(self, nodule_id: str, processing_time: float) -> ProcessingResult:
        """Create empty result for failed processing."""
        return ProcessingResult(
            nodule_id=nodule_id,
            ground_truth=-1,  # Indeterminate
            predicted_class=0,
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
        
        # Binary accuracy (exclude indeterminate ground truth = -1)
        binary_results = [
            r for r in self.results 
            if r.ground_truth != -1
        ]
        
        # Calculate binary accuracy
        # predicted_class: map to binary (prob > 0.5 = abnormal)
        binary_correct = sum(
            1 for r in binary_results
            if (r.malignancy_probability >= 0.5 and r.ground_truth == 1) or
               (r.malignancy_probability < 0.5 and r.ground_truth == 0)
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
        
        # Count ground truth distribution
        gt_dist = {"abnormal": 0, "normal": 0, "indeterminate": 0}
        for r in self.results:
            if r.ground_truth == 1:
                gt_dist["abnormal"] += 1
            elif r.ground_truth == 0:
                gt_dist["normal"] += 1
            else:
                gt_dist["indeterminate"] += 1
        
        return {
            "total_cases": total,
            "binary_accuracy": binary_correct / len(binary_results) if binary_results else 0,
            "evaluable_cases": len(binary_results),
            "ground_truth_distribution": gt_dist,
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
            gt_label = {1: "Abnormal", 0: "Normal", -1: "Indeterminate"}.get(result.ground_truth, "Unknown")
            pred_label = "Abnormal" if result.malignancy_probability >= 0.5 else "Normal"
            print(f"  Ground Truth:    {gt_label}")
            print(f"  Predicted:       {pred_label}")
            print(f"  Probability:     {result.malignancy_probability:.3f}")
            print(f"  Confidence:      {result.confidence:.3f}")
            print(f"  Lung-RADS:       Category {result.lung_rads}")
            print(f"  Time:            {result.processing_time:.2f}s")
            
            if result.ground_truth != -1:
                pred_binary = 1 if result.malignancy_probability >= 0.5 else 0
                match = "✓" if pred_binary == result.ground_truth else "✗"
                print(f"  Match:           {match}")
            else:
                print(f"  Match:           N/A (no ground truth)")
        
        # Summary
        summary = self.get_summary()
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total Cases:         {summary['total_cases']}")
        print(f"Binary Accuracy:     {summary['binary_accuracy']:.1%} ({summary['evaluable_cases']} evaluable cases)")
        print(f"Avg Confidence:      {summary['average_confidence']:.1%}")
        print(f"Avg Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"\nGround Truth Distribution:")
        for label, count in summary['ground_truth_distribution'].items():
            print(f"  {label.capitalize()}: {count}")
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
    print("2. Multiple Agents: 3 Radiologists + 3 Pathologists + 1 Oncologist")
    print("3. Weighted Consensus: Disagreement resolution via voting")
    print("4. Internal Actions: Python ML/NLP called from AgentSpeak plans")
    print()
    
    # Create MAS with multiple agents
    mas = SPADEMedicalMAS(
        num_radiologists=3,
        num_pathologists=2,
        verbose=True
    )
    
    # Select cases for demo
    ids = mas.loader.get_case_ids()
    if not ids:
        print("No cases found in data directory.")
        return
    case_ids = ids[:3] # Process first 3 cases for demo
    
    for case_id in case_ids:
        result = mas.process_nodule(case_id)
        
        print(f"\n{'='*50}")
        print(f"Result for {case_id}:")
        gt_label = {1: "Abnormal", 0: "Normal", -1: "Indeterminate"}.get(result.ground_truth, "Unknown")
        pred_label = "Abnormal" if result.malignancy_probability >= 0.5 else "Normal"
        print(f"  Ground Truth:    {gt_label}")
        print(f"  Prediction:      {pred_label}")
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
        case_ids = mas.loader.get_case_ids()[:5]
        for case_id in case_ids:
            mas.process_nodule(case_id)
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
