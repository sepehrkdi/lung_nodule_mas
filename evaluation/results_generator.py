"""
Results Generator

Generates formatted tables, visualizations, and reports from ablation results.
Outputs in multiple formats: Markdown, LaTeX, JSON, and console.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class TableConfig:
    """Configuration for generating result tables."""
    title: str
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc_roc", "pr_auc"])
    show_std: bool = True
    show_ci: bool = True
    precision: int = 3
    highlight_best: bool = True
    sort_by: Optional[str] = None
    sort_descending: bool = True


class ResultsGenerator:
    """
    Generates formatted results from ablation studies.
    
    Produces:
    - Markdown tables for documentation
    - LaTeX tables for academic papers
    - JSON for programmatic access
    - Console output for debugging
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize results generator.
        
        Args:
            output_dir: Directory for output files (default: results/)
        """
        self.output_dir = output_dir or Path(__file__).parent.parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all(
        self,
        ablation_results: Dict[str, Dict[str, Any]],
        baseline_results: Dict[str, Dict[str, Any]],
        cv_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate all result formats.
        
        Args:
            ablation_results: Results from ablation studies
            baseline_results: Results from baseline predictors
            cv_results: Cross-validation results (optional)
            
        Returns:
            Dict mapping format name to file path
        """
        generated = {}
        
        # Generate baseline comparison table
        if baseline_results:
            md_path = self.generate_baseline_table_markdown(baseline_results)
            generated["baseline_markdown"] = str(md_path)
            
            latex_path = self.generate_baseline_table_latex(baseline_results)
            generated["baseline_latex"] = str(latex_path)
        
        # Generate ablation matrices
        if ablation_results:
            md_path = self.generate_ablation_matrix_markdown(ablation_results)
            generated["ablation_markdown"] = str(md_path)
            
            latex_path = self.generate_ablation_table_latex(ablation_results)
            generated["ablation_latex"] = str(latex_path)
        
        # Generate CV summary
        if cv_results:
            md_path = self.generate_cv_summary_markdown(cv_results)
            generated["cv_markdown"] = str(md_path)
        
        # Generate combined JSON
        json_path = self._save_all_results_json(
            ablation_results, baseline_results, cv_results
        )
        generated["all_json"] = str(json_path)
        
        return generated
    
    def generate_baseline_table_markdown(
        self,
        results: Dict[str, Dict[str, Any]],
        config: Optional[TableConfig] = None
    ) -> Path:
        """
        Generate markdown table comparing baselines to full system.
        
        Args:
            results: Dict mapping baseline name to metrics dict
            config: Table configuration
            
        Returns:
            Path to generated markdown file
        """
        config = config or TableConfig(title="Baseline Comparison")
        
        lines = [
            f"# {config.title}",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Metrics Table",
            "",
        ]
        
        # Create header
        headers = ["Method"] + [m.upper() for m in config.metrics]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Sort results if specified
        items = list(results.items())
        if config.sort_by and config.sort_by in config.metrics:
            items.sort(
                key=lambda x: x[1].get(config.sort_by, 0),
                reverse=config.sort_descending
            )
        
        # Find best values for highlighting
        best_values = {}
        if config.highlight_best:
            for metric in config.metrics:
                values = [r.get(metric, 0) for _, r in items if r.get(metric) is not None]
                if values:
                    best_values[metric] = max(values)
        
        # Add rows
        for name, metrics in items:
            row = [name]
            for metric in config.metrics:
                val = metrics.get(metric)
                if val is None:
                    row.append("-")
                elif isinstance(val, tuple):  # CI or mean±std
                    row.append(f"{val[0]:.{config.precision}f}–{val[1]:.{config.precision}f}")
                else:
                    formatted = f"{val:.{config.precision}f}"
                    if config.highlight_best and metric in best_values:
                        if abs(val - best_values[metric]) < 1e-6:
                            formatted = f"**{formatted}**"
                    row.append(formatted)
            lines.append("| " + " | ".join(row) + " |")
        
        lines.extend(["", "## Notes", "- Bold values indicate best performance"])
        
        output_path = self.output_dir / "baseline_comparison.md"
        output_path.write_text("\n".join(lines))
        return output_path
    
    def generate_baseline_table_latex(
        self,
        results: Dict[str, Dict[str, Any]],
        config: Optional[TableConfig] = None
    ) -> Path:
        """
        Generate LaTeX table for academic papers.
        
        Args:
            results: Dict mapping baseline name to metrics dict
            config: Table configuration
            
        Returns:
            Path to generated .tex file
        """
        config = config or TableConfig(title="Baseline Comparison")
        
        lines = [
            "% Auto-generated baseline comparison table",
            f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{config.title}}}",
            r"\label{tab:baseline-comparison}",
            r"\begin{tabular}{l" + "c" * len(config.metrics) + "}",
            r"\toprule",
        ]
        
        # Header
        headers = ["Method"] + [m.upper().replace("_", "-") for m in config.metrics]
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")
        
        # Find best values
        best_values = {}
        if config.highlight_best:
            for metric in config.metrics:
                values = [r.get(metric, 0) for _, r in results.items() if r.get(metric) is not None]
                if values:
                    best_values[metric] = max(values)
        
        # Rows
        for name, metrics in results.items():
            row = [name.replace("_", " ")]
            for metric in config.metrics:
                val = metrics.get(metric)
                if val is None:
                    row.append("-")
                else:
                    formatted = f"{val:.{config.precision}f}"
                    if config.highlight_best and metric in best_values:
                        if abs(val - best_values[metric]) < 1e-6:
                            formatted = r"\textbf{" + formatted + "}"
                    row.append(formatted)
            lines.append(" & ".join(row) + r" \\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        output_path = self.output_dir / "baseline_comparison.tex"
        output_path.write_text("\n".join(lines))
        return output_path
    
    def generate_ablation_matrix_markdown(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Path:
        """
        Generate ablation study results as markdown matrix.
        
        Args:
            results: Dict mapping ablation config name to metrics
            
        Returns:
            Path to generated markdown file
        """
        lines = [
            "# Ablation Study Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Group by category
        categories = {}
        for name, metrics in results.items():
            cat = metrics.get("category", "other")
            if cat not in categories:
                categories[cat] = {}
            categories[cat][name] = metrics
        
        for category, cat_results in categories.items():
            lines.extend([
                f"## {category.title()} Ablations",
                "",
                "| Configuration | Accuracy | F1 | AUC-ROC | PR-AUC | Δ Accuracy |",
                "|--------------|----------|-----|---------|--------|------------|"
            ])
            
            # Get baseline accuracy for delta
            baseline_acc = None
            for name, metrics in cat_results.items():
                if "baseline" in name.lower() or "full" in name.lower():
                    baseline_acc = metrics.get("accuracy", 0)
                    break
            
            for name, metrics in cat_results.items():
                acc = metrics.get("accuracy", 0)
                f1 = metrics.get("f1", 0)
                auc = metrics.get("auc_roc", 0)
                pr_auc = metrics.get("pr_auc", 0)
                
                delta = ""
                if baseline_acc is not None:
                    diff = acc - baseline_acc
                    delta = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
                
                lines.append(
                    f"| {name} | {acc:.3f} | {f1:.3f} | {auc:.3f} | {pr_auc:.3f} | {delta} |"
                )
            
            lines.append("")
        
        output_path = self.output_dir / "ablation_results.md"
        output_path.write_text("\n".join(lines))
        return output_path
    
    def generate_ablation_table_latex(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Path:
        """
        Generate LaTeX ablation table for papers.
        
        Args:
            results: Dict mapping ablation config name to metrics
            
        Returns:
            Path to generated .tex file
        """
        lines = [
            "% Auto-generated ablation study table",
            f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation-study}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Category & Configuration & Accuracy & F1 & AUC-ROC & PR-AUC \\",
            r"\midrule"
        ]
        
        # Group by category
        categories = {}
        for name, metrics in results.items():
            cat = metrics.get("category", "Other")
            if cat not in categories:
                categories[cat] = {}
            categories[cat][name] = metrics
        
        for category, cat_results in categories.items():
            first = True
            for name, metrics in cat_results.items():
                cat_col = category.title() if first else ""
                first = False
                
                acc = metrics.get("accuracy", 0)
                f1 = metrics.get("f1", 0)
                auc = metrics.get("auc_roc", 0)
                pr_auc = metrics.get("pr_auc", 0)
                
                lines.append(
                    f"{cat_col} & {name.replace('_', ' ')} & "
                    f"{acc:.3f} & {f1:.3f} & {auc:.3f} & {pr_auc:.3f} \\\\"
                )
            lines.append(r"\midrule")
        
        # Remove last midrule
        lines[-1] = r"\bottomrule"
        
        lines.extend([
            r"\end{tabular}",
            r"\end{table*}"
        ])
        
        output_path = self.output_dir / "ablation_results.tex"
        output_path.write_text("\n".join(lines))
        return output_path
    
    def generate_cv_summary_markdown(
        self,
        cv_results: Dict[str, Any]
    ) -> Path:
        """
        Generate cross-validation summary.
        
        Args:
            cv_results: CV results dict with fold metrics
            
        Returns:
            Path to generated markdown file
        """
        lines = [
            "# Cross-Validation Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"- Number of folds: {cv_results.get('n_folds', 'N/A')}",
            f"- Total cases: {cv_results.get('total_cases', 'N/A')}",
            "",
            "## Summary Statistics",
            ""
        ]
        
        acc_mean = cv_results.get("accuracy_mean", 0)
        acc_std = cv_results.get("accuracy_std", 0)
        acc_ci = cv_results.get("accuracy_ci95", (0, 0))
        
        lines.extend([
            f"| Metric | Mean | Std | 95% CI |",
            f"|--------|------|-----|--------|",
            f"| Accuracy | {acc_mean:.4f} | {acc_std:.4f} | [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}] |",
            ""
        ])
        
        # Per-fold results
        fold_metrics = cv_results.get("fold_metrics", [])
        if fold_metrics:
            lines.extend([
                "## Per-Fold Results",
                "",
                "| Fold | Accuracy | Cases |",
                "|------|----------|-------|"
            ])
            
            for i, fold in enumerate(fold_metrics):
                acc = fold.get("accuracy", 0)
                cases = fold.get("cases_with_ground_truth", 0)
                lines.append(f"| {i+1} | {acc:.4f} | {cases} |")
        
        output_path = self.output_dir / "cv_summary.md"
        output_path.write_text("\n".join(lines))
        return output_path
    
    def _save_all_results_json(
        self,
        ablation_results: Dict[str, Dict[str, Any]],
        baseline_results: Dict[str, Dict[str, Any]],
        cv_results: Optional[Dict[str, Any]]
    ) -> Path:
        """Save all results to a single JSON file."""
        data = {
            "generated": datetime.now().isoformat(),
            "baselines": baseline_results or {},
            "ablations": ablation_results or {},
            "cross_validation": cv_results or {}
        }
        
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            return obj
        
        output_path = self.output_dir / "all_results.json"
        with open(output_path, "w") as f:
            json.dump(convert(data), f, indent=2)
        
        return output_path
    
    def generate_claim_verification_report(
        self,
        verification_report: Dict[str, Any]
    ) -> Path:
        """
        Generate claim verification report in markdown.
        
        Args:
            verification_report: Output from ClaimVerifier
            
        Returns:
            Path to generated report
        """
        lines = [
            "# Architectural Claim Verification Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- Total claims: {verification_report.get('total_claims', 0)}",
            f"- Verified: {verification_report.get('verified_count', 0)}",
            f"- Failed: {verification_report.get('failed_count', 0)}",
            f"- Skipped: {verification_report.get('skipped_count', 0)}",
            "",
            "## Claim Details",
            ""
        ]
        
        claims = verification_report.get("claims", {})
        for claim_id, claim_data in claims.items():
            status = claim_data.get("status", "UNKNOWN")
            status_emoji = {"PASS": "✓", "FAIL": "✗", "SKIPPED": "○"}.get(status, "?")
            
            lines.extend([
                f"### {status_emoji} {claim_id}",
                "",
                f"**Status:** {status}",
                f"**Description:** {claim_data.get('description', 'N/A')}",
                ""
            ])
            
            if claim_data.get("evidence"):
                lines.append("**Evidence:**")
                for k, v in claim_data["evidence"].items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
            
            if claim_data.get("reason"):
                lines.append(f"**Reason:** {claim_data['reason']}")
                lines.append("")
        
        output_path = self.output_dir / "claim_verification_report.md"
        output_path.write_text("\n".join(lines))
        return output_path


def generate_summary_statistics(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute summary statistics across all results.
    
    Args:
        results: Dict mapping name to metrics dict
        
    Returns:
        Summary statistics dict
    """
    if not results:
        return {}
    
    accuracies = [r.get("accuracy", 0) for r in results.values() if r.get("accuracy")]
    f1_scores = [r.get("f1", 0) for r in results.values() if r.get("f1")]
    
    return {
        "n_configurations": len(results),
        "accuracy": {
            "mean": np.mean(accuracies) if accuracies else 0,
            "std": np.std(accuracies) if accuracies else 0,
            "min": min(accuracies) if accuracies else 0,
            "max": max(accuracies) if accuracies else 0
        },
        "f1": {
            "mean": np.mean(f1_scores) if f1_scores else 0,
            "std": np.std(f1_scores) if f1_scores else 0,
            "min": min(f1_scores) if f1_scores else 0,
            "max": max(f1_scores) if f1_scores else 0
        }
    }
