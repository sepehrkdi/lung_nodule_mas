
import sys
from pathlib import Path
import numpy as np

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from data.nlmcxr_parser import parse_all_nlmcxr_cases

def main():
    xml_dir = project_root / "data" / "NLMCXR" / "ecgen-radiology"
    if not xml_dir.exists():
        print(f"Error: Directory {xml_dir} not found.")
        return

    print(f"Loading cases from {xml_dir}...")
    cases = parse_all_nlmcxr_cases(xml_dir)
    
    lengths = []
    case_objects = []

    for cid, case in cases.items():
        text = f"{case.findings} {case.impression}".strip()
        l = len(text)
        lengths.append(l)
        case_objects.append((l, cid, text))

    if not lengths:
        print("No cases found.")
        return

    lengths = np.array(lengths)
    print(f"\nTotal Reports: {len(lengths)}")
    print(f"Min: {np.min(lengths)}")
    print(f"Max: {np.max(lengths)}")
    print(f"Mean: {np.mean(lengths):.2f}")
    print(f"Median: {np.median(lengths)}")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(lengths, percentiles)
    print("\nPercentiles:")
    for p, v in zip(percentiles, perc_vals):
        print(f"  {p}th: {v:.1f}")

    # Count how many below thresholds
    print("\nThreshold Counts:")
    for t in [40, 60, 80, 90, 100]:
        count = np.sum(lengths < t)
        perc = (count / len(lengths)) * 100
        print(f"  < {t} chars: {count} ({perc:.2f}%)")

    # Show examples around thresholds
    case_objects.sort(key=lambda x: x[0])
    
    def print_examples_near(target, k=3):
        print(f"\n--- Examples near {target} chars ---")
        # Find closest
        closest_indices = np.argsort(np.abs(lengths - target))[:k]
        for idx in closest_indices:
            l, cid, text = case_objects[idx] # Wait, this sorting is on separate list
            # Re-find in case_objects
            # Better to just search in sorted case_objects
            pass

    # Easier way to find examples
    print("\n--- Qualitative Examples ---")
    targets = [60, 80, 90]
    for t in targets:
        print(f"\n[Target: ~{t} chars]")
        # filter those around t +/- 2
        candidates = [c for c in case_objects if t-2 <= c[0] <= t+5]
        for i, (l, cid, text) in enumerate(candidates[:3]):
             print(f"  ({l} chars) [{cid}]: {text}")

if __name__ == "__main__":
    main()
