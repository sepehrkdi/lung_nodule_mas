#!/usr/bin/env python3
"""
Extract NLMCXR dataset archives.

This script extracts the compressed NLMCXR dataset files:
- NLMCXR_reports.tgz: Contains 3,956 XML report files
- NLMCXR_png.tgz: Contains 7,472 PNG image files
"""

import tarfile
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return current.parent.parent


def extract_nlmcxr(base_dir: Optional[Path] = None) -> None:
    """
    Extract NLMCXR dataset archives.

    Args:
        base_dir: Base directory containing the archives.
                  Defaults to PROJECT_ROOT/data/NLMCXR
    """
    if base_dir is None:
        project_root = find_project_root()
        base_dir = project_root / "data" / "NLMCXR"

    if not base_dir.exists():
        raise FileNotFoundError(f"NLMCXR directory not found at {base_dir}")

    logger.info(f"Extracting NLMCXR dataset from {base_dir}")

    # Extract reports archive
    reports_archive = base_dir / "NLMCXR_reports.tgz"
    if not reports_archive.exists():
        raise FileNotFoundError(f"Reports archive not found at {reports_archive}")

    logger.info(f"Extracting {reports_archive.name}...")
    with tarfile.open(reports_archive, "r:gz") as tar:
        tar.extractall(base_dir)
        logger.info(f"  Extracted {len(tar.getmembers())} files")

    # Extract images archive
    images_archive = base_dir / "NLMCXR_png.tgz"
    if not images_archive.exists():
        raise FileNotFoundError(f"Images archive not found at {images_archive}")

    logger.info(f"Extracting {images_archive.name}...")
    images_dir = base_dir / "images"
    images_dir.mkdir(exist_ok=True)

    with tarfile.open(images_archive, "r:gz") as tar:
        tar.extractall(images_dir)
        logger.info(f"  Extracted {len(tar.getmembers())} files")

    # Verify extraction
    xml_dir = base_dir / "ecgen-radiology"
    if xml_dir.exists():
        xml_count = len(list(xml_dir.glob("*.xml")))
        logger.info(f"Verification: Found {xml_count} XML files in {xml_dir.name}/")
    else:
        logger.warning(f"Warning: Expected directory {xml_dir} not found after extraction")

    png_count = len(list(images_dir.rglob("*.png")))
    logger.info(f"Verification: Found {png_count} PNG files in images/")

    logger.info("Extraction complete!")


def main():
    """Main entry point."""
    try:
        extract_nlmcxr()
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
