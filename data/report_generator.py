"""
Report Generator - Converts Structured Annotations to Natural Language
=======================================================================

EDUCATIONAL PURPOSE - NLP CONCEPTS:

This module generates natural language pathology reports from structured
nodule annotations. This serves two purposes:

1. PROVIDES INPUT FOR NLP AGENT:
   The Pathologist agent needs text to analyze, demonstrating:
   - Tokenization
   - Named Entity Recognition (NER)
   - Pattern matching
   - Information extraction

2. DEMONSTRATES TEXT GENERATION:
   Shows how structured data can be converted to natural language,
   which is the inverse of what NLP systems typically do.

SEMANTIC MAPPINGS:
We map numeric LIDC scores to clinical terminology that appears
in real radiology reports, allowing the NLP agent to practice
extracting meaningful information.
"""

from typing import Dict, Any, Optional
import random


class ReportGenerator:
    """
    Generates natural language radiology reports from nodule features.
    
    EDUCATIONAL PURPOSE:
    This class creates realistic-looking pathology reports that the
    Pathologist agent can analyze using NLP techniques. The reports
    follow standard radiology report structure:
    
    1. FINDINGS: Description of what was observed
    2. IMPRESSION: Clinical interpretation
    3. RECOMMENDATION: Suggested next steps
    
    Usage:
        generator = ReportGenerator()
        report = generator.generate(features_dict)
    """
    
    # Semantic mappings for LIDC scores
    # These match the terminology used in real radiology reports
    
    TEXTURE_DESCRIPTIONS = {
        1: "pure ground-glass opacity (GGO)",
        2: "predominantly ground-glass with minimal solid component",
        3: "part-solid nodule with ground-glass and solid components",
        4: "predominantly solid with minor ground-glass component",
        5: "solid"
    }
    
    MARGIN_DESCRIPTIONS = {
        1: "poorly defined, indistinct margins",
        2: "near poorly defined margins",
        3: "moderately well-defined margins",
        4: "near sharp, well-defined margins",
        5: "sharp, well-circumscribed margins"
    }
    
    SPICULATION_DESCRIPTIONS = {
        1: "no spiculation",
        2: "minimal spiculation",
        3: "moderate spiculation",
        4: "significant spiculation present",
        5: "marked spiculation with corona radiata sign"
    }
    
    LOBULATION_DESCRIPTIONS = {
        1: "no lobulation",
        2: "minimal lobulation",
        3: "moderate lobulation",
        4: "significant lobular contour",
        5: "marked lobulation with scalloped borders"
    }
    
    CALCIFICATION_DESCRIPTIONS = {
        1: "popcorn calcification pattern (suggestive of hamartoma)",
        2: "laminated calcification pattern (benign appearance)",
        3: "solid calcification",
        4: "eccentric/non-central calcification (concerning)",
        5: "central calcification (typically benign)",
        6: "no calcification identified"
    }
    
    SPHERICITY_DESCRIPTIONS = {
        1: "markedly elongated, linear morphology",
        2: "ovoid/elongated shape",
        3: "ovoid morphology",
        4: "nearly round/ovoid shape",
        5: "round, spherical morphology"
    }
    
    MALIGNANCY_IMPRESSIONS = {
        1: "Highly likely benign. Features strongly suggest benign etiology.",
        2: "Probably benign. Low suspicion for malignancy.",
        3: "Indeterminate. Unable to reliably characterize; further evaluation recommended.",
        4: "Moderately suspicious for malignancy. Concerning features present.",
        5: "Highly suspicious for malignancy. Features strongly suggestive of primary lung cancer."
    }
    
    LOCATION_PHRASES = [
        "in the {location}",
        "located in the {location}",
        "identified within the {location}",
        "visualized in the {location}",
        "present in the {location}"
    ]
    
    def __init__(self, include_measurements: bool = True, formal_style: bool = True):
        """
        Initialize the report generator.
        
        Args:
            include_measurements: Include exact size measurements
            formal_style: Use formal medical report style
        """
        self.include_measurements = include_measurements
        self.formal_style = formal_style
        
    def generate(self, features: Dict[str, Any]) -> str:
        """
        Generate a complete radiology report from nodule features.
        
        Args:
            features: Dictionary containing nodule annotations
            
        Returns:
            Formatted natural language report
        """
        sections = []
        
        # Header
        nodule_id = features.get("nodule_id", "unknown")
        sections.append(f"CHEST CT - PULMONARY NODULE EVALUATION")
        sections.append(f"Nodule Reference: {nodule_id}")
        sections.append("")
        
        # FINDINGS section
        sections.append("FINDINGS:")
        sections.append(self._generate_findings(features))
        sections.append("")
        
        # IMPRESSION section
        sections.append("IMPRESSION:")
        sections.append(self._generate_impression(features))
        sections.append("")
        
        # RECOMMENDATION section
        sections.append("RECOMMENDATION:")
        sections.append(self._generate_recommendation(features))
        
        return "\n".join(sections)
    
    def _generate_findings(self, features: Dict[str, Any]) -> str:
        """Generate the FINDINGS section of the report."""
        findings = []
        
        # Size and location
        diameter = features.get("diameter_mm", 0)
        location = features.get("location", "lung parenchyma")
        
        location_phrase = random.choice(self.LOCATION_PHRASES).format(location=location)
        
        if self.include_measurements:
            findings.append(
                f"A {diameter:.1f} mm pulmonary nodule is identified {location_phrase}."
            )
        else:
            size_desc = self._size_to_description(diameter)
            findings.append(
                f"A {size_desc} pulmonary nodule is identified {location_phrase}."
            )
        
        # Texture/attenuation
        texture = features.get("texture", 5)
        texture_desc = self.TEXTURE_DESCRIPTIONS.get(texture, "solid")
        findings.append(f"The nodule demonstrates {texture_desc} attenuation.")
        
        # Margins
        margin = features.get("margin", 5)
        margin_desc = self.MARGIN_DESCRIPTIONS.get(margin, "well-defined margins")
        findings.append(f"Margins are {margin_desc}.")
        
        # Spiculation (if present)
        spiculation = features.get("spiculation", 1)
        if spiculation >= 2:
            spic_desc = self.SPICULATION_DESCRIPTIONS.get(spiculation, "no spiculation")
            findings.append(f"The nodule demonstrates {spic_desc}.")
        
        # Lobulation (if present)
        lobulation = features.get("lobulation", 1)
        if lobulation >= 2:
            lob_desc = self.LOBULATION_DESCRIPTIONS.get(lobulation, "no lobulation")
            findings.append(f"There is {lob_desc} of the nodule contour.")
        
        # Calcification
        calcification = features.get("calcification", 6)
        calc_desc = self.CALCIFICATION_DESCRIPTIONS.get(calcification, "no calcification")
        if calcification != 6:  # Only mention if calcification is present
            findings.append(f"Calcification: {calc_desc}.")
        else:
            findings.append("No internal calcification is identified.")
        
        # Sphericity/shape
        sphericity = features.get("sphericity", 5)
        if sphericity <= 3:  # Only mention if notably non-spherical
            shape_desc = self.SPHERICITY_DESCRIPTIONS.get(sphericity, "irregular shape")
            findings.append(f"Morphology: {shape_desc}.")
        
        return " ".join(findings)
    
    def _generate_impression(self, features: Dict[str, Any]) -> str:
        """Generate the IMPRESSION section of the report."""
        impressions = []
        
        diameter = features.get("diameter_mm", 0)
        malignancy = features.get("malignancy", 3)
        location = features.get("location", "lung")
        
        # Size classification
        if diameter < 6:
            size_class = "small"
        elif diameter < 15:
            size_class = "intermediate-sized"
        else:
            size_class = "large"
        
        impressions.append(
            f"{size_class.capitalize()} pulmonary nodule in the {location}."
        )
        
        # Malignancy assessment
        mal_impression = self.MALIGNANCY_IMPRESSIONS.get(malignancy, "Indeterminate.")
        impressions.append(mal_impression)
        
        # Add Lung-RADS category estimation
        lung_rads = self._estimate_lung_rads(features)
        impressions.append(f"Estimated Lung-RADS Category: {lung_rads}.")
        
        return " ".join(impressions)
    
    def _generate_recommendation(self, features: Dict[str, Any]) -> str:
        """Generate the RECOMMENDATION section of the report."""
        malignancy = features.get("malignancy", 3)
        diameter = features.get("diameter_mm", 0)
        
        if malignancy <= 1:
            return "Annual low-dose CT screening recommended. No short-term follow-up required."
        elif malignancy == 2:
            if diameter < 6:
                return "Annual CT surveillance recommended per Fleischner Society guidelines."
            else:
                return "Follow-up CT in 6-12 months to assess stability. Consider annual surveillance if stable."
        elif malignancy == 3:
            return "Follow-up CT in 3-6 months recommended. If stable, continue annual surveillance. Consider PET-CT for further characterization."
        elif malignancy == 4:
            return "Short-interval follow-up CT in 3 months OR PET-CT recommended for metabolic characterization. Consider tissue sampling if suspicious on PET."
        else:  # malignancy == 5
            return "PET-CT strongly recommended. Consider CT-guided biopsy or surgical consultation for tissue diagnosis. Recommend multidisciplinary tumor board discussion."
    
    def _size_to_description(self, diameter: float) -> str:
        """Convert diameter to descriptive size."""
        if diameter < 4:
            return "tiny"
        elif diameter < 6:
            return "small"
        elif diameter < 10:
            return "subcentimeter"
        elif diameter < 15:
            return "moderate-sized"
        elif diameter < 30:
            return "large"
        else:
            return "very large"
    
    def _estimate_lung_rads(self, features: Dict[str, Any]) -> str:
        """Estimate Lung-RADS category from features."""
        diameter = features.get("diameter_mm", 0)
        texture = features.get("texture", 5)
        spiculation = features.get("spiculation", 1)
        margin = features.get("margin", 5)
        calcification = features.get("calcification", 6)
        
        # Benign calcification patterns
        if calcification in [1, 2, 5]:  # Popcorn, laminated, or central
            return "2 (Benign)"
        
        is_solid = texture >= 4
        is_part_solid = texture == 3
        is_ground_glass = texture <= 2
        
        # Ground glass nodules
        if is_ground_glass:
            if diameter < 30:
                return "2 (Benign)"
            else:
                return "3 (Probably Benign)"
        
        # Part-solid nodules
        if is_part_solid:
            if diameter < 6:
                return "2 (Benign)"
            elif diameter < 8:
                return "3 (Probably Benign)"
            else:
                return "4A (Suspicious)"
        
        # Solid nodules
        if is_solid:
            if diameter < 6:
                return "2 (Benign)"
            elif diameter < 8:
                return "3 (Probably Benign)"
            elif diameter < 15:
                # Check for high-risk features
                if spiculation >= 4 and margin <= 2:
                    return "4B (Suspicious)"
                return "4A (Suspicious)"
            else:
                return "4B (Suspicious)"
        
        return "3 (Probably Benign)"
    
    def generate_brief(self, features: Dict[str, Any]) -> str:
        """
        Generate a brief one-paragraph report.
        
        Useful for quick testing and demonstration.
        
        Args:
            features: Nodule features dictionary
            
        Returns:
            Brief report string
        """
        diameter = features.get("diameter_mm", 0)
        location = features.get("location", "lung parenchyma")
        texture = features.get("texture", 5)
        margin = features.get("margin", 5)
        malignancy = features.get("malignancy", 3)
        spiculation = features.get("spiculation", 1)
        
        texture_desc = self.TEXTURE_DESCRIPTIONS.get(texture, "solid")
        margin_desc = self.MARGIN_DESCRIPTIONS.get(margin, "well-defined")
        mal_label = features.get("malignancy_label", "indeterminate")
        
        spic_text = ""
        if spiculation >= 3:
            spic_text = f" with {self.SPICULATION_DESCRIPTIONS.get(spiculation, 'spiculation')}"
        
        return (
            f"CT imaging reveals a {diameter:.1f}mm {texture_desc} pulmonary nodule "
            f"in the {location}. The margins are {margin_desc}{spic_text}. "
            f"Overall impression: {mal_label} for malignancy."
        )


def generate_report(features: Dict[str, Any], brief: bool = False) -> str:
    """
    Convenience function to generate a report.
    
    Args:
        features: Nodule features dictionary
        brief: If True, generate brief one-paragraph report
        
    Returns:
        Generated report string
    """
    generator = ReportGenerator()
    if brief:
        return generator.generate_brief(features)
    return generator.generate(features)


if __name__ == "__main__":
    # Demo usage
    print("=== Report Generator Demo ===\n")
    
    # Sample features (suspicious nodule)
    sample_features = {
        "nodule_id": "007",
        "diameter_mm": 18.7,
        "malignancy": 4,
        "malignancy_label": "Moderately Suspicious",
        "spiculation": 4,
        "margin": 2,
        "texture": 5,
        "lobulation": 4,
        "calcification": 6,
        "sphericity": 2,
        "location": "right upper lobe"
    }
    
    generator = ReportGenerator()
    
    print("--- Full Report ---")
    print(generator.generate(sample_features))
    print("\n--- Brief Report ---")
    print(generator.generate_brief(sample_features))
    
    # Benign example
    print("\n\n=== Benign Nodule Example ===\n")
    benign_features = {
        "nodule_id": "001",
        "diameter_mm": 8.5,
        "malignancy": 1,
        "malignancy_label": "Highly Unlikely",
        "spiculation": 1,
        "margin": 5,
        "texture": 5,
        "lobulation": 1,
        "calcification": 1,  # Popcorn - benign
        "sphericity": 5,
        "location": "right upper lobe"
    }
    
    print("--- Brief Report (Benign) ---")
    print(generator.generate_brief(benign_features))
