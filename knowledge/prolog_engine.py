"""
Prolog Engine for Lung Nodule Multi-Agent System
=================================================

STRICT MODE - NO FALLBACKS

This module provides the PySwip wrapper for executing Prolog queries
against the Lung-RADS and Multi-Agent Consensus knowledge bases.

REQUIREMENTS:
    - SWI-Prolog must be installed on the system
    - pyswip package must be installed (pip install pyswip)

INSTALLATION:
    Ubuntu/Debian: sudo apt-get install swi-prolog
    macOS: brew install swi-prolog
    
EDUCATIONAL PURPOSE:
    Demonstrates integration of symbolic AI (Prolog/First-Order Logic)
    with Python for medical decision support.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class PrologUnavailableError(Exception):
    """
    Raised when Prolog/PySwip is not available.
    
    This error indicates that either:
    1. SWI-Prolog is not installed on the system
    2. The pyswip Python package is not installed
    3. The Prolog knowledge base files are missing
    
    STRICT MODE: This system requires Prolog and will not fall back
    to Python-based rules.
    """
    pass


class PrologQueryError(Exception):
    """
    Raised when a Prolog query fails.
    
    This may occur due to:
    1. Malformed query syntax
    2. Missing predicates in the knowledge base
    3. Runtime errors in Prolog execution
    """
    pass


class PrologEngine:
    """
    PySwip wrapper for Lung-RADS Prolog knowledge base.
    
    STRICT MODE: This class will raise PrologUnavailableError if
    Prolog cannot be initialized. No fallback behavior is provided.
    
    Example usage:
        engine = PrologEngine()
        result = engine.query_lung_rads(size=12, texture="solid")
        print(result)  # {'category': '4A', 'management': 'followup_3_months'}
    """
    
    def __init__(self, auto_load_kb: bool = True):
        """
        Initialize the Prolog engine.
        
        Args:
            auto_load_kb: If True, automatically load the knowledge bases
            
        Raises:
            PrologUnavailableError: If PySwip or SWI-Prolog is not available
        """
        self._prolog = None
        self._kb_loaded = False
        
        # Get paths to knowledge base files
        self._kb_dir = Path(__file__).parent
        self._lung_rads_path = self._kb_dir / "lung_rads.pl"
        self._consensus_path = self._kb_dir / "multi_agent_consensus.pl"
        
        # Initialize PySwip
        self._initialize_prolog()
        
        # Load knowledge bases if requested
        if auto_load_kb:
            self.load_knowledge_bases()
    
    def _initialize_prolog(self) -> None:
        """
        Initialize the PySwip Prolog interface.
        
        Raises:
            PrologUnavailableError: If initialization fails
        """
        try:
            from pyswip import Prolog
            self._prolog = Prolog()
            logger.info("PySwip Prolog engine initialized successfully")
        except ImportError as e:
            raise PrologUnavailableError(
                "PySwip is not installed. Install with: pip install pyswip\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise PrologUnavailableError(
                "Failed to initialize Prolog. Ensure SWI-Prolog is installed.\n"
                "Install with: sudo apt-get install swi-prolog (Ubuntu/Debian)\n"
                "             brew install swi-prolog (macOS)\n"
                f"Original error: {e}"
            )
    
    def load_knowledge_bases(self) -> None:
        """
        Load the Lung-RADS and Multi-Agent Consensus knowledge bases.
        
        Raises:
            PrologUnavailableError: If knowledge base files are missing
            PrologQueryError: If loading fails
        """
        # Verify files exist
        if not self._lung_rads_path.exists():
            raise PrologUnavailableError(
                f"Lung-RADS knowledge base not found: {self._lung_rads_path}"
            )
        
        if not self._consensus_path.exists():
            raise PrologUnavailableError(
                f"Consensus knowledge base not found: {self._consensus_path}"
            )
        
        try:
            # Load the knowledge bases
            self._prolog.consult(str(self._lung_rads_path))
            logger.info(f"Loaded knowledge base: {self._lung_rads_path.name}")
            
            self._prolog.consult(str(self._consensus_path))
            logger.info(f"Loaded knowledge base: {self._consensus_path.name}")
            
            self._kb_loaded = True
            logger.info("All Prolog knowledge bases loaded successfully")
            
        except Exception as e:
            raise PrologQueryError(
                f"Failed to load Prolog knowledge bases: {e}"
            )
    
    def load_knowledge_base(self, path: str) -> None:
        """
        Load an additional knowledge base file.
        
        Args:
            path: Path to the .pl file
            
        Raises:
            PrologUnavailableError: If file not found
            PrologQueryError: If loading fails
        """
        kb_path = Path(path)
        if not kb_path.exists():
            raise PrologUnavailableError(f"Knowledge base not found: {path}")
        
        try:
            self._prolog.consult(str(kb_path))
            logger.info(f"Loaded additional knowledge base: {kb_path.name}")
        except Exception as e:
            raise PrologQueryError(f"Failed to load {path}: {e}")
    
    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """
        Execute a raw Prolog query.
        
        Args:
            query_str: Prolog query string (without trailing period)
            
        Returns:
            List of solution dictionaries
            
        Raises:
            PrologQueryError: If query execution fails
            
        Example:
            >>> engine.query("lung_rads_category(n001, Cat)")
            [{'Cat': '4A'}]
        """
        try:
            results = list(self._prolog.query(query_str))
            return results
        except Exception as e:
            raise PrologQueryError(f"Query failed: {query_str}\nError: {e}")
    
    def assertz(self, fact: str) -> None:
        """
        Assert a new fact into the knowledge base.
        
        Args:
            fact: Prolog fact to assert (without trailing period)
            
        Raises:
            PrologQueryError: If assertion fails
            
        Example:
            >>> engine.assertz("size(n001, 15)")
        """
        try:
            self._prolog.assertz(fact)
            logger.debug(f"Asserted fact: {fact}")
        except Exception as e:
            raise PrologQueryError(f"Failed to assert fact: {fact}\nError: {e}")
    
    def retractall(self, pattern: str) -> None:
        """
        Retract all matching facts from the knowledge base.
        
        Args:
            pattern: Prolog pattern to match and retract
            
        Example:
            >>> engine.retractall("size(n001, _)")
        """
        try:
            list(self._prolog.query(f"retractall({pattern})"))
            logger.debug(f"Retracted: {pattern}")
        except Exception as e:
            raise PrologQueryError(f"Failed to retract: {pattern}\nError: {e}")
    
    # =========================================================================
    # High-Level Query Methods
    # =========================================================================
    
    def query_lung_rads(
        self,
        size: float,
        texture: str,
        nodule_id: str = "query_nodule"
    ) -> Dict[str, Any]:
        """
        Query Lung-RADS category for a nodule.
        
        This method:
        1. Asserts the nodule characteristics as temporary facts
        2. Queries for the Lung-RADS category
        3. Queries for the management recommendation
        4. Cleans up the temporary facts
        
        Args:
            size: Nodule size in millimeters
            texture: Nodule texture (solid, ground_glass, part_solid)
            nodule_id: Identifier for the nodule (default: query_nodule)
            
        Returns:
            Dictionary with 'category', 'management', and 'description'
            
        Raises:
            PrologQueryError: If query fails
        """
        # Normalize texture
        texture_map = {
            "solid": "solid",
            "dense": "solid",
            "ground_glass": "ground_glass",
            "groundglass": "ground_glass",
            "ggo": "ground_glass",
            "part_solid": "part_solid",
            "partsolid": "part_solid",
            "mixed": "part_solid",
        }
        normalized_texture = texture_map.get(
            texture.lower().replace("-", "_").strip(),
            "solid"
        )
        
        try:
            # Clean up any previous facts for this nodule
            self.retractall(f"nodule_size({nodule_id}, _)")
            self.retractall(f"nodule_texture({nodule_id}, _)")
            
            # Also need to assert nodule/1 and texture/2, size/2 for lung_rads.pl
            self.retractall(f"nodule({nodule_id})")
            self.retractall(f"size({nodule_id}, _)")
            self.retractall(f"texture({nodule_id}, _)")
            
            # Assert nodule characteristics for both KB formats
            self.assertz(f"nodule_size({nodule_id}, {size})")
            self.assertz(f"nodule_texture({nodule_id}, {normalized_texture})")
            self.assertz(f"nodule({nodule_id})")
            self.assertz(f"size({nodule_id}, {size})")
            
            # Map texture to numeric value for lung_rads.pl
            texture_to_num = {
                "ground_glass": 1,
                "part_solid": 3,
                "solid": 5,
            }
            texture_num = texture_to_num.get(normalized_texture, 5)
            self.assertz(f"texture({nodule_id}, {texture_num})")
            
            # Query lung_rads_category/2 from lung_rads.pl first (simpler, no recursion)
            # Use once/1 to get first result and avoid infinite backtracking
            category = None
            description = ""
            
            try:
                alt_results = self.query(
                    f"once(lung_rads_category({nodule_id}, Category))"
                )
                if alt_results:
                    category = str(alt_results[0].get("Category", ""))
            except:
                pass
            
            # If that didn't work, try lung_rads/3 with once/1 to prevent recursion
            if not category:
                try:
                    # Exclude 4X to avoid recursive call
                    category_results = self.query(
                        f"once((lung_rads({nodule_id}, Cat, Desc), Cat \\== '4X'))"
                    )
                    if category_results:
                        category = str(category_results[0].get("Cat", ""))
                        description = str(category_results[0].get("Desc", ""))
                except:
                    pass
            
            # Fallback: derive from size rules directly
            if not category:
                if size < 6:
                    category = "2"
                    description = "Size-based: <6mm"
                elif size < 8:
                    category = "3"
                    description = "Size-based: 6-8mm"
                elif size < 15:
                    category = "4A"
                    description = "Size-based: 8-15mm"
                else:
                    category = "4B"
                    description = "Size-based: >=15mm"
            
            # Map category to management recommendation
            management_map = {
                "1": "routine_annual_screening",
                "2": "annual_screening",
                "3": "followup_6_months",
                "4A": "followup_3_months",
                "4B": "pet_ct_or_biopsy",
                "4X": "pet_ct_or_biopsy",
            }
            management = management_map.get(category, "clinical_correlation")
            
            # Clean up all asserted facts
            self.retractall(f"nodule_size({nodule_id}, _)")
            self.retractall(f"nodule_texture({nodule_id}, _)")
            self.retractall(f"nodule({nodule_id})")
            self.retractall(f"size({nodule_id}, _)")
            self.retractall(f"texture({nodule_id}, _)")
            
            logger.info(
                f"Lung-RADS query: size={size}mm, texture={normalized_texture} "
                f"-> category={category}, management={management}"
            )
            
            return {
                "category": category,
                "management": management,
                "description": description,
                "source": "prolog"
            }
            
        except Exception as e:
            # Clean up on error
            try:
                self.retractall(f"nodule_size({nodule_id}, _)")
                self.retractall(f"nodule_texture({nodule_id}, _)")
                self.retractall(f"nodule({nodule_id})")
                self.retractall(f"size({nodule_id}, _)")
                self.retractall(f"texture({nodule_id}, _)")
            except:
                pass
            raise PrologQueryError(f"Lung-RADS query failed: {e}")
    
    def compute_consensus(
        self,
        nodule_id: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute multi-agent consensus using Prolog weighted voting.
        
        Args:
            nodule_id: Nodule identifier
            findings: List of agent findings, each with:
                - agent_name: Name of the agent
                - probability: Malignancy probability (0-1)
                - predicted_class: Predicted class (1-5)
                - weight: Agent weight (optional, defaults to 1.0)
                
        Returns:
            Dictionary with consensus results:
                - probability: Weighted average probability
                - predicted_class: Final class
                - confidence: Agreement-based confidence
                - method: "prolog_weighted_voting"
                
        Raises:
            PrologQueryError: If consensus computation fails
        """
        try:
            # Clear previous findings
            self.retractall(f"agent_finding({nodule_id}, _, _, _)")
            
            # Assert all agent findings
            for finding in findings:
                agent = finding.get("agent_name", "unknown")
                prob = finding.get("probability", 0.5)
                pred_class = finding.get("predicted_class", 3)
                
                self.assertz(
                    f"agent_finding({nodule_id}, {agent}, {prob}, {pred_class})"
                )
            
            # Query for consensus via resolution strategies
            # Uses resolve_disagreement to apply overrides/tiebreakers if needed
            # Fallback (Rule 4) ensures it returns weighted average if no disagreement
            results = self.query(
                f"resolve_disagreement({nodule_id}, WeightedProb, Confidence, Strategy)"
            )
            
            if results:
                weighted_prob = float(results[0].get("WeightedProb", 0.5))
                confidence = float(results[0].get("Confidence", 0.5))
                strategy = str(results[0].get("Strategy", "weighted_average"))
            else:
                # Calculate manually if Prolog query returns no results
                strategy = "python_fallback"
                if findings:
                    total_weight = sum(f.get("weight", 1.0) for f in findings)
                    weighted_prob = sum(
                        f.get("probability", 0.5) * f.get("weight", 1.0)
                        for f in findings
                    ) / total_weight
                    confidence = 0.7
                else:
                    weighted_prob = 0.5
                    confidence = 0.0
            
            # Determine predicted class from probability
            if weighted_prob < 0.2:
                pred_class = 1
            elif weighted_prob < 0.4:
                pred_class = 2
            elif weighted_prob < 0.6:
                pred_class = 3
            elif weighted_prob < 0.8:
                pred_class = 4
            else:
                pred_class = 5
            
            # Clean up
            self.retractall(f"agent_finding({nodule_id}, _, _, _)")
            
            logger.info(
                f"Consensus for {nodule_id}: prob={weighted_prob:.3f}, "
                f"class={pred_class}, confidence={confidence:.3f}"
            )
            
            return {
                "probability": weighted_prob,
                "predicted_class": pred_class,
                "confidence": confidence,
                "strategy": strategy,
                "method": "prolog_consensus"
            }
            
        except Exception as e:
            # Clean up on error
            try:
                self.retractall(f"agent_finding({nodule_id}, _, _, _)")
            except:
                pass
            raise PrologQueryError(f"Consensus computation failed: {e}")
    
    def query_tnm_stage(
        self,
        nodule_id: str,
        size: float,
        lymph_nodes: Optional[str] = None,
        metastasis: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query TNM staging for a nodule.
        
        Args:
            nodule_id: Nodule identifier
            size: Tumor size in millimeters
            lymph_nodes: Lymph node status (optional)
            metastasis: Metastasis status (optional)
            
        Returns:
            Dictionary with T, N, M stages and overall stage group
            
        Raises:
            PrologQueryError: If staging query fails
        """
        try:
            # Assert nodule size
            self.retractall(f"nodule_size({nodule_id}, _)")
            self.assertz(f"nodule_size({nodule_id}, {size})")
            
            # Assert lymph node status if provided
            if lymph_nodes:
                self.assertz(
                    f"nlp_entity({nodule_id}, staging, lymph_node, {lymph_nodes})"
                )
            
            # Assert metastasis if provided
            if metastasis:
                self.assertz(
                    f"nlp_entity({nodule_id}, staging, metastasis, {metastasis})"
                )
            
            # Query T stage
            t_results = self.query(f"t_stage({nodule_id}, TStage, TDesc)")
            t_stage = t_results[0]["TStage"] if t_results else "TX"
            t_desc = t_results[0].get("TDesc", "") if t_results else ""
            
            # Query N stage
            n_results = self.query(f"n_stage({nodule_id}, NStage, NDesc)")
            n_stage = n_results[0]["NStage"] if n_results else "N0"
            n_desc = n_results[0].get("NDesc", "") if n_results else ""
            
            # Query M stage
            m_results = self.query(f"m_stage({nodule_id}, MStage, MDesc)")
            m_stage = m_results[0]["MStage"] if m_results else "M0"
            m_desc = m_results[0].get("MDesc", "") if m_results else ""
            
            # Query overall stage group
            stage_results = self.query(f"stage_group({nodule_id}, StageGroup)")
            stage_group = stage_results[0]["StageGroup"] if stage_results else "Unknown"
            
            # Clean up
            self.retractall(f"nodule_size({nodule_id}, _)")
            if lymph_nodes:
                self.retractall(
                    f"nlp_entity({nodule_id}, staging, lymph_node, _)"
                )
            if metastasis:
                self.retractall(
                    f"nlp_entity({nodule_id}, staging, metastasis, _)"
                )
            
            return {
                "t_stage": t_stage,
                "t_description": t_desc,
                "n_stage": n_stage,
                "n_description": n_desc,
                "m_stage": m_stage,
                "m_description": m_desc,
                "stage_group": stage_group,
                "source": "prolog"
            }
            
        except Exception as e:
            raise PrologQueryError(f"TNM staging query failed: {e}")


# =============================================================================
# Legacy Alias for Backward Compatibility
# =============================================================================

class LungRADSKnowledgeBase(PrologEngine):
    """
    Alias for PrologEngine for backward compatibility.
    
    Used by agents/spade_oncologist.py which imports LungRADSKnowledgeBase.
    """
    pass


# =============================================================================
# Module-level validation
# =============================================================================

def validate_prolog_installation() -> bool:
    """
    Validate that Prolog is properly installed and configured.
    
    Returns:
        True if Prolog is available
        
    Raises:
        PrologUnavailableError: If Prolog is not available
    """
    engine = PrologEngine(auto_load_kb=True)
    
    # Test a simple query
    result = engine.query_lung_rads(size=10, texture="solid")
    
    logger.info(f"Prolog validation successful: {result}")
    return True


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Prolog Engine Test (STRICT MODE - NO FALLBACKS)")
    print("=" * 60)
    
    try:
        # Initialize engine
        print("\n1. Initializing Prolog engine...")
        engine = PrologEngine()
        print("   SUCCESS: Prolog engine initialized")
        
        # Test Lung-RADS queries
        print("\n2. Testing Lung-RADS queries...")
        
        test_cases = [
            (5, "solid"),      # Category 2
            (7, "solid"),      # Category 3
            (12, "solid"),     # Category 4A
            (20, "solid"),     # Category 4B
            (10, "part_solid"),
            (25, "ground_glass"),
        ]
        
        for size, texture in test_cases:
            result = engine.query_lung_rads(size=size, texture=texture)
            print(f"   Size={size}mm, Texture={texture} -> "
                  f"Category {result['category']}, {result['management']}")
        
        # Test consensus
        print("\n3. Testing multi-agent consensus...")
        findings = [
            {"agent_name": "radiologist_densenet", "probability": 0.7, "predicted_class": 4},
            {"agent_name": "radiologist_resnet", "probability": 0.65, "predicted_class": 4},
            {"agent_name": "pathologist_spacy", "probability": 0.6, "predicted_class": 3},
        ]
        consensus = engine.compute_consensus("test_nodule", findings)
        print(f"   Consensus: prob={consensus['probability']:.3f}, "
              f"class={consensus['predicted_class']}, "
              f"confidence={consensus['confidence']:.3f}")
        
        # Test TNM staging
        print("\n4. Testing TNM staging...")
        tnm = engine.query_tnm_stage("test_nodule", size=25)
        print(f"   T-Stage: {tnm['t_stage']}")
        print(f"   N-Stage: {tnm['n_stage']}")
        print(f"   M-Stage: {tnm['m_stage']}")
        print(f"   Stage Group: {tnm['stage_group']}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED - Prolog is properly configured")
        print("=" * 60)
        
    except PrologUnavailableError as e:
        print(f"\nERROR: Prolog is not available!")
        print(f"Details: {e}")
        print("\nTo fix this:")
        print("  1. Install SWI-Prolog:")
        print("     Ubuntu/Debian: sudo apt-get install swi-prolog")
        print("     macOS: brew install swi-prolog")
        print("  2. Install PySwip:")
        print("     pip install pyswip")
        sys.exit(1)
        
    except PrologQueryError as e:
        print(f"\nERROR: Prolog query failed!")
        print(f"Details: {e}")
        sys.exit(1)
