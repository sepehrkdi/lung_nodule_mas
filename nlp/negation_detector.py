"""
NegEx-style negation and uncertainty detection for clinical NLP.
"""

import re
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Certainty(Enum):
    """Certainty status for a medical entity mention."""
    AFFIRMED = "affirmed"      # Positively stated
    NEGATED = "negated"        # Explicitly denied
    UNCERTAIN = "uncertain"    # Hedged or uncertain


@dataclass
class Trigger:
    """
    A trigger phrase for negation or uncertainty detection.
    
    Attributes:
        phrase: The trigger text (e.g., "no evidence of")
        category: What this trigger indicates (NEGATED or UNCERTAIN)
        direction: Scope direction ("forward", "backward", "bidirectional")
    """
    phrase: str
    category: Certainty
    direction: str  # "forward", "backward", or "bidirectional"


@dataclass 
class EntityCertainty:
    """Result of certainty detection for an entity."""
    text: str
    start: int
    end: int
    certainty: Certainty
    trigger: Optional[str] = None
    trigger_position: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "certainty": self.certainty.value,
            "trigger": self.trigger
        }


# =============================================================================
# TRIGGER WORD LISTS
# Based on NegEx (Chapman et al., 2001) and ConText (Harkema et al., 2009)
# =============================================================================

NEGATION_TRIGGERS = [
    # Pre-negation triggers (scope goes FORWARD)
    Trigger("no evidence of", Certainty.NEGATED, "forward"),
    Trigger("no signs of", Certainty.NEGATED, "forward"),
    Trigger("no sign of", Certainty.NEGATED, "forward"),
    Trigger("no", Certainty.NEGATED, "forward"),
    Trigger("without evidence of", Certainty.NEGATED, "forward"),
    Trigger("without", Certainty.NEGATED, "forward"),
    Trigger("negative for", Certainty.NEGATED, "forward"),
    Trigger("denies", Certainty.NEGATED, "forward"),
    Trigger("denied", Certainty.NEGATED, "forward"),
    Trigger("no evidence", Certainty.NEGATED, "forward"),
    Trigger("never", Certainty.NEGATED, "forward"),
    Trigger("absent", Certainty.NEGATED, "forward"),
    Trigger("absence of", Certainty.NEGATED, "forward"),
    Trigger("rules out", Certainty.NEGATED, "forward"),
    Trigger("ruled out", Certainty.NEGATED, "forward"),
    Trigger("rule out", Certainty.NEGATED, "forward"),
    Trigger("free of", Certainty.NEGATED, "forward"),
    Trigger("unremarkable for", Certainty.NEGATED, "forward"),
    Trigger("unremarkable", Certainty.NEGATED, "forward"),
    Trigger("not demonstrate", Certainty.NEGATED, "forward"),
    Trigger("not see", Certainty.NEGATED, "forward"),
    Trigger("not identify", Certainty.NEGATED, "forward"),
    Trigger("not", Certainty.NEGATED, "forward"),
    Trigger("fails to reveal", Certainty.NEGATED, "forward"),
    Trigger("failed to reveal", Certainty.NEGATED, "forward"),
    Trigger("clear of", Certainty.NEGATED, "forward"),
    
    # Post-negation triggers (scope goes BACKWARD)
    Trigger("is ruled out", Certainty.NEGATED, "backward"),
    Trigger("was ruled out", Certainty.NEGATED, "backward"),
    Trigger("are ruled out", Certainty.NEGATED, "backward"),
    Trigger("have been ruled out", Certainty.NEGATED, "backward"),
    Trigger("has been ruled out", Certainty.NEGATED, "backward"),
    Trigger("unlikely", Certainty.NEGATED, "backward"),
    Trigger("is unlikely", Certainty.NEGATED, "backward"),
    Trigger("was not seen", Certainty.NEGATED, "backward"),
    Trigger("not seen", Certainty.NEGATED, "backward"),
    Trigger("not identified", Certainty.NEGATED, "backward"),
    Trigger("not demonstrated", Certainty.NEGATED, "backward"),
    Trigger("is absent", Certainty.NEGATED, "backward"),
    Trigger("are absent", Certainty.NEGATED, "backward"),
]

UNCERTAINTY_TRIGGERS = [
    # Pre-uncertainty triggers (scope goes FORWARD)
    Trigger("possible", Certainty.UNCERTAIN, "forward"),
    Trigger("possibly", Certainty.UNCERTAIN, "forward"),
    Trigger("probable", Certainty.UNCERTAIN, "forward"),
    Trigger("probably", Certainty.UNCERTAIN, "forward"),
    Trigger("may represent", Certainty.UNCERTAIN, "forward"),
    Trigger("may be", Certainty.UNCERTAIN, "forward"),
    Trigger("might be", Certainty.UNCERTAIN, "forward"),
    Trigger("might represent", Certainty.UNCERTAIN, "forward"),
    Trigger("could be", Certainty.UNCERTAIN, "forward"),
    Trigger("could represent", Certainty.UNCERTAIN, "forward"),
    Trigger("cannot exclude", Certainty.UNCERTAIN, "forward"),
    Trigger("can not exclude", Certainty.UNCERTAIN, "forward"),
    Trigger("cannot rule out", Certainty.UNCERTAIN, "forward"),
    Trigger("can not rule out", Certainty.UNCERTAIN, "forward"),
    Trigger("questionable", Certainty.UNCERTAIN, "forward"),
    Trigger("question of", Certainty.UNCERTAIN, "forward"),
    Trigger("equivocal", Certainty.UNCERTAIN, "forward"),
    Trigger("suspicious for", Certainty.UNCERTAIN, "forward"),
    Trigger("suspicion of", Certainty.UNCERTAIN, "forward"),
    Trigger("suspicion for", Certainty.UNCERTAIN, "forward"),
    Trigger("suggestive of", Certainty.UNCERTAIN, "forward"),
    Trigger("suggests", Certainty.UNCERTAIN, "forward"),
    Trigger("suggesting", Certainty.UNCERTAIN, "forward"),
    Trigger("appears to be", Certainty.UNCERTAIN, "forward"),
    Trigger("appears to represent", Certainty.UNCERTAIN, "forward"),
    Trigger("consistent with", Certainty.UNCERTAIN, "forward"),
    Trigger("compatible with", Certainty.UNCERTAIN, "forward"),
    Trigger("differential includes", Certainty.UNCERTAIN, "forward"),
    Trigger("differential diagnosis includes", Certainty.UNCERTAIN, "forward"),
    Trigger("reportedly", Certainty.UNCERTAIN, "forward"),
    Trigger("presumably", Certainty.UNCERTAIN, "forward"),
    Trigger("likely", Certainty.UNCERTAIN, "forward"),
    Trigger("favor", Certainty.UNCERTAIN, "forward"),
    Trigger("favoring", Certainty.UNCERTAIN, "forward"),
    Trigger("consider", Certainty.UNCERTAIN, "forward"),
    Trigger("versus", Certainty.UNCERTAIN, "bidirectional"),
    Trigger("vs", Certainty.UNCERTAIN, "bidirectional"),
    Trigger("vs.", Certainty.UNCERTAIN, "bidirectional"),
    Trigger("or", Certainty.UNCERTAIN, "bidirectional"),
    
    # Post-uncertainty triggers (scope goes BACKWARD)
    Trigger("is suspected", Certainty.UNCERTAIN, "backward"),
    Trigger("is questionable", Certainty.UNCERTAIN, "backward"),
    Trigger("cannot be excluded", Certainty.UNCERTAIN, "backward"),
    Trigger("cannot be ruled out", Certainty.UNCERTAIN, "backward"),
    Trigger("should be considered", Certainty.UNCERTAIN, "backward"),
    Trigger("remains uncertain", Certainty.UNCERTAIN, "backward"),
    Trigger("is uncertain", Certainty.UNCERTAIN, "backward"),
    Trigger("is indeterminate", Certainty.UNCERTAIN, "backward"),
]

# Scope termination terms - these END the trigger's scope
SCOPE_TERMINATORS = [
    # Conjunctions that change context
    "but", "however", "although", "though", "except",
    "apart from", "aside from", "nevertheless", "nonetheless",
    "yet", "still", "whereas", "while",
    
    # Causal terms (change subject)
    "cause", "causing", "caused by", "causes",
    "secondary to", "due to", "because", "because of",
    "reason", "etiology", "as a result",
    
    # Relative clauses
    "which", "who", "that", "whose",
    
    # Punctuation (converted to words for matching)
    ".", ";", ":", "\n"
]


# =============================================================================
# NEGEX DETECTOR CLASS
# =============================================================================

class NegExDetector:
    """NegEx-style detector for negation and uncertainty in clinical text."""
    
    def __init__(self, scope_window: int = 6):
        """
        Initialize the detector.
        
        Args:
            scope_window: Maximum number of words to look for entities
                         after/before a trigger (default: 6, from original NegEx)
        """
        self.scope_window = scope_window
        
        # Combine negation and uncertainty triggers
        self.triggers = NEGATION_TRIGGERS + UNCERTAINTY_TRIGGERS
        
        # Sort by phrase length (longest first) for greedy matching
        self.triggers.sort(key=lambda t: len(t.phrase), reverse=True)
        
        # Precompile terminator pattern
        self._terminator_pattern = self._build_terminator_pattern()
    
    def _build_terminator_pattern(self) -> re.Pattern:
        """Build regex pattern for scope terminators."""
        # Escape special characters and build pattern
        terms = []
        for term in SCOPE_TERMINATORS:
            if term in [".", ";", ":", "\n"]:
                terms.append(re.escape(term))
            else:
                terms.append(r'\b' + re.escape(term) + r'\b')
        
        return re.compile('|'.join(terms), re.IGNORECASE)
    
    def find_triggers(self, text: str) -> List[Tuple[int, int, Trigger]]:
        """
        Find all trigger phrases in the text.
        
        Args:
            text: The input text
            
        Returns:
            List of (start, end, Trigger) tuples, sorted by position
        """
        found = []
        text_lower = text.lower()
        
        for trigger in self.triggers:
            # Build pattern for this trigger
            pattern = r'\b' + re.escape(trigger.phrase.lower()) + r'\b'
            
            for match in re.finditer(pattern, text_lower):
                found.append((match.start(), match.end(), trigger))
        
        # Sort by position
        found.sort(key=lambda x: x[0])
        
        # Remove overlapping triggers (keep longest = first due to sort order)
        non_overlapping = []
        last_end = -1
        for start, end, trigger in found:
            if start >= last_end:
                non_overlapping.append((start, end, trigger))
                last_end = end
        
        return non_overlapping
    
    def _find_scope_end_forward(self, text: str, trigger_end: int) -> int:
        """
        Find where the forward scope ends.
        
        Scope ends at:
        1. A termination term
        2. End of sentence (punctuation)
        3. Maximum window size (word count)
        """
        remaining = text[trigger_end:]
        
        # Check for terminator
        term_match = self._terminator_pattern.search(remaining)
        if term_match:
            # Scope ends at terminator
            terminator_pos = trigger_end + term_match.start()
        else:
            terminator_pos = len(text)
        
        # Also limit by word count
        words = remaining.split()
        if len(words) > self.scope_window:
            # Find position after scope_window words
            word_limit_pos = trigger_end
            for i, word in enumerate(words[:self.scope_window]):
                word_limit_pos = text.find(word, word_limit_pos) + len(word) + 1
            terminator_pos = min(terminator_pos, word_limit_pos)
        
        return terminator_pos
    
    def _find_scope_start_backward(self, text: str, trigger_start: int) -> int:
        """
        Find where the backward scope starts.
        
        Scope starts at:
        1. A termination term (going backward)
        2. Start of sentence
        3. Maximum window size (word count)
        """
        preceding = text[:trigger_start]
        
        # Check for terminator (find last occurrence)
        term_matches = list(self._terminator_pattern.finditer(preceding))
        if term_matches:
            # Scope starts after last terminator
            last_term = term_matches[-1]
            terminator_pos = last_term.end()
        else:
            terminator_pos = 0
        
        # Also limit by word count
        words = preceding.split()
        if len(words) > self.scope_window:
            # Start from word at scope_window position from end
            start_words = words[-self.scope_window:]
            word_limit_pos = preceding.rfind(start_words[0])
            terminator_pos = max(terminator_pos, word_limit_pos)
        
        return terminator_pos
    
    def _is_entity_in_scope(
        self,
        entity_start: int,
        entity_end: int,
        trigger_start: int,
        trigger_end: int,
        scope_boundary: int,
        direction: str
    ) -> bool:
        """Check if an entity falls within the trigger's scope."""
        
        if direction == "forward":
            # Entity must be after trigger and before scope boundary
            return trigger_end <= entity_start <= scope_boundary
        
        elif direction == "backward":
            # Entity must be before trigger and after scope boundary
            return scope_boundary <= entity_end <= trigger_start
        
        else:  # bidirectional
            # Check both directions with smaller window
            forward_ok = trigger_end <= entity_start <= trigger_end + 100
            backward_ok = trigger_start - 100 <= entity_end <= trigger_start
            return forward_ok or backward_ok
    
    def detect(
        self,
        text: str,
        entities: List[Tuple[str, int, int]]
    ) -> List[EntityCertainty]:
        """
        Detect negation and uncertainty for a list of entities.
        
        Args:
            text: The full report text
            entities: List of (entity_text, start, end) tuples
            
        Returns:
            List of EntityCertainty objects with certainty labels
        """
        results = []
        triggers = self.find_triggers(text)
        
        for ent_text, ent_start, ent_end in entities:
            certainty = Certainty.AFFIRMED
            matched_trigger = None
            matched_trigger_pos = None
            
            # Check each trigger for scope overlap
            for trig_start, trig_end, trigger in triggers:
                # Calculate scope boundary
                if trigger.direction == "forward":
                    scope_boundary = self._find_scope_end_forward(text, trig_end)
                elif trigger.direction == "backward":
                    scope_boundary = self._find_scope_start_backward(text, trig_start)
                else:
                    scope_boundary = trig_end  # Handle bidirectional in _is_entity_in_scope
                
                # Check if entity is in scope
                if self._is_entity_in_scope(
                    ent_start, ent_end,
                    trig_start, trig_end,
                    scope_boundary, trigger.direction
                ):
                    # NEGATED takes precedence over UNCERTAIN
                    if trigger.category == Certainty.NEGATED:
                        certainty = Certainty.NEGATED
                        matched_trigger = trigger.phrase
                        matched_trigger_pos = (trig_start, trig_end)
                        break  # Negation is definitive
                    elif trigger.category == Certainty.UNCERTAIN and certainty != Certainty.NEGATED:
                        certainty = Certainty.UNCERTAIN
                        matched_trigger = trigger.phrase
                        matched_trigger_pos = (trig_start, trig_end)
            
            results.append(EntityCertainty(
                text=ent_text,
                start=ent_start,
                end=ent_end,
                certainty=certainty,
                trigger=matched_trigger,
                trigger_position=matched_trigger_pos
            ))
        
        return results
    
    def detect_in_text(self, text: str, entity_pattern: str) -> List[EntityCertainty]:
        """
        Convenience method: find entities matching a pattern and detect their certainty.
        
        Args:
            text: The input text
            entity_pattern: Regex pattern to find entities
            
        Returns:
            List of EntityCertainty for all matched entities
        """
        # Find all matches of the entity pattern
        entities = []
        for match in re.finditer(entity_pattern, text, re.IGNORECASE):
            entities.append((match.group(), match.start(), match.end()))
        
        return self.detect(text, entities)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_negated(text: str, entity_text: str) -> bool:
    """
    Quick check if an entity is negated in the text.
    
    Args:
        text: Full text
        entity_text: The entity to check
        
    Returns:
        True if entity is negated
    """
    detector = NegExDetector()
    
    # Find entity position
    match = re.search(re.escape(entity_text), text, re.IGNORECASE)
    if not match:
        return False
    
    entities = [(entity_text, match.start(), match.end())]
    results = detector.detect(text, entities)
    
    return results[0].certainty == Certainty.NEGATED if results else False


def is_uncertain(text: str, entity_text: str) -> bool:
    """
    Quick check if an entity is uncertain in the text.
    
    Args:
        text: Full text
        entity_text: The entity to check
        
    Returns:
        True if entity is uncertain
    """
    detector = NegExDetector()
    
    match = re.search(re.escape(entity_text), text, re.IGNORECASE)
    if not match:
        return False
    
    entities = [(entity_text, match.start(), match.end())]
    results = detector.detect(text, entities)
    
    return results[0].certainty == Certainty.UNCERTAIN if results else False


def get_certainty(text: str, entity_text: str) -> str:
    """
    Get the certainty label for an entity.
    
    Args:
        text: Full text
        entity_text: The entity to check
        
    Returns:
        "affirmed", "negated", or "uncertain"
    """
    detector = NegExDetector()
    
    match = re.search(re.escape(entity_text), text, re.IGNORECASE)
    if not match:
        return "affirmed"
    
    entities = [(entity_text, match.start(), match.end())]
    results = detector.detect(text, entities)
    
    return results[0].certainty.value if results else "affirmed"


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NegEx-Style Negation and Uncertainty Detection Demo")
    print("=" * 60)
    
    detector = NegExDetector()
    
    # Test cases demonstrating various scenarios
    test_cases = [
        # --- NEGATION EXAMPLES ---
        ("No pulmonary nodule identified.", "nodule", Certainty.NEGATED),
        ("Without evidence of malignancy.", "malignancy", Certainty.NEGATED),
        ("Patient denies chest pain.", "chest pain", Certainty.NEGATED),
        ("Lungs are clear. No nodules or masses.", "nodules", Certainty.NEGATED),
        ("Nodule was ruled out.", "Nodule", Certainty.NEGATED),
        ("The mass is unlikely.", "mass", Certainty.NEGATED),
        
        # --- UNCERTAINTY EXAMPLES ---
        ("Possible pulmonary nodule in RUL.", "nodule", Certainty.UNCERTAIN),
        ("Cannot exclude small nodule.", "nodule", Certainty.UNCERTAIN),
        ("May represent granuloma versus malignancy.", "granuloma", Certainty.UNCERTAIN),
        ("Findings suspicious for malignancy.", "malignancy", Certainty.UNCERTAIN),
        ("Questionable nodule in left lower lobe.", "nodule", Certainty.UNCERTAIN),
        ("Consistent with pneumonia.", "pneumonia", Certainty.UNCERTAIN),
        
        # --- AFFIRMED EXAMPLES ---
        ("12mm solid nodule in right upper lobe.", "nodule", Certainty.AFFIRMED),
        ("There is a spiculated mass.", "mass", Certainty.AFFIRMED),
        ("Nodule measures 15mm.", "Nodule", Certainty.AFFIRMED),
        
        # --- EDGE CASES (scope termination) ---
        ("No fever but nodule is present.", "nodule", Certainty.AFFIRMED),  # "but" terminates scope
        ("Possible infection. Nodule is stable.", "Nodule", Certainty.AFFIRMED),  # Different sentence
        ("No evidence of pneumonia; however, nodule noted.", "nodule", Certainty.AFFIRMED),
    ]
    
    print("\n--- Test Results ---\n")
    print(f"{'Text':<55} | {'Entity':<15} | {'Expected':<10} | {'Actual':<10} | {'Pass?'}")
    print("-" * 105)
    
    passed = 0
    total = len(test_cases)
    
    for text, entity, expected in test_cases:
        result = get_certainty(text, entity)
        actual = Certainty(result)
        passed_test = actual == expected
        passed += 1 if passed_test else 0
        
        # Truncate text for display
        text_display = text[:52] + "..." if len(text) > 55 else text
        status = "✓" if passed_test else "✗"
        
        print(f"{text_display:<55} | {entity:<15} | {expected.value:<10} | {actual.value:<10} | {status}")
    
    print("-" * 105)
    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")
