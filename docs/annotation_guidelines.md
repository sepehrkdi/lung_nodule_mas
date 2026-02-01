# Annotation Guidelines for Lung Nodule NLP Evaluation

## Overview

These guidelines describe how to annotate radiology reports for evaluating the NLP pipeline's performance in extracting lung nodule information.

---

## 1. Entity Annotation

### 1.1 Target Entities

Annotate the following entity types:

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| `NODULE` | Any mention of pulmonary nodule/mass/lesion | "nodule", "mass", "lesion", "opacity" |
| `SIZE` | Numeric size measurement | "15mm", "1.5 cm", "15 x 12 mm" |
| `LOCATION` | Anatomical location | "right upper lobe", "RUL", "lingula" |
| `TEXTURE` | Nodule texture/density | "solid", "ground-glass", "part-solid" |

### 1.2 Annotation Format

```
[TEXT](ENTITY_TYPE)
```

**Example:**
```
A [15mm](SIZE) [solid](TEXTURE) [nodule](NODULE) in the [right upper lobe](LOCATION).
```

---

## 2. Certainty Annotation

### 2.1 Certainty Labels

For each `NODULE` entity, assign a certainty label:

| Label | Description | Trigger Examples |
|-------|-------------|------------------|
| `AFFIRMED` | Positively stated | "nodule is present", "demonstrates nodule" |
| `NEGATED` | Explicitly denied | "no nodule", "without nodule", "ruled out" |
| `UNCERTAIN` | Hedged/uncertain | "possible nodule", "cannot exclude", "suspicious" |

### 2.2 Annotation Format

```
[TEXT](ENTITY_TYPE:CERTAINTY)
```

**Examples:**
```
No [nodule](NODULE:NEGATED) identified.
Possible [nodule](NODULE:UNCERTAIN) in RUL.
A 15mm [nodule](NODULE:AFFIRMED) is present.
```

---

## 3. Edge Cases

### 3.1 Scope Termination

Negation scope is terminated by conjunctions ("but", "however").

**Example:**
```
No fever but [nodule](NODULE:AFFIRMED) is present.
```
Here, "but" terminates the negation scope, so "nodule" is AFFIRMED.

### 3.2 Multiple Mentions

Annotate each mention separately:

```
Multiple [nodules](NODULE:AFFIRMED) are present. The largest [nodule](NODULE:AFFIRMED) measures 15mm.
```

### 3.3 Comparative Statements

Previous exam references don't affect certainty:

```
Compared to prior, [nodule](NODULE:AFFIRMED) is unchanged.
```

---

## 4. Inter-Annotator Agreement

- Each report should be annotated by 2 annotators independently
- Disagreements resolved by discussion
- Cohen's Kappa ≥ 0.7 required before using annotations for evaluation

---

## 5. Quick Reference Card

| Pattern | Certainty Label |
|---------|-----------------|
| "no [entity]" | NEGATED |
| "without [entity]" | NEGATED |
| "[entity] ruled out" | NEGATED |
| "possible [entity]" | UNCERTAIN |
| "cannot exclude [entity]" | UNCERTAIN |
| "suspicious for [entity]" | UNCERTAIN |
| "[size] [entity]" | AFFIRMED |
| "[entity] is present" | AFFIRMED |
