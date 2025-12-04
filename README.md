# Multi-Agent System for Lung Nodule Classification

**EDUCATIONAL PROJECT**

This project demonstrates the integration of:
- **Natural Language Processing (NLP)** - Medical text analysis with scispaCy + regex
- **Symbolic AI (Prolog)** - First-Order Logic reasoning with PySwip
- **Distributed AI (BDI Agents)** - Multi-agent collaboration with belief-desire-intention architecture
- **SPADE-BDI Framework** - Proper AgentSpeak(L) interpreter for genuine BDI semantics
- **Diverse Agent Architecture** - Multiple specialized agents with different approaches

## BDI Framework

### SPADE-BDI Implementation (Recommended)
This project uses **SPADE-BDI** as the primary BDI framework:
- **SPADE**: Smart Python Agent Development Environment
- **AgentSpeak(L)**: Proper plan syntax with triggering events
- **Internal Actions**: Python functions callable from AgentSpeak plans
- **Multi-Agent Consensus**: Weighted voting across multiple agents

### Why SPADE-BDI?
1. **Proper AgentSpeak Interpreter**: Unlike custom implementations, SPADE-BDI provides genuine AgentSpeak(L) semantics
2. **Standard Communication**: Uses XMPP for agent messaging (FIPA-compliant)
3. **Academic Credibility**: Well-documented framework with published papers
4. **Python Integration**: Easy to combine with ML/NLP libraries

## Architecture

### Extended 5-Agent Architecture (Recommended)

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    EXTENDED MAS ARCHITECTURE                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   RADIOLOGIST AGENTS (Image Analysis) - 3 Approaches           │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
    │   │  DenseNet121 │  │   ResNet50   │  │  Rule-Based  │         │
    │   │   W = 1.0    │  │   W = 1.0    │  │   W = 0.7    │         │
    │   │   (Deep CNN) │  │  (Deep CNN)  │  │ (Heuristics) │         │
    │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
    │          │                 │                  │                 │
    │          └────────────────┬┴──────────────────┘                │
    │                           │                                     │
    │   PATHOLOGIST AGENTS (Report Analysis) - 2 Approaches          │
    │   ┌──────────────────────┐  ┌──────────────────────┐           │
    │   │    Regex-Based       │  │    spaCy NER + Rules │           │
    │   │     W = 0.8          │  │       W = 0.9        │           │
    │   │  (Pattern Match)     │  │   (Statistical NLP)  │           │
    │   └──────────┬───────────┘  └──────────┬───────────┘           │
    │              │                          │                       │
    │              └────────────┬─────────────┘                      │
    │                           │                                     │
    │   ┌───────────────────────┴───────────────────────┐            │
    │   │              PROLOG CONSENSUS                  │            │
    │   │   • Weighted voting (agent expertise weights)  │            │
    │   │   • Disagreement detection & resolution        │            │
    │   │   • Confidence interval computation            │            │
    │   │   • Dempster-Shafer evidence combination       │            │
    │   └───────────────────────┬───────────────────────┘            │
    │                           │                                     │
    │                    ┌──────┴──────┐                             │
    │                    │   FINAL     │                             │
    │                    │ DIAGNOSIS   │                             │
    │                    │ + LUNG-RADS │                             │
    │                    └─────────────┘                             │
    └─────────────────────────────────────────────────────────────────┘
```

### Agent Diversity Rationale

**Why Different Radiologist Approaches?**
- **DenseNet121**: Dense connections, excellent for texture features
- **ResNet50**: Skip connections, robust gradient flow
- **Rule-Based**: Interpretable baseline following Lung-RADS guidelines

**Why Different Pathologist Approaches?**
- **Regex**: Fast, explicit patterns, fully interpretable
- **spaCy NER**: Robust to variations, contextual understanding, medical entities

## Agents

### 1. Radiologist Agents (Computer Vision) - x3

| Agent | Architecture | Weight | Approach |
|-------|--------------|--------|----------|
| RadiologistDenseNet | DenseNet121 | 1.0 | Deep CNN with dense connections |
| RadiologistResNet | ResNet50 | 1.0 | Deep CNN with residual blocks |
| RadiologistRuleBased | Heuristic | 0.7 | Size/texture rules (Lung-RADS) |

### 2. Pathologist Agents (NLP) - x2

| Agent | Method | Weight | Approach |
|-------|--------|--------|----------|
| PathologistRegex | Pattern Matching | 0.8 | Regular expressions for extraction |
| PathologistSpacy | spaCy NER | 0.9 | Medical NER + semantic rules |

### 3. Oncologist/Consensus (Symbolic Reasoning)
- Uses **Prolog** via PySwip
- Combines findings from ALL 5 agents
- Applies **Lung-RADS** classification rules
- Implements **weighted consensus voting**
- Handles disagreement resolution
- Generates clinical recommendations

## BDI Architecture

This project implements the Belief-Desire-Intention (BDI) model using SPADE-BDI:

- **Beliefs**: Current knowledge about nodules
  - Size, texture, margins from NLP
  - Classification probability from CV
  - Rules from Prolog knowledge base
  - Annotated with source agent for provenance

- **Desires**: Goals agents want to achieve
  - Analyze image (Radiologist)
  - Extract information (Pathologist)

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd lung_nodule_mas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install scispaCy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

# Install SWI-Prolog (required for PySwip)
# Ubuntu/Debian:
sudo apt-get install swi-prolog

# macOS:
brew install swi-prolog
```

## Dataset

### Primary: Open-I Indiana University Chest X-ray Collection

This project uses the **Open-I Indiana University Chest X-ray Collection** - a dataset containing paired chest X-ray images with corresponding radiology reports.

**Why Open-I?**
- ✅ **Paired Data**: Images + reports for both CV and NLP agents
- ✅ **Real Reports**: Authentic radiology language for NLP training
- ✅ **Manageable Size**: Can use 20-50 cases for educational project
- ✅ **Free Access**: No special access required

**Dataset Source**: https://openi.nlm.nih.gov/

**Downloading the Dataset**:
```bash
# Create data directory
mkdir -p data/openi/images data/openi/reports

# Download from Open-I (manual process):
# 1. Go to https://openi.nlm.nih.gov/
# 2. Search for "pulmonary nodule" or "lung nodule"
# 3. Download images and XML reports
# 4. Place PNGs in data/openi/images/
# 5. Place XMLs in data/openi/reports/

# Alternatively, use the provided download script:
python -m data.download_openi --n_cases 50
```

**Data Structure**:
```
data/openi/
├── images/
│   ├── CXR111_IM-0076-1001.png
│   └── ...
├── reports/
│   ├── CXR111_IM-0076-1001.xml
│   └── ...
└── manifest.json  (auto-generated)
```

### Fallback Dataset (Included)
10 synthetic cases with pre-defined features for zero-setup demo:
```
data/fallback/
├── nodule_001.json  # Malignancy 1 (benign)
├── nodule_005.json  # Malignancy 3 (indeterminate)
├── nodule_010.json  # Malignancy 5 (suspicious)
└── ...
```

Each fallback case includes:
- Structured features (size, texture, location)
- Synthetic radiology report text
- Ground truth malignancy score

## Usage

### Extended 5-Agent Demo (Recommended)
```bash
python main_extended.py --demo
```

### Extended 5-Agent Evaluation
```bash
python main_extended.py --evaluate
```

### Extended with Export
```bash
python main_extended.py --evaluate --export results.json
```

### Quick Demo (Original Implementation)
```bash
python main.py --demo
```

### SPADE-BDI Demo
```bash
python spade_main.py --demo
```

### SPADE-BDI with Custom Agent Count
```bash
python spade_main.py --num-radiologists 5 --num-pathologists 3
```

### Process All Nodules
```bash
python spade_main.py --all
```

### Process Specific Nodule
```bash
python spade_main.py --nodule nodule_001
```

### With Evaluation Metrics
```bash
python spade_main.py --all --evaluate
```

### Save Results
```bash
python spade_main.py --all --output results.json
```

## Project Structure

```
lung_nodule_mas/
├── main.py                     # Original orchestrator
├── main_extended.py            # Extended 5-agent orchestrator (recommended)
├── spade_main.py               # SPADE-BDI orchestrator
├── orchestrator.py             # Multi-agent orchestrator with Prolog consensus
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── agents/                     # BDI Agents
│   ├── base_agent.py          # Original BDI base class
│   ├── spade_base.py          # SPADE-BDI base class
│   ├── radiologist_agent.py   # Original CV agent
│   ├── pathologist_agent.py   # Original NLP agent
│   ├── oncologist_agent.py    # Original Prolog agent
│   ├── spade_radiologist.py   # SPADE-BDI CV agent
│   ├── spade_pathologist.py   # SPADE-BDI NLP agent
│   ├── spade_oncologist.py    # SPADE-BDI reasoning agent
│   ├── radiologist_variants.py # DenseNet, ResNet, Rule-based
│   └── pathologist_variants.py # Regex, spaCy NER
│
├── asl/                        # AgentSpeak(L) plans
│   ├── radiologist.asl        # CV plans (SPADE-BDI syntax)
│   ├── pathologist.asl        # NLP plans (SPADE-BDI syntax)
│   └── oncologist.asl         # Reasoning plans (SPADE-BDI syntax)
│
├── knowledge/                  # Prolog Knowledge Bases
│   ├── lung_rads.pl           # Lung-RADS classification rules
│   ├── multi_agent_consensus.pl # Weighted voting & disagreement resolution
│   └── prolog_engine.py       # PySwip interface
│
├── communication/              # Agent messaging
│   └── message_queue.py       # FIPA-ACL broker
│
├── models/                     # ML models
│   └── classifier.py          # DenseNet121
│
├── nlp/                        # NLP components
│   └── extractor.py           # scispaCy extractor
│
├── data/                       # Data handling
│   ├── lidc_loader.py         # LIDC data loader
│   ├── openi_loader.py        # Open-I dataset loader
│   ├── report_generator.py    # Report synthesis
│   ├── prepare_dataset.py     # Dataset prep
│   └── fallback/              # Demo nodules
│       ├── nodule_001.json
│       └── ...
│
└── evaluation/                 # Metrics
    └── metrics.py             # Accuracy, ROC, etc.
```

## Educational Concepts Demonstrated

### 1. NLP (Pathologist Agents)
- **Tokenization**: Breaking text into words
- **POS Tagging**: Identifying word types
- **NER**: Named Entity Recognition for medical terms
- **Regex Patterns**: Extracting measurements and keywords
- **Statistical vs Symbolic NLP**: spaCy vs regex comparison
- **First-Order Logic**: Predicates and rules
- **Unification**: Pattern matching with variables
- **Backtracking**: Exploring solution space
- **Resolution**: Deriving conclusions from rules

### 3. Distributed AI (All Agents)
- **BDI Architecture**: Bratman's model
- **AgentSpeak**: Plan specification language
- **Speech Acts**: FIPA-ACL performatives
- **Message Passing**: Asynchronous communication

## Lung-RADS Categories

| Category | Description | Action |
|----------|-------------|--------|
| 1 | Negative | Annual screening |
| 2 | Benign appearance | Annual screening |
| 3 | Probably benign | 6-month follow-up |
| 4A | Suspicious | 3-month follow-up or PET |
| 4B | Very suspicious | PET-CT, consider biopsy |
| 4X | Additional features | Immediate evaluation |

## Example Output

```
=== SPADE-BDI MULTI-AGENT SYSTEM RESULTS ===
Agents: 3 Radiologists, 2 Pathologists, 1 Oncologist

--- nodule_001 ---
  Ground Truth:    Malignancy 1
  Predicted:       Malignancy 1
  Probability:     0.123
  Confidence:      0.892
  Lung-RADS:       Category 2
  Match:           ✓

  Agent Votes:
    Radiologist_1: 12.5%
    Radiologist_2: 11.8%
    Radiologist_3: 13.2%
    Pathologist_1: 15.0%
    Pathologist_2: 14.2%

--- nodule_005 ---
  Ground Truth:    Malignancy 5
  Predicted:       Malignancy 5
  Probability:     0.847
  Confidence:      0.765
  Lung-RADS:       Category 4B
  Match:           ✓

=== SUMMARY ===
Total Nodules:       10
5-Class Accuracy:    70.0%
Binary Accuracy:     87.5%
Avg Confidence:      81.2%
```

## References

### BDI and Multi-Agent Systems
- Bratman, M. (1987). "Intention, Plans, and Practical Reason"
- Rao, A. S., & Georgeff, M. P. (1995). "BDI Agents: From Theory to Practice"
- Bordini, R. H., et al. (2007). "Programming Multi-Agent Systems in AgentSpeak using Jason"

### SPADE-BDI Framework
- SPADE: https://spade-mas.readthedocs.io/
- SPADE-BDI Extension: https://github.com/javipalanca/spade_bdi
- AgentSpeak(L): Rao, A. S. (1996). "AgentSpeak(L): BDI Agents Speak Out in a Logical Computable Language"

### Medical AI
- Armato III, S. G., et al. (2011). "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)"
- American College of Radiology. (2019). "Lung-RADS Assessment Categories"

### NLP for Medical Text
- Neumann, M., et al. (2019). "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"

## License

Educational use only. Not for clinical use.

## Disclaimer

This is an educational demonstration project. It is NOT intended for clinical use. Real medical AI systems require:
- Rigorous validation on large datasets
- Clinical trials and regulatory approval (FDA/CE)
- Integration with clinical workflows
- Continuous monitoring and updates
