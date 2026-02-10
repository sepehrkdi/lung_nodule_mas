"""
Pydantic Schemas for API Request/Response Models
=================================================
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class AnalysisStatus(str, Enum):
    """Status of an analysis session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class AgentType(str, Enum):
    """Type of agent."""
    RADIOLOGIST = "radiologist"
    PATHOLOGIST = "pathologist"


class AgentFindingResponse(BaseModel):
    """Response model for a single agent's finding."""
    agent_name: str
    agent_type: str
    approach: str
    weight: float
    probability: float
    predicted_class: int
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class WeightRationaleResponse(BaseModel):
    """Response model for dynamic weight rationale."""
    radiology_richness: float = 0.5
    pathology_richness: float = 0.5
    radiology_components: Dict[str, float] = Field(default_factory=dict)
    pathology_components: Dict[str, float] = Field(default_factory=dict)
    dynamic_weights: Dict[str, float] = Field(default_factory=dict)
    base_weights: Dict[str, float] = Field(default_factory=dict)
    scale_floor: float = 0.5


class ConsensusResultResponse(BaseModel):
    """Response model for consensus result."""
    nodule_id: str
    final_probability: float
    final_class: int
    confidence: float
    agreement_level: str
    disagreement_agents: List[str] = Field(default_factory=list)
    radiologist_findings: List[AgentFindingResponse] = Field(default_factory=list)
    pathologist_findings: List[AgentFindingResponse] = Field(default_factory=list)
    lung_rads_category: Optional[str] = None
    recommendation: Optional[str] = None
    weight_rationale: Optional[WeightRationaleResponse] = None


class AnalysisStateResponse(BaseModel):
    """Response model for analysis state (polling endpoint)."""
    session_id: str
    nodule_id: str
    status: AnalysisStatus
    total_agents: int = 6  # 3 radiologists + 3 pathologists
    completed_count: int = 0
    completed_agents: List[AgentFindingResponse] = Field(default_factory=list)
    consensus: Optional[ConsensusResultResponse] = None
    error_message: Optional[str] = None
    started_at: str
    updated_at: str


class NoduleListResponse(BaseModel):
    """Response model for list of nodules."""
    nodule_ids: List[str]
    total_count: int


class NoduleFeaturesResponse(BaseModel):
    """Response model for nodule features."""
    nodule_id: str
    diameter_mm: Optional[float] = None
    malignancy: Optional[int] = None
    malignancy_label: Optional[str] = None
    texture: Optional[str] = None  # String for NLMCXR (e.g., "solid", "ground_glass")
    texture_label: Optional[str] = None
    margin: Optional[str] = None  # String for NLMCXR
    margin_label: Optional[str] = None
    spiculation: Optional[str] = None  # String for NLMCXR
    spiculation_label: Optional[str] = None
    lobulation: Optional[str] = None  # String for NLMCXR
    calcification: Optional[str] = None  # String for NLMCXR
    sphericity: Optional[int] = None
    subtlety: Optional[int] = None
    internal_structure: Optional[int] = None
    source: Optional[str] = None
    is_synthetic: bool = False


class ReportResponse(BaseModel):
    """Response model for generated report."""
    nodule_id: str
    report_type: str = "full"  # "full" or "brief"
    report_text: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    nodule_ids: List[str]


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    session_id: str
    total_cases: int
    status: AnalysisStatus


class MetricsResponse(BaseModel):
    """Response model for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_cases: int
    unanimous_count: int
    majority_count: int
    split_count: int
    confusion_matrix: List[List[int]]
    class_labels: List[str] = ["Normal", "Abnormal"]


class MetricsStatusResponse(BaseModel):
    """Response model for metrics computation progress."""
    status: str  # "running", "completed", "error"
    processed: int = 0
    total: int = 0
    metrics: Optional[MetricsResponse] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    agents_available: int = 6  # 3 radiologists + 3 pathologists
