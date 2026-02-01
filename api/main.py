"""
FastAPI Backend for Lung Nodule Multi-Agent System
===================================================

Provides REST API endpoints for:
- Listing and retrieving nodule data
- Running multi-agent analysis with polling for real-time updates
- Serving CT images
- Generating reports
- Computing evaluation metrics
"""

import asyncio
import logging
import io
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from PIL import Image

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    AnalysisStatus,
    AnalysisStateResponse,
    NoduleListResponse,
    NoduleFeaturesResponse,
    ReportResponse,
    ConsensusResultResponse,
    AgentFindingResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    MetricsResponse,
    HealthResponse,
)
from api.analysis_state import analysis_manager, AgentResult
from data.lidc_loader import LIDCLoader
from data.report_generator import ReportGenerator
from orchestrator import MultiAgentOrchestrator, ConsensusResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lung Nodule Multi-Agent System API",
    description="REST API for multi-agent lung nodule classification with real-time analysis updates",
    version="1.0.0",
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy initialization)
_loader: Optional[LIDCLoader] = None
_report_generator: Optional[ReportGenerator] = None
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_loader() -> LIDCLoader:
    """Get or create the data loader."""
    global _loader
    if _loader is None:
        _loader = LIDCLoader()
    return _loader


def get_report_generator() -> ReportGenerator:
    """Get or create the report generator."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing MultiAgentOrchestrator...")
        _orchestrator = MultiAgentOrchestrator()
        logger.info("Orchestrator initialized successfully")
    return _orchestrator


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents_available=5
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint - redirect to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# =============================================================================
# NODULE DATA ENDPOINTS
# =============================================================================

@app.get("/nodules", response_model=NoduleListResponse, tags=["Nodules"])
async def list_nodules():
    """List all available nodule IDs."""
    try:
        loader = get_loader()
        nodule_ids = loader.get_nodule_ids()
        return NoduleListResponse(
            nodule_ids=nodule_ids,
            total_count=len(nodule_ids)
        )
    except Exception as e:
        logger.error(f"Failed to list nodules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodules/{nodule_id}/features", response_model=NoduleFeaturesResponse, tags=["Nodules"])
async def get_nodule_features(nodule_id: str):
    """Get features/metadata for a specific nodule."""
    try:
        loader = get_loader()
        _, features = loader.load_nodule(nodule_id)
        
        return NoduleFeaturesResponse(
            nodule_id=nodule_id,
            diameter_mm=features.get("diameter_mm"),
            malignancy=features.get("malignancy"),
            malignancy_label=features.get("malignancy_label"),
            texture=features.get("texture"),
            texture_label=features.get("texture_label"),
            margin=features.get("margin"),
            margin_label=features.get("margin_label"),
            spiculation=features.get("spiculation"),
            spiculation_label=features.get("spiculation_label"),
            lobulation=features.get("lobulation"),
            calcification=features.get("calcification"),
            sphericity=features.get("sphericity"),
            subtlety=features.get("subtlety"),
            internal_structure=features.get("internal_structure"),
            source=features.get("source"),
            is_synthetic=features.get("source") != "XML Import"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Nodule {nodule_id} not found")
    except Exception as e:
        logger.error(f"Failed to get nodule features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodules/{nodule_id}/image", tags=["Nodules"])
async def get_nodule_image(nodule_id: str, colormap: str = "gray", upscale: int = 4):
    """
    Get CT image for a specific nodule as PNG.
    
    Args:
        nodule_id: The nodule identifier
        colormap: Color mapping ('gray', 'bone', 'hot')
        upscale: Upscaling factor for display (default 4x)
    """
    try:
        loader = get_loader()
        image, _ = loader.load_nodule(nodule_id)
        
        if image is None:
            raise HTTPException(status_code=404, detail="Image not available")
        
        # Convert from float [0,1] to uint8 [0,255]
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply colormap
        if colormap == "bone":
            # Simple bone-like colormap: increase contrast in mid-tones
            img_uint8 = np.clip(img_uint8 * 1.2, 0, 255).astype(np.uint8)
        elif colormap == "hot":
            # Convert to RGB hot colormap
            from matplotlib import cm
            colored = cm.hot(image)
            img_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_uint8, mode='RGB')
        
        if colormap != "hot":
            pil_image = Image.fromarray(img_uint8, mode='L')
        
        # Upscale for better display
        if upscale > 1:
            new_size = (pil_image.width * upscale, pil_image.height * upscale)
            pil_image = pil_image.resize(new_size, Image.Resampling.NEAREST)
        
        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Nodule {nodule_id} not found")
    except Exception as e:
        logger.error(f"Failed to get nodule image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodules/{nodule_id}/report", response_model=ReportResponse, tags=["Nodules"])
async def get_nodule_report(nodule_id: str, report_type: str = "full"):
    """
    Generate a radiology report for a specific nodule.
    
    Args:
        nodule_id: The nodule identifier
        report_type: 'full' for complete report, 'brief' for summary
    """
    try:
        loader = get_loader()
        _, features = loader.load_nodule(nodule_id)
        
        generator = get_report_generator()
        
        if report_type == "brief":
            report_text = generator.generate_brief(features)
        else:
            report_text = generator.generate(features)
        
        return ReportResponse(
            nodule_id=nodule_id,
            report_type=report_type,
            report_text=report_text
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Nodule {nodule_id} not found")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================

async def run_analysis_task(session_id: str, nodule_id: str):
    """
    Background task to run multi-agent analysis.
    Updates analysis_manager as each agent completes.
    """
    try:
        loader = get_loader()
        image, features = loader.load_nodule(nodule_id)
        
        generator = get_report_generator()
        report = generator.generate(features)
        
        orchestrator = get_orchestrator()
        
        # Callback to update state manager when each agent completes
        async def on_agent_complete(agent_name: str, result: dict):
            analysis_manager.add_agent_result(session_id, agent_name, result)
        
        # Run analysis with callback
        result = await orchestrator.analyze_case(
            nodule_id=nodule_id,
            image_array=image,
            report=report,
            features=features,
            on_agent_complete=on_agent_complete
        )
        
        # Convert consensus result to dict
        consensus_dict = {
            "nodule_id": result.nodule_id,
            "final_probability": result.final_probability,
            "final_class": result.final_class,
            "confidence": result.confidence,
            "agreement_level": result.agreement_level,
            "disagreement_agents": result.disagreement_agents,
            "radiologist_findings": [
                {
                    "agent_name": f.agent_name,
                    "agent_type": f.agent_type,
                    "approach": f.approach,
                    "weight": f.weight,
                    "probability": f.probability,
                    "predicted_class": f.predicted_class,
                    "details": f.details,
                }
                for f in result.radiologist_findings
            ],
            "pathologist_findings": [
                {
                    "agent_name": f.agent_name,
                    "agent_type": f.agent_type,
                    "approach": f.approach,
                    "weight": f.weight,
                    "probability": f.probability,
                    "predicted_class": f.predicted_class,
                    "details": f.details,
                }
                for f in result.pathologist_findings
            ],
            "prolog_reasoning": result.prolog_reasoning,
        }
        
        # Add Lung-RADS category based on final class
        lung_rads_map = {1: "1", 2: "2", 3: "3", 4: "4A", 5: "4B"}
        consensus_dict["lung_rads_category"] = lung_rads_map.get(result.final_class, "3")
        
        # Add recommendation
        recommendations = {
            1: "No further follow-up needed for this nodule.",
            2: "Annual low-dose CT screening recommended.",
            3: "Short-term follow-up CT in 6 months recommended.",
            4: "Short-term follow-up CT in 3 months or PET-CT recommended.",
            5: "Tissue sampling (biopsy) or surgical consultation recommended.",
        }
        consensus_dict["recommendation"] = recommendations.get(result.final_class, "Clinical correlation recommended.")
        
        analysis_manager.set_consensus(session_id, consensus_dict)
        
    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {e}")
        analysis_manager.set_error(session_id, str(e))


@app.post("/analyze/{nodule_id}", tags=["Analysis"])
async def start_analysis(nodule_id: str, background_tasks: BackgroundTasks):
    """
    Start multi-agent analysis for a nodule.
    
    Returns a session_id that can be used to poll for status updates.
    The analysis runs in the background with agents completing one by one.
    """
    try:
        # Verify nodule exists
        loader = get_loader()
        loader.load_nodule(nodule_id)  # Will raise if not found
        
        # Create session
        session_id = analysis_manager.start_analysis(nodule_id)
        
        # Start background analysis task
        background_tasks.add_task(run_analysis_task, session_id, nodule_id)
        
        return {
            "session_id": session_id,
            "nodule_id": nodule_id,
            "status": "running",
            "message": "Analysis started. Poll /analyze/status/{session_id} for updates."
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Nodule {nodule_id} not found")
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/status/{session_id}", tags=["Analysis"])
async def get_analysis_status(session_id: str):
    """
    Poll for analysis status and partial results.
    
    Returns the current state including:
    - Status (running/completed/error)
    - Number of agents completed
    - Partial results from completed agents
    - Final consensus (when complete)
    """
    state = analysis_manager.get_state(session_id)
    
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return state


@app.delete("/analyze/{session_id}", tags=["Analysis"])
async def delete_analysis_session(session_id: str):
    """Delete a completed analysis session."""
    if analysis_manager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# =============================================================================
# BATCH ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/batch/analyze", tags=["Batch"])
async def start_batch_analysis(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start batch analysis for multiple nodules.
    
    Returns session IDs for each nodule analysis.
    """
    try:
        loader = get_loader()
        sessions = []
        
        for nodule_id in request.nodule_ids:
            try:
                loader.load_nodule(nodule_id)  # Verify exists
                session_id = analysis_manager.start_analysis(nodule_id)
                background_tasks.add_task(run_analysis_task, session_id, nodule_id)
                sessions.append({
                    "nodule_id": nodule_id,
                    "session_id": session_id,
                    "status": "running"
                })
            except FileNotFoundError:
                sessions.append({
                    "nodule_id": nodule_id,
                    "session_id": None,
                    "status": "error",
                    "error": f"Nodule {nodule_id} not found"
                })
        
        return {
            "total_requested": len(request.nodule_ids),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Failed to start batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# METRICS ENDPOINTS
# =============================================================================

@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """
    Compute evaluation metrics from all analyzed cases.
    
    Note: This requires having run analyses first. Returns metrics based
    on comparing predicted classes against ground truth malignancy scores.
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        loader = get_loader()
        orchestrator = get_orchestrator()
        
        # Collect results for all nodules
        y_true = []
        y_pred = []
        unanimous_count = 0
        majority_count = 0
        split_count = 0
        
        nodule_ids = loader.get_nodule_ids()
        
        for nodule_id in nodule_ids:
            try:
                image, features = loader.load_nodule(nodule_id)
                generator = get_report_generator()
                report = generator.generate(features)
                
                result = await orchestrator.analyze_case(
                    nodule_id=nodule_id,
                    image_array=image,
                    report=report,
                    features=features
                )
                
                ground_truth = features.get("malignancy", 3)
                y_true.append(ground_truth)
                y_pred.append(result.final_class)
                
                if result.agreement_level == "unanimous":
                    unanimous_count += 1
                elif result.agreement_level == "majority":
                    majority_count += 1
                else:
                    split_count += 1
                    
            except Exception as e:
                logger.warning(f"Skipping nodule {nodule_id} in metrics: {e}")
                continue
        
        if len(y_true) == 0:
            raise HTTPException(status_code=400, detail="No data available for metrics")
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix (5x5 for classes 1-5)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
        
        return MetricsResponse(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_cases=len(y_true),
            unanimous_count=unanimous_count,
            majority_count=majority_count,
            split_count=split_count,
            confusion_matrix=cm.tolist(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
