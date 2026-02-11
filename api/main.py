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
from typing import Optional, List, Dict, Any
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
    MetricsStatusResponse,
    HealthResponse,
)
from api.analysis_state import analysis_manager, AgentResult
from data.nlmcxr_loader import NLMCXRLoader
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
_loader: Optional[NLMCXRLoader] = None
_orchestrator: Optional[MultiAgentOrchestrator] = None

# Constants
MAX_EVALUATION_CASES = 500


def get_loader() -> NLMCXRLoader:
    """Get or create the data loader."""
    global _loader
    if _loader is None:
        _loader = NLMCXRLoader()
    return _loader


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
        agents_available=6  # 3 radiologists + 3 pathologists
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
    """List all available case IDs."""
    try:
        loader = get_loader()
        # ALIGNMENT WITH REPORT: Use pre-filtered list of nodule cases
        # Limit to MAX_EVALUATION_CASES as per specifications
        case_ids = loader.get_nodule_case_ids(limit=MAX_EVALUATION_CASES)
        return NoduleListResponse(
            nodule_ids=case_ids,
            total_count=len(case_ids)
        )
    except Exception as e:
        logger.error(f"Failed to list cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodules/{nodule_id}/features", response_model=NoduleFeaturesResponse, tags=["Nodules"])
async def get_nodule_features(nodule_id: str):
    """Get features/metadata for a specific case."""
    try:
        loader = get_loader()
        images, metadata = loader.load_case(nodule_id)
        
        # Extract features from NLMCXR metadata
        nlp_features = metadata.get("nlp_features", {})
        
        return NoduleFeaturesResponse(
            nodule_id=nodule_id,
            diameter_mm=nlp_features.get("size_mm"),
            malignancy=metadata.get("ground_truth"),
            malignancy_label=metadata.get("ground_truth_label"),
            texture=nlp_features.get("texture"),
            texture_label=nlp_features.get("texture"),
            margin=nlp_features.get("margin"),
            margin_label=nlp_features.get("margin"),
            spiculation=nlp_features.get("spiculation"),
            spiculation_label=nlp_features.get("spiculation"),
            lobulation=nlp_features.get("lobulation"),
            calcification=nlp_features.get("calcification"),
            sphericity=None,
            subtlety=None,
            internal_structure=None,
            source="NLMCXR",
            is_synthetic=False
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case {nodule_id} not found")
    except Exception as e:
        logger.error(f"Failed to get case features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodules/{nodule_id}/image", tags=["Nodules"])
async def get_nodule_image(nodule_id: str, colormap: str = "gray", upscale: int = 4):
    """
    Get chest X-ray image for a specific case as PNG.
    
    Args:
        nodule_id: The case identifier
        colormap: Color mapping ('gray', 'bone', 'hot')
        upscale: Upscaling factor for display (default 4x)
    """
    try:
        loader = get_loader()
        images, _ = loader.load_case(nodule_id)
        
        if not images:
            raise HTTPException(status_code=404, detail="Image not available")
        
        # Use first image (PA view typically)
        image = images[0]
        
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
    Get radiology report for a specific case.
    
    Args:
        nodule_id: The case identifier
        report_type: 'full' for complete report, 'brief' for summary
    """
    try:
        loader = get_loader()
        _, metadata = loader.load_case(nodule_id)
        
        # Use real NLMCXR report text
        findings = metadata.get("findings", "")
        impression = metadata.get("impression", "")
        
        if report_type == "brief":
            report_text = impression if impression else findings[:200]
        else:
            report_text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        
        return ReportResponse(
            nodule_id=nodule_id,
            report_type=report_type,
            report_text=report_text
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case {nodule_id} not found")
    except Exception as e:
        logger.error(f"Failed to get report: {e}")
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
        images, metadata = loader.load_case(nodule_id)
        
        # Use real NLMCXR report text
        findings = metadata.get("findings", "")
        impression = metadata.get("impression", "")
        report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        
        # Build features from metadata
        features = metadata.get("nlp_features", {})
        features.update({
            "case_id": nodule_id,
            "ground_truth": metadata.get("ground_truth"),
            "num_images": len(images)
        })
        
        # Use first image or list for multi-image
        image = images[0] if len(images) == 1 else images
        
        orchestrator = get_orchestrator()
        
        # Callback to update state manager when each agent completes
        async def on_agent_complete(agent_name: str, result: dict):
            analysis_manager.add_agent_result(session_id, agent_name, result)
        
        # Get image metadata for multi-image aggregation
        image_metadata = metadata.get("images_metadata", [])
        
        # Run analysis with callback
        result = await orchestrator.analyze_case(
            case_id=nodule_id,
            image_array=image,
            report=report,
            features=features,
            image_metadata=image_metadata,
            case_metadata=metadata,  # Pass full metadata for dynamic weight computation
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
            "thinking_process": result.thinking_process,
            "weight_rationale": result.weight_rationale,
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
        # Verify case exists
        loader = get_loader()
        loader.load_case(nodule_id)  # Will raise if not found
        
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
                loader.load_case(nodule_id)  # Verify exists
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
                    "error": f"Case {nodule_id} not found"
                })
        
        return {
            "total_requested": len(request.nodule_ids),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Failed to start batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# METRICS ENDPOINTS (async with polling for progress)
# =============================================================================

# In-memory state for metrics computation
_metrics_state: Dict[str, Any] = {
    "status": "idle",     # idle | running | completed | error
    "processed": 0,
    "total": 0,
    "result": None,
    "error": None,
}

async def _compute_metrics_background():
    """Background task that computes metrics and updates _metrics_state."""
    global _metrics_state
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import concurrent.futures

        loader = get_loader()
        orchestrator = get_orchestrator()

        y_true = []
        y_pred = []
        unanimous_count = 0
        majority_count = 0
        split_count = 0

        split_count = 0

        case_ids = loader.get_nodule_case_ids(limit=MAX_EVALUATION_CASES)
        _metrics_state["total"] = len(case_ids)
        _metrics_state["processed"] = 0

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        for idx, nodule_id in enumerate(case_ids):
            try:
                images, metadata = loader.load_case(nodule_id)

                findings = metadata.get("findings", "")
                impression = metadata.get("impression", "")
                report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

                features = metadata.get("nlp_features", {})
                features["ground_truth"] = metadata.get("ground_truth")

                image = images[0] if len(images) == 1 else images

                # Run CPU-heavy analysis in thread pool so the event loop
                # remains free to serve /health and /metrics/status requests
                def _run_analysis(cid, img, rpt, feat, meta):
                    import asyncio as _aio
                    _loop = _aio.new_event_loop()
                    try:
                        return _loop.run_until_complete(
                            orchestrator.analyze_case(
                                case_id=cid,
                                image_array=img,
                                report=rpt,
                                features=feat,
                                case_metadata=meta,
                            )
                        )
                    finally:
                        _loop.close()

                result = await loop.run_in_executor(
                    executor,
                    _run_analysis, nodule_id, image, report, features, metadata
                )

                ground_truth = metadata.get("ground_truth", -1)
                if ground_truth != -1:
                    y_true.append(ground_truth)
                    y_pred.append(1 if result.final_probability >= 0.5 else 0)

                if result.agreement_level == "unanimous":
                    unanimous_count += 1
                elif result.agreement_level == "majority":
                    majority_count += 1
                else:
                    split_count += 1

            except Exception as e:
                logger.warning(f"Skipping nodule {nodule_id} in metrics: {e}")

            _metrics_state["processed"] = idx + 1

        if len(y_true) == 0:
            _metrics_state["status"] = "error"
            _metrics_state["error"] = "No data available for metrics"
            return

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        # Binary confusion matrix: 0=Normal, 1=Abnormal
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        _metrics_state["result"] = MetricsResponse(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            total_cases=len(y_true),
            unanimous_count=unanimous_count,
            majority_count=majority_count,
            split_count=split_count,
            confusion_matrix=cm.tolist(),
        )
        _metrics_state["status"] = "completed"
        logger.info("Metrics computation completed successfully")

    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        _metrics_state["status"] = "error"
        _metrics_state["error"] = str(e)


@app.post("/metrics/start", tags=["Metrics"])
async def start_metrics():
    """
    Start computing evaluation metrics in the background.
    Poll /metrics/status for progress.
    """
    global _metrics_state
    if _metrics_state["status"] == "running":
        return {"message": "Metrics computation already running", "status": "running"}

    _metrics_state = {
        "status": "running",
        "processed": 0,
        "total": 0,
        "result": None,
        "error": None,
    }
    asyncio.ensure_future(_compute_metrics_background())
    return {"message": "Metrics computation started", "status": "running"}


@app.get("/metrics/status", response_model=MetricsStatusResponse, tags=["Metrics"])
async def metrics_status():
    """Poll for metrics computation progress."""
    return MetricsStatusResponse(
        status=_metrics_state["status"],
        processed=_metrics_state["processed"],
        total=_metrics_state["total"],
        metrics=_metrics_state["result"],
        error_message=_metrics_state.get("error"),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """
    Return the last computed metrics (does NOT recompute).
    Use POST /metrics/start to trigger computation.
    """
    if _metrics_state["status"] == "completed" and _metrics_state["result"]:
        return _metrics_state["result"]
    elif _metrics_state["status"] == "running":
        raise HTTPException(status_code=202, detail="Metrics computation still in progress")
    else:
        raise HTTPException(status_code=400, detail="No metrics available. POST /metrics/start first.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
