"""
Case Analysis Page
==================

Main analysis page with:
- Nodule selection
- CT image display
- Real-time agent updates via polling
- Consensus result display
"""

import streamlit as st
import requests
import time
from typing import Optional, Dict, Any

# Import components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.agent_panel import render_agent_panel, render_consensus_panel
from ui.components.image_viewer import render_image_viewer, render_features_table
from ui.components.report_viewer import render_report_viewer

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_connection() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False


def get_nodule_list() -> list:
    """Fetch list of available nodules from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/nodules", timeout=10)
        if response.status_code == 200:
            return response.json().get("nodule_ids", [])
    except:
        pass
    return []


def get_nodule_features(nodule_id: str) -> Optional[Dict[str, Any]]:
    """Fetch nodule features from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/nodules/{nodule_id}/features", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_nodule_report(nodule_id: str) -> Optional[str]:
    """Fetch generated report from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/nodules/{nodule_id}/report", timeout=10)
        if response.status_code == 200:
            return response.json().get("report_text", "")
    except:
        pass
    return None


def start_analysis(nodule_id: str) -> Optional[str]:
    """Start analysis and return session ID."""
    try:
        response = requests.post(f"{API_BASE_URL}/analyze/{nodule_id}", timeout=10)
        if response.status_code == 200:
            return response.json().get("session_id")
    except Exception as e:
        st.error(f"Failed to start analysis: {e}")
    return None


def get_analysis_status(session_id: str) -> Optional[Dict[str, Any]]:
    """Poll for analysis status."""
    try:
        response = requests.get(f"{API_BASE_URL}/analyze/status/{session_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def render_case_analysis_page():
    """Render the case analysis page."""
    
    st.title("🔬 Case Analysis")
    st.markdown("Analyze lung nodules with our 6-agent multi-agent system")
    
    # Check API connection
    api_connected = check_api_connection()
    
    if not api_connected:
        st.error("""
        ⚠️ **API Not Connected**
        
        The FastAPI backend is not running. Please start it first:
        ```bash
        python run_api.py
        ```
        Or run both API and UI together:
        ```bash
        python run_all.py
        ```
        """)
        return
    
    st.success("✅ Connected to API")
    
    # Initialize session state
    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "selected_nodule" not in st.session_state:
        st.session_state.selected_nodule = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "last_status" not in st.session_state:
        st.session_state.last_status = None
    
    # Sidebar for nodule selection
    st.sidebar.header("📁 Select Nodule")
    
    nodule_ids = get_nodule_list()
    
    if not nodule_ids:
        st.sidebar.warning("No nodules found")
        return
    
    selected_nodule = st.sidebar.selectbox(
        "Nodule ID",
        options=nodule_ids,
        format_func=lambda x: f"Nodule {x}",
        key="nodule_selector"
    )
    
    # Reset analysis state when nodule changes
    if selected_nodule != st.session_state.selected_nodule:
        st.session_state.selected_nodule = selected_nodule
        st.session_state.analyzing = False
        st.session_state.session_id = None
        st.session_state.analysis_complete = False
        st.session_state.last_status = None
    
    # Load nodule data
    features = get_nodule_features(selected_nodule)
    report_text = get_nodule_report(selected_nodule)
    
    if not features:
        st.error("Failed to load nodule data")
        return
    
    # Main content area - two columns
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # CT Image
        render_image_viewer(
            selected_nodule,
            api_base_url=API_BASE_URL,
            is_synthetic=features.get("is_synthetic", False),
            show_controls=False
        )
        
        # Features table
        render_features_table(features)
    
    with col_right:
        # Report
        if report_text:
            render_report_viewer(
                report_text,
                selected_nodule,
                highlight=True,
                show_raw=True
            )
        else:
            st.warning("Report not available")
    
    st.markdown("---")
    
    # Analysis section
    st.header("🤖 Multi-Agent Analysis")
    
    # Analyze button
    analyze_col1, analyze_col2 = st.columns([1, 3])
    
    with analyze_col1:
        analyze_button = st.button(
            "🚀 Analyze Nodule",
            disabled=st.session_state.analyzing,
            type="primary",
            use_container_width=True
        )
    
    # with analyze_col2:
    #     if st.session_state.analyzing:
    #         st.info("⏳ Analysis in progress... Results will appear below as agents complete.")
    
    # Start analysis
    if analyze_button and not st.session_state.analyzing:
        session_id = start_analysis(selected_nodule)
        if session_id:
            st.session_state.session_id = session_id
            st.session_state.analyzing = True
            st.session_state.analysis_complete = False
            st.session_state.last_status = None
            st.rerun()
    
    # Polling loop for analysis status
    if st.session_state.analyzing and st.session_state.session_id:
        status = get_analysis_status(st.session_state.session_id)
        
        if status:
            st.session_state.last_status = status
            
            # Display agent panel with current results
            completed_agents = status.get("completed_agents", [])
            render_agent_panel(completed_agents)
            
            # Check if complete
            if status.get("status") == "completed":
                st.session_state.analyzing = False
                st.session_state.analysis_complete = True
                st.rerun()
                
            elif status.get("status") == "error":
                st.session_state.analyzing = False
                st.error(f"❌ Analysis failed: {status.get('error_message', 'Unknown error')}")
                
            else:
                # Still running - schedule rerun
                time.sleep(0.5)
                st.rerun()
        else:
            st.error("Failed to get analysis status")
            st.session_state.analyzing = False
    
    # Show previous results if analysis is complete
    elif st.session_state.analysis_complete and st.session_state.last_status:
        st.success("✅ Analysis complete!")
        status = st.session_state.last_status
        completed_agents = status.get("completed_agents", [])
        render_agent_panel(completed_agents)
        
        consensus = status.get("consensus")
        if consensus:
            st.markdown("---")
            # Pass ground truth info for comparison display
            ground_truth_info = None
            if features:
                gt = features.get("malignancy")
                if gt is not None:
                    ground_truth_info = {
                        "ground_truth": gt,
                        "ground_truth_label": features.get("malignancy_label", "unknown"),
                    }
            render_consensus_panel(consensus, ground_truth_info=ground_truth_info)


if __name__ == "__main__":
    render_case_analysis_page()
