"""
Batch Processing Page
=====================

Process multiple nodules at once with progress tracking and export.
"""

import streamlit as st
import requests
import time
import pandas as pd
from typing import List, Dict, Any
import json
from datetime import datetime

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_connection() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_nodule_list() -> list:
    """Fetch list of available nodules from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/nodules", timeout=5)
        if response.status_code == 200:
            return response.json().get("nodule_ids", [])
    except:
        pass
    return []


def start_batch_analysis(nodule_ids: List[str]) -> List[Dict[str, Any]]:
    """Start batch analysis and return session info."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch/analyze",
            json={"nodule_ids": nodule_ids},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("sessions", [])
    except Exception as e:
        st.error(f"Failed to start batch analysis: {e}")
    return []


def get_analysis_status(session_id: str) -> Dict[str, Any]:
    """Get analysis status for a session."""
    try:
        response = requests.get(f"{API_BASE_URL}/analyze/status/{session_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def render_batch_processing_page():
    """Render the batch processing page."""
    
    st.title("📦 Batch Processing")
    st.markdown("Process multiple nodules and export results")
    
    # Check API connection
    if not check_api_connection():
        st.error("""
        ⚠️ **API Not Connected**
        
        Please start the API server first:
        ```bash
        python run_api.py
        ```
        """)
        return
    
    st.success("✅ Connected to API")
    
    # Initialize session state
    if "batch_running" not in st.session_state:
        st.session_state.batch_running = False
    if "batch_sessions" not in st.session_state:
        st.session_state.batch_sessions = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    
    # Get available nodules
    nodule_ids = get_nodule_list()
    
    if not nodule_ids:
        st.warning("No nodules available")
        return
    
    # Selection section
    st.header("📋 Select Nodules")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_nodules = st.multiselect(
            "Select nodules to analyze",
            options=nodule_ids,
            default=nodule_ids[:5] if len(nodule_ids) >= 5 else nodule_ids,
            format_func=lambda x: f"Nodule {x}"
        )
    
    with col2:
        st.metric("Selected", len(selected_nodules))
        
        # Quick select buttons
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_all = True
            st.rerun()
        
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_all = False
            st.rerun()
    
    st.markdown("---")
    
    # Run batch analysis
    st.header("🚀 Run Analysis")
    
    run_col1, run_col2 = st.columns([1, 3])
    
    with run_col1:
        run_button = st.button(
            "Start Batch Analysis",
            disabled=st.session_state.batch_running or len(selected_nodules) == 0,
            type="primary",
            use_container_width=True
        )
    
    with run_col2:
        if len(selected_nodules) == 0:
            st.warning("Please select at least one nodule")
        elif st.session_state.batch_running:
            st.info("⏳ Batch analysis in progress...")
    
    # Start batch
    if run_button and not st.session_state.batch_running:
        sessions = start_batch_analysis(selected_nodules)
        if sessions:
            st.session_state.batch_sessions = sessions
            st.session_state.batch_running = True
            st.session_state.batch_results = []
            st.rerun()
    
    # Progress tracking
    if st.session_state.batch_running and st.session_state.batch_sessions:
        st.subheader("📊 Progress")
        
        total = len(st.session_state.batch_sessions)
        completed = 0
        results = []
        all_done = True
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check status of each session
        for session_info in st.session_state.batch_sessions:
            session_id = session_info.get("session_id")
            nodule_id = session_info.get("nodule_id")
            
            if session_id:
                status = get_analysis_status(session_id)
                
                if status.get("status") == "completed":
                    completed += 1
                    consensus = status.get("consensus", {})
                    results.append({
                        "nodule_id": nodule_id,
                        "status": "completed",
                        "final_class": consensus.get("final_class"),
                        "probability": consensus.get("final_probability"),
                        "confidence": consensus.get("confidence"),
                        "agreement": consensus.get("agreement_level"),
                        "lung_rads": consensus.get("lung_rads_category"),
                    })
                elif status.get("status") == "error":
                    completed += 1
                    results.append({
                        "nodule_id": nodule_id,
                        "status": "error",
                        "error": status.get("error_message"),
                    })
                else:
                    all_done = False
                    results.append({
                        "nodule_id": nodule_id,
                        "status": "running",
                        "progress": f"{status.get('completed_count', 0)}/5 agents"
                    })
            else:
                completed += 1
                results.append({
                    "nodule_id": nodule_id,
                    "status": "error",
                    "error": session_info.get("error", "Failed to start")
                })
        
        # Update progress
        progress_bar.progress(completed / total)
        status_text.write(f"Completed: {completed}/{total} cases")
        
        # Show results table
        if results:
            st.session_state.batch_results = results
            
            df = pd.DataFrame(results)
            
            # Style the dataframe
            def style_status(val):
                if val == "completed":
                    return "background-color: #d4edda"
                elif val == "error":
                    return "background-color: #f8d7da"
                else:
                    return "background-color: #fff3cd"
            
            styled_df = df.style.applymap(style_status, subset=["status"])
            st.dataframe(styled_df, use_container_width=True)
        
        if all_done:
            st.session_state.batch_running = False
            st.success(f"✅ Batch analysis complete! Processed {total} cases.")
        else:
            time.sleep(0.5)
            st.rerun()
    
    # Show results if available
    elif st.session_state.batch_results:
        st.subheader("📊 Results")
        
        df = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary")
        
        completed_results = [r for r in st.session_state.batch_results if r.get("status") == "completed"]
        
        if completed_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processed", len(completed_results))
            
            with col2:
                avg_prob = sum(r.get("probability", 0) for r in completed_results) / len(completed_results)
                st.metric("Avg Probability", f"{avg_prob:.1%}")
            
            with col3:
                avg_conf = sum(r.get("confidence", 0) for r in completed_results) / len(completed_results)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            with col4:
                unanimous = sum(1 for r in completed_results if r.get("agreement") == "unanimous")
                st.metric("Unanimous", f"{unanimous}/{len(completed_results)}")
            
            # Class distribution
            st.subheader("🎯 Class Distribution")
            
            class_counts = {}
            for r in completed_results:
                cls = r.get("final_class", 0)
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            chart_data = pd.DataFrame({
                "Class": [f"Class {k}" for k in sorted(class_counts.keys())],
                "Count": [class_counts[k] for k in sorted(class_counts.keys())]
            })
            
            st.bar_chart(chart_data.set_index("Class"))
    
    # Export section
    st.markdown("---")
    st.header("💾 Export Results")
    
    if st.session_state.batch_results:
        completed_results = [r for r in st.session_state.batch_results if r.get("status") == "completed"]
        
        if completed_results:
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                df = pd.DataFrame(completed_results)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON export
                json_str = json.dumps(completed_results, indent=2)
                
                st.download_button(
                    "📥 Download JSON",
                    data=json_str,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("Run a batch analysis to enable export")


if __name__ == "__main__":
    render_batch_processing_page()
