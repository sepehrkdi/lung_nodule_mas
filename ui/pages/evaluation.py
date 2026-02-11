"""
Evaluation Dashboard Page
=========================

Display metrics, confusion matrix, and agreement statistics.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_connection() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False


def start_metrics_computation() -> bool:
    """Kick off background metrics computation."""
    try:
        response = requests.post(f"{API_BASE_URL}/metrics/start", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error starting metrics: {e}")
    return False


def poll_metrics_status() -> Optional[Dict[str, Any]]:
    """Poll for metrics computation progress."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/status", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def render_evaluation_page():
    """Render the evaluation dashboard page."""
    
    st.title("📊 Evaluation Dashboard")
    st.markdown("View performance metrics and agent agreement statistics")
    
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
    
    # Warning about computation time
    st.warning("""
    ⚠️ **Note**: Computing metrics requires analyzing all 50 nodule cases.
    This may take several minutes depending on hardware.
    """)
    
    # Compute metrics button
    if st.button("🔄 Compute Metrics", type="primary"):
        if start_metrics_computation():
            progress_bar = st.progress(0, text="Starting metrics computation...")
            status_text = st.empty()
            
            while True:
                status = poll_metrics_status()
                if status is None:
                    st.error("Lost connection to API")
                    break
                
                state = status["status"]
                processed = status.get("processed", 0)
                total = status.get("total", 0)
                
                if total > 0:
                    pct = processed / total
                    progress_bar.progress(pct, text=f"Analyzing case {processed}/{total}...")
                    status_text.caption(f"Processing... {processed} of {total} cases completed")
                
                if state == "completed":
                    progress_bar.progress(1.0, text="Done!")
                    status_text.empty()
                    st.session_state.metrics = status.get("metrics")
                    st.success("✅ Metrics computed successfully!")
                    break
                elif state == "error":
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Metrics computation failed: {status.get('error_message', 'Unknown error')}")
                    break
                
                time.sleep(2)
            
            st.rerun()
        else:
            st.error("Failed to start metrics computation")
    
    # Display metrics if available
    if "metrics" in st.session_state and st.session_state.metrics:
        metrics = st.session_state.metrics
        
        st.markdown("---")
        
        # Main metrics
        st.header("🎯 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.1%}",
                help="Overall classification accuracy"
            )
        
        with col2:
            st.metric(
                "Precision",
                f"{metrics['precision']:.1%}",
                help="Weighted average precision"
            )
        
        with col3:
            st.metric(
                "Recall",
                f"{metrics['recall']:.1%}",
                help="Weighted average recall"
            )
        
        with col4:
            st.metric(
                "F1 Score",
                f"{metrics['f1_score']:.1%}",
                help="Weighted average F1 score"
            )
        
        st.markdown("---")
        
        # Confusion Matrix
        st.header("🔢 Confusion Matrix")
        
        cm = np.array(metrics["confusion_matrix"])
        labels = metrics.get("class_labels", ["Benign", "Malignant"])
        
        # Create annotated heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True
        )
        
        fig.update_layout(
            title="Predicted vs Actual Classification (Benign / Malignant)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        # Reverse y-axis to have Class 1 at top
        fig.update_yaxes(autorange="reversed")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Agreement Statistics
        st.header("🤝 Agent Agreement Statistics")
        
        total_cases = metrics["total_cases"]
        unanimous = metrics["unanimous_count"]
        majority = metrics["majority_count"]
        split = metrics["split_count"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases", total_cases)
        
        with col2:
            st.metric(
                "Unanimous",
                unanimous,
                f"{unanimous/total_cases:.1%}" if total_cases > 0 else "0%"
            )
        
        with col3:
            st.metric(
                "Majority",
                majority,
                f"{majority/total_cases:.1%}" if total_cases > 0 else "0%"
            )
        
        with col4:
            st.metric(
                "Split",
                split,
                f"{split/total_cases:.1%}" if total_cases > 0 else "0%"
            )
        
        # Agreement pie chart
        agreement_data = {
            "Type": ["Unanimous", "Majority", "Split"],
            "Count": [unanimous, majority, split],
            "Color": ["#28a745", "#ffc107", "#dc3545"]
        }
        
        fig_pie = px.pie(
            agreement_data,
            values="Count",
            names="Type",
            title="Agent Agreement Distribution",
            color="Type",
            color_discrete_map={
                "Unanimous": "#28a745",
                "Majority": "#ffc107",
                "Split": "#dc3545"
            }
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Interpretation
        st.header("📝 Interpretation")
        
        st.markdown(f"""
        ### Summary
        
        - **Total Cases Analyzed**: {total_cases}
        - **Overall Accuracy**: {metrics['accuracy']:.1%}
        
        ### Agreement Analysis
        
        - **Unanimous Agreement** ({unanimous/total_cases:.1%}): All 6 agents agreed on the classification
        - **Majority Agreement** ({majority/total_cases:.1%}): At least 4 of 6 agents agreed
        - **Split Decision** ({split/total_cases:.1%}): No clear majority among agents
        
        ### Key Insights
        
        {"✅ **High Agreement**: Most cases show unanimous or majority agreement, indicating consistent agent behavior." if (unanimous + majority) / total_cases > 0.7 else "⚠️ **Moderate Agreement**: Consider reviewing agent configurations for better consistency."}
        
        {"✅ **Good Accuracy**: The system achieves strong classification performance." if metrics['accuracy'] > 0.7 else "⚠️ **Room for Improvement**: Consider fine-tuning models or adding more training data."}
        """)
        
        # Detailed metrics table
        with st.expander("📊 View Raw Metrics"):
            st.json(metrics)
    
    else:
        st.info("👆 Click 'Compute Metrics' to generate evaluation statistics")
        
        # Show placeholder visualizations
        st.markdown("---")
        st.header("📊 Sample Visualizations")
        
        st.markdown("""
        Once metrics are computed, you'll see:
        
        1. **Performance Metrics**: Accuracy, Precision, Recall, F1 Score
        2. **Confusion Matrix**: Visual comparison of predicted vs actual classes
        3. **Agreement Statistics**: How often agents agree on classifications
        4. **Interpretation**: Automated insights about system performance
        """)


if __name__ == "__main__":
    render_evaluation_page()
