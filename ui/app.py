"""
Lung Nodule Multi-Agent System - Streamlit Dashboard
=====================================================

Main entry point for the Streamlit web interface.

Run with:
    streamlit run ui/app.py
    
Or use the launcher:
    python run_ui.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Lung Nodule MAS",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/lung_nodule_mas",
        "Report a bug": "https://github.com/your-repo/lung_nodule_mas/issues",
        "About": """
        # Lung Nodule Multi-Agent System
        
        A multi-agent system for lung nodule classification using:
        - 3 Radiologist agents (DenseNet-121, ResNet-50, Rule-based)
        - 2 Pathologist agents (Regex, spaCy NER)
        - Prolog-based consensus mechanism
        
        Educational demonstration of AI in medical imaging.
        """
    }
)

# Import pages
from ui.pages.case_analysis import render_case_analysis_page
from ui.pages.batch_processing import render_batch_processing_page
from ui.pages.evaluation import render_evaluation_page


def render_sidebar():
    """Render the sidebar with system info."""
    
    # System info
    st.sidebar.markdown("### 🤖 Agent Configuration")
    st.sidebar.markdown("""
    **Radiologists:**
    - DenseNet-121 (W=1.0)
    - ResNet-50 (W=1.0)
    - Rule-based (W=0.7)
    
    **Pathologists:**
    - Regex Parser (W=0.8)
    - spaCy NER (W=0.9)
    """)


def render_header():
    """Render the page header."""
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(
            "https://img.icons8.com/fluency/96/lungs.png",
            width=80
        )
    with col2:
        st.title("🫁 Lung Nodule MAS")
        st.markdown("Multi-Agent Classification System")
    
    st.markdown("---")


def main():
    """Main application entry point."""
    
    # Render header at top of page
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Tab-based navigation in main area
    tab1, tab2, tab3 = st.tabs([
        "🔬 Case Analysis",
        "📦 Batch Processing", 
        "📊 Evaluation Dashboard"
    ])
    
    with tab1:
        render_case_analysis_page()
    
    with tab2:
        render_batch_processing_page()
    
    with tab3:
        render_evaluation_page()


if __name__ == "__main__":
    main()
