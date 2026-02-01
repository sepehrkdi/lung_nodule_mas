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
    """Render the sidebar navigation."""
    
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/lungs.png",
        width=80
    )
    
    st.sidebar.title("🫁 Lung Nodule MAS")
    st.sidebar.markdown("Multi-Agent Classification System")
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🔬 Case Analysis",
            "📦 Batch Processing",
            "📊 Evaluation Dashboard"
        ],
        index=0,
        key="navigation"
    )
    
    st.sidebar.markdown("---")
    
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
    
    st.sidebar.markdown("---")
    
    # Help section
    with st.sidebar.expander("ℹ️ Help"):
        st.markdown("""
        **Quick Start:**
        1. Start the API: `python run_api.py`
        2. Select a nodule from the dropdown
        3. Click "Analyze Nodule"
        4. Watch agents process in real-time
        
        **API Status:**
        - Green = Connected
        - Red = Start API server
        
        **Questions?**
        Check the README.md for detailed documentation.
        """)
    
    return page


def main():
    """Main application entry point."""
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "🔬 Case Analysis":
        render_case_analysis_page()
    elif page == "📦 Batch Processing":
        render_batch_processing_page()
    elif page == "📊 Evaluation Dashboard":
        render_evaluation_page()


if __name__ == "__main__":
    main()
