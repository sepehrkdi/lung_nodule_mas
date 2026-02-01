"""
Report Viewer Component
=======================

Displays generated radiology reports with highlighted key findings.
"""

import streamlit as st
import re
from typing import Optional


def highlight_findings(report_text: str) -> str:
    """
    Add highlighting to key findings in the report.
    
    Highlights:
    - Size measurements (e.g., "12.5 mm")
    - Suspicious terms
    - Benign indicators
    - Lung-RADS categories
    """
    # Highlight measurements
    report_text = re.sub(
        r'(\d+\.?\d*)\s*(mm|cm)',
        r'<span style="background-color: #fff3cd; padding: 1px 4px; border-radius: 3px; font-weight: bold;">\1 \2</span>',
        report_text
    )
    
    # Highlight suspicious terms (red)
    suspicious_terms = [
        'suspicious', 'malignant', 'concerning', 'highly suspicious',
        'spiculated', 'spiculation', 'irregular', 'poorly defined'
    ]
    for term in suspicious_terms:
        pattern = re.compile(rf'\b({term})\b', re.IGNORECASE)
        report_text = pattern.sub(
            r'<span style="background-color: #f8d7da; color: #721c24; padding: 1px 4px; border-radius: 3px;">\1</span>',
            report_text
        )
    
    # Highlight benign terms (green)
    benign_terms = [
        'benign', 'likely benign', 'probably benign', 'well-defined',
        'smooth margins', 'calcified', 'stable'
    ]
    for term in benign_terms:
        pattern = re.compile(rf'\b({term})\b', re.IGNORECASE)
        report_text = pattern.sub(
            r'<span style="background-color: #d4edda; color: #155724; padding: 1px 4px; border-radius: 3px;">\1</span>',
            report_text
        )
    
    # Highlight Lung-RADS
    report_text = re.sub(
        r'(Lung-RADS\s*(?:Category)?:?\s*[1-4][AB]?)',
        r'<span style="background-color: #cce5ff; color: #004085; padding: 2px 6px; border-radius: 3px; font-weight: bold;">\1</span>',
        report_text,
        flags=re.IGNORECASE
    )
    
    return report_text


def render_report_viewer(
    report_text: str,
    nodule_id: str,
    highlight: bool = True,
    show_raw: bool = False
):
    """
    Render the radiology report viewer.
    
    Args:
        report_text: The report text to display
        nodule_id: The nodule ID
        highlight: Whether to highlight key terms
        show_raw: Whether to show raw text option
    """
    st.subheader("📋 Radiology Report")
    
    # Report header
    st.markdown(f"""
    <div style="
        background-color: #f8f9fa;
        border-left: 4px solid #4A90D9;
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
    ">
        <strong>Report ID:</strong> {nodule_id} | 
        <strong>Type:</strong> Pulmonary Nodule Evaluation
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    if show_raw:
        view_mode = st.radio(
            "View Mode",
            options=["Highlighted", "Raw Text"],
            horizontal=True,
            key=f"report_mode_{nodule_id}"
        )
        highlight = view_mode == "Highlighted"
    
    # Process and display report
    if highlight:
        processed_text = highlight_findings(report_text)
        
        # Convert newlines to HTML breaks and wrap in styled div
        processed_text = processed_text.replace('\n', '<br>')
        
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            font-family: 'Georgia', serif;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
        ">
            {processed_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Legend
        with st.expander("🔍 Highlighting Legend"):
            st.markdown("""
            - <span style="background-color: #fff3cd; padding: 1px 4px;">Yellow</span> - Measurements
            - <span style="background-color: #f8d7da; color: #721c24; padding: 1px 4px;">Red</span> - Suspicious findings
            - <span style="background-color: #d4edda; color: #155724; padding: 1px 4px;">Green</span> - Benign indicators
            - <span style="background-color: #cce5ff; color: #004085; padding: 1px 4px;">Blue</span> - Lung-RADS category
            """, unsafe_allow_html=True)
    else:
        st.text_area(
            "Report Text",
            value=report_text,
            height=400,
            disabled=True,
            key=f"report_raw_{nodule_id}"
        )


def render_lung_rads_badge(category: str, size: str = "large"):
    """
    Render a Lung-RADS category badge.
    
    Args:
        category: Lung-RADS category (1, 2, 3, 4A, 4B, etc.)
        size: Badge size ("small", "medium", "large")
    """
    # Color mapping based on category risk
    color_map = {
        "1": ("#28a745", "Negative"),
        "2": ("#5cb85c", "Benign Appearance"),
        "3": ("#ffc107", "Probably Benign"),
        "4A": ("#fd7e14", "Suspicious"),
        "4B": ("#dc3545", "Very Suspicious"),
        "4X": ("#dc3545", "Additional Features"),
    }
    
    color, description = color_map.get(category, ("#6c757d", "Unknown"))
    
    size_styles = {
        "small": "font-size: 1em; padding: 5px 10px;",
        "medium": "font-size: 1.5em; padding: 10px 20px;",
        "large": "font-size: 2em; padding: 15px 30px;",
    }
    
    style = size_styles.get(size, size_styles["medium"])
    
    st.markdown(f"""
    <div style="
        display: inline-block;
        background-color: {color};
        color: white;
        {style}
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    ">
        Lung-RADS {category}
        <div style="font-size: 0.5em; font-weight: normal; margin-top: 5px;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)
