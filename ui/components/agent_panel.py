"""
Agent Panel Component
=====================

Displays agent findings in card format with real-time updates.
Shows 5 agent cards: 3 radiologists + 2 pathologists
"""

import streamlit as st
from typing import Dict, Any, List, Optional


# Agent metadata for display
AGENT_INFO = {
    "radiologist_densenet": {
        "display_name": "DenseNet-121",
        "type": "Radiologist",
        "icon": "🔬",
        "approach": "Deep Learning CNN",
        "color": "#4A90D9",
    },
    "radiologist_resnet": {
        "display_name": "ResNet-50",
        "type": "Radiologist", 
        "icon": "🔬",
        "approach": "Deep Learning CNN",
        "color": "#5A9FE8",
    },
    "radiologist_rulebased": {
        "display_name": "Rule-Based",
        "type": "Radiologist",
        "icon": "📏",
        "approach": "Heuristic Rules",
        "color": "#6AAFFF",
    },
    "pathologist_regex": {
        "display_name": "Regex Parser",
        "type": "Pathologist",
        "icon": "📝",
        "approach": "Pattern Matching",
        "color": "#7BC47F",
    },
    "pathologist_spacy": {
        "display_name": "spaCy NER",
        "type": "Pathologist",
        "icon": "🧠",
        "approach": "NLP / NER",
        "color": "#8BD48F",
    },
}

# All agent names in expected order
ALL_AGENTS = [
    "radiologist_densenet",
    "radiologist_resnet", 
    "radiologist_rulebased",
    "pathologist_regex",
    "pathologist_spacy",
]


def get_class_color(predicted_class: int) -> str:
    """Get color based on malignancy class."""
    colors = {
        1: "#28a745",  # Green - Highly Unlikely
        2: "#5cb85c",  # Light Green - Moderately Unlikely
        3: "#ffc107",  # Yellow - Indeterminate
        4: "#fd7e14",  # Orange - Moderately Suspicious
        5: "#dc3545",  # Red - Highly Suspicious
    }
    return colors.get(predicted_class, "#6c757d")


def get_class_label(predicted_class: int) -> str:
    """Get label for malignancy class."""
    labels = {
        1: "Highly Unlikely",
        2: "Moderately Unlikely", 
        3: "Indeterminate",
        4: "Moderately Suspicious",
        5: "Highly Suspicious",
    }
    return labels.get(predicted_class, "Unknown")


def render_agent_card(
    agent_name: str,
    finding: Optional[Dict[str, Any]] = None,
    is_completed: bool = False
):
    """
    Render a single agent card.
    
    Args:
        agent_name: Name of the agent
        finding: Agent's finding dictionary (if completed)
        is_completed: Whether the agent has completed analysis
    """
    info = AGENT_INFO.get(agent_name, {
        "display_name": agent_name,
        "type": "Unknown",
        "icon": "❓",
        "approach": "Unknown",
        "color": "#6c757d",
    })
    
    with st.container():
        # Card styling based on status
        if is_completed and finding:
            prob = finding.get("probability", 0.5)
            pred_class = finding.get("predicted_class", 3)
            weight = finding.get("weight", 1.0)
            
            class_color = get_class_color(pred_class)
            class_label = get_class_label(pred_class)
            
            st.markdown(f"""
            <div style="
                border: 2px solid {info['color']};
                border-radius: 10px;
                padding: 15px;
                margin: 5px 0;
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.5em;">{info['icon']}</span>
                    <span style="
                        background-color: #28a745;
                        color: white;
                        padding: 2px 8px;
                        border-radius: 10px;
                        font-size: 0.75em;
                    ">✓ Complete</span>
                </div>
                <h4 style="margin: 10px 0 5px 0; color: #333;">{info['display_name']}</h4>
                <p style="margin: 0; color: #666; font-size: 0.85em;">{info['type']} • {info['approach']}</p>
                <hr style="margin: 10px 0; border-color: #eee;">
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-size: 0.9em;">Probability:</span>
                        <strong>{prob:.1%}</strong>
                    </div>
                    <div style="
                        background-color: #e9ecef;
                        border-radius: 5px;
                        height: 8px;
                        overflow: hidden;
                    ">
                        <div style="
                            background-color: {class_color};
                            width: {prob*100}%;
                            height: 100%;
                        "></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <span style="font-size: 0.9em;">Prediction:</span>
                        <span style="
                            background-color: {class_color};
                            color: white;
                            padding: 2px 8px;
                            border-radius: 5px;
                            font-size: 0.85em;
                        ">Class {pred_class}</span>
                    </div>
                    <div style="text-align: center; margin-top: 5px; font-size: 0.8em; color: #666;">
                        {class_label}
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-size: 0.85em; color: #888;">Weight:</span>
                        <span style="font-size: 0.85em; color: #888;">{weight:.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Waiting state
            st.markdown(f"""
            <div style="
                border: 2px dashed #dee2e6;
                border-radius: 10px;
                padding: 15px;
                margin: 5px 0;
                background-color: #f8f9fa;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.5em; opacity: 0.5;">{info['icon']}</span>
                    <span style="
                        background-color: #6c757d;
                        color: white;
                        padding: 2px 8px;
                        border-radius: 10px;
                        font-size: 0.75em;
                    ">⏳ Waiting</span>
                </div>
                <h4 style="margin: 10px 0 5px 0; color: #999;">{info['display_name']}</h4>
                <p style="margin: 0; color: #aaa; font-size: 0.85em;">{info['type']} • {info['approach']}</p>
                <hr style="margin: 10px 0; border-color: #eee;">
                <div style="text-align: center; padding: 20px 0; color: #999;">
                    <div class="spinner" style="
                        border: 3px solid #f3f3f3;
                        border-top: 3px solid #3498db;
                        border-radius: 50%;
                        width: 24px;
                        height: 24px;
                        margin: 0 auto 10px;
                        animation: spin 1s linear infinite;
                    "></div>
                    Analyzing...
                </div>
            </div>
            <style>
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
            """, unsafe_allow_html=True)


def render_agent_panel(
    completed_agents: List[Dict[str, Any]],
    show_all: bool = True
):
    """
    Render the full agent panel with all 5 agent cards.
    
    Args:
        completed_agents: List of completed agent findings
        show_all: Whether to show all agents or only completed ones
    """
    st.subheader("🤖 Agent Analysis")
    
    # Create lookup for completed agents
    completed_lookup = {
        a.get("agent_name"): a for a in completed_agents
    }
    
    # Progress indicator
    completed_count = len(completed_agents)
    total_count = len(ALL_AGENTS)
    
    # Clamp progress to valid range [0.0, 1.0]
    progress_value = min(completed_count / total_count, 1.0) if total_count > 0 else 0.0
    
    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        st.progress(progress_value)
    with progress_col2:
        st.write(f"**{min(completed_count, total_count)}/{total_count}** agents")
    
    st.markdown("---")
    
    # Radiologists section
    st.markdown("#### 🔬 Radiologists (Image Analysis)")
    rad_cols = st.columns(3)
    
    radiologist_agents = [a for a in ALL_AGENTS if "radiologist" in a]
    for i, agent_name in enumerate(radiologist_agents):
        with rad_cols[i]:
            finding = completed_lookup.get(agent_name)
            is_completed = agent_name in completed_lookup
            render_agent_card(agent_name, finding, is_completed)
    
    st.markdown("---")
    
    # Pathologists section
    st.markdown("#### 📝 Pathologists (Report Analysis)")
    path_cols = st.columns(2)
    
    pathologist_agents = [a for a in ALL_AGENTS if "pathologist" in a]
    for i, agent_name in enumerate(pathologist_agents):
        with path_cols[i]:
            finding = completed_lookup.get(agent_name)
            is_completed = agent_name in completed_lookup
            render_agent_card(agent_name, finding, is_completed)


def render_consensus_panel(consensus: Dict[str, Any]):
    """
    Render the final consensus result panel.
    
    Args:
        consensus: Consensus result dictionary
    """
    st.subheader("🎯 Consensus Result")
    
    final_class = consensus.get("final_class", 3)
    final_prob = consensus.get("final_probability", 0.5)
    confidence = consensus.get("confidence", 0.5)
    agreement = consensus.get("agreement_level", "unknown")
    lung_rads = consensus.get("lung_rads_category", "3")
    recommendation = consensus.get("recommendation", "")
    
    class_color = get_class_color(final_class)
    class_label = get_class_label(final_class)
    
    # Agreement badge
    agreement_colors = {
        "unanimous": "#28a745",
        "majority": "#ffc107",
        "split": "#dc3545",
    }
    agreement_color = agreement_colors.get(agreement, "#6c757d")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {class_color}22 0%, {class_color}11 100%);
            border: 2px solid {class_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        ">
            <div style="font-size: 0.9em; color: #666;">Final Prediction</div>
            <div style="font-size: 2.5em; font-weight: bold; color: {class_color};">
                Class {final_class}
            </div>
            <div style="font-size: 1em; color: #333;">{class_label}</div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                Probability: <strong>{final_prob:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border: 2px solid #dee2e6;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        ">
            <div style="font-size: 0.9em; color: #666;">Lung-RADS Category</div>
            <div style="font-size: 2.5em; font-weight: bold; color: #333;">
                {lung_rads}
            </div>
            <div style="margin-top: 10px;">
                <span style="
                    background-color: {agreement_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 15px;
                    font-size: 0.85em;
                ">{agreement.capitalize()} Agreement</span>
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                Confidence: <strong>{confidence:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
            border: 2px solid #90caf9;
            border-radius: 15px;
            padding: 20px;
        ">
            <div style="font-size: 0.9em; color: #666; margin-bottom: 10px;">📋 Recommendation</div>
            <div style="font-size: 0.95em; color: #333; line-height: 1.5;">
                {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show disagreement details if any
    disagreement_agents = consensus.get("disagreement_agents", [])
    if disagreement_agents:
        st.markdown("---")
        st.warning(f"⚠️ **Disagreement detected** among agents: {', '.join(disagreement_agents)}")
