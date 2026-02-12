"""
Agent Panel Component
=====================

Displays agent findings in card format with real-time updates.
Shows 6 agent cards: 3 radiologists + 3 pathologists
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
    "pathologist_context": {
        "display_name": "Context Analyzer",
        "type": "Pathologist",
        "icon": "🔍",
        "approach": "Negation/Uncertainty",
        "color": "#9BE49F",
    },
}

# All agent names in expected order
ALL_AGENTS = [
    "radiologist_densenet",
    "radiologist_resnet", 
    "radiologist_rulebased",
    "pathologist_regex",
    "pathologist_spacy",
    "pathologist_context",
]


def get_class_color(predicted_class: int) -> str:
    """Get color based on binary classification (0=benign, 1=malignant)."""
    colors = {
        0: "#28a745",  # Green - Benign
        1: "#dc3545",  # Red - Malignant
    }
    return colors.get(predicted_class, "#6c757d")


def get_class_label(predicted_class: int) -> str:
    """Get label for binary class."""
    labels = {
        0: "Benign",
        1: "Malignant",
    }
    return labels.get(predicted_class, "Unknown")


def get_classification_category(predicted_class: int) -> Dict[str, str]:
    """Get benign/malignant category for a predicted class."""
    if predicted_class == 0:
        return {"label": "Benign", "color": "#28a745", "icon": "✅"}
    else:
        return {"label": "Malignant", "color": "#dc3545", "icon": "🔴"}


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
            pred_class = finding.get("predicted_class", 0)
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
                        <span style="font-size: 0.85em; color: #555; font-weight: bold;">{weight:.2f}</span>
                    </div>
                    <div style="
                        background-color: #e9ecef;
                        border-radius: 3px;
                        height: 4px;
                        overflow: hidden;
                        margin-top: 3px;
                    ">
                        <div style="
                            background-color: #17a2b8;
                            width: {min(weight * 100, 100)}%;
                            height: 100%;
                        "></div>
                    </div>
                    <div style="text-align: right; font-size: 0.7em; color: #aaa; margin-top: 2px;">
                        dynamic (per-case)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Structured Evidence if available (Module 2)
            nodule_frames = finding.get("details", {}).get("nodule_frames", [])
            if nodule_frames:
                with st.expander(f"🧩 Structured Evidence ({len(nodule_frames)})"):
                    # Custom CSS for tree view
                    st.markdown("""
                    <style>
                    .tree-container { margin-left: 10px; font-family: 'Inter', sans-serif; }
                    .tree-node { position: relative; padding-left: 25px; margin-bottom: 5px; }
                    .tree-node::before {
                        content: '';
                        position: absolute;
                        left: 0;
                        top: 0;
                        bottom: -15px;
                        width: 2px;
                        background: #dee2e6;
                    }
                    .tree-node:last-child::before { height: 12px; }
                    .tree-node::after {
                        content: '';
                        position: absolute;
                        left: 0;
                        top: 12px;
                        width: 20px;
                        height: 2px;
                        background: #dee2e6;
                    }
                    .tree-label { 
                        display: inline-block;
                        padding: 2px 10px;
                        border-radius: 6px;
                        background: #f8f9fa;
                        border: 1px solid #e9ecef;
                        font-size: 0.9em;
                    }
                    .tree-root { font-weight: bold; background: #e3f2fd; border-color: #bbdefb; }
                    .tree-leaf { color: #495057; }
                    .tree-attribute { color: #6c757d; font-size: 0.8em; margin-right: 5px; }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    for i, frame in enumerate(nodule_frames):
                         anchor = frame.get("anchor", "Nodule")
                         
                         st.markdown(f"""
                         <div class="tree-container">
                             <div class="tree-node">
                                 <span class="tree-label tree-root">🔍 {anchor.capitalize()}</span>
                             </div>
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf"><span class="tree-attribute">📏 Size:</span> {frame["size_mm"]} mm</span></div>' for _ in [1] if frame.get("size_mm")])}
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf"><span class="tree-attribute">📍 Loc:</span> {frame["location"]}</span></div>' for _ in [1] if frame.get("location")])}
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf"><span class="tree-attribute">✨ Texture:</span> {frame["texture"]}</span></div>' for _ in [1] if frame.get("texture")])}
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf"><span class="tree-attribute">➖ Margins:</span> {frame["margins"]}</span></div>' for _ in [1] if frame.get("margins")])}
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf">💎 Calcified</span></div>' for _ in [1] if frame.get("calcification")])}
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf">⛔ Negated</span></div>' for _ in [1] if frame.get("negated")])}
                             {"".join([f'<div class="tree-node"><span class="tree-label tree-leaf">❓ Uncertain</span></div>' for _ in [1] if frame.get("uncertain")])}
                         </div>
                         <div style="margin-bottom: 15px;"></div>
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
    Render the full agent panel with all 6 agent cards.
    
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
    path_cols = st.columns(3)
    
    pathologist_agents = [a for a in ALL_AGENTS if "pathologist" in a]
    for i, agent_name in enumerate(pathologist_agents):
        with path_cols[i]:
            finding = completed_lookup.get(agent_name)
            is_completed = agent_name in completed_lookup
            render_agent_card(agent_name, finding, is_completed)


def render_consensus_panel(consensus: Dict[str, Any], ground_truth_info: Optional[Dict[str, Any]] = None):
    """
    Render the final consensus result panel.
    
    Args:
        consensus: Consensus result dictionary
        ground_truth_info: Optional dict with 'ground_truth' (0/1/-1) and 'ground_truth_label'
    """
    st.subheader("🎯 Oncologist Consensus Result")
    
    final_class = consensus.get("final_class", 3)
    final_prob = consensus.get("final_probability", 0.5)
    confidence = consensus.get("confidence", 0.5)
    agreement = consensus.get("agreement_level", "unknown")
    lung_rads = consensus.get("lung_rads_category", "3")
    recommendation = consensus.get("recommendation", "")
    
    class_color = get_class_color(final_class)
    class_label = get_class_label(final_class)
    category = get_classification_category(final_class)
    
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
            <div style="margin-top: 8px;">
                <span style="
                    background-color: {category['color']};
                    color: white;
                    padding: 3px 12px;
                    border-radius: 12px;
                    font-size: 0.9em;
                    font-weight: bold;
                ">{category['icon']} {category['label']}</span>
            </div>
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
    
    # Show ground truth comparison if available
    if ground_truth_info:
        gt_value = ground_truth_info.get("ground_truth")
        gt_label = ground_truth_info.get("ground_truth_label", "unknown")
        
        if gt_value is not None and gt_value != -1:
            # Map ground truth to category
            gt_display = "Abnormal" if gt_value == 1 else "Normal"
            gt_color = "#dc3545" if gt_value == 1 else "#28a745"
            gt_icon = "🔴" if gt_value == 1 else "✅"
            
            # Determine if prediction matches ground truth
            # final_class is already binary: 0=Benign, 1=Malignant
            pred_binary = final_class if final_class in [0, 1] else -1 
            
            if pred_binary == -1:
                match_icon = "⚠️"
                match_text = "Indeterminate — cannot directly compare"
                match_color = "#ffc107"
            elif pred_binary == gt_value:
                match_icon = "✅"
                match_text = "Prediction matches ground truth"
                match_color = "#28a745"
            else:
                match_icon = "❌"
                match_text = "Prediction does NOT match ground truth"
                match_color = "#dc3545"
            
            st.markdown("---")
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%);
                border: 2px solid #b0bec5;
                border-radius: 15px;
                padding: 20px;
            ">
                <div style="font-size: 1.1em; font-weight: bold; color: #333; margin-bottom: 15px;">
                    📊 Ground Truth Comparison
                </div>
                <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.85em; color: #666;">Prediction</div>
                        <div style="
                            font-size: 1.3em; font-weight: bold;
                            color: {category['color']};
                            margin-top: 4px;
                        ">{category['icon']} {category['label']}</div>
                        <div style="font-size: 0.8em; color: #888;">Class {final_class}</div>
                    </div>
                    <div style="font-size: 2em; color: #999;">vs</div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.85em; color: #666;">Ground Truth</div>
                        <div style="
                            font-size: 1.3em; font-weight: bold;
                            color: {gt_color};
                            margin-top: 4px;
                        ">{gt_icon} {gt_display}</div>
                        <div style="font-size: 0.8em; color: #888;">NLP-derived from report</div>
                    </div>
                </div>
                <div style="
                    text-align: center; margin-top: 15px;
                    padding: 8px 16px;
                    background-color: {match_color}22;
                    border: 1px solid {match_color};
                    border-radius: 10px;
                    color: {match_color};
                    font-weight: bold;
                ">{match_icon} {match_text}</div>
            </div>
            """, unsafe_allow_html=True)
        elif gt_value == -1:
            st.markdown("---")
            st.info("📊 **Ground Truth:** Indeterminate — the NLP analysis could not derive a clear label from the report text.")
    
    # Show disagreement details if any
    disagreement_agents = consensus.get("disagreement_agents", [])
    if disagreement_agents:
        st.markdown("---")
        st.warning(f"⚠️ **Disagreement detected** among agents: {', '.join(disagreement_agents)}")
    
    # Show Dynamic Weight Rationale
    weight_rationale = consensus.get("weight_rationale", {})
    if weight_rationale:
        render_weight_rationale(weight_rationale)
    
    # Render Thinking Process (BDI)
    print(f"DEBUG: Consensus keys: {list(consensus.keys())}")
    if "thinking_process" in consensus:
        render_thinking_process(consensus["thinking_process"])


def render_weight_rationale(rationale: Dict[str, Any]):
    """
    Render the dynamic weight rationale panel, showing how per-case
    information richness influenced agent weights.
    
    Args:
        rationale: Weight rationale dict from DynamicWeightCalculator
    """
    st.markdown("---")
    st.subheader("⚖️ Dynamic Weight Assignment")
    
    rad_richness = rationale.get("radiology_richness", 0.5)
    path_richness = rationale.get("pathology_richness", 0.5)
    rad_components = rationale.get("radiology_components", {})
    path_components = rationale.get("pathology_components", {})
    dynamic_weights = rationale.get("dynamic_weights", {})
    base_weights = rationale.get("base_weights", {})
    
    with st.expander("See how agent weights were adapted for this case", expanded=False):
        rcol1, rcol2 = st.columns(2)
        
        with rcol1:
            st.markdown(f"""
            <div style="
                border: 2px solid #4A90D9;
                border-radius: 10px;
                padding: 15px;
                background: linear-gradient(135deg, #e8f0fe 0%, #ffffff 100%);
            ">
                <div style="font-weight: bold; color: #4A90D9; margin-bottom: 8px;">
                    🔬 Radiology Richness: {rad_richness:.1%}
                </div>
                <div style="
                    background-color: #e9ecef;
                    border-radius: 5px;
                    height: 10px;
                    overflow: hidden;
                    margin-bottom: 10px;
                ">
                    <div style="
                        background-color: #4A90D9;
                        width: {rad_richness*100}%;
                        height: 100%;
                    "></div>
                </div>
            """, unsafe_allow_html=True)
            
            for comp_name, comp_val in rad_components.items():
                label = comp_name.replace("_", " ").title()
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin: 3px 0;">
                    <span style="color: #666;">{label}:</span>
                    <span style="color: #333; font-weight: bold;">{comp_val:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with rcol2:
            st.markdown(f"""
            <div style="
                border: 2px solid #7BC47F;
                border-radius: 10px;
                padding: 15px;
                background: linear-gradient(135deg, #e8f8e8 0%, #ffffff 100%);
            ">
                <div style="font-weight: bold; color: #7BC47F; margin-bottom: 8px;">
                    📝 Pathology Richness: {path_richness:.1%}
                </div>
                <div style="
                    background-color: #e9ecef;
                    border-radius: 5px;
                    height: 10px;
                    overflow: hidden;
                    margin-bottom: 10px;
                ">
                    <div style="
                        background-color: #7BC47F;
                        width: {path_richness*100}%;
                        height: 100%;
                    "></div>
                </div>
            """, unsafe_allow_html=True)
            
            for comp_name, comp_val in path_components.items():
                label = comp_name.replace("_", " ").title()
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin: 3px 0;">
                    <span style="color: #666;">{label}:</span>
                    <span style="color: #333; font-weight: bold;">{comp_val:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Weight comparison table
        st.markdown("##### Agent Weight Adjustments")
        weight_data = []
        for agent_name in sorted(dynamic_weights.keys()):
            base_w = base_weights.get(agent_name, 0.5)
            dyn_w = dynamic_weights[agent_name]
            change = dyn_w - base_w
            change_str = f"+{change:.3f}" if change >= 0 else f"{change:.3f}"
            agent_type = "🔬 Rad" if "radiologist" in agent_name else "📝 Path"
            display_name = agent_name.replace("radiologist_", "").replace("pathologist_", "")
            weight_data.append({
                "Type": agent_type,
                "Agent": display_name,
                "Base": f"{base_w:.2f}",
                "Dynamic": f"{dyn_w:.3f}",
                "Change": change_str,
            })
        
        if weight_data:
            import pandas as pd
            df = pd.DataFrame(weight_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_thinking_process(steps: List[Dict[str, str]]):
    """
    Render the agent's internal thinking process (BDI).
    
    Args:
        steps: List of reasoning steps
    """
    st.markdown("---")
    st.subheader("🧠 Agent Thinking Process")
    
    with st.expander("See internal BDI reasoning (Beliefs, Desires, Intentions)", expanded=False):
        for step in steps:
            step_type = step.get("type", "info")
            icon = "ℹ️"
            color = "#17a2b8"
            
            if step_type == "belief":
                icon = "👁️"  # Perception
                color = "#28a745"
            elif step_type == "reasoning":
                icon = "⚙️"  # Processing/Logic
                color = "#ffc107"
            elif step_type == "deliberation":
                icon = "⚖️"  # Weighing options
                color = "#17a2b8"
            elif step_type == "intention":
                icon = "🎯"  # Goal/Action
                color = "#dc3545"
            
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: flex-start;
                margin-bottom: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                border-left: 4px solid {color};
                border-radius: 4px;
            ">
                <div style="font-size: 1.2em; margin-right: 15px;">{icon}</div>
                <div>
                    <div style="font-weight: bold; color: {color}; font-size: 0.85em; text-transform: uppercase;">
                        {step.get('step')}
                    </div>
                    <div style="color: #333;">
                        {step.get('description')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
