"""
Femtech Privacy Policy Deception Detection - Streamlit Demo
============================================================
Run with: streamlit run demo/streamlit_app.py

Author: Alexis Prieto
Course: ML/Deep Learning Research - Fall 2025
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Page config
st.set_page_config(
    page_title="Femtech Privacy Detector",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Privacy Policy Deception Detection")
st.markdown("**Detecting contradictions between privacy policy claims and actual practices using NLI**")
st.markdown("---")

# Load model (cached)
@st.cache_resource
def load_model():
    model_name = "microsoft/deberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

with st.spinner("Loading DeBERTa model... (this may take a minute)"):
    tokenizer, model, device = load_model()

st.success(f"Model loaded! Running on: {device}")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This tool uses **Natural Language Inference (NLI)** to detect 
contradictions between privacy policy claims and documented practices.

**Key Results:**
- 76.4% average contradiction score
- 100% regulatory alignment
- 93.3% HIGH/CRITICAL detection

**Author:** Alexis Prieto  
**Course:** ML/Deep Learning Research
""")

# Detection function
def detect_contradiction(policy_claim, violation):
    inputs = tokenizer(
        violation,
        policy_claim,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    return {
        'contradiction': probs[0].item(),
        'neutral': probs[1].item(),
        'entailment': probs[2].item()
    }

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📜 Privacy Policy Claim")
    policy_claim = st.text_area(
        "What the policy says:",
        value="We do not share users' health information with third parties",
        height=100
    )

with col2:
    st.subheader("⚠️ Documented Practice")
    violation = st.text_area(
        "What actually happened:",
        value="Shared pregnancy and period data with Facebook Analytics from 2016-2019",
        height=100
    )

# Analyze button
if st.button("🔍 Analyze for Contradiction", type="primary"):
    with st.spinner("Analyzing..."):
        scores = detect_contradiction(policy_claim, violation)
    
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Score display
    c_score = scores['contradiction']
    
    # Severity
    if c_score >= 0.75:
        severity = "🔴 CRITICAL"
        color = "red"
    elif c_score >= 0.55:
        severity = "🟠 HIGH"
        color = "orange"
    elif c_score >= 0.35:
        severity = "🟡 MEDIUM"
        color = "yellow"
    else:
        severity = "🟢 LOW"
        color = "green"
    
    # Display metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Contradiction Score", f"{c_score:.3f}")
    m2.metric("Severity", severity)
    m3.metric("Neutral Score", f"{scores['neutral']:.3f}")
    
    # Progress bar
    st.progress(c_score)
    
    # Interpretation
    if c_score >= 0.55:
        st.error(f"**High contradiction detected!** The policy claim appears to contradict the documented practice.")
    elif c_score >= 0.35:
        st.warning(f"**Moderate contradiction.** Some inconsistency between policy and practice.")
    else:
        st.success(f"**Low contradiction.** Policy appears consistent with practice.")

# Example violations
st.markdown("---")
st.subheader("📋 Pre-loaded Examples")

examples = {
    "Flo - Facebook Sharing ($59.5M Settlement)": {
        "policy": "We do not share users' health information with third parties",
        "violation": "Shared pregnancy and period data with Facebook Analytics from 2016-2019"
    },
    "Ovia - Employer Sharing": {
        "policy": "Your health data is kept strictly confidential",
        "violation": "Shared detailed health data with employers through corporate wellness programs"
    },
    "Glow - Security Failure ($250K Fine)": {
        "policy": "Your data is securely stored using encryption and access controls",
        "violation": "User data was accessible to anyone who knew the user's email address"
    },
    "Apple Health - Control (Good)": {
        "policy": "Health data stays on your device and is not sent to Apple",
        "violation": "Health data is processed and stored on device without transmission to Apple servers"
    }
}

selected = st.selectbox("Select an example:", list(examples.keys()))

if st.button("Load Example"):
    st.session_state['policy'] = examples[selected]['policy']
    st.session_state['violation'] = examples[selected]['violation']
    st.experimental_rerun()
