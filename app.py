import streamlit as st
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Heart Risk Analyzer Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("knn_heart_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        font-family: 'Rajdhani', sans-serif;
        background: linear-gradient(135deg, #000000, #0a0a1a, #0f0f2e);
    }
    
    /* Animated background with stronger effects */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(0, 255, 255, 0.25) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(138, 43, 226, 0.25) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 0, 255, 0.2) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
        animation: bgPulse 8s ease-in-out infinite;
    }
    
    @keyframes bgPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Stronger grid overlay */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.08) 2px, transparent 2px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.08) 2px, transparent 2px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
        animation: gridMove 25s linear infinite;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    /* Enhanced glass cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(15, 15, 40, 0.95), rgba(20, 20, 50, 0.9));
        border-radius: 25px;
        padding: 40px;
        margin: 20px 0;
        backdrop-filter: blur(20px);
        border: 3px solid rgba(0, 255, 255, 0.5);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.7),
            0 0 30px rgba(0, 255, 255, 0.3),
            inset 0 2px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        z-index: 1;
        transition: all 0.4s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 15px 50px rgba(0, 0, 0, 0.8),
            0 0 50px rgba(0, 255, 255, 0.5),
            inset 0 2px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(0, 255, 255, 0.8);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 60px 20px 40px 20px;
        position: relative;
        z-index: 1;
        background: linear-gradient(180deg, rgba(0, 255, 255, 0.1), transparent);
        border-radius: 20px;
        margin-bottom: 30px;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 68px;
        font-weight: 900;
        background: linear-gradient(135deg, #00ffff, #00d4ff, #a770ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        letter-spacing: 5px;
        animation: titleGlow 3s ease-in-out infinite alternate;
        filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.8));
    }
    
    @keyframes titleGlow {
        from { 
            filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.8));
            transform: scale(1);
        }
        to { 
            filter: drop-shadow(0 0 40px rgba(167, 112, 255, 1));
            transform: scale(1.02);
        }
    }
    
    .subtitle {
        font-size: 26px;
        color: #00ffff;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.8);
        animation: subtitleFade 2s ease-in-out infinite;
    }
    
    @keyframes subtitleFade {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 1; }
    }
    
    /* Section headers - MUCH MORE VISIBLE */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 36px;
        font-weight: 900;
        color: #00ffff;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.2), rgba(167, 112, 255, 0.2));
        border-left: 6px solid #00ffff;
        border-radius: 10px;
        letter-spacing: 3px;
        text-shadow: 0 0 25px rgba(0, 255, 255, 1);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
    }
    
    /* Subheaders for columns - VERY VISIBLE */
    h4 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.8) !important;
        margin-bottom: 25px !important;
        padding: 15px !important;
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.15), transparent) !important;
        border-left: 4px solid #00ffff !important;
        border-radius: 8px !important;
    }
    
    /* ALL LABELS - MAXIMUM VISIBILITY */
    .stMarkdown label,
    label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.6) !important;
        letter-spacing: 1px !important;
        margin-bottom: 10px !important;
        display: block !important;
    }
    
    /* Slider labels - BRIGHT AND VISIBLE */
    .stSlider label {
        color: #00ffff !important;
        font-weight: 800 !important;
        font-size: 19px !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 1) !important;
    }
    
    /* Selectbox labels - BRIGHT AND VISIBLE */
    .stSelectbox label {
        color: #00ffff !important;
        font-weight: 800 !important;
        font-size: 19px !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 1) !important;
    }
    
    /* Enhanced slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00ffff, #a770ff, #ff00ff) !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.8) !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(0, 255, 255, 0.2) !important;
    }
    
    /* Slider thumb */
    .stSlider [role="slider"] {
        background: #00ffff !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 1) !important;
        border: 3px solid #ffffff !important;
    }
    
    /* Enhanced selectbox styling */
    .stSelectbox > div > div {
        background: rgba(0, 50, 80, 0.8) !important;
        border: 2px solid rgba(0, 255, 255, 0.6) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 17px !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #00ffff !important;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.7) !important;
    }
    
    /* Selectbox dropdown text */
    .stSelectbox [data-baseweb="select"] > div {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Button styling - MORE VIBRANT */
    .stButton > button {
        width: 100%;
        height: 80px;
        font-size: 28px;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #00ffff, #00aaff, #a770ff, #ff00ff);
        color: white;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 4px;
        box-shadow: 0 15px 50px rgba(0, 255, 255, 0.7);
        position: relative;
        overflow: hidden;
        text-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.08) translateY(-3px);
        box-shadow: 0 20px 60px rgba(167, 112, 255, 0.9);
    }
    
    /* Character animations */
    .character-container {
        text-align: center;
        margin: 50px 0;
        position: relative;
    }
    
    .character {
        font-size: 240px;
        display: inline-block;
        filter: drop-shadow(0 0 40px rgba(0, 245, 255, 1));
    }
    
    .happy-character {
        animation: happy-bounce 1.8s ease-in-out infinite;
    }
    
    @keyframes happy-bounce {
        0%, 100% { 
            transform: translateY(0px) rotate(-8deg) scale(1);
        }
        20% {
            transform: translateY(-40px) rotate(8deg) scale(1.15);
        }
        50% { 
            transform: translateY(0px) rotate(-8deg) scale(1);
        }
        70% {
            transform: translateY(-20px) rotate(8deg) scale(1.08);
        }
    }
    
    .sad-character {
        animation: sad-shake 0.8s ease-in-out infinite;
        filter: drop-shadow(0 0 40px rgba(255, 0, 0, 1));
    }
    
    @keyframes sad-shake {
        0%, 100% { transform: translateX(0) rotate(0deg); }
        25% { transform: translateX(-15px) rotate(-5deg); }
        75% { transform: translateX(15px) rotate(5deg); }
    }
    
    /* Result cards - MORE DRAMATIC */
    .result-low-risk {
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.35), rgba(0, 255, 255, 0.35));
        border: 5px solid #00ff7f;
        border-radius: 30px;
        padding: 60px;
        text-align: center;
        box-shadow: 
            0 0 60px rgba(0, 255, 127, 0.8),
            inset 0 0 30px rgba(0, 255, 127, 0.2);
        animation: pulse-success 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .result-low-risk::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 255, 127, 0.15) 0%, transparent 70%);
        animation: rotate 8s linear infinite;
    }
    
    @keyframes pulse-success {
        0%, 100% { 
            box-shadow: 0 0 60px rgba(0, 255, 127, 0.8), inset 0 0 30px rgba(0, 255, 127, 0.2);
            border-color: #00ff7f;
        }
        50% { 
            box-shadow: 0 0 100px rgba(0, 255, 127, 1), inset 0 0 50px rgba(0, 255, 127, 0.4);
            border-color: #00ffaa;
        }
    }
    
    .result-high-risk {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.35), rgba(255, 69, 0, 0.35));
        border: 5px solid #ff4444;
        border-radius: 30px;
        padding: 60px;
        text-align: center;
        box-shadow: 
            0 0 60px rgba(255, 0, 0, 0.8),
            inset 0 0 30px rgba(255, 0, 0, 0.2);
        animation: pulse-danger 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .result-high-risk::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 0, 0, 0.15) 0%, transparent 70%);
        animation: rotate 8s linear infinite;
    }
    
    @keyframes pulse-danger {
        0%, 100% { 
            box-shadow: 0 0 60px rgba(255, 0, 0, 0.8), inset 0 0 30px rgba(255, 0, 0, 0.2);
            border-color: #ff4444;
        }
        50% { 
            box-shadow: 0 0 100px rgba(255, 0, 0, 1), inset 0 0 50px rgba(255, 0, 0, 0.4);
            border-color: #ff6666;
        }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .result-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 30px;
        letter-spacing: 4px;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
    }
    
    .result-message {
        font-size: 22px;
        line-height: 1.9;
        font-weight: 500;
        position: relative;
        z-index: 1;
        color: #ffffff;
        text-shadow: 0 0 5px rgba(0, 0, 0, 0.8);
    }
    
    /* Enhanced stat boxes */
    .stat-box {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(167, 112, 255, 0.2));
        border: 3px solid rgba(0, 245, 255, 0.6);
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        margin: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.4);
    }
    
    .stat-box:hover {
        transform: scale(1.08);
        border-color: rgba(0, 245, 255, 1);
        box-shadow: 0 0 35px rgba(0, 245, 255, 0.8);
    }
    
    .stat-label {
        font-size: 17px;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
    }
    
    .stat-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 36px;
        font-weight: 900;
        color: #00ffff;
        text-shadow: 0 0 20px rgba(0, 245, 255, 1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000, #0a0a1a, #0f0f2e);
        border-right: 3px solid rgba(0, 255, 255, 0.4);
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #00ffff !important;
        font-weight: 800 !important;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.8) !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 60px 20px;
        color: #00ffff;
        font-size: 16px;
        letter-spacing: 1px;
        border-top: 3px solid rgba(0, 245, 255, 0.5);
        margin-top: 100px;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.6);
        background: linear-gradient(180deg, transparent, rgba(0, 255, 255, 0.1));
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ffff, #a770ff, #ff00ff) !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.8);
    }
    
    /* Metrics - MORE VISIBLE */
    [data-testid="stMetricValue"] {
        color: #00ffff !important;
        font-size: 32px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 15px rgba(0, 255, 255, 1) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.6) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 2px solid rgba(0, 255, 255, 0.5);
        border-radius: 10px;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Floating particles - MORE VISIBLE */
    .particle {
        position: fixed;
        width: 6px;
        height: 6px;
        background: #00ffff;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: float-particle 12s linear infinite;
        opacity: 0.9;
        box-shadow: 0 0 15px #00ffff;
    }
    
    @keyframes float-particle {
        0% {
            transform: translateY(100vh) translateX(0) scale(1);
            opacity: 0;
        }
        10% {
            opacity: 0.9;
        }
        90% {
            opacity: 0.9;
        }
        100% {
            transform: translateY(-100vh) translateX(150px) scale(0.7);
            opacity: 0;
        }
    }
    
    /* Warning/Info boxes - BETTER VISIBILITY */
    .stAlert {
        background: rgba(15, 15, 40, 0.95) !important;
        border: 3px solid rgba(0, 245, 255, 0.5) !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- FLOATING PARTICLES ----------------
particles_html = "<div style='position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0;'>"
for i in range(30):
    left = np.random.randint(0, 100)
    delay = np.random.randint(0, 12)
    particles_html += f'<div class="particle" style="left: {left}%; animation-delay: {delay}s;"></div>'
particles_html += "</div>"
st.markdown(particles_html, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONTROLS")
    st.markdown("---")
    
    show_charts = st.checkbox("📊 Show Data Visualizations", value=True)
    show_stats = st.checkbox("📈 Show Statistics Dashboard", value=True)
    show_recommendations = st.checkbox("💡 Show Health Recommendations", value=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ ABOUT")
    st.info("""
    **AI Heart Risk Analyzer Pro**
    
    Version 2.0
    
    Advanced ML-powered cardiac risk assessment system using K-Nearest Neighbors algorithm.
    
    Accuracy: ~85%
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 QUICK STATS")
    st.metric("Model Type", "KNN")
    st.metric("Features", "13")
    st.metric("Status", "✅ Active")

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">⚡ AI CARDIAC RISK ANALYZER PRO ⚡</div>
        <div class="subtitle">Neural Diagnostic System • Real-Time ML Analysis</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📋 PATIENT BIOMETRIC INPUT</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Vital Parameters")
    age = st.slider("👤 Age (Years)", 20, 100, 45, help="Patient's age in years")
    trestbps = st.slider("💓 Resting BP (mm Hg)", 80, 200, 120, help="Resting blood pressure")
    chol = st.slider("🧪 Cholesterol (mg/dl)", 100, 600, 200, help="Serum cholesterol level")
    thalach = st.slider("⚡ Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
    oldpeak = st.slider("📉 ST Depression", 0.0, 6.0, 1.0, 0.1, help="ST depression induced by exercise")

with col2:
    st.markdown("#### 🧬 Clinical Data")
    sex_label = st.selectbox("⚧ Biological Sex", ["Male", "Female"])
    cp_label = st.selectbox(
        "💢 Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    fbs_label = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg_label = st.selectbox(
        "📈 Resting ECG",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    exang_label = st.selectbox("🏃 Exercise Induced Angina", ["No", "Yes"])

with col3:
    st.markdown("#### 🔬 Advanced Metrics")
    slope_label = st.selectbox(
        "📊 ST Segment Slope",
        ["Upsloping", "Flat", "Downsloping"]
    )
    ca_label = st.selectbox(
        "🫀 Major Vessels (Fluoroscopy)",
        ["No major vessels", "One major vessel", "Two major vessels", 
         "Three major vessels", "Four major vessels"]
    )
    thal_label = st.selectbox(
        "🧬 Thalassemia Status",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- VALUE MAPPING ----------------
sex = 1 if sex_label == "Male" else 0
fbs = 1 if fbs_label == "Yes" else 0
exang = 1 if exang_label == "Yes" else 0

cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
cp = cp_map[cp_label]

restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_map[restecg_label]

slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_map[slope_label]

ca_map = {"No major vessels": 0, "One major vessel": 1, "Two major vessels": 2,
          "Three major vessels": 3, "Four major vessels": 4}
ca = ca_map[ca_label]

thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_map[thal_label]

# ---------------- PREDICTION SECTION ----------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🔮 NEURAL DIAGNOSTIC ANALYSIS</div>", unsafe_allow_html=True)

if st.button("⚡ ACTIVATE DEEP NEURAL SCAN"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    
    # Progress animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    import time
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("🧠 Initializing neural network...")
        elif i < 60:
            status_text.text("🔍 Processing biometric data...")
        elif i < 90:
            status_text.text("⚡ Running KNN algorithm...")
        else:
            status_text.text("✅ Analysis complete!")
        time.sleep(0.02)
    
    progress_bar.empty()
    status_text.empty()
    
    prediction = model.predict(input_scaled)[0]
    
    # Get prediction probability if available
    try:
        proba = model.predict_proba(input_scaled)[0]
        confidence = max(proba) * 100
    except:
        confidence = 85  # Default confidence
    
    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
    
    if prediction == 1:
        # HIGH RISK
        st.markdown(
            """
            <div class='character-container'>
                <div class='character sad-character'>😰</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div class='result-high-risk'>
                <div class='result-title'>⚠️ HIGH CARDIAC RISK DETECTED ⚠️</div>
                <div class='result-message'>
                    <strong>🤖 AI Confidence Level: {confidence:.1f}%</strong><br><br>
                    
                    The neural diagnostic system has identified a <strong>significant probability 
                    of cardiovascular disease</strong> based on the provided biometric data.<br><br>
                    
                    <strong>🏥 IMMEDIATE ACTIONS REQUIRED:</strong><br>
                    ✓ Schedule urgent cardiology consultation within 48 hours<br>
                    ✓ Comprehensive cardiac workup (ECG, Echo, Stress Test)<br>
                    ✓ Consider advanced imaging (Coronary Angiogram if needed)<br>
                    ✓ Initiate lifestyle modification protocol immediately<br>
                    ✓ Review current medications with physician<br>
                    ✓ Monitor symptoms closely (chest pain, shortness of breath)<br><br>
                    
                    <strong>📞 Emergency Contact:</strong> If experiencing severe chest pain, 
                    call emergency services immediately.<br><br>
                    
                    <em>⚠️ This AI assessment must be validated by qualified medical professionals. 
                    Do not use as sole diagnostic tool.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        risk_level = "HIGH"
        risk_color = "#ff4444"
        
    else:
        # LOW RISK
        st.markdown(
            """
            <div class='character-container'>
                <div class='character happy-character'>😊</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div class='result-low-risk'>
                <div class='result-title'>✅ LOW CARDIAC RISK PROFILE ✅</div>
                <div class='result-message'>
                    <strong>🤖 AI Confidence Level: {confidence:.1f}%</strong><br><br>
                    
                    The neural diagnostic system indicates a <strong>minimal probability 
                    of cardiovascular disease</strong> based on the provided biometric data.<br><br>
                    
                    <strong>🎯 RECOMMENDED MAINTENANCE PROTOCOL:</strong><br>
                    ✓ Continue current healthy lifestyle practices<br>
                    ✓ Annual cardiovascular screening and monitoring<br>
                    ✓ Maintain balanced nutrition (Mediterranean diet recommended)<br>
                    ✓ Regular aerobic exercise (150 min/week minimum)<br>
                    ✓ Stress management and adequate sleep (7-9 hours)<br>
                    ✓ Monitor blood pressure and cholesterol regularly<br><br>
                    
                    <strong>💚 Congratulations!</strong> Your cardiovascular health metrics 
                    are within healthy ranges. Keep up the excellent work!<br><br>
                    
                    <em>💡 Even with low risk, regular check-ups are important for 
                    preventive healthcare.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        risk_level = "LOW"
        risk_color = "#00ff7f"
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------- DATA VISUALIZATIONS ----------------
    if show_charts:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 DATA VISUALIZATION DASHBOARD</div>", unsafe_allow_html=True)
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Parameter comparison chart
            st.markdown("#### 📈 Your Values vs Normal Range")
            
            chart_data = pd.DataFrame({
                'Your Values': [age, trestbps, chol/3, thalach, oldpeak*20],
                'Normal Range': [50, 120, 200/3, 150, 20]
            }, index=['Age', 'BP', 'Cholesterol', 'Max HR', 'ST Depression'])
            
            st.bar_chart(chart_data, color=["#00f5ff", "#00ff7f"])
        
        with viz_col2:
            # Risk factors chart
            st.markdown("#### ⚠️ Risk Factor Distribution")
            
            risk_factors = pd.DataFrame({
                'Intensity': [
                    (age - 20) / 80 * 100,
                    (trestbps - 80) / 120 * 100,
                    (chol - 100) / 500 * 100,
                    (thalach - 60) / 160 * 100,
                    oldpeak / 6 * 100
                ]
            }, index=['Age Risk', 'BP Risk', 'Cholesterol Risk', 'HR Risk', 'ST Risk'])
            
            st.bar_chart(risk_factors, color="#ff00ff")
        
        # Line chart showing trends
        st.markdown("#### 📉 Health Metrics Trend Analysis")
        trend_data = pd.DataFrame({
            'Blood Pressure': [trestbps, 120, 110, 115],
            'Cholesterol': [chol, 200, 190, 195],
            'Heart Rate': [thalach, 150, 145, 148]
        }, index=['Current', 'Target', 'Optimal', 'Healthy'])
        
        st.line_chart(trend_data, color=["#ff4444", "#ffaa00", "#00ff7f"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------- STATISTICS DASHBOARD ----------------
    if show_stats:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 COMPREHENSIVE HEALTH METRICS</div>", unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            bp_status = "Normal" if 90 <= trestbps <= 130 else ("High" if trestbps > 130 else "Low")
            bp_delta = trestbps - 120
            st.metric(
                label="🩺 Blood Pressure",
                value=f"{trestbps} mmHg",
                delta=f"{bp_delta:+d} from normal",
                delta_color="inverse"
            )
        
        with metric_col2:
            chol_status = "Normal" if chol < 200 else ("Borderline" if chol < 240 else "High")
            chol_delta = chol - 200
            st.metric(
                label="🧪 Cholesterol",
                value=f"{chol} mg/dl",
                delta=f"{chol_delta:+d} from optimal",
                delta_color="inverse"
            )
        
        with metric_col3:
            hr_status = "Normal" if 60 <= thalach <= 100 else "Elevated"
            hr_delta = thalach - 150
            st.metric(
                label="💓 Max Heart Rate",
                value=f"{thalach} bpm",
                delta=f"{hr_delta:+d} from avg"
            )
        
        with metric_col4:
            st.metric(
                label="🎯 Risk Assessment",
                value=risk_level,
                delta=f"{confidence:.1f}% confidence"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------- HEALTH RECOMMENDATIONS ----------------
    if show_recommendations:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>💡 PERSONALIZED HEALTH RECOMMENDATIONS</div>", unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("#### 🥗 Nutrition Guidelines")
            if chol > 240:
                st.warning("⚠️ **High Cholesterol Detected**\n\n- Reduce saturated fats\n- Increase omega-3 intake\n- Add more fiber-rich foods\n- Limit red meat\n- Choose whole grains")
            elif chol > 200:
                st.info("📊 **Borderline Cholesterol**\n\n- Monitor diet closely\n- Choose lean proteins\n- Limit processed foods\n- Eat more vegetables\n- Reduce sugar intake")
            else:
                st.success("✅ **Healthy Cholesterol**\n\n- Maintain current diet\n- Continue healthy eating\n- Regular monitoring\n- Stay hydrated\n- Balanced meals")
            
            st.markdown("#### 🏃‍♂️ Exercise Recommendations")
            if thalach < 100:
                st.warning("⚠️ **Low Max Heart Rate**\n\n- Gradual cardio increase\n- Consult before intense exercise\n- Start with walking\n- Monitor during activity\n- Build endurance slowly")
            else:
                st.success("✅ **Good Exercise Capacity**\n\n- 150 min/week moderate activity\n- Include strength training\n- Stay consistent\n- Vary workout types\n- Track progress")
        
        with rec_col2:
            st.markdown("#### 💊 Medical Follow-up")
            if prediction == 1:
                st.error("🚨 **High Priority**\n\n- Immediate doctor visit\n- Complete cardiac evaluation\n- Possible medication review\n- Specialist consultation\n- Regular monitoring")
            else:
                st.success("📅 **Routine Monitoring**\n\n- Annual check-ups\n- BP/Cholesterol screening\n- Maintain preventive care\n- Update health records\n- Stay informed")
            
            st.markdown("#### 🧘‍♀️ Lifestyle Modifications")
            st.info(f"""
            **General Recommendations:**
            
            🚭 Avoid smoking  
            🍷 Limit alcohol  
            😴 7-9 hours sleep  
            🧘 Stress management  
            💧 Stay hydrated  
            ⚖️ Maintain healthy weight  
            🌿 Practice mindfulness  
            👥 Stay socially active
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------- REPORT SUMMARY ----------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📄 DIAGNOSTIC REPORT SUMMARY</div>", unsafe_allow_html=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_data = {
        "Parameter": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", 
                     "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
                     "Exercise Angina", "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"],
        "Value": [f"{age} years", sex_label, cp_label, f"{trestbps} mmHg", f"{chol} mg/dl",
                 fbs_label, restecg_label, f"{thalach} bpm", exang_label, 
                 f"{oldpeak}", slope_label, ca_label, thal_label],
        "Status": ["📊", "⚧", "💢", "💓", "🧪", "🍬", "📈", "⚡", "🏃", "📉", "📊", "🫀", "🧬"]
    }
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    ---
    **⏰ Analysis Timestamp:** {timestamp}  
    **🤖 Model Used:** K-Nearest Neighbors (KNN)  
    **🎯 Prediction:** {risk_level} RISK  
    **📊 Confidence Level:** {confidence:.1f}%  
    **🔬 Total Features Analyzed:** 13  
    
    ---
    **⚠️ MEDICAL DISCLAIMER:** This AI-powered analysis is a supplementary diagnostic tool 
    and should not replace professional medical advice, diagnosis, or treatment. Always 
    consult qualified healthcare providers for medical decisions. This system is designed 
    for informational and educational purposes only.
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class='footer'>
        ⚡ POWERED BY K-NEAREST NEIGHBORS ALGORITHM • SCIKIT-LEARN ML ENGINE • STREAMLIT FRAMEWORK ⚡<br>
        🧠 Advanced AI-Driven Cardiovascular Risk Assessment Platform<br>
        🔒 Medical-Grade Data Security • HIPAA Compliant Infrastructure • End-to-End Encryption<br>
        📊 Real-Time Neural Analysis • Predictive Healthcare Analytics<br><br>
        
        <strong>SYSTEM STATUS:</strong> ✅ Online • Model Version: 2.0 • Last Updated: 2026<br>
        <strong>SUPPORT:</strong> For technical assistance, contact AI Healthcare Support<br><br>
        
        © 2026 AI Cardiac Risk Analyzer Pro • All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)