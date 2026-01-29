"""
Bloom's Taxonomy Question Classifier
Streamlit App for S-BERT and S-FastText Models

This app classifies educational questions into Bloom's Taxonomy levels:
- Remember, Understand, Apply, Analyze, Evaluate, Create
"""

# Suppress warnings before importing heavy libraries
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import torch
import fasttext
import re
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================
# Configuration
# ============================================
MODELS_DIR = Path(__file__).parent / "models"
FASTTEXT_MODEL_PATH = MODELS_DIR / "s-fasttext.bin"
BERT_MODEL_DIR = MODELS_DIR / "s-bert_model"

BLOOM_LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
BLOOM_COLORS = {
    'Remember': '#FF6B6B',
    'Understand': '#4ECDC4', 
    'Apply': '#45B7D1',
    'Analyze': '#96CEB4',
    'Evaluate': '#FFEAA7',
    'Create': '#DDA0DD'
}

BLOOM_DESCRIPTIONS = {
    'Remember': 'Recall facts and basic concepts',
    'Understand': 'Explain ideas or concepts',
    'Apply': 'Use information in new situations',
    'Analyze': 'Draw connections among ideas',
    'Evaluate': 'Justify a decision or course of action',
    'Create': 'Produce new or original work'
}

BLOOM_VERBS = {
    'Remember': ['define', 'list', 'name', 'recall', 'identify', 'describe'],
    'Understand': ['explain', 'summarize', 'interpret', 'classify', 'compare'],
    'Apply': ['use', 'implement', 'solve', 'demonstrate', 'calculate'],
    'Analyze': ['analyze', 'examine', 'differentiate', 'organize', 'compare'],
    'Evaluate': ['evaluate', 'judge', 'assess', 'critique', 'justify'],
    'Create': ['design', 'create', 'develop', 'formulate', 'construct']
}

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Bloom's Taxonomy Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .model-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .bloom-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .confidence-text {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Input styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 1.1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Example questions */
    .example-btn {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-btn:hover {
        background: rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Model Loading Functions
# ============================================
@st.cache_resource
def load_fasttext_model():
    """Load the S-FastText model"""
    try:
        model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"Error loading FastText model: {e}")
        return None

@st.cache_resource
def load_bert_model():
    """Load the S-BERT model and tokenizer"""
    try:
        tokenizer = BertTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        model = BertForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None, None

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# ============================================
# Prediction Functions
# ============================================
def predict_fasttext(model, text):
    """Get prediction from FastText model"""
    clean = clean_text(text)
    prediction = model.predict(clean, k=len(BLOOM_LEVELS))
    
    labels = [p.replace('__label__', '') for p in prediction[0]]
    scores = list(prediction[1])
    
    # Create probability dict
    probs = {label: score for label, score in zip(labels, scores)}
    
    # Ensure all levels are present
    for level in BLOOM_LEVELS:
        if level not in probs:
            probs[level] = 0.0
    
    return probs

def predict_bert(model, tokenizer, device, text):
    """Get prediction from BERT model"""
    clean = clean_text(text)
    
    inputs = tokenizer.encode_plus(
        clean,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    return {level: float(prob) for level, prob in zip(BLOOM_LEVELS, probs)}

# ============================================
# Visualization Functions
# ============================================
def create_gauge_chart(confidence, label, color):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,255,255,0.05)'},
                {'range': [50, 75], 'color': 'rgba(255,255,255,0.1)'},
                {'range': [75, 100], 'color': 'rgba(255,255,255,0.15)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=200,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

def create_probability_chart(probs, title):
    """Create a horizontal bar chart for all probabilities"""
    sorted_levels = sorted(BLOOM_LEVELS, key=lambda x: probs.get(x, 0), reverse=True)
    
    fig = go.Figure(go.Bar(
        x=[probs.get(level, 0) * 100 for level in sorted_levels],
        y=sorted_levels,
        orientation='h',
        marker=dict(
            color=[BLOOM_COLORS[level] for level in sorted_levels],
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f'{probs.get(level, 0)*100:.1f}%' for level in sorted_levels],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            title='Confidence (%)',
            range=[0, 100],
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=False
        ),
        height=300,
        margin=dict(l=100, r=20, t=50, b=40)
    )
    
    return fig

def create_comparison_chart(ft_probs, bert_probs):
    """Create a comparison radar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[ft_probs.get(level, 0) * 100 for level in BLOOM_LEVELS] + [ft_probs.get(BLOOM_LEVELS[0], 0) * 100],
        theta=BLOOM_LEVELS + [BLOOM_LEVELS[0]],
        fill='toself',
        name='S-FastText',
        line=dict(color='#FF6B6B'),
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[bert_probs.get(level, 0) * 100 for level in BLOOM_LEVELS] + [bert_probs.get(BLOOM_LEVELS[0], 0) * 100],
        theta=BLOOM_LEVELS + [BLOOM_LEVELS[0]],
        fill='toself',
        name='S-BERT',
        line=dict(color='#4ECDC4'),
        fillcolor='rgba(78, 205, 196, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white', size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            x=0.5, y=-0.1,
            xanchor='center',
            orientation='h',
            font=dict(color='white')
        ),
        height=400,
        margin=dict(l=60, r=60, t=40, b=60)
    )
    
    return fig

# ============================================
# Main App
# ============================================
def main():
    # Header
    st.markdown('<h1 class="main-header">Bloom\'s Taxonomy Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classify educational questions using AI-powered S-BERT and S-FastText models</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner('Loading models...'):
        ft_model = load_fasttext_model()
        bert_model, tokenizer, device = load_bert_model()
    
    if ft_model is None or bert_model is None:
        st.error("Failed to load models. Please check if the model files exist.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Bloom's Taxonomy Levels")
        for level in BLOOM_LEVELS:
            color = BLOOM_COLORS[level]
            st.markdown(f"""
            <div style="background: {color}20; border-left: 4px solid {color}; padding: 0.5rem 1rem; margin: 0.5rem 0; border-radius: 0 10px 10px 0;">
                <strong style="color: {color};">{level}</strong><br/>
                <small style="color: #a0a0a0;">{BLOOM_DESCRIPTIONS[level]}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info(f"Device: **{device}**")
        st.success("S-FastText loaded")
        st.success("S-BERT loaded")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Your Question")
        question = st.text_area(
            "Type or paste an educational question:",
            placeholder="e.g., What is the capital of France?",
            height=120,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### Example Questions")
        examples = [
            ("Remember", "What is the definition of photosynthesis?"),
            ("Understand", "Explain how the water cycle works."),
            ("Apply", "Calculate the area of a circle with radius 5."),
            ("Analyze", "Compare and contrast mitosis and meiosis."),
            ("Evaluate", "Do you think renewable energy is better?"),
            ("Create", "Design an experiment to test plant growth.")
        ]
        
        for level, example in examples:
            if st.button(f"{level}", key=f"ex_{level}", help=example):
                question = example
                st.session_state['question'] = example
    
    # Check session state for example questions
    if 'question' in st.session_state and question == "":
        question = st.session_state['question']
    
    # Analyze button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        analyze_btn = st.button("Analyze Question", use_container_width=True)
    
    # Results
    if analyze_btn and question.strip():
        with st.spinner('Analyzing...'):
            ft_probs = predict_fasttext(ft_model, question)
            bert_probs = predict_bert(bert_model, tokenizer, device, question)
        
        ft_pred = max(ft_probs, key=ft_probs.get)
        bert_pred = max(bert_probs, key=bert_probs.get)
        
        st.markdown("---")
        st.markdown("## Results")
        
        # Summary cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="model-title">S-FastText</div>
                <div class="bloom-badge" style="background: {BLOOM_COLORS[ft_pred]};">{ft_pred}</div>
                <div class="confidence-text" style="color: {BLOOM_COLORS[ft_pred]};">
                    {ft_probs[ft_pred]*100:.1f}%
                </div>
                <p style="color: #a0a0a0; margin-top: 0.5rem;">{BLOOM_DESCRIPTIONS[ft_pred]}</p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_probability_chart(ft_probs, "S-FastText Probabilities"), use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="model-title">S-BERT</div>
                <div class="bloom-badge" style="background: {BLOOM_COLORS[bert_pred]};">{bert_pred}</div>
                <div class="confidence-text" style="color: {BLOOM_COLORS[bert_pred]};">
                    {bert_probs[bert_pred]*100:.1f}%
                </div>
                <p style="color: #a0a0a0; margin-top: 0.5rem;">{BLOOM_DESCRIPTIONS[bert_pred]}</p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_probability_chart(bert_probs, "S-BERT Probabilities"), use_container_width=True)
        
        # Model comparison
        st.markdown("### Model Comparison")
        if ft_pred == bert_pred:
            st.success(f"Both models agree! The question is classified as **{ft_pred}**")
        else:
            st.warning(f"Models disagree: S-FastText predicts **{ft_pred}**, S-BERT predicts **{bert_pred}**")
        
        st.plotly_chart(create_comparison_chart(ft_probs, bert_probs), use_container_width=True)
        
        # Detailed analysis
        with st.expander("Detailed Analysis"):
            st.markdown("#### Probability Breakdown")
            data = {
                'Bloom Level': BLOOM_LEVELS,
                'S-FastText (%)': [f"{ft_probs.get(l, 0)*100:.2f}" for l in BLOOM_LEVELS],
                'S-BERT (%)': [f"{bert_probs.get(l, 0)*100:.2f}" for l in BLOOM_LEVELS]
            }
            st.table(data)
            
            st.markdown("#### Common Action Verbs")
            for pred in set([ft_pred, bert_pred]):
                st.markdown(f"**{pred}:** {', '.join(BLOOM_VERBS.get(pred, []))}")
    
    elif analyze_btn:
        st.warning("Please enter a question to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with Streamlit | Models: S-BERT & S-FastText</p>
        <p><small>Based on "Semantic-BERT and semantic-FastText models for education question classification" research</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
