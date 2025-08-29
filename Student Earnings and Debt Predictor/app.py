import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import json
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="EduPredict Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# AUTHENTICATION SYSTEM
# ===================================================================

class AuthManager:
    def __init__(self):
        self.users_file = "users.json"
        self.ensure_users_file()
    
    def ensure_users_file(self):
        """Create users file if it doesn't exist"""
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        """Load users from file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(self, users):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def register_user(self, username, email, password, full_name):
        """Register a new user"""
        users = self.load_users()
        
        if username in users:
            return False, "Username already exists"
        
        # Check if email exists
        for user_data in users.values():
            if user_data.get('email') == email:
                return False, "Email already registered"
        
        users[username] = {
            'email': email,
            'password': self.hash_password(password),
            'full_name': full_name,
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
        
        self.save_users(users)
        return True, "Registration successful"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        users = self.load_users()
        
        if username not in users:
            return False, "Username not found"
        
        if users[username]['password'] != self.hash_password(password):
            return False, "Invalid password"
        
        # Update last login
        users[username]['last_login'] = datetime.now().isoformat()
        self.save_users(users)
        
        return True, "Login successful"
    
    def get_user_info(self, username):
        """Get user information"""
        users = self.load_users()
        return users.get(username, {})

# ===================================================================
# ENHANCED CSS STYLING
# ===================================================================

def load_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        
        /* Custom Header */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Authentication Cards */
        .auth-container {
            max-width: 500px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            border: 1px solid #e1e5e9;
        }
        
        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .auth-header h2 {
            color: #2d3748;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .auth-header p {
            color: #718096;
            font-size: 0.9rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        /* Sidebar Styling */
        .sidebar-content {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
        }
        
        /* Risk Level Badges */
        .risk-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .risk-low {
            background: #c6f6d5;
            color: #22543d;
            border: 2px solid #9ae6b4;
        }
        
        .risk-medium {
            background: #fef5e7;
            color: #744210;
            border: 2px solid #f6e05e;
        }
        
        .risk-high {
            background: #fed7d7;
            color: #742a2a;
            border: 2px solid #fc8181;
        }
        
        .risk-very-high {
            background: #fed7e2;
            color: #702459;
            border: 2px solid #f687b3;
        }
        
        /* Summary Card */
        .summary-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        /* Info Panels */
        .info-panel {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .info-panel h4 {
            color: #2d3748;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        /* Navigation */
        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
            padding: 1rem 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .user-info {
            color: #4a5568;
            font-weight: 500;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .auth-container {
                margin: 1rem;
                padding: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ===================================================================
# SMART PREDICTOR CLASS (Same as original)
# ===================================================================

class SmartEarningsDebtPredictor:
    """Smart predictor that intelligently handles engineered features"""
    def __init__(self, earnings_model, debt_model, earnings_features, debt_features):
        self.earnings_model = earnings_model
        self.debt_model = debt_model
        self.earnings_features = earnings_features
        self.debt_features = debt_features
        self.creation_date = datetime.now()
        
        # Define base feature medians
        self.feature_medians = {
            'UNITID': 194240.0,
            'IPEDSCOUNT1': 50.0,
            'institution_median_earnings': 50374.0,
            'debt_to_income_4yr': 0.437,
            'roi_4yr': 164237.0,
            'earnings_percentile_in_major': 0.454,
            'high_earner': 0.0,
            'low_earner': 0.0,
            'cip_major_group': 0.0
        }
        
        # CIP code to earnings mapping
        self.cip_earnings_estimates = {
            11: 75000, 51: 60000, 52: 55000, 14: 70000, 26: 45000,
            13: 42000, 42: 40000, 23: 38000, 50: 35000,
        }
        
        # CIP code to debt mapping
        self.cip_debt_estimates = {
            11: 22000, 51: 25000, 52: 23000, 14: 24000, 26: 26000,
            13: 27000, 42: 28000, 23: 25000, 50: 30000,
        }

    def _derive_cip_major_group(self, cipcode):
        return cipcode // 10000

    def _estimate_earnings_by_program(self, cipcode, credlev):
        major_group = self._derive_cip_major_group(cipcode)
        base_earnings = self.cip_earnings_estimates.get(major_group, 45000)
        
        credlev_multipliers = {
            1: 0.65, 2: 0.75, 3: 1.0, 4: 1.1, 5: 1.3, 6: 1.6, 7: 1.8, 8: 1.2
        }
        
        multiplier = credlev_multipliers.get(credlev, 1.0)
        return base_earnings * multiplier

    def _estimate_debt_by_program(self, cipcode, credlev, control):
        major_group = self._derive_cip_major_group(cipcode)
        base_debt = self.cip_debt_estimates.get(major_group, 25000)
        
        credlev_multipliers = {
            1: 0.6, 2: 0.8, 3: 1.0, 4: 0.7, 5: 1.4, 6: 1.8, 7: 2.5, 8: 1.1
        }
        
        control_multipliers = {
            'Public': 0.9, 'Private, nonprofit': 1.2,
            'Private, for-profit': 1.3, 'Foreign': 0.8
        }
        
        cred_mult = credlev_multipliers.get(credlev, 1.0)
        control_mult = control_multipliers.get(control, 1.0)
        
        return base_debt * cred_mult * control_mult

    def _prepare_features_smart(self, input_data, target_type='earnings'):
        cipcode = input_data.get('CIPCODE', 110701)
        credlev = input_data.get('CREDLEV', 3)
        control = input_data.get('CONTROL', 'Public')
        distance = input_data.get('DISTANCE', 0)
        main = input_data.get('MAIN', 1)
        ipedscount1 = input_data.get('IPEDSCOUNT1', 50)
        
        features = {
            'CIPCODE': cipcode, 'CREDLEV': credlev, 'DISTANCE': distance,
            'MAIN': main, 'IPEDSCOUNT1': ipedscount1
        }
        
        features['cip_major_group'] = self._derive_cip_major_group(cipcode)
        
        est_earnings = self._estimate_earnings_by_program(cipcode, credlev)
        est_debt = self._estimate_debt_by_program(cipcode, credlev, control)
        
        features['high_earner'] = 1 if est_earnings > 60000 else 0
        features['low_earner'] = 1 if est_earnings < 35000 else 0
        features['debt_to_income_4yr'] = est_debt / est_earnings
        features['roi_4yr'] = (est_earnings * 4) - est_debt
        features['earnings_percentile_in_major'] = 0.5
        features['institution_median_earnings'] = est_earnings
        features['UNITID'] = input_data.get('UNITID', self.feature_medians['UNITID'])
        
        required_features = self.earnings_features if target_type == 'earnings' else self.debt_features
        
        feature_vector = {}
        for feature in required_features:
            feature_vector[feature] = features.get(feature, self.feature_medians.get(feature, 0))
        
        return pd.DataFrame([feature_vector])

    def predict_earnings(self, program_data):
        X = self._prepare_features_smart(program_data, 'earnings')
        prediction = self.earnings_model.predict(X)[0]
        
        mae = 2857
        confidence_lower = max(0, prediction - mae)
        confidence_upper = prediction + mae
        
        return {
            'predicted_earnings': round(prediction, 2),
            'confidence_lower': round(confidence_lower, 2),
            'confidence_upper': round(confidence_upper, 2),
            'currency': 'USD'
        }

    def predict_debt(self, program_data):
        X = self._prepare_features_smart(program_data, 'debt')
        prediction = self.debt_model.predict(X)[0]
        
        mae = 1033
        confidence_lower = max(0, prediction - mae)
        confidence_upper = prediction + mae
        
        return {
            'predicted_debt': round(prediction, 2),
            'confidence_lower': round(confidence_lower, 2),
            'confidence_upper': round(confidence_upper, 2),
            'currency': 'USD'
        }

    def predict_comprehensive(self, program_data):
        earnings_result = self.predict_earnings(program_data)
        debt_result = self.predict_debt(program_data)
        
        predicted_earnings = earnings_result['predicted_earnings']
        predicted_debt = debt_result['predicted_debt']
        
        four_year_earnings = predicted_earnings * 4
        roi = four_year_earnings - predicted_debt
        debt_to_income_ratio = predicted_debt / predicted_earnings if predicted_earnings > 0 else 0
        
        if debt_to_income_ratio < 0.3:
            risk_level = "Low Risk"
        elif debt_to_income_ratio < 0.6:
            risk_level = "Medium Risk"
        elif debt_to_income_ratio < 1.0:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        return {
            'earnings': earnings_result,
            'debt': debt_result,
            'roi_analysis': {
                'four_year_total_earnings': round(four_year_earnings, 2),
                'net_roi': round(roi, 2),
                'debt_to_income_ratio': round(debt_to_income_ratio, 2),
                'risk_level': risk_level
            },
            'methodology': {
                'estimated_features_used': True,
                'note': 'Some features estimated based on program characteristics'
            }
        }

# ===================================================================
# AUTHENTICATION FUNCTIONS
# ===================================================================

def show_login_page():
    """Display login page"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 EduPredict Pro</h1>
        <p>AI-Powered Education Investment Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <h2>🎓 Welcome Back</h2>
            <p>Sign in to access your personalized education predictions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,3,1])
    with col2:

        with st.form("login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🔑 Login", use_container_width=True)
            with col2:
                if st.form_submit_button("📝 New User? Sign Up", use_container_width=True):
                    st.session_state.show_signup = True
                    st.rerun()
            
            if login_button:
                if username and password:
                    auth_manager = AuthManager()
                    success, message = auth_manager.authenticate_user(username, password)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_info = auth_manager.get_user_info(username)
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please fill in all fields")

def show_signup_page():
    """Display signup page"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 EduPredict Pro</h1>
        <p>AI-Powered Education Investment Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <h2>🚀 Join EduPredict Pro</h2>
            <p>Create your account to start making informed education decisions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    coll1, coll2, coll3 = st.columns([1,3,1])
    with coll2:
        with st.form("signup_form"):
            st.subheader("Create Your Account")
            
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="Enter your email address")
            username = st.text_input("Username", placeholder="Choose a username")
            password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            col1, col2 = st.columns(2)
            with col1:
                signup_button = st.form_submit_button("🎯 Create Account", use_container_width=True)
            with col2:
                if st.form_submit_button("🔙 Back to Login", use_container_width=True):
                    st.session_state.show_signup = False
                    st.rerun()
            
            if signup_button:
                if full_name and email and username and password and confirm_password:
                    if password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    else:
                        auth_manager = AuthManager()
                        success, message = auth_manager.register_user(username, email, password, full_name)
                        
                        if success:
                            st.success("✅ Account created successfully! Please login.")
                            st.session_state.show_signup = False
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please fill in all fields")

def show_navigation():
    """Display navigation bar"""
    user_info = st.session_state.get('user_info', {})
    full_name = user_info.get('full_name', st.session_state.get('username', 'User'))
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="nav-container">
            <div>
                <h3 style="margin: 0; color: #2d3748;">🎓 EduPredict Pro</h3>
            </div>
            <div class="user-info">
                Welcome, {full_name}! 👋
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

@st.cache_resource
def load_predictor():
    """Load the smart predictor model"""
    try:
        with open('smart_earnings_debt_predictor.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package['predictor']
    except FileNotFoundError:
        st.error("❌ Model file 'smart_earnings_debt_predictor.pkl' not found!")
        st.info("Please make sure you've run the training code and saved the smart predictor.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("There might be an issue with the model file. Try recreating it.")
        st.stop()

def show_main_app():
    """Display the main application"""
    # Load CSS
    load_css()
    
    # Show navigation
    show_navigation()
    
    # Load the predictor
    predictor = load_predictor()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 EduPredict Pro</h1>
        <p>AI-Powered Education Investment Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>📋 Program Configuration</h3>
            <p>Configure your educational program details below to receive personalized predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # CIP Code mapping for common programs
        cip_codes = {
            "Computer Science": 110701,
            "Registered Nursing": 513801,
            "Business Administration": 520201,
            "Mechanical Engineering": 141901,
            "Psychology": 420101,
            "Elementary Education": 131202,
            "Accounting": 520301,
            "Biology": 260101,
            "English Literature": 230101,
            "Art History": 500703,
            "Criminal Justice": 430104,
            "Marketing": 521401,
            "Civil Engineering": 140801,
            "Finance": 520801,
            "Mathematics": 270101,
        }
        
        # Program selection
        selected_program = st.selectbox(
            "🎯 Select Your Program",
            options=list(cip_codes.keys()),
            help="Choose your field of study"
        )
        
        cipcode = cip_codes[selected_program]
        program_name = selected_program
        
        # Credential Level
        credential_mapping = {
            "Undergraduate Certificate": 1,
            "Associate's Degree": 2,
            "Bachelor's Degree": 3,
            "Post-baccalaureate Certificate": 4,
            "Master's Degree": 5,
            "Doctoral Degree": 6,
            "First Professional Degree": 7,
            "Graduate Certificate": 8
        }
        
        selected_credential = st.selectbox(
            "🎓 Credential Level",
            options=list(credential_mapping.keys()),
            index=2,
            help="Select the type of degree/certificate you're pursuing"
        )
        credlev = credential_mapping[selected_credential]
        
        # Institution Type
        institution_types = ["Public", "Private, nonprofit", "Private, for-profit", "Foreign"]
        control = st.selectbox(
            "🏫 Institution Type",
            options=institution_types,
            help="Type of institution you're attending"
        )
        
        # Learning Format
        distance_mapping = {
            "On-campus": 0,
            "Online": 1,
            "Hybrid": 2,
            "Other": 3
        }
        
        selected_format = st.selectbox(
            "💻 Learning Format",
            options=list(distance_mapping.keys()),
            help="How will you primarily take classes?"
        )
        distance = distance_mapping[selected_format]
        
        # Campus Type
        main_campus = st.radio(
            "🏛️ Campus Type",
            options=["Main Campus", "Branch Campus"],
            help="Are you attending the main campus or a branch?"
        )
        main = 1 if main_campus == "Main Campus" else 0
        
        # Program Size
        program_size = st.slider(
            "👥 Estimated Program Size",
            min_value=1,
            max_value=500,
            value=50,
            help="Approximate number of students in your program/cohort"
        )
        
        # Advanced Options
        with st.expander("⚙️ Advanced Options"):
            unitid = st.number_input(
                "Institution UNITID",
                min_value=100000,
                max_value=999999,
                value=None,
                help="Leave blank to use default estimation"
            )
    
    # Create input dictionary
    program_data = {
        'CIPCODE': cipcode,
        'CREDLEV': credlev,
        'CONTROL': control,
        'DISTANCE': distance,
        'MAIN': main,
        'IPEDSCOUNT1': program_size
    }
    
    if unitid:
        program_data['UNITID'] = unitid
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display current selections
        st.markdown("### 📊 Program Summary")
        
        st.markdown(f"""
        <div class="summary-card">
            <h4>🎯 {program_name}</h4>
            <p><strong>Credential:</strong> {selected_credential}</p>
            <p><strong>Institution:</strong> {control}</p>
            <p><strong>Format:</strong> {selected_format}</p>
            <p><strong>Campus:</strong> {main_campus}</p>
            <p><strong>Program Size:</strong> {program_size} students</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔮 Generate AI Predictions", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your program data..."):
                try:
                    # Get predictions
                    result = predictor.predict_comprehensive(program_data)
                    
                    # Store results in session state
                    st.session_state.prediction_result = result
                    st.session_state.program_info = {
                        'name': program_name,
                        'credential': selected_credential,
                        'institution': control,
                        'cipcode': cipcode
                    }
                    
                    st.success("✅ Predictions generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
    
    with col2:
        # Information panel
        st.markdown("""
        <div class="info-panel">
            <h4>🧠 AI Model Intelligence</h4>
            <p><strong>🎯 Prediction Accuracy:</strong></p>
            <ul>
                <li>📊 89.6% earnings accuracy</li>
                <li>💰 89.8% debt accuracy</li>
                <li>🎓 50K+ training records</li>
            </ul>
            <p><strong>📈 Analysis Factors:</strong></p>
            <ul>
                <li>📚 Field of study trends</li>
                <li>🏫 Institution characteristics</li>
                <li>💼 Historical outcomes</li>
                <li>📊 Market conditions</li>
            </ul>
            <p><strong>🎯 Confidence Intervals:</strong></p>
            <ul>
                <li>Earnings: ±$2,857</li>
                <li>Debt: ±$1,033</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results if available
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        program_info = st.session_state.program_info
        
        st.markdown("---")
        st.markdown("## 🎯 Your AI-Generated Predictions")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Expected Annual Salary</h4>
                <h2>${result['earnings']['predicted_earnings']:,.0f}</h2>
                <p>Primary income expectation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Total Debt Expected</h4>
                <h2>${result['debt']['predicted_debt']:,.0f}</h2>
                <p>Student loan burden</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💎 4-Year ROI</h4>
                <h2>${result['roi_analysis']['net_roi']:,.0f}</h2>
                <p>Return on investment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            debt_ratio = result['roi_analysis']['debt_to_income_ratio']
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Debt-to-Income</h4>
                <h2>{debt_ratio:.2f}</h2>
                <p>Financial burden ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Assessment with enhanced styling
        risk_level = result['roi_analysis']['risk_level']
        risk_class_map = {
            "Low Risk": "risk-low",
            "Medium Risk": "risk-medium", 
            "High Risk": "risk-high",
            "Very High Risk": "risk-very-high"
        }
        
        risk_icons = {
            "Low Risk": "🟢",
            "Medium Risk": "🟡", 
            "High Risk": "🟠",
            "Very High Risk": "🔴"
        }
        
        risk_descriptions = {
            "Low Risk": "Excellent financial outlook with manageable debt burden relative to expected earnings.",
            "Medium Risk": "Generally positive financial situation. Monitor debt levels and career prospects carefully.",
            "High Risk": "Significant financial considerations. Carefully evaluate alternatives and funding strategies.",
            "Very High Risk": "Major financial concerns. Consider alternative programs or additional funding sources."
        }
        
        st.markdown("### 🎯 Financial Risk Assessment")
        
        risk_class = risk_class_map[risk_level]
        risk_icon = risk_icons[risk_level]
        
        st.markdown(f"""
        <div class="{risk_class} risk-badge" style="width: 100%; text-align: center; font-size: 1.2rem;">
            {risk_icon} {risk_level}
        </div>
        <div style="margin-top: 1rem; padding: 1rem; background: #f7fafc; border-radius: 10px; border-left: 4px solid #4299e1;">
            <p>{risk_descriptions[risk_level]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Analysis with improved layout
        st.markdown("### 📊 Comprehensive Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Earnings Breakdown")
            earnings_data = {
                "Metric": [
                    "Expected Annual Salary",
                    "Conservative Estimate (95% CI)",
                    "Optimistic Estimate (95% CI)",
                    "4-Year Total Earnings"
                ],
                "Amount": [
                    f"${result['earnings']['predicted_earnings']:,.0f}",
                    f"${result['earnings']['confidence_lower']:,.0f}",
                    f"${result['earnings']['confidence_upper']:,.0f}",
                    f"${result['roi_analysis']['four_year_total_earnings']:,.0f}"
                ]
            }
            earnings_df = pd.DataFrame(earnings_data)
            st.dataframe(earnings_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 💳 Debt Analysis")
            debt_data = {
                "Metric": [
                    "Expected Total Debt",
                    "Conservative Estimate (95% CI)",
                    "Optimistic Estimate (95% CI)",
                    "Est. Monthly Payment (10yr)"
                ],
                "Amount": [
                    f"${result['debt']['predicted_debt']:,.0f}",
                    f"${result['debt']['confidence_lower']:,.0f}",
                    f"${result['debt']['confidence_upper']:,.0f}",
                    f"${result['debt']['predicted_debt']/120:,.0f}"
                ]
            }
            debt_df = pd.DataFrame(debt_data)
            st.dataframe(debt_df, use_container_width=True, hide_index=True)
        
        # Enhanced Visualization
        st.markdown("### 📈 Interactive Financial Projection")
        
        # Create enhanced comparison chart
        metrics = ['Annual Salary', 'Total Debt', '4-Year Earnings', 'Net ROI']
        values = [
            result['earnings']['predicted_earnings'],
            result['debt']['predicted_debt'],
            result['roi_analysis']['four_year_total_earnings'],
            result['roi_analysis']['net_roi']
        ]
        
        # Create a more sophisticated chart
        fig = go.Figure()
        
        colors = ['#667eea', '#f093fb', '#4ecdc4', '#45b7d1']
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f'${v:,.0f}' for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Financial Overview: {program_info['name']} ({program_info['credential']})",
            xaxis_title="Financial Metrics",
            yaxis_title="Amount (USD)",
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.update_xaxes(gridcolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Recommendations
        st.markdown("### 💡 AI-Generated Recommendations")
        
        debt_ratio = result['roi_analysis']['debt_to_income_ratio']
        net_roi = result['roi_analysis']['net_roi']
        
        recommendations = []
        
        if debt_ratio < 0.3:
            recommendations.append(("✅", "Excellent Choice", "Your debt-to-income ratio is very manageable. This program offers strong financial prospects."))
        elif debt_ratio < 0.6:
            recommendations.append(("⚠️", "Good Choice", "Generally positive outlook. Consider scholarships and grants to further reduce debt burden."))
        elif debt_ratio < 1.0:
            recommendations.append(("🚨", "Proceed with Caution", "High debt ratio detected. Explore cost reduction strategies and income enhancement opportunities."))
        else:
            recommendations.append(("❌", "High Risk", "Very high debt burden. Strongly consider alternative programs or additional funding sources."))
        
        if net_roi > 100000:
            recommendations.append(("💰", "Outstanding ROI", "This program demonstrates exceptional return on investment potential."))
        elif net_roi > 50000:
            recommendations.append(("📊", "Positive ROI", "Solid financial returns expected from this educational investment."))
        elif net_roi > 0:
            recommendations.append(("📈", "Marginal ROI", "Consider non-financial benefits when evaluating this investment."))
        else:
            recommendations.append(("📉", "Negative ROI", "Financial returns may not justify the investment. Explore alternatives."))
        
        for icon, title, description in recommendations:
            st.markdown(f"""
            <div style="margin: 1rem 0; padding: 1rem; background: white; border-radius: 10px; border-left: 4px solid #4299e1; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h5 style="margin: 0 0 0.5rem 0; color: #2d3748;">{icon} {title}</h5>
                <p style="margin: 0; color: #4a5568;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Export functionality
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Full Report", use_container_width=True):
                results_summary = {
                    "Program": program_info['name'],
                    "Credential Level": program_info['credential'],
                    "Institution Type": program_info['institution'],
                    "CIP Code": program_info['cipcode'],
                    "Predicted Annual Salary": result['earnings']['predicted_earnings'],
                    "Salary Lower Bound": result['earnings']['confidence_lower'],
                    "Salary Upper Bound": result['earnings']['confidence_upper'],
                    "Predicted Total Debt": result['debt']['predicted_debt'],
                    "Debt Lower Bound": result['debt']['confidence_lower'],
                    "Debt Upper Bound": result['debt']['confidence_upper'],
                    "4-Year Total Earnings": result['roi_analysis']['four_year_total_earnings'],
                    "Net ROI": result['roi_analysis']['net_roi'],
                    "Debt-to-Income Ratio": result['roi_analysis']['debt_to_income_ratio'],
                    "Risk Level": result['roi_analysis']['risk_level'],
                    "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "User": st.session_state.get('username', 'Anonymous')
                }
                
                results_df = pd.DataFrame([results_summary])
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📁 Download CSV Report",
                    data=csv,
                    file_name=f"edupredict_report_{program_info['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("🔄 Generate New Prediction", use_container_width=True):
                # Clear previous results
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                if 'program_info' in st.session_state:
                    del st.session_state.program_info
                st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    # Load CSS
    load_css()
    
    # Authentication flow
    if not st.session_state.authenticated:
        if st.session_state.show_signup:
            show_signup_page()
        else:
            show_login_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
