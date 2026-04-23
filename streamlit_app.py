import streamlit as st
import pandas as pd
import joblib
import yaml
import os

# --- 1. Load Config and Models ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@st.cache_resource
def load_models():
    model = joblib.load(config['paths']['model_path'])
    fa_model = joblib.load(config['paths']['factor_analysis_model_path'])
    encoders = joblib.load(config['paths']['encoder_path'])
    return model, fa_model, encoders

model, fa_model, encoders = load_models()

# --- 2. Streamlit UI Layout ---
st.title("🍎 M1 Mac Purchase Predictor")
st.write("Predict customer purchase behavior based on feedback and demographics.")

with st.form("prediction_form"):
    st.header("Customer Information")
    col1, col2 = st.columns(2)
    
    with col1:
        trust_apple = st.radio("Do you trust Apple?", ["Yes", "No"])
        user_pcmac = st.radio("Current Computer Type", ["Apple", "PC"])
        familiarity_m1 = st.radio("Familiar with M1 chip?", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        status = st.selectbox("Employment Status", ["Student", "Employed", "Self-Employed", "Retired", "Unemployed"])
        
    with col2:
        interest_computers = st.slider("Interest in Computers", 1, 5, 3)
        age_computer = st.slider("Age of current computer (years)", 0, 10, 2)
        appleproducts_count = st.slider("Apple products owned", 0, 10, 1)
        age_group = st.slider("Age Group (encoded)", 1, 10, 3)
        income_group = st.slider("Income Group (encoded)", 1, 7, 3)

    st.header("Feature Importance (1-5)")
    f_cols = st.columns(3)
    f_batterylife = f_cols[0].slider("Battery Life", 1, 5, 3)
    f_price = f_cols[1].slider("Price", 1, 5, 3)
    f_size = f_cols[2].slider("Size/Portability", 1, 5, 3)
    f_multitasking = f_cols[0].slider("Multitasking", 1, 5, 3)
    f_noise = f_cols[1].slider("Quietness", 1, 5, 3)
    f_performance = f_cols[2].slider("Raw Performance", 1, 5, 3)
    f_neural = f_cols[0].slider("Neural/AI Tasks", 1, 5, 3)
    f_synergy = f_cols[1].slider("Ecosystem Synergy", 1, 5, 3)
    f_performanceloss = f_cols[2].slider("Performance on Battery", 1, 5, 3)

    domain = st.selectbox("Primary Domain", ["IT & Technology", "Business", "Engineering", "Arts & Culture", "Education", "Healthcare", "Other"])

    submit = st.form_submit_button("Predict Purchase Likelihood")

# --- 3. Prediction Logic ---
if submit:
    try:
        if os.path.exists(config['paths']['selected_features_output']):
            with open(config['paths']['selected_features_output'], "r") as f:
                SELECTED_FEATURES = [line.strip() for line in f]
        else:
            SELECTED_FEATURES = model.feature_names_in_.tolist()
        # Create DataFrame
        input_data = {
            "trust_apple": trust_apple, "interest_computers": interest_computers,
            "age_computer": age_computer, "user_pcmac": user_pcmac,
            "appleproducts_count": appleproducts_count, "familiarity_m1": familiarity_m1,
            "f_batterylife": f_batterylife, "f_price": f_price, "f_size": f_size,
            "f_multitasking": f_multitasking, "f_noise": f_noise,
            "f_performance": f_performance, "f_neural": f_neural,
            "f_synergy": f_synergy, "f_performanceloss": f_performanceloss,
            "gender": gender, "age_group": age_group, "income_group": income_group,
            "status": status, "domain": domain
        }
        df = pd.DataFrame([input_data])

        # Inject missing cols for encoder
        df['m1_purchase'] = 0
        df['m1_consideration'] = "No"

        # Manual Cleaning
        df['trust_apple'] = df['trust_apple'].apply(lambda x: 1 if x == "Yes" else 0)
        df['user_pcmac'] = df['user_pcmac'].apply(lambda x: 1 if x == "Apple" else 0)
        df['familiarity_m1'] = df['familiarity_m1'].apply(lambda x: 1 if x == "Yes" else 0)
        df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else 0)

        # Encoding
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    df[col] = 0 
        fa_cols = config['dataset']['feature_cols']
        fa_cols = [col for col in fa_cols if col in SELECTED_FEATURES]
        factors = fa_model.transform(df[fa_cols])
        # Transform
        factor_df = pd.DataFrame(
            factors, 
            columns=[f'factor_{i+1}' for i in range(fa_model.n_components)], 
            index=df.index
        )

        df_dropped = df.drop(columns=fa_cols)
        df_final = pd.concat([df_dropped, factor_df], axis=1)
        df_final = df_final[model.feature_names_in_]

        # Predict
        prediction = model.predict(df_final)
        prob = model.predict_proba(df_final)[0][1]

        if prediction[0] == 1:
            st.success(f"Likely to Purchase! (Confidence: {prob:.2%})")
        else:
            st.error(f"Unlikely to Purchase. (Confidence: {1-prob:.2%})")

    except Exception as e:
        st.error(f"Error: {e}")