import os
import yaml
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config", "config.yaml")
config = yaml.load(open(config_path), Loader=yaml.FullLoader)

model = joblib.load(config['paths']['model_path'])
fa_model = joblib.load(config['paths']['factor_analysis_model_path'])
encoders = joblib.load(config['paths']['encoder_path']) 

# 2. Load Selected Features (Once at startup)
if os.path.exists(config['paths']['selected_features_output']):
    with open(config['paths']['selected_features_output'], "r") as f:
        SELECTED_FEATURES = [line.strip() for line in f]
else:
    SELECTED_FEATURES = model.feature_names_in_.tolist()

def predict_m1_purchase(*args):
    try:
        # Map Gradio inputs to a Dictionary
        input_keys = [
            "trust_apple", "interest_computers", "age_computer", "user_pcmac", 
            "appleproducts_count", "familiarity_m1", "f_batterylife", "f_price", 
            "f_size", "f_multitasking", "f_noise", "f_performance", "f_neural", 
            "f_synergy", "f_performanceloss", "gender", "age_group", 
            "income_group", "status", "domain"
        ]
        input_dict = dict(zip(input_keys, args))
        df = pd.DataFrame([input_dict])

        # Manual Cleaning (Must match your training CleanData logic)
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

        # Factor Analysis
        fa_cols = config['dataset']['feature_cols']
        fa_cols = [col for col in fa_cols if col in SELECTED_FEATURES]
        factors = fa_model.transform(df[fa_cols])

        factor_df = pd.DataFrame(
            factors, 
            columns=[f'factor_{i+1}' for i in range(fa_model.n_components)], 
            index=df.index
        )

        # Combine and Filter
        df_dropped = df.drop(columns=fa_cols)
        df_final = pd.concat([df_dropped, factor_df], axis=1)
        
        # Keep column order
        df_final = df_final[model.feature_names_in_]

        # Prediction
        prediction = model.predict(df_final)
        prob = model.predict_proba(df_final)[0][1]
        
        result = "Likely to Purchase" if prediction[0] == 1 else "Unlikely to Purchase"
        return f"{result} (Confidence: {prob:.2%})"

    except Exception as e:
        return f"Error: {str(e)}"

# 8. Gradio UI Definition
inputs = [
    gr.Radio(["Yes", "No"], label="Do you trust Apple?"),
    gr.Slider(1, 5, step=1, label="Interest in Computers"),
    gr.Slider(0, 10, step=1, label="Age of current computer (years)"),
    gr.Radio(["Apple", "PC"], label="Current Computer Type"),
    gr.Slider(0, 10, step=1, label="Number of Apple products owned"),
    gr.Radio(["Yes", "No"], label="Are you familiar with the M1 chip?"),
    gr.Slider(1, 5, step=1, label="Importance: Battery Life"),
    gr.Slider(1, 5, step=1, label="Importance: Price"),
    gr.Slider(1, 5, step=1, label="Importance: Size/Portability"),
    gr.Slider(1, 5, step=1, label="Importance: Multitasking"),
    gr.Slider(1, 5, step=1, label="Importance: Quietness (Noise)"),
    gr.Slider(1, 5, step=1, label="Importance: Raw Performance"),
    gr.Slider(1, 5, step=1, label="Importance: Neural/AI Tasks"),
    gr.Slider(1, 5, step=1, label="Importance: Ecosystem Synergy"),
    gr.Slider(1, 5, step=1, label="Importance: No Performance Loss on Battery"),
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Slider(1, 10, step=1, label="Age Group"),
    gr.Slider(1, 7, step=1, label="Income Group"),
    gr.Dropdown(["Student", "Employed", "Self-Employed", "Retired", "Student and employed", "Unemployed"], label="Employment Status"),
    gr.Dropdown([
        "IT & Technology", "Marketing", "Business", "Engineering", "Finance", 
        "Science", "Arts & Culture", "Social Sciences", "Education", 
        "Hospitality", "Politics", "Administration & Public Services", 
        "Healthcare", "Economics", "Retired", "Law", "Agriculture", 
        "Communication", "Realestate", "Logistics", "Consulting", "Retail"
    ], label="Primary Domain/Usage")
]

demo = gr.Interface(
    fn=predict_m1_purchase,
    inputs=inputs,
    outputs="text",
    title="M1 Mac Purchase Predictor",
    description="Predict customer purchase behavior based on feedback and demographics."
)

if __name__ == "__main__":
    demo.launch()