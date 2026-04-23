import joblib
import pandas as pd
import yaml
from src.clean_data import DataPreprocessor
import logging
from src.model_training import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("M1_Feedback_Analysis_and_Sales_Prediction")
config = yaml.load(open("config/config.yaml"), Loader=yaml.FullLoader)

def load_data():
    try:
        data = pd.read_csv(config['paths']['raw_data'])
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    preprocessor = DataPreprocessor(config)

    df = load_data()
    df = preprocessor.clean_data(df)
    df, encoders_dict = preprocessor.encoding_data(df)
    encoder_path = config['paths']['encoder_path']
    joblib.dump(encoders_dict, 'models/encoder.pkl')
    logger.info(f"Encoder saved to {encoder_path}")

    df, selected_features = preprocessor.feature_selection(df)
    selected_features_path = config['paths'].get('selected_features_output', 'selected_features.txt')
    with open(selected_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    df_final, fa = preprocessor.factor_analysis(df)
    model_path = config['paths']['factor_analysis_model_path']
    joblib.dump(fa, model_path)
    logger.info(f"Factor analysis model saved to {model_path}")
    # Save at the end
    df_final.to_csv(config['paths']['updated_data'], index=False)

    model_trainer = ModelTrainer(config)
    model = model_trainer.train_model(df_final)
    model_path = config['paths']['model_path']
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")