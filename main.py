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
    df = preprocessor.encoding_data(df)
    df = preprocessor.feature_selection(df)
    df_final = preprocessor.factor_analysis(df)

    # Save at the end
    df_final.to_csv(config['paths']['updated_data'], index=False)

    model_trainer = ModelTrainer(config)
    model = model_trainer.train_model(df_final)
    model_path = config['paths']['model_path']
    pd.to_pickle(model, model_path)
    logger.info(f"Model saved to {model_path}")