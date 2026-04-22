import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight

logging = logging.getLogger("M1_Feedback_Analysis_and_Sales_Prediction")

class ModelTrainer:
    def __init__(self, config):
        self.config = config    

    def train_model(self, df):
        logging.info("Training model...")

        x = df.drop(columns=self.config['dataset']['cols_to_drop'])
        y = df[self.config['dataset']['target_col']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Calculate class weights
        weights = compute_sample_weight('balanced', y_train)
        # Train the model with class weights
        model=GaussianNB(var_smoothing = 0.0003511191734215131)
        model.fit(x_train, y_train, sample_weight=weights)

        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased'])
        report_path = self.config['paths'].get('report_output', 'classification_report.txt')
        
        with open(report_path, "w") as f:
            f.write("Classification Report for M1 Purchase Prediction\n")
            f.write("="*50 + "\n")
            f.write(report)
        
        logging.info(f"Classification report saved to {report_path}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Purchased', 'Purchased'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix: M1 Purchase Prediction")
        
        plot_path = self.config['paths'].get('plot_output', 'confusion_matrix.png')
        plt.savefig(plot_path)
        logging.info(f"Confusion matrix saved to {plot_path}")
        plt.close()

        return model