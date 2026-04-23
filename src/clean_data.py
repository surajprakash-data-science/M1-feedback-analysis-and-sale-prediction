import pandas as pd
import logging
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("M1_Feedback_Analysis_and_Sales_Prediction")

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def clean_data(self, df):
        logger.info("Cleaning data...")
        df = df.drop_duplicates()
        df = df.dropna()
        logger.info("Data cleaned successfully.")
        return df
    
    def encoding_data(self, df):
        logger.info("Encoding binary columns")
        df['m1_purchase'] = df['m1_purchase'].apply(lambda x: 1 if x== "Yes" else 0)
        df['trust_apple'] = df['trust_apple'].apply(lambda x: 1 if x== "Yes" else 0)
        df['user_pcmac'] = df['user_pcmac'].apply(lambda x: 1 if x== "Apple" else 0)
        df['familiarity_m1'] = df['familiarity_m1'].apply(lambda x: 1 if x== "Yes" else 0)
        df['gender'] = df['gender'].apply(lambda x: 1 if x== "Male" else 0)

        logger.info("Encoding data with LabelEncoder...")
        encoders_dict = {}
        encoded_df = df.copy()

        cols_to_encode = self.config['dataset']['category_cols']
        for col in cols_to_encode:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str)) # Ensure string type to avoid errors
            encoders_dict[col] = le
            
        logger.info("Data encoded successfully.")
        # Returning the df and the dict of encoders
        return encoded_df, encoders_dict
        
    def feature_selection(self, df):
        logger.info("Performing feature selection...")
        x = df.drop(columns=self.config['dataset']['target_col'])
        y = df[self.config['dataset']['target_col']]
        # Remove Quasi-Constant (threshold = 0.25 means 99% same values)
        sel = VarianceThreshold(threshold=0.25)
        X_filter = sel.fit_transform(df[self.config['dataset']['feature_cols']])
        # Filter: Statistical Selection (Top 10 features)
        input_cols = self.config['dataset']['numerical_cols']
        stat_sel = SelectKBest(score_func=f_classif, k=10)
        stat_sel.fit(x[input_cols], y)
        stat_features_num = pd.Index(input_cols)[stat_sel.get_support()]
        
        input_cols = self.config['dataset']['categorical_cols']
        stat_sel = SelectKBest(score_func=chi2, k='all')
        stat_sel.fit(x[input_cols], y)
        stat_features_cat = pd.Index(input_cols)[stat_sel.get_support()]

        # Embedded: Tree-based Importance
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(x, y)
        tree_features = x.columns[rf.feature_importances_ > rf.feature_importances_.mean()]
        # Wrapper: Recursive Feature Elimination
        rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
        rfe.fit(x, y)
        rfe_features = x.columns[rfe.support_]
        # Combine all selected features with votes
        logger.info("Combining selected features from all methods...")
        votes = pd.DataFrame({'Feature': x.columns})
        votes['Stat_cat'] = votes['Feature'].isin(stat_features_cat)
        votes['Stat_num'] = votes['Feature'].isin(stat_features_num)
        votes['Tree'] = votes['Feature'].isin(tree_features)
        votes['RFE'] = votes['Feature'].isin(rfe_features)
        votes['Total'] = votes[['Stat_cat', 'Stat_num', 'Tree', 'RFE']].sum(axis=1)
        best_features = votes.sort_values(by='Total', ascending=False)
        selected_cols = best_features[best_features['Total'] >= 1]['Feature'].tolist()
        df_final = df[selected_cols + self.config['dataset']['target_col']]
        
        logger.info("Feature selection completed successfully.")

        return df_final, df[selected_cols]

    def factor_analysis(self, df):
        logger.info("Performing factor analysis...")

        fa_cols = [col for col in self.config['dataset']['feature_cols'] if col in df.columns]
        n_comp = self.config['factor_analysis_params']['n_components']
        fa = FactorAnalysis(n_components=n_comp,
                            random_state=self.config['factor_analysis_params']['random_state'])
        
        factors = fa.fit_transform(df[fa_cols].values)

        factor_df = pd.DataFrame(
            factors, 
            columns=[f'factor_{i+1}' for i in range(self.config['factor_analysis_params']['n_components'])], 
            index=df.index
        )

        df_dropped = df.drop(columns=fa_cols)
        df_final = pd.concat([df_dropped, factor_df], axis=1)

        logger.info("Factor analysis completed successfully.")
        return df_final, fa