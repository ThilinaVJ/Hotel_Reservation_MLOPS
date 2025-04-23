import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exeption import CustomExeption
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    
    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data processing step")

            logger.info("Dropping the columns")
            df.drop(columns='Booking_ID', inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Label Encoding")
            label_encorder = LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col] = label_encorder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encorder.classes_, label_encorder.transform(label_encorder.classes_))}

            logger.info("Label Mappings are:")
            for col,mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Doing Skewness Handling")

            skew_threshold = self.config["data_processing"]["skewness_threshold"]

            skewness = df[num_cols].apply(lambda X:X.skew())

            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        
        except Exception as e:
            logger.error(f"Error during processing step {e}")
            raise CustomExeption("Error while preprocess data", e)

    
    def balance_data(self, df):
        try:
            logger.info("Handiling Imbalanced data")
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Data balanced sucessfully!")
            
            return balanced_df

        except Exception as e:
            logger.error(f"Error during balancing data {e}")
            raise CustomExeption("Error while balancing data", e)

    
    def feature_selection(self, df):

        try:
            logger.info("Starting feature selection process")
            
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            })
            feature_importance_df.sort_values(by="importance", ascending=False)
            top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

            no_of_features_to_select = self.config["data_processing"]["no_of_features"]

            top_features = top_features_importance_df["feature"].head(no_of_features_to_select).values
            top_df = df[top_features.tolist() + ["booking_status"]]

            logger.info(f"Features Selected: {top_features}")

            logger.info("feature selection completed sucessfully!")

            return top_df


        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomExeption("Error while feature selection", e)

    
    def save_data(self,df,file_path):
        try:
            logger.info("Saving our data in processed folder")

            df.to_csv(file_path, index=False)

            logger.info(f"Data Sucessfully saved!: {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data {e}")
            raise CustomExeption("Error while saving data", e)

        
    def process(self):
        try:
            logger.info("Loading the data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns] # be careful, has to select the same columns as the train df

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data preprocessing completed successfully!")

        except Exception as e:
            logger.error(f"Error during data preprocessing pipeline {e}")
            raise CustomExeption("Error while data preprocessing pipeline", e)


if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
