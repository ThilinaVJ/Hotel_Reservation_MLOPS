import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import get_logger
from src.custom_exeption import CustomExeption
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data splitted successfully for Model training")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error during load and split data in Model training {e}")
            raise CustomExeption("Error while data loading and spliting in Model training", e)

    
    def train_lgb(self, X_train, y_train):
        try:
            logger.info("Initializing our model")

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Starting our Hyperparameter Tuning...")

            random_search = RandomizedSearchCV(
                estimator = lgbm_model,
                param_distributions = self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs = self.random_search_params["n_jobs"],
                verbose = self.random_search_params["verbose"],
                random_state = self.random_search_params["random_state"]
            )

            logger.info("Starting our Hyper parameter tuninig...")

            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed!")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are: {best_params}")

            return best_lgbm_model

        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomExeption("Failed to train model", e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"precision: {precision}")
            logger.info(f"recall: {recall}")
            logger.info(f"f1 score: {f1}")

            return {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
            }

        except Exception as e:
            logger.error(f"Error while model evaluataion {e}")
            raise CustomExeption("Failed to evaluate the model", e)      


    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("saving the model")

            joblib.dump(model, self.model_output_path)

            logger.info(f"Model safed to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving the model {e}")
            raise CustomExeption("Failed to saving the model", e)
    

    def run(self):
        try:
            logger.info("Starting our model training pipeline...")

            X_train, X_test, y_train, y_test = self.load_and_split_data()
            best_lgbm_model = self.train_lgb(X_train, y_train)
            model_metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
            self.save_model(best_lgbm_model)

            logger.info("Model Training Successfully completed!")
        
        except Exception as e:
            logger.error(f"Error while running the model training pipeline {e}")
            raise CustomExeption("Failed to run the model training pipeline", e)


if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()