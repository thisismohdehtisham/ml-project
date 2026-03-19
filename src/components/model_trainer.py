import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def intitiate_model_trainer(self, train_array, test_array, preprocessor_path):   #output of data transformation is passed as input to model trainer
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            # using hyper parameter tuning to find the best hyperparameters for each model using GridSearchCV or RandomizedSearchCV
            

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)

            # to get the best model score from the dictionary
            best_model_score = max([v['test_score'] for v in model_report.values()])

            # to get the best model name from dict

            best_model_name = list(model_report.keys())[
                [v['test_score'] for v in model_report.values()].index(best_model_score)
            ]

            

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            preprocessing_obj = preprocessor_path

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)  

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # logging.info("Exception occurred at model training")
            raise CustomException(e, sys)