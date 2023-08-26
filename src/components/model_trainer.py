# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os
sys.path.append("src")


# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import catboost
from catboost import CatBoostRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass 
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_training(self,train_array,test_array):

   try:
        logging.info("Splitting training and test array")
        x_train,y_train,x_test,y_test = (
           train_array[:,:-1],
           train_array[:,-1],
           test_array[:,:-1],
           test_array[:,-1]
        )
        models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
  
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}
        model_report:dict = evaluate_model(x_train=x_train,y_train=
                                      y_train,x_test=x_test,y_test=y_test,models=models)  
        
        #Get best model score from dictionary
        best_model_score = max(sorted(model_report.values()))

        #Get best model name

        best_model_name = list(model_report.keys())[
           list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        
        if best_model_score < 0.6:
           raise CustomException("No best model found")
        
        logging.info(f"Best model found on both training and testing dataset")

        save_obj(
           file_path = self.model_trainer_config.train_model_file_path,
           obj = best_model
        )

        predicted_= best_model.predict(x_test)
        r2_score_ = r2_score(y_test,predicted_)

        return r2_score_


   except Exception as e:
             raise  CustomException(e,sys)
       
       

