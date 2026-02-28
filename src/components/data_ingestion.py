import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass  # Using dataclass to define the configuration for data ingestion[act as a decorator]
class DataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv') # os.path.join is used to create a path by joining the specified folder and file name. In this case, it creates a path for the training data file named 'train.csv' inside the 'artifacts' folder.
        self.test_data_path = os.path.join('artifacts', 'test.csv') # In this case, it creates a path for the testing data file named 'test.csv' inside the 'artifacts' folder.
        self.raw_data_path = os.path.join('artifacts', 'data.csv') # In this case, it creates a path for the raw data file named 'data.csv' inside the 'artifacts' folder.


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion methos or componenet")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe') # reading the dataset

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path,index=False, header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(("Ingestion of the data is completed"))

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
                
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
