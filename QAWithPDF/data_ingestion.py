from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data(data_path):
    try:
        logging.info("data loading started...")

        loader = SimpleDirectoryReader(input_files=[data_path])

        documents = loader.load_data()

        logging.info("data loading completed...")
        return documents

    except Exception as e:
        raise customexception(e, sys)