import os
import sys

from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

from exception import customexception

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def load_model():
    """
    Load Gemini LLM
    """
    try:
        model = Gemini(
            model_name="models/gemini-2.5-flash",
            api_key=GOOGLE_API_KEY
        )

        return model

    except Exception as e:
        raise customexception(e, sys)