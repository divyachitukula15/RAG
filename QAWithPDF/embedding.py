from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding

import sys

from exception import customexception
from logger import logging


def download_gemini_embedding(model, document):
    """
    Create vector index and query engine
    """

    try:
        logging.info("Loading Gemini Embedding Model...")

        gemini_embed_model = GeminiEmbedding(
            model_name="models/gemini-embedding-001"
        )

        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Creating Vector Index...")

        index = VectorStoreIndex.from_documents(document)

        index.storage_context.persist(persist_dir="./storage")

        logging.info("Vector Index Created Successfully")

        query_engine = index.as_query_engine()

        return query_engine

    except Exception as e:
        raise customexception(e, sys)