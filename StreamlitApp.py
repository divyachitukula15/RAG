import streamlit as st
import os

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model


def main():

    st.set_page_config(page_title="QA with Documents")

    st.header("QA with Documents (Information Retrieval)")

    doc = st.file_uploader("Upload your document")

    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):

        if doc is None:
            st.error("Please upload a file.")
            return

        with st.spinner("Processing..."):

            # Save uploaded PDF
            os.makedirs("Data", exist_ok=True)

            file_path = os.path.join("Data", doc.name)

            with open(file_path, "wb") as f:
                f.write(doc.getbuffer())

            # Load document
            document = load_data(file_path)

            # Load model
            model = load_model()

            # Create query engine
            query_engine = download_gemini_embedding(
                model,
                document
            )

            # Ask question
            response = query_engine.query(user_question)

            st.write(response.response)


if __name__ == "__main__":
    main()