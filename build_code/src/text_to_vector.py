import time
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import TokenTextSplitter

from data_body import EmbeddingOutputBody
from models import EmbeddingModel
from helpers import logger


def format_input_passage(input_text):

    """
    Take a text and format it to the standard text format for the emdedding model.
    """
    # text_splitter2 = TokenTextSplitter(chunk_size = 512)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, 
        chunk_overlap=200, 
        length_function=len, 
        is_separator_regex=False
    )
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     model_name="gpt-4",
    #     chunk_size=500,
    #     chunk_overlap=20,
    # )

    chunks = text_splitter.split_text(input_text)
    text = []
    for text_chunk in chunks:
        passage = "passage: "
        text.append(passage + text_chunk)

    return text, chunks

def format_input_query(input_text):

    """
    Take a query and format it to the standard text format for the emdedding model.
    """
    query = "query: "
    return [query + input_text]

def embed_documents(input_text, task, extra_data, response_url):
    """
    Transform the input data and generate embeddings
    """
    try:
        start_time = time.time()
        if task == "ingest":
            
            formated_chunked_text, chunked_text = format_input_passage(input_text=input_text)
            embedding_result = EmbeddingModel.predict(formated_chunked_text)

            response = {
                "dim":embedding_result.shape,
                "text": chunked_text,
                "vectors": embedding_result.tolist()
            }

        if task == "retrieve":

            formated_text= format_input_query(input_text=input_text)
            embedding_result = EmbeddingModel.predict(formated_text)

            response = {
                "dim":embedding_result.shape,
                "vectors": embedding_result.tolist()
            }
            
        response["extra_data"] = extra_data
        payload = json.loads(EmbeddingOutputBody(**response).json())
        payload =  {"data": payload}
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time} seconds")
        requests.post(url=f"{response_url}/documents/embedding", json=payload
        )
        return payload
    
    except Exception as e:
        logger.info(f"Error: {e}")
        return {"error": str(e)}, 500

