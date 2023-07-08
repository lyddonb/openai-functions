import json
import uuid
import chromadb
from chromadb.config import Settings
import openai
import tiktoken


DB_PATH = "data/mem_store"


CHROMA_CLIENT = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_PATH
))


def get_or_create_collection(collection_name):
    return CHROMA_CLIENT.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})


def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']


def add_vector(collection, text, metadata):
    id = uuid.uuid4().hex
    embedding = get_embedding(text)

    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
        ids=[id]
    )


def save_memory(memory):
    collection = get_or_create_collection("memories")
    add_vector(collection, memory, {})
    CHROMA_CLIENT.persist()


def query_vectors(collection, query, n):
    query_embedding = get_embedding(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )


def retrieve_memories(query):
    collection = get_or_create_collection("memories")
    res = query_vectors(collection, query, 5)

    print(">>> retrieved memories: ") 
    print(res["documents"])
    return res["documents"]


def get_completion(messages):

    functions = [
        {
        "name": "save_memory",
        "description": """Use this function if I mention something which you think would be useful in the future and should be saved as a memory.  
        Saved memories will allow you to retrieve snippets of past conversations when needed.""",
        "parameters": {
            "type": "object",
            "properties": {
            "memory": {
                "type": "string",
                "description": "A short string describing the memory to be saved"
            },
            },
            "required": ["memory"]
        }
        },
        {
        "name": "retrieve_memories",
        "description": """Use this function to query and retrieve memories of important conversation snippets that we had in the past.
        Use this function if the information you require is not in the current prompt or you need additional information to refresh your memory.""",
        "parameters": {
            "type": "object",
            "properties": {
            "query": {
                "type": "string",
                "description": "The query to be used to look up memories from a vector database"
            },
            },
            "required": ["query"]
        }
        },    
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        #model="gpt-4-0613",
        messages=messages,
        functions=functions,
        max_tokens=200,
        stop=None,
        temperature=0.5,
        function_call="auto"
    )

    return response


def process_input(user_input=None, messages=None):
    if not messages and user_input:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]

    response = get_completion(messages)

    if response["choices"][0]["finish_reason"] == "stop":
        return response.choices[0].message["content"].strip()

    if response["choices"][0]["finish_reason"] == "function_call":
        function_name = response["choices"][0].message["function_call"]["name"]
        function_parameters = response["choices"][0].message["function_call"]["arguments"]
        arguments = json.loads(function_parameters)
        function_result = ""

        if function_name == "save_memory":
            function_result = save_memory(**arguments)
        elif function_name == "retrieve_memories":
            function_result = retrieve_memories(**arguments)

        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {"name": function_name, "arguments": function_parameters}
        })
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": f'{{"result": {str(function_result)}}}'
            }
        )

        return process_input(messages=messages)


if __name__ == "__main__":
    # First session
    # message = process_input(user_input="Hi, can you remeember that my birthday is May 5th, 1980")

    # Second session
    message = process_input(user_input="Hi, do you remember when my birthday is?")

    print(message)