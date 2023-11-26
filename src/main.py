import json
import pandas as pd
from datetime import datetime
from src.rag import RAG
from src.embedding import Embedding
from src.summarize import Summarize
from src.vectorstore import VectorStore
from src.languae_model_wrapper import WrapLlm


def export_data(data, output_file_name, output_folder_path):
    '''helper function to export data'''
    # Get today's date with the hour
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save to_summarise_df to a CSV file with the current timestamp
    csv_filename = f'/{output_file_name}_{current_time}.csv'
    csv_data_path = output_folder_path + csv_filename
    data.to_csv(csv_data_path)


def extract_summarize(paragraph, data, llm):
    '''function to combine the extraction and summarization process'''
    doc_search_result = rag.doc_search(paragraph, top_k=3)

    # prepare the data to be summarized
    to_summarise_df = (pd.merge(doc_search_result, data, on=['file_name', 'chunk_title'])
                .groupby(['file_name', 'chunk_title'])
                .first()
                .reset_index()[['file_name', 'chunk_title', 'doc', 'similarity_score']]
                .sort_values(by='similarity_score', ascending=False))

    # Get all the rows to be summarized for the extracted documents:
    # Add a dummy 'similarity_score' column to the data dataframe
    data['similarity_score'] = None

    # Merge the two dataframes based on the "file_name" column
    merged_df = pd.merge(data, to_summarise_df[['file_name']], on='file_name')

    # Filter the merged dataframe to keep only relevant columns
    final_df = merged_df[['file_name', 'chunk_title', 'doc', 'similarity_score', 'chunk']]

    # Apply the summarize_text_chunk method to each row
    print('Summarizing the extracted chunks')
    final_df['summarized_chunk'] = final_df['chunk'].apply(lambda x: summarize.generate_summary(x, llm, how="chunk"))

    # Group by 'file_name' and aggregate the 'summarized_chunk' into a list
    grouped_df = final_df.groupby('file_name')['summarized_chunk'].agg(list).reset_index()

    # Merge the grouped dataframe back to to_summarise_df
    to_summarise_df = pd.merge(to_summarise_df, grouped_df, on='file_name', how='left')

    # We use this exception handling in case we encounter a out of memory issue
    # In this case, we get the full summary by joining the summaries of all chunks
    print('Summarizing the extracted papers')
    try:
        to_summarise_df['doc_summary'] = to_summarise_df['summarized_chunk'].apply(lambda text_list: rag.generate_summary(text_list, llm, how="list"))
        print('Documents were summarized using an LLM')
    except Exception as e:
        print(f"Exception during summarization: {e}")
        to_summarise_df['doc_summary'] = to_summarise_df['summarized_chunk'].apply(lambda text_list: '\n'.join(text_list))
        print('Documents were summarized using joining of summarized chunks')
    summarized_retrieved_data = to_summarise_df
    return summarized_retrieved_data


# Specify the path to settings.local.json file
settings_file_path = '../settings.local.json'

# Read JSON data from the file
with open(settings_file_path, 'r') as file:
    settings = json.load(file)

# Extract Input data from the loaded json
input_documents_data_path = settings['data_paths']['inputs']['documents_folder_path']

llama_2_7b_model_name = settings['llama2_7b_settings']['model_name']
hf_auth_token = settings['huggingface_settings']['hf_auth_token']

# Extract Pinecone settings from the loaded JSON
pinecone_api_key = settings['pinecone_settings']['api_key']
pinecone_environment = settings['pinecone_settings']['environment']
pinecone_index_name = settings['pinecone_settings']['index_name']
# Extract Hugging Face embedding model name from the loaded JSON
hf_embedding_model_name = settings['huggingface_settings']['model_name']

wrap_llm = WrapLlm(hf_auth=hf_auth_token, model_name=llama_2_7b_model_name)
llm = wrap_llm.create_llama_wrapper()

# Initialize VectorStore and Pinecone index
vectorstore = VectorStore()
pinecone_index = vectorstore.initialize_pinecone_index(pinecone_api_key, pinecone_environment, pinecone_index_name)

# Initialize Embedding and Hugging Face embedding model
embedding = Embedding()
embed_model = embedding.initialize_hf_embeddings(hf_embedding_model_name)

# Push the Embedded data into the Pinecone vectorstore
data = vectorstore.read_documents(input_documents_data_path)
pinecone_index = vectorstore.create_pinecone_vectorstore(data, pinecone_index, embed_model)

rag = RAG(pinecone_index, embed_model)
summarize = Summarize()

# TO DO: Read this from the API request body (i.e. we wrap this as a Fast API)
paragraph = 'mRNA vaccines have become a versatile technology for the prevention of infectious diseases and the treatment of cancers.'

# extract the most similar papers to 'parahraph' and summarize them
summarized_retrieved_data = extract_summarize(paragraph, data, llm)

# Export the summarized data
output_file_name = 'summarized_retrieved_data'
output_folder_path = settings['data_paths']['outputs']['summarized_retrieved_data_path']

summarized_documemts = summarized_retrieved_data[['file_name', 'chunk_title', 'similarity_score', 'doc_summary']]
# summarised_documemts = to_summarise_df[['file_name', 'chunk_title', 'similarity_score', 'summarized_chunk']]

export_data(summarized_documemts, output_file_name, output_folder_path)
