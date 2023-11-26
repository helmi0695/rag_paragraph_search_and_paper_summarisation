import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from src.rag import RAG
from src.embedding import Embedding
from src.vectorstore import VectorStore


class Validate():
    def __init__(self):
        pass

    def predict(self, rag, val_df):
        # Get the most similar document
        val_df['top_3_doc'] = val_df['chunk'].apply(lambda query: rag.get_top_k_documents(query, k=3))

        # Update the similarity score to be 0 or 1:
        # All scores >= to 0.5 are considered 1
        # Note: I set the threshold to 0.5 based on my experiments, but it can be updated upon further inspection, new data or other factors
        val_df['is_similar_pred'] = val_df['top_3_doc'].apply(lambda d: 0 if d[0][-1] < 0.5 else 1)
        return val_df

    def get_performance_metrics(self, val_df, output_metrics_path, rag):
        val_df = self.predict(rag, val_df)

        # Get today's date with the hour
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Evaluate precision, recall, and F1 score
        precision = precision_score(val_df['is_similar'], val_df['is_similar_pred'])
        recall = recall_score(val_df['is_similar'], val_df['is_similar_pred'])
        f1 = f1_score(val_df['is_similar'], val_df['is_similar_pred'])

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score' : f1
        }

        # Export the metrics as JSON
        with open(output_metrics_path + f'/validation_metrics_{current_time}.json', 'w') as file:
            json.dump(metrics, file, indent=4)
        return metrics
    
    def export_data(self, data, output_file_name, output_folder_path):
        '''helper function to export data'''
        # Get today's date with the hour
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save to_summarise_df to a CSV file with the current timestamp
        csv_filename = f'/{output_file_name}_{current_time}.csv'
        csv_data_path = output_folder_path + csv_filename
        data.to_csv(csv_data_path)


if __name__ == "__main__":
    # Specify the path to settings.local.json file
    settings_file_path = '../settings.local.json'

    # Read JSON data from the file
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)

    # inputs:
    # Extract Pinecone settings from the loaded JSON
    pinecone_api_key = settings['pinecone_settings']['api_key']
    pinecone_environment = settings['pinecone_settings']['environment']
    pinecone_index_name = settings['pinecone_settings']['index_name']

    # Extract input documents data path from the loaded JSON
    input_documents_data_path = settings['data_paths']['inputs']['documents_folder_path']

    # Extract Hugging Face embedding model name from the loaded JSON
    hf_embedding_model_name = settings['huggingface_settings']['model_name']

    # outputs
    output_metrics_path = settings['data_paths']['outputs']['metrics_path']
    output_file_name = 'val_data'
    predicted_validation_data_path = settings['data_paths']['outputs']['predicted_validation_data_path']

    # Initialize VectorStore and Pinecone index
    vectorstore = VectorStore()
    pinecone_index = vectorstore.initialize_pinecone_index(pinecone_api_key, pinecone_environment, pinecone_index_name)
    
    # Initialize Embedding and Hugging Face embedding model
    embedding = Embedding()
    embed_model = embedding.initialize_hf_embeddings(hf_embedding_model_name)

    # Initialize RAG with Pinecone index and Hugging Face embedding model
    rag = RAG(pinecone_index, embed_model)

    # Read the validation data
    validation_data_path = settings['data_paths']['inputs']['validation_data_path']
    val_df = pd.read_excel(validation_data_path)

    
    # Run the validation
    validate = Validate()
    metrics = validate.get_performance_metrics(val_df, output_metrics_path, rag)

    # Export the validation data with predictions
    validate.export_data(data=val_df, output_file_name=output_file_name, output_folder_path=predicted_validation_data_path)
