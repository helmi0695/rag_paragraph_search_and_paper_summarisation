import os
import pinecone
import time
import logging
import re
import glob
import pandas as pd


class VectorStore():
    def __init__(self):
        # Set up logging
        # logging.basicConfig(level=logging.INFO) # In Google Colab, logs created using logging module are not directly displayed in the output cell
        pass

    def read_documents(self, folder_path):
        # Initialize an empty list to store data
        data_content = []

        # Get a list of all .txt files in the folder
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

        # Loop through each file, read its content, and append to the list
        for doc_id, txt_file in enumerate(txt_files):
            try:
                file_path = os.path.join(folder_path, txt_file)
                print(f'Importing {file_path}')
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    # Split content into documents based on "----"
                    documents = re.split(r'----', content)
                    file_name = os.path.basename(txt_file)

                    # Process each document
                    for chunk_id, document in enumerate(documents):
                        # Extract chunks based on "TITLE PARAGRAPH:"
                        chunks = re.split(r'TITLE PARAGRAPH:', document)

                        # Process each chunk
                        for sub_chunk_id, chunk in enumerate(chunks):
                            # Skip empty chunks
                            if not chunk.strip():
                                continue

                            # Extract chunk title
                            title_match = re.search(r'(.*?)\n', chunk)
                            chunk_title = title_match.group(1).strip() if title_match else None

                            data_content.append({
                                'file_name': file_name,
                                'chunk_id': f'{doc_id}-{chunk_id}-{sub_chunk_id}',
                                'doc_id': doc_id,
                                'chunk_title': chunk_title,
                                'chunk': chunk.strip(),
                                'chunk_length': len(chunk),
                                'doc': content,
                                'doc_length': len(content)
                            })
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")

        # Create a Pandas DataFrame from the list
        data = pd.DataFrame(data_content)
        return data

    def initialize_pinecone_index(self, pinecone_api_key, pinecone_environment, index_name):
        # get API key from app.pinecone.io and environment from console
        pinecone.init(
            api_key=os.environ.get('PINECONE_API_KEY') or pinecone_api_key,
            environment=os.environ.get('PINECONE_ENVIRONMENT') or pinecone_environment
        )

        # Index initialisation
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=384,
                metric='cosine'
            )
            # wait for index to finish initialization
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)

        # connect to the index:
        index = pinecone.Index(index_name)

        # Log the index stats
        # logging.info("Pinecone index stats: %s", index.describe_index_stats())
        print(("Pinecone index stats: %s", index.describe_index_stats()))
        return index

    def create_pinecone_vectorstore(self, data, index, embed_model):
        # Embed and index the documents - This must only be done once we have new data to inject into the index
        # Note this method can be modified according to the use case: i.e. we can use the 'update' method to update an existing index
        batch_size = 32

        for i in range(0, len(data), batch_size):
            i_end = min(len(data), i+batch_size)
            batch = data.iloc[i:i_end]
            ids = [f"{x['chunk_id']}" for i, x in batch.iterrows()]
            texts = [x['chunk'] for i, x in batch.iterrows()]
            embeds = embed_model.embed_documents(texts)
            # get metadata to store in Pinecone
            metadata = [
                {'text': x['chunk'],
                 'chunk_title': x['chunk_title'],
                 'file_name': x['file_name'],
                 'doc_id': x['doc_id']
                 } for i, x in batch.iterrows()
            ]
            # add to Pinecone
            index.upsert(vectors=zip(ids, embeds, metadata))
        print(("Pinecone index stats: %s", index.describe_index_stats()))
        return index
