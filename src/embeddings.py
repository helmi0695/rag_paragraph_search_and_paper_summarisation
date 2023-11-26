# embedding.py
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class Embedding():
    def __init__(self):
        pass

    def initialize_hf_embeddings(self, embedding_model_name):

        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        embed_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )
        return embed_model
