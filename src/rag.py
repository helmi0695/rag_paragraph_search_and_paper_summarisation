import pandas as pd
from langchain.vectorstores import Pinecone


class RAG():
    def __init__(self,
                 index,
                 embed_model):
        self.index = index
        self.embed_model = embed_model

    def get_top_k_documents(self, query, k=3):
        text_field = 'text'  # field in metadata that contains text content

        vectorstore = Pinecone(
            self.index, self.embed_model.embed_query, text_field
        )

        top_k_docs = vectorstore.similarity_search_with_score(
            query,  # the search query
            k=k  # returns top 3 most relevant chunks of text
        )
        return top_k_docs

    def doc_search(self, query, top_k=3):
        search_results = list()
        metadata = dict()

        documents = self.get_top_k_documents(query, k=top_k)
        # Loop through the documents and get the metadata_cotent and the score
        for doc in documents:
            score = doc[-1]
            metadata = doc[0].metadata
            metadata['similarity_score'] = score
            search_results.append(metadata)

        # Create a result DataFrame
        res_df = pd.DataFrame(search_results)
        return res_df
