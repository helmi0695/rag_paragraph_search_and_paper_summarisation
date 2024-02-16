# RAG paragraph search in documents & summarisation using LLMs

## Genral Description

In this Repo we are covering the use of LLMs to retrive similar papers based on a paragraph following these steps:

- Creating a vector database with LLM and documents (pdf or extracted text).
- Given the input (a paper’s paragraph), getting the top 3 most similar papers in the database.
- For each paper, providing a summary.
- Providing an evaluation pipeline that helps assess the quality and accuracy of the output.

## Similar Documents Retrieval

### Strategy

- Initialize a Pinecone index.
- Read text documents and break them into chunks (paragraphs).
- Generate embeddings using a [Hugging Face](https://huggingface.co/) model.
- Update the created index using the 'Upsert' method.
- Perform a similarity search to retrieve the top K documents (K=3) with matching scores.
- Results are ranked in descending order and stored in a dataframe for further processing.
- Retrieve all chunks belonging to the top 3 paragraphs.

### Challenges

- Storing entire documents is challenging due to large text sizes (around 50K characters).

### Solutions

- Store documents in chunks.

### Potential Improvements

- Experiment with different embedding models (e.g., Ada from [OpenAI](https://www.openai.com/)).
- Try different indexes (e.g., [Elasticsearch](https://www.elastic.co/)).
- Use a threshold to retrieve only documents with a high similarity score (e.g., 0.6).

## Summarization

### Strategy

- Based on the retrieval output, summarize paragraphs of the top 3 papers using an LLM.
- Utilize chunk summarization to fit the model and text into memory.
- Merge summarized chunks into one text and save the output as a CSV file.

### Challenges

- Unable to summarize whole papers as they may not fit into memory.

### Solutions

- Summarize chunks of documents instead of the whole paper.

### Potential Improvements

- Benchmark different models (e.g., [GPT-3.5-turbo](https://www.openai.com/gpt-3)) or [GPT-4](https://www.openai.com/research/gpt-4).
- Prompt engineering: experiment with different prompts to get the best-performing one.
- Hyperparameter tuning: experiment with different parameters and keep the best (e.g., adding max_tokens to control the number of input tokens).

## Validation Pipeline

### Strategy

- Build the validation dataset:
  - Use random paragraphs from the papers.
  - Paragraphs can have different lengths.
  - Label them by assigning 1 to the true labels.
  - Add negatives: Added random paragraphs related and not related to vaccines. Labeled them by assigning 0 as a target value.
- Run this through the RAG and compare the results with the ground truth (positive results are paragraphs returned having a similarity score > 0.6).
- Calculate precision, recall, and F1-score.

### Further Validation Ideas

Given more powerful resources (GPU), we can use an LLM to also validate and check the quality of the summarization like so:

- Using an LLM, check if a summary describes the document (use a high max_tokens parameter to handle long paragraphs).

### Interpretation

- Precision: 1.0 (Every document predicted as relevant is indeed relevant. No false positives).
- Recall: 0.9167 (The model has captured 91.67% of all relevant documents).
- F1 Score: 0.9565 (High overall performance, balancing precision and recall).

## Class Descriptions

In this project, I wrapped each component described above into at least one class.

- Each component is designed to be easily expanded:
  - Easily plug a new LLM to benchmark and compare it with the LLAMA 2 model.
  - Use another embedding model by adding a new method in the Embedding() class.
  - Add another vector store (like Elasticsearch) to store embeddings and perform similarity. This can be implemented in the VectorStore() class.

Note:

- Each component can be used as a separate endpoint to only perform summarization or only extraction or a combination of both.

## Setting Up the Environment

### Instructions

- **LLAMA2:**
  - To access Llama 2 models, one must first request access via [this form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) (access is typically granted instantly).

- **HuggingFace:**
  - Generating a HuggingFace Auth token: to do so, please follow [this link](https://huggingface.co/docs/transformers.js/guides/private#:~:text=To%20generate%20an%20access%20token,copy%20it%20to%20your%20clipboard).

- **Pinecone:**
  - Please follow the link to generate a [free Pinecone API key](https://app.pinecone.io/).

- **Cloning the GitHub Repo:**
  - If the plan is to run the files locally: Follow the normal process of cloning a repo from GitHub.
  - If the plan is to clone the private GitHub repo to Colab:
    - Clone the repo locally and open it.
    - Open a new terminal.
    - Generate an SSH key following the instructions found [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
    - Add the generated public SSH key (found in the .pub file) to the repo under Settings > Deploy keys > Add deploy key.
    - For simplicity reasons, copy the two generated files to a folder in Google Drive named 'deploy_keys'.
    - Open Colab and open a notebook from the /notebooks folder.
    - Mount Google Drive.
    - Run the following commands in a Colab cell:

    ```bash
    !mkdir -p /root/.ssh/
    !cp /content/drive/MyDrive/deploy_keys/id_ed25519* /root/.ssh/
    !ssh-keyscan github.com >> /root/.ssh/known_hosts
    !ssh -T git@github.com
    !git clone git@github.com:helmi0695/instadeep-llm-technical-test.git
    ```

Now that we have the repo cloned, follow these steps to run the code.

- Create a 'settings.local.json' file under the root folder where you can fill all the secrets and other environment variables.
  
  ```json
  {
      "llama2_7b_settings": {
          "model_name": "meta-llama/Llama-2-7b-chat-hf"
          // Specifies the model name for the Llama 2-7b chat model from Hugging Face.
      },
      "huggingface_settings": {
          "hf_auth_token": "YOUR_HUGGINGFACE_AUTH_TOKEN"
          // Insert your Hugging Face authentication token for accessing models here.
          "model_name": "sentence-transformers/all-MiniLM-L6-v2"
          // Specifies the model name for the Sentence Transformers MiniLM-L6-v2 model from Hugging Face.
      },
      "pinecone_settings": {
          "api_key": "YOUR_PINECONE_API_KEY"
          // Insert your Pinecone API key for accessing the Pinecone service here.
          "environment": "gcp-starter"
          // Specifies the Pinecone environment, in this case, "gcp-starter".
          "index_name": "llama-2-rag"
          // Specifies the name of the index in Pinecone for the Llama 2 Rag model.
      },
      "data_paths": {
          "inputs": {
              "documents_folder_path": "/content/instadeep-llm-technical-test/ressources/data/inputs/raw_text"
              // Specifies the path to the folder containing raw text documents for input.
              "validation_data_path": "/content/instadeep-llm-technical-test/ressources/data/inputs/validation/val_data.xlsx"
              // Specifies the path to the validation data in Excel format.
          },
          "outputs": {
              "summarized_retrieved_data_path": "/content/instadeep-llm-technical-test/ressources/data/outputs/summarized_docs"
              // Specifies the path to store the summarized retrieved documents.
              "metrics_path": "/content/instadeep-llm-technical-test/ressources/data/outputs/validation"
              // Specifies the path to store metrics related to validation.
              "predicted_validation_data_path": "/content/instadeep-llm-technical-test/ressources/data/outputs/validation"
              // Specifies the path to store the predicted validation data.
          }
      }
  }

Notes:

- Please refer to 'settings.local.json' to create the local file.
- In practice it would be safer to use a keyvault to store the secrets on the cloud.

## Running the code

There are 3 ways to run the code:

- Not recommended: Locally in case you have enough Ressources:

  - Create a virtual environment (i.e. using: python3 -m venv myenv)
  - Activate it.
  - install the requirements.txt using 'pip install -r requirements.txt'
  - Run 'python main.py'

- Recommended: In Colab:
  - Full walkthrough with comments and examples: notebooks/colab_full_walkthrough_InstaDeep_llm_task_llama_2_7b_chat_agent.ipynb
  - Running with the Class structure: notebooks/colab_class_structure_InstaDeep_llm_task_llama_2_7b_chat_agent.ipynb

