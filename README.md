# instadeep-llm-technical-test
 RAG - summarisation using LLMs

 Setting up the environment:

 -Cloning the github repo:
    -If the plan is to run the files locally: Follow the normal process of cloning a repo from github
    Note:  I do not recommand this option as I used the LLAMA model for this experiment
    -If the plan is to clone the private github repo to colab:
        -Clone the repo locally and open it.
        -Open a new terminal.
        -Generate an SSH key following the instructions found in this page: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
        -Add the generated public SSH key (found in the .pub file) to the repo under Setting > Deploy keys > Add deploy key
        -For simplicty reasons, copy the two generated files to a folder in google drive named 'deploy_keys'
        -Open colab and open a new notebook
        -Run this command: !mkdir -p /root/.ssh/
        -Mount Google drive
        -Run the following command: !cp /content/drive/MyDrive/deploy_keys/id_ed25519* /root/.ssh/
        -Run this command to enable github on Colab: !ssh-keyscan github.com >> /root/.ssh/known_hosts
        -Make sure that you are successfully authenticated by running: !ssh -T git@github.com
        -Clone the repo using the following command: !git clone git@github.com:helmi0695/instadeep-llm-technical-test.git


Now that we have the repo cloned, follow these steps to run the code.
 -Create an .env.local file under the root folder where you can fill all the secrets and other environment variables
 -

