from langchain import PromptTemplate,  LLMChain


class Summarize():
    def __init__(self):
        pass

    def generate_summary(self, text, llm, how="chunk"):
        """
        Used mainly to summarize text.
        the text can be under 3 diffrent formats:
            - chunk: a single paragraph
            - list : a list of paragraphs
            - full : a full document - This is not recommended if we have large document that do not fit into memory
        Input: text_chunk, llm, how:("chunk","list", "full")
        Output: summary of text_chunk
        """
        # Defining the template to generate summary
        template = """
        Write a concise summary of the text, return your responses with 1-2 sentences that cover the key points of the text without generating any extra content.
        ```{text}```
        SUMMARY:
        """
        if how == "list":
            template = """
            Write a concise summary based the list of texts provided, return a coherent summary that covers the key points of the text without generating any extra content.
            ```{text}```
            SUMMARY:
            """
        elif how == "full":
            template = """
            Write a concise summary of the text, return your responses with 5 paragraphs that cover the key points of the text without generating any extra content.
            ```{text}```
            SUMMARY:
            """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        summary = llm_chain.run(text)
        return summary
