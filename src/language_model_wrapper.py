import transformers
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline


class WrapLlm():
    def __init__(self,
                 hf_auth,
                 model_name):
        self.hf_auth = hf_auth
        self.model_name = model_name

    def initialize_llama_model_tokenizer(self):
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.hf_auth
        )
        model.eval()
        print(f"Model loaded on {device}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_auth
        )
        return model, tokenizer

    def create_llama_wrapper(self):
        model, tokenizer = self.initialize_llama_model_tokenizer()

        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=1048,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        llm = HuggingFacePipeline(pipeline=generate_text)
        return llm
