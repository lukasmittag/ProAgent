from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff

HF_MODEL_MAP = {
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
}

_MODEL_CACHE = {}


def _get_shared_model(model_name):
    if model_name not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo_id = HF_MODEL_MAP.get(model_name, model_name)
        rprint(f"[green][HF][/green]: loading {repo_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype="auto", device_map="auto"
        )
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]

TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "qwen2.5:3b": 32768,
    "qwen2.5:7b": 32768,
}


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    """
    def __init__(self, 
                 role_messages, 
                 model="gpt-3.5-turbo-0301",
                 retrival_method="recent_k",
                 K=3):
        '''
        args:  
        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        self.chat_model = True
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

        self.hf_tokenizer, self.hf_model = _get_shared_model(self.model)

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    def query(self, key=None, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        import torch

        rec = self.K
        if trace == True:
            self.K = 0
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False:
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again."
        self.K = rec

        prompt = self.hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoding = self.hf_tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        gen_kwargs = {"max_new_tokens": 256}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output_ids = self.hf_model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            stops = [stop] if isinstance(stop, str) else stop
            for s in stops:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []

