import openai
from loguru import logger

from src.llms.base import BaseLLM
from importlib import import_module
from openai import OpenAI

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")

class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: str) -> str:
        openai.api_key = conf.GPT_api_key
        if conf.GPT_api_base and conf.GPT_api_base.strip():
            openai.base_url = conf.GPT_api_base
        res = openai.chat.completions.create(
            model = self.params['model_name'],
            messages = [{"role": "user","content": query}],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
        )
        real_res = res.choices[0].message.content

        token_consumed = res.usage.total_tokens
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res



class DeepSeek(BaseLLM):
    def __init__(self, model_name='deepseek-chat', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        from src.configs import config
        self.client = OpenAI(api_key=config.GPT_api_key, base_url=config.GPT_api_base)

    def request(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
        ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        real_res = response.choices[0].message.content
        return real_res