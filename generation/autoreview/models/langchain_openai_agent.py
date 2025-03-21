# Also see https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db
import os
import json
import re
import time
import tiktoken

from langchain.callbacks import get_openai_callback
from bs4 import BeautifulSoup
from langchain import LLMChain
from openai.error import RateLimitError, InvalidRequestError, Timeout
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI

API_KEY = os.environ["OPENAI_API_KEY"]

class OpenAIAgent:
    """
    Main class for OpenAI agents.
    """

    def __init__(self, model_name="gpt-3.5-turbo", prompt_name="prompt_action_gpt"):
        """
        Create an OpenAI agent based on a model name and prompt name.
        """
        super().__init__()
        # params = {"temperature": 0, 
        params = {"max_tokens": 500}  # "temperature": 0.2}
        open_ai_params = {
            "top_p": 1,
        }
            # "frequency_penalty": 0.0,
            # "presence_penalty": 0.0,
        self.model_name = model_name
        self.model = ChatOpenAI(
            model_name=model_name,
            **params,
            model_kwargs=open_ai_params,
            openai_api_key=API_KEY,
            max_retries=0,
        )
        self.executed_actions = []
        self.budget_spent = 0
        print(f"Running the experiment for {model_name} using Langchain")

    def description(self) -> str:
        """
        return the description of the agent
        """
        return super().description()

    def get_state_dict(self):
        return {"budget_spent": self.budget_spent}

    def get_response(self, prompt):
        template = "You are a helpful assistant."
        # template = "You are an autoregressive language model that completes user's sentences. You should not conversate with user."
        # https://github.com/langchain-ai/langchain/issues/2212
        prompt = prompt.replace("{", "{{")  # todo : fix this
        prompt = prompt.replace("}", "}}")  # todo : fix this

        # print(f"Length of prompt in OpenAI: {len(prompt.split())}")

        while True:
            # this loop breaks if the request is successful
            try:
                # get the full prompt
                system_message_prompt = SystemMessagePromptTemplate.from_template(template)
                human_message_prompt = HumanMessagePromptTemplate.from_template(prompt)

                chat_prompt = ChatPromptTemplate.from_messages(
                    [system_message_prompt, human_message_prompt]
                )

                # use callback to get
                with get_openai_callback() as cb:
                    openai_chain = LLMChain(prompt=chat_prompt, llm=self.model)
                    response = openai_chain.run({})
                    n_tokens = cb.total_tokens
                    total_cost = cb.total_cost

                break
            except RateLimitError as e:
                # Deal with rate limit error
                print(e._message)

                n_seconds = 0.2
                # print(f"Retrying in {n_seconds} seconds")

                time.sleep(n_seconds)

            except InvalidRequestError as e:
                # Deal with maximum length error
                print(e._message)

                # Handle maximum length error
                max_len_str = "This model's maximum context length is "
                assert max_len_str in e._message

                # get the number of tokens for chat_prompt
                max_length = int(
                    e._message[e._message.index(max_len_str) + len(max_len_str) :].split(" ")[0]
                )

                # shred from prompt and get template + prompt to max length
                while True:
                    n_tokens = count_tokens(template + prompt, model=self.model_name)
                    if n_tokens < max_length - 200:
                        print(f"Shredded to max length {max_length}")
                        break
                    else:
                        prompt = prompt[:-2000]

                time.sleep(0.2)

            except Timeout as e:
                # For urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out. (read timeout=600)
                print(e._message)

                n_seconds = 0.2
                print(f"Retrying in {n_seconds} seconds")

                time.sleep(n_seconds)

            except Exception as e:
                print(f"Exception occured as: {e}")
                n_seconds = 0.2
                print(f"Retrying in {n_seconds} seconds")
                time.sleep(n_seconds)                  
                # raise ValueError("Error in get_response")

        # return the response, cost, # tokens, and prompt
        response_dict = {
            "response": response,
            "total_cost": total_cost,
            "prompt": template + "\n\n" + prompt,
            "n_tokens": cb.total_tokens,
        }

        # add to budget
        self.budget_spent += total_cost

        return response_dict


def get_price(num_tokens, price_per_1ktokens):
    return num_tokens * price_per_1ktokens / 1000



def count_tokens(text, model="chatgpt"):
    model_map = {
        "chatgpt": "gpt-3.5-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
        "gpt-4": "gpt-4",
        "davinci": "text-davinci-003",
        "ada": "ada",
        "babbage": "babbage",
        "curie": "curie",
        "davinci1": "davinci",
        "davinci2": "text-davinci-002",
    }

    enc = tiktoken.encoding_for_model(model_map.get(model))
    return len(enc.encode(text))