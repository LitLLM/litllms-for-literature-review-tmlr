#!/usr/bin/env python3
# Copyright (c) ServiceNow Research and its affiliates.
"""
This file defines wrapper for ChatGPT API
"""
from typing import Any
import openai
import tiktoken
from .base_model import BaseMLModel
import os

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY


class ChatGPTModel(BaseMLModel):
    """
    Class to act as interface for ChatGPT
    """

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        openai.api_key = OPENAI_API_KEY

    def gen_response(self, json_data: Any) -> str:
        """
        This wrapper allows to pre-process, truncate or post-process the response
        """
        response = self.run_open_ai_api(json_data)
        return response

    def truncate_tokens(self, input_txt: str, max_tokens: int = 3000) -> str:
        """
        This allows to truncate the input text to the max tokens ingested by the model

        The maximum number of tokens to generate in the chat completion.
        The total length of input tokens and generated tokens is limited by the model's context length
        See https://platform.openai.com/docs/api-reference/chat/create
        """
        # enc = tiktoken.get_encoding("cl100k_base")
        encoding = tiktoken.encoding_for_model(self.model_name)
        tokens = encoding.encode(input_txt)
        num_tokens = len(tokens)
        if num_tokens > max_tokens:
            tokens = tokens[:max_tokens]
            truncated_txt = encoding.decode(tokens)
        else:
            truncated_txt = input_txt
        return truncated_txt

    def run_open_ai_api(self, json_data: Any, max_gen_tokens: int = 500, temperature: float = 0.2) -> str:
        """
        This function actually calls the OpenAI API

        :param json_data:
        :return:
        """
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            max_tokens=max_gen_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{json_data['prompt']}"},
            ],
        )

        return completion["choices"][0]["message"]["content"]
