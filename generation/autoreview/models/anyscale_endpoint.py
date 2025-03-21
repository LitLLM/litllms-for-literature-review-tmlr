import time
import openai
import os
ANYSCALE_ENDPOINT_API_KEY = os.environ['ANYSCALE_ENDPOINT_API_KEY']


def anyscale_chat_complete(prompt, engine, temp=0.7, max_tokens=500):
    # (prompt, n, engine, temp, top_p, max_tokens=256, echo=False, logprobs=None,max_retries=3):
    system_content = "You are a helpful assistant."
    prompt = prompt.replace("{", "{{")  # todo : fix this
    prompt = prompt.replace("}", "}}")  # todo : fix this

    completion = None
    if engine in ["meta-llama/Llama-2-7b-chat-hf", 
                    "meta-llama/Llama-2-13b-chat-hf",
                    "meta-llama/Llama-2-70b-chat-hf",
                    "codellama/CodeLlama-34b-Instruct-hf"]:
        openai.api_key = ANYSCALE_ENDPOINT_API_KEY
        openai.api_base = "https://api.endpoints.anyscale.com/v1"
    else:
        openai.api_key = "your openai key"
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=engine,
                messages=[{"role": "system", "content": system_content}, {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp)
                # n=1 if n is None else n,
                # echo=echo,
                # logprobs=logprobs,
                # top_p=1 if not top_p else top_p,          
                # stop=None if n is None else ["\n"],     

            # return [c.message.content.strip() for c in completion.choices]
            response = completion["choices"][0]["message"]["content"].strip()
            return response
        except openai.error.RateLimitError as e:
            print("RateLimitError, Sleeping for 5 seconds...")
            time.sleep(5)
        except openai.error.APIError as e:
            print(f"APIError, {e}\nSleeping for 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"{e}, Sleeping for 5 seconds...")
            time.sleep(5)    
    
    # response = completion["choices"][0]["message"]["content"]
    # return response
            