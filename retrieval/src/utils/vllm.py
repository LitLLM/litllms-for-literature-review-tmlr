import requests, os

BASE_URL = {
    # "llama-3-70b": "http://0.0.0.0:8000/v1/completions",
    "llama-3-8b": os.getenv("LLAMA_8b_URL"),
    "llama-3-70b": os.getenv("LLAMA_70b_URL"),
    "llama-3.1-70b": os.getenv("LLAMA_70b_URL"),
    "llama-3.1-405b": os.getenv("LLAMA_405b_URL"),
}
AUTH_TOKEN = {
    "llama-3-70b": os.getenv("LLAMA_70b_BEARER"),
    "llama-3.1-70b": os.getenv("LLAMA_70b_BEARER"),
    "llama-3.1-405b": os.getenv("LLAMA_405b_BEARER"),
    "llama-3-8b": os.getenv("LLAMA_8b_BEARER"),
}
MODELS = {
    "llama-3.1-70b": os.getenv("LLAMA_70b_MODEL_NAME"),
    "llama-3.1-405b": os.getenv("LLAMA_405b_MODEL_NAME"),
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def infer_vllm(
    prompt,
    max_tokens=2000,
    end_point="llama-3-70b",
    top_logprobs=None,
    return_full_json=False,
    temperature=0,
):
    data = {
        "model": MODELS[end_point],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": top_logprobs,
    }
    headers = {
        "Content-Type": "application/json",
        "Cookie": "sessiona=1687876608.234.49.972136|78cabb3f310793e5a58a141fe9058709",
        "Authorization": f"{AUTH_TOKEN[end_point]}",
    }
    try:
        result = requests.post(BASE_URL[end_point], headers=headers, json=data)
        # print(result)
        result = result.json()
    except Exception as e:
        result = {
            "choices": [{"text": f"Error in generation here is the full json: {e}"}]
        }
    if return_full_json:
        return result
    try:
        return result["choices"][0]["text"]
    except KeyError as e:
        print(f"Error in making requests for {end_point}: {e}")
        print(result)
        return result["message"]
    except Exception as e:
        print(f"Error in making requests for {end_point}: {e}")
        print(result)
        return result["message"]


def main():
    prompt = "Generate an example from the WikiHow summarization dataset."
    result = infer_vllm(prompt, end_point="llama-3.1-405b")
    print(result)


if __name__ == "__main__":
    main()
