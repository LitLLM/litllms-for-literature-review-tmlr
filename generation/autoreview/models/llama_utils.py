


def llama_tokenizer_utils(tokenizer):
    # TODO SA https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
    # https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da?permalink_comment_id=4658424#gistcomment-4658424
    tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation 
    tokenizer.padding_side = "right"
    tokenizer.pad_token = "<PAD>"
    return tokenizer                       


def llama_model_utils(model):
    # TODO SA https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 
    model.config.use_cache = False

