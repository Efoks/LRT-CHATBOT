import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from src import config as cfg

if __name__ == '__main__':

    if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        print('Flash-Attention-2 is available')
        attn_implementation = 'flash_attention_2'
    else:
        print('Flash-Attention-2 is not available')
        attn_implementation = 'sdpa'

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=cfg.LLM_MODEL,
                                              token = cfg.TOKEN)
    llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cfg.LLM_MODEL,
                                               token = cfg.TOKEN,
                                               torch_dtype=torch.float16,
                                               low_cpu_mem_usage=False).to('cuda')

    print('LLM model loaded successfully')
    inpurt_text = 'How could you describe Lithuanian Argriculture sectore?'
    print(f'Input text: {inpurt_text}')

    chat_template = [{
        'role': 'user',
        'content': inpurt_text
    }]

    promt = tokenizer.apply_chat_template(conversation=chat_template,
                                          tokenize=False,
                                          add_generation_prompt=True)

    tokezined_input = tokenizer(promt,
                                return_tensors='pt').to('cuda')

    outputs = llm.generate(**tokezined_input,
                           max_new_tokens = 256)
    # print(outputs[0])
    decoded_outputs = tokenizer.decode(outputs[0])

    print(f'Output: {decoded_outputs}')