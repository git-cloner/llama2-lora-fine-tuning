import os
import sys
import argparse

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=100
)

def generate() :
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    args = parser.parse_args()
    if args.load_8bit is None:
        load_8bit = False
    else:
        load_8bit = args.load_8bit
    if args.base_model is None:
        base_model = "./model/llama-7b"
    else:
        base_model = args.base_model
    if args.lora_weights is None:
        lora_weights = "./model/llama-peft"
    else:
        lora_weights = args.lora_weights

    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    device_map_cpu = {
        "transformer.word_embeddings": "cpu",
        "transformer.word_embeddings_layernorm": "cpu",
        "lm_head": "cpu",
        "transformer.h": "cpu",
        "transformer.ln_f": "cpu",
        "model.embed_tokens": "cpu",
        "model.layers":"cpu",
        "model.norm":"cpu"
    }

    device_map_cpu = {"": "cpu"}

    device_map_gpu = "auto"

    # load model
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            quantization_config=bnb_config_4bit if load_8bit else bnb_config_8bit,
            torch_dtype=torch.float16,
            device_map=device_map_gpu if load_8bit else device_map_cpu
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # generate
    while True:
        input_text = input("Input:")
        if len(input_text.strip()) == 0:
            break
        inputs = tokenizer(input_text, return_tensors="pt")
        generation_output = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config
        )
        s = generation_output[0]
        response = tokenizer.decode(s, skip_special_tokens=True)
        print("Response: ", response)
        print("\n")

if __name__ == '__main__':
    with torch.autocast("cuda"):    
        generate()