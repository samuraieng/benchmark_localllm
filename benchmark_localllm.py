import logging
import os
import traceback

logging.basicConfig(level=logging.INFO)
#os.environ["DISABLE_TQDM"] = "0"

from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForMultimodalLM, utils
utils.logging.set_verbosity_info()

#MODEL_ID = "google/gemma-4-31B-it"
MODEL_ID = "sbintuitions/sarashina2.2-3b-instruct-v0.1"

# Load model
#processor = AutoProcessor.from_pretrained(MODEL_ID)

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
except Exception:
    traceback.print_exc()

model = AutoModelForCausalLM.from_pretrained(
#model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

pass

# Prompt
messages = [
    {"role": "system", "content": "日本語上級者で応答は日本語で行う"},
    {"role": "user", "content": "きれっきれのダジャレを一発かましてくれ"},
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"{response}")

# Parse output
#textout = processor.parse_response(response, prefix=inputs["input_ids"])
#print(f"{textout}")

pass
