

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

torch.set_default_device("cuda")

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)

#Set inputs
text = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nwhat is shown in the page in detail? ASSISTANT:"
image = Image.open("./trial/image/wss.png")

input_ids = tokenizer(text, return_tensors='pt').input_ids
image_tensor = model.image_preprocess(image)

#Generate the answer
output_ids = model.generate(
    input_ids,
    max_new_tokens=1000,
    images=image_tensor,
    use_cache=True)[0]
print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())