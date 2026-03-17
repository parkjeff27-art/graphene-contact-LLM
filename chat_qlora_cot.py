import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./_qwen25_7b_qlora_cot/final_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=bnb_config, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Model loaded!\n")

system_prompt = "You are a domain-specific physics assistant specialized in graphene-metal interfaces and contact resistance. Provide detailed, physics-grounded explanations based on experimental evidence from TLCD/TECD comparative studies. Use step-by-step reasoning."

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        break
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nAssistant: {response}")

print("Bye!")
