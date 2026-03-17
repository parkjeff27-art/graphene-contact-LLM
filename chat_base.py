import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 박사님이 사용하신 베이스 모델 ID (또는 로컬 경로)
model_id = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading Base Model: {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Base Model Loaded Successfully! (No Adapter attached)")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("User (Base Model): ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Assistant: {response}\n")

