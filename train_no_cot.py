import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
model_id = "Qwen/Qwen2.5-7B-Instruct"
data_file = "thesis_contact_resistance_no_cot.json"
output_dir = "./_qwen25_7b_contact_resistance_no_cot"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print(">>> Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=bnb_config, trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
print(">>> Applying QDoRA Config...")
peft_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", use_dora=True
)
print(">>> Loading Dataset...")
dataset = load_dataset("json", data_files=data_file, split="train")
def preprocess_function(examples):
    texts = []
    for message in examples['messages']:
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}
print(">>> Preprocessing Dataset...")
dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
print(">>> Configuring SFTConfig...")
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    num_train_epochs=15,
    save_strategy="epoch",
    logging_steps=1,
    max_steps=80,
    optim="paged_adamw_8bit",
    bf16=True,
    report_to="none",
    gradient_checkpointing=True,
    max_length=2048,
    dataset_text_field="text",
)
print(">>> Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
    peft_config=peft_config,
)
print(">>> Starting QDoRA Training (Contact Resistance NO CoT - 49 samples)...")
trainer.train()
print(f">>> Saving adapter to {output_dir}/final_adapter")
trainer.model.save_pretrained(f"{output_dir}/final_adapter")
tokenizer.save_pretrained(f"{output_dir}/final_adapter")
print(">>> Training Complete!")
