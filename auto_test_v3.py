import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./_qwen25_7b_contact_resistance_v3/final_adapter"

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

system_prompt = "You are a domain-specific physics assistant specialized in graphene-metal interfaces and contact resistance. Provide detailed, physics-grounded explanations based on experimental evidence from TLCD/TECD comparative studies. Use step-by-step reasoning. Always respond in English only."

questions = [
    "Describe the physical structure of TLCD and TECD devices.",
    "Why does increasing contact area reduce contact resistance in graphene-metal interfaces?",
    "What were the quantitative contact resistance values from four-probe measurements for TLCD and TECD?",
    "Why is the BTH model necessary in addition to the Landauer model for graphene-metal contacts?",
    "What causes the sub-linear relationship between contact area and contact resistance reduction?",
    "How does specific contact resistivity (ρc) differ from edge resistance (Rgg') in carrier transport?",
    "Why does gate voltage modulate contact resistance in both TLCD and TECD devices?",
    "Describe the metal-on-bottom fabrication process used in this study.",
    "Why can't edge current crowding alone explain the lower contact resistance in TECD?",
    "What is the key novelty of comparing TLCD and TECD structures?",
    "What are the key geometric differences between TLCD and TECD in terms of contact length and contact area?",
    "How does the electrode pad design in TECD ensure continuous graphene coverage over the entire metal surface?",
    "How does the transfer length (LT) relate to the saturation behavior of contact resistance with increasing contact area?",
    "What is the role of the tunneling matrix element |Mmg|² in the BTH model for graphene-metal contacts?",
    "Why does the Landauer model alone fail to explain the contact resistance difference between TLCD and TECD?",
    "How does wavefunction overlap between metal and graphene states influence specific contact resistivity (ρc)?",
    "What is the physical meaning of the distributed contact resistance model used to explain the sub-linear Rc reduction?",
    "Why does current crowding occur preferentially at the contact edge in TLCD devices?",
    "How do Rgg' and ρc contribute differently to total contact resistance in TLCD versus TECD?",
    "What physical mechanism explains why a 625× area increase yields only a 3.9× reduction in contact resistance?",
    "What channel lengths were used in the TLM measurements, and how was 2Rc extracted from the data?",
    "How does the four-probe contact resistance compare between TLCD and TECD?",
    "What substrate and dielectric layer were used for device fabrication, and what was the SiO₂ thickness?",
    "How do the two-probe and four-probe methods independently confirm the contact area effect on Rc?",
    "Why was the metal-on-bottom process chosen over the conventional metal-on-top approach?",
    "What interfacial contamination issues does the metal-on-bottom process avoid compared to metal-on-top structures?",
    "What role does reactive ion etching (RIE) play in defining the channel geometry, and how is the pad region protected?",
    "What does the conclusion suggest about extending this contact area engineering approach to other applications?",
    "Why is this study considered the first experimental verification of area-dependent contact resistance in graphene?",
    "How does the comparison between TLCD and TECD provide evidence that distributed charge transfer governs contact resistance?",
]

results = []

for i, q in enumerate(questions):
    print(f"[{i+1}/30] {q[:60]}...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results.append({"question_number": i+1, "question": q, "answer": response})
    print(f"  Done.\n")

output_file = f"/home/jsp/test_results_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Also save readable text version
txt_file = output_file.replace(".json", ".txt")
with open(txt_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"Q{r['question_number']}: {r['question']}\n")
        f.write(f"A: {r['answer']}\n")
        f.write("-" * 80 + "\n\n")

print(f"\nAll 30 questions complete!")
print(f"JSON: {output_file}")
print(f"TXT:  {txt_file}")
