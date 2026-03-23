# Graphene-Contact-LLM

**Small Data, Small Model, Superior Domain Accuracy:**
Fine-Tuning a 7B Language Model as a Condensed Matter Physics Expert with 64 Examples

## Key Result

| Model | Parameters | Condition | Correct (✓) | Partial (△) | Wrong (✗) | Accuracy |
|-------|-----------|-----------|-------------|-------------|-----------|----------|
| Base 7B | 7B | Open-book | 9 | 17 | 4 | 30% |
| **FT 7B** | **7B** | **Closed-book** | **22** | **5** | **3** | **73%** |
| Qwen3-235B | 235B | Open-book | 18 | 9 | 3 | 60% |

> A fine-tuned 7B model with no access to the source paper outperforms a 33× larger model that receives the paper's content as a prompt.

## Overview

This repository contains the complete codebase for fine-tuning and evaluating a domain-specific language model for condensed matter physics research. The model is trained on only **64 question–answer pairs** from a single research paper on graphene–metal contact resistance, using **QDoRA (Quantized Weight-Decomposed Low-Rank Adaptation)** with **Chain-of-Thought reasoning** on a single consumer GPU.


## Research Article
Small Data, Small Model, Superior Domain Accuracy: Fine-Tuning a 7B Language Model as a Condensed Matter Physics Expert with 64 Examples
(https://www.preprints.org/manuscript/202603.1691)


## Fine-Tuned Model

The final fine-tuned adapter is available on Hugging Face:

🤗 [pjspjs0987/Qwen2.5-7B-GrapheneContact-QDoRA-CoT](https://huggingface.co/pjspjs0987/Qwen2.5-7B-GrapheneContact-QDoRA-CoT)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-7B-Instruct |
| Adapter | QDoRA (`use_dora=True`) |
| Rank (r) / Alpha (α) | 64 / 128 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit NF4, double quantization, bfloat16 |
| Max Steps | 80 |
| Learning Rate | 5×10⁻⁴ |
| Effective Batch Size | 8 (batch=1, gradient accumulation=8) |
| Training Time | ~20 minutes |

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers peft bitsandbytes trl
```

### 2. Fine-Tune Your Own Model
```bash
python train_contact_resistance_v3.py
```

### 3. Run Interactive Chat
```bash
# Fine-tuned model
python chat_contact_resistance_v1.py

# Base model (for comparison)
python chat_base.py
```

### 4. Run Automated Evaluation
```bash
python auto_test_v3.py
```

### 5. Load from Hugging Face
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "pjspjs0987/Qwen2.5-7B-GrapheneContact-QDoRA-CoT"
)
model = PeftModel.from_pretrained(
    base_model,
    "pjspjs0987/Qwen2.5-7B-GrapheneContact-QDoRA-CoT"
)
```

## Ablation Study

Four fine-tuning configurations were compared using 49 QA pairs to isolate the effects of adapter type and Chain-of-Thought reasoning:

| Experiment | Adapter | CoT | Script |
|-----------|---------|-----|--------|
| Exp 1 | QDoRA | ✓ | `train_contact_resistance_v1.py` |
| Exp 2 | QDoRA | ✗ | `train_no_cot.py` |
| Exp 3 | QLoRA | ✓ | `train_qlora_cot.py` |
| Exp 4 | QLoRA | ✗ | `train_qlora_no_cot.py` |

Key findings: QDoRA provides slightly more stable performance than QLoRA, while CoT has minimal impact on accuracy. Data quality is the primary driver of model performance.

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| CPU | AMD Threadripper PRO 5975WX (32 cores) |
| RAM | 256 GB DDR4-3200 ECC |
| OS | Ubuntu |

## Citation
```bibtex
@article{park2026small,
  title={Small Data, Small Model, Superior Domain Accuracy: Fine-Tuning a 7B Language Model as a Condensed Matter Physics Expert with 64 Examples},
  author={Park, Junsu},
  year={2026},
  institution={Gwangju Institute of Science and Technology (GIST)}
}
```

## Author

**Junsu Park**
- Department of Physics and Photon Science, GIST (Gwangju Institute of Science and Technology), Korea
- Hugging Face: [pjspjs0987](https://huggingface.co/pjspjs0987)
- GitHub: [parkjeff27-art](https://github.com/parkjeff27-art)
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
