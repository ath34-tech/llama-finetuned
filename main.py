import json
import re
import os
import random
from typing import List, Dict, Any

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ============================
# OPTIONAL: Add HF Token
# ============================
# Either set env var or run: huggingface-cli login
os.environ.setdefault("HF_TOKEN", "YOUR_TOKEN")  # <-- replace if needed

# ============================
# MODEL/OUTPUT CONFIG
# ============================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./upsc_evaluator_finetuned"
HAS_CUDA = torch.cuda.is_available()


# =====================================================================
# -----------------------   DATA PROCESSOR   ---------------------------
# =====================================================================

class UPSCDataProcessor:
    """Extract and clean UPSC answer data"""

    def __init__(self):
        self.underline_patterns = [
            r'\[UNDERLINED:\s*["\']?(.*?)["\']?\]',
            r'\[UNDERLINED:\s*(.*?)\]'
        ]

    def extract_underlined_text(self, text: str) -> List[str]:
        underlined = []
        for pattern in self.underline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            underlined.extend([m.strip() for m in matches if m.strip()])

        seen = set()
        uniq = []
        for x in underlined:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(x)
        return uniq

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[✓✗✔✘⊕⊗○●◯◉]', '', text)
        return text.strip()

    def extract_question(self, question_field: str) -> str:
        q = re.sub(r'^\d+\.\s*', '', question_field)
        q = re.sub(r'\[QUESTION\]\s*', '', q)
        q = re.sub(r'\[.*?\]', '', q)
        questions = [s.strip() for s in q.split("\n") if s.strip() and len(s.strip()) > 20]
        return questions[0] if questions else question_field.strip()

    def process_raw_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed = []
        for idx, item in enumerate(raw_data):
            try:
                question = self.extract_question(item.get("question", ""))
                answer = item.get("answer", "")
                max_marks = item.get("max_marks", 10)

                if len(answer) < 50 or not question:
                    continue

                underlined = self.extract_underlined_text(answer)
                clean_answer = self.clean_text(answer)

                processed.append({
                    "id": f"answer_{idx+1}",
                    "question": question,
                    "answer": clean_answer,
                    "key_concepts_highlighted": underlined,
                    "max_marks": max_marks
                })
            except Exception as e:
                print("Error processing:", idx, e)
                continue
        return processed


# =====================================================================
# -----------------------   RATING GENERATOR   ------------------------
# =====================================================================

class RatingGenerator:
    def __init__(self):
        self.parameters = [
            "relevancy", "usefulness", "depth_of_content",
            "conceptual_clarity", "structure_organization",
            "use_of_examples", "critical_analysis"
        ]

    def generate_coherent_ratings(self, answer_length: int, num_concepts: int):
        if answer_length > 800 and num_concepts >= 5:
            base = random.randint(3, 5)
        elif answer_length > 400 and num_concepts >= 3:
            base = random.randint(2, 4)
        else:
            base = random.randint(1, 3)

        ratings = {}
        for p in self.parameters:
            variance = random.randint(-1, 1)
            score = max(1, min(5, base + variance))
            ratings[p] = score
        return ratings

    def calculate_total_score(self, ratings, max_marks):
        avg = sum(ratings.values()) / len(ratings)
        perc_map = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.80, 5: 0.95}
        p = perc_map[round(avg)]
        return round(p * max_marks, 1)

    def generate_evaluation(self, answer, concepts, max_marks):
        ratings = self.generate_coherent_ratings(len(answer), len(concepts))
        total_score = self.calculate_total_score(ratings, max_marks)
        feedback = self._generate_feedback(ratings, concepts)
        return {
            "ratings": ratings,
            "total_score": total_score,
            "max_marks": max_marks,
            "feedback": feedback,
            "concepts_identified": len(concepts)
        }

    def _generate_feedback(self, ratings, concepts):
        fb = []
        avg = sum(ratings.values()) / len(ratings)
        if avg >= 4:
            fb.append("Excellent answer with comprehensive coverage.")
        elif avg >= 3:
            fb.append("Good answer with adequate coverage.")
        else:
            fb.append("Answer needs improvement in several areas.")

        strengths = [k for k, v in ratings.items() if v >= 4]
        weaknesses = [k for k, v in ratings.items() if v <= 2]
        if strengths:
            fb.append("Strong in: " + ", ".join(strengths).replace("_"," ") + ".")
        if weaknesses:
            fb.append("Needs improvement in: " + ", ".join(weaknesses).replace("_"," ") + ".")
        if len(concepts) >= 5:
            fb.append("Good use of key concepts.")
        elif len(concepts) >= 2:
            fb.append("Some key concepts covered.")
        else:
            fb.append("Limited coverage of key concepts.")
        return " ".join(fb)


# =====================================================================
# -----------------------   PROMPT FORMATTER   ------------------------
# =====================================================================

def format_training_prompt(item):
    eval_data = item["evaluation"]
    ratings_str = "\n".join(
        [f"- {k.replace('_',' ').title()}: {v}/5 stars" for k, v in eval_data["ratings"].items()]
    )
    return f"""
You are an expert UPSC evaluator.

### Question:
{item['question']}

### Student Answer:
{item['answer']}

### Key Concepts:
{', '.join(item['key_concepts_highlighted']) if item['key_concepts_highlighted'] else 'None'}

### Ratings:
{ratings_str}

### Total Score:
{eval_data['total_score']}/{eval_data['max_marks']}

### Feedback:
{eval_data['feedback']}
""".strip()


# =====================================================================
# -----------------------   DATASET CREATION   ------------------------
# =====================================================================

def create_training_dataset(raw_json_data):
    print("Processing raw data...")
    processor = UPSCDataProcessor()
    processed = processor.process_raw_data(raw_json_data)
    print("✓", len(processed), "answers processed")

    rating_gen = RatingGenerator()
    for item in processed:
        item["evaluation"] = rating_gen.generate_evaluation(
            item["answer"],
            item["key_concepts_highlighted"],
            item["max_marks"]
        )

    formatted = [{"text": format_training_prompt(i)} for i in processed]
    ds = Dataset.from_pandas(pd.DataFrame(formatted))
    print(ds)
    return ds, processed


# =====================================================================
# ----------------------------- MODEL SETUP ----------------------------
# =====================================================================

def setup_model_and_tokenizer():
    """
    Windows + RTX 3060 (CUDA 12.7 drivers). Use FP16 on GPU (no bitsandbytes/4-bit).
    """
    print("➡ Loading model in FP16 (Windows + RTX 3060 compatible)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
        device_map="auto" if HAS_CUDA else None,
        trust_remote_code=True
    )
    return model, tokenizer


def setup_lora_config():
    # Standard FP16 LoRA config
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
    )


def setup_training_args():
    # Batch size 1 recommended for 6GB VRAM; use gradient accumulation
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=HAS_CUDA,  # FP16 only if GPU present
        save_steps=50,
        logging_steps=20,
        save_total_limit=2,
        report_to="none"
    )


# =====================================================================
# ----------------------------- TRAINING -------------------------------
# =====================================================================

def train_model(raw_data):
    dataset, processed = create_training_dataset(raw_data)

    with open("processed_training_data.json", "w", encoding="utf8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print("Loading model...")
    model, tokenizer = setup_model_and_tokenizer()

    print("Applying LoRA...")
    lora_cfg = setup_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = setup_training_args()

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=args,
        dataset_text_field="text",   # ✅ REQUIRED FIX
    )

    trainer.train()

    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")
    return model, tokenizer, processed


# =====================================================================
# ------------------------- EVALUATION FUNCTION ------------------------
# =====================================================================

def evaluate_new_answer(model, tokenizer, question, answer, max_marks=10):
    prompt = f"""
You are an expert UPSC evaluator.

### Question:
{question}

### Answer:
{answer}

### Maximum Marks: {max_marks}

### Give evaluation:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt")
    if HAS_CUDA:
        inputs = inputs.to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# =====================================================================
# ------------------------------- MAIN --------------------------------
# =====================================================================

SAMPLE_DATA = [
    {
        "question": "3. Just when the caterpillar thought the world was over, it became a butterfly.",
        "answer": """The Samkhya school talks of [UNDERLINED: purusha] and [UNDERLINED: prakriti].
        The [UNDERLINED: end is never really the end], symbolised by the
        [UNDERLINED: caterpillar transforming into a butterfly]. Our blind
        [UNDERLINED: ignorance] is the cocoon we must break using [UNDERLINED: meditation].""",
        "max_marks": 10
    }
]

def main():
    print("CUDA Available:", HAS_CUDA)
    model, tokenizer, processed = train_model(SAMPLE_DATA)

    q = "Discuss the importance of resilience in facing life’s challenges."
    a = "Resilience allows people to bounce back from difficulties and adapt positively."
    print("\nGenerated Evaluation:\n")
    print(evaluate_new_answer(model, tokenizer, q, a))


if __name__ == "__main__":
    main()
