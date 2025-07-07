# PromptPilot
PromptForge is an interactive playground to craft, compare, and evaluate different prompting techniques for Large Language Models (LLMs) such as GPT-4, using real-time feedback, context engineering, and tool integration.
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_dataset(model_name="gpt2", split="train[:1%]", max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("yelp_review_full", split=split)

    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding="max_length", max_length=max_length)

    return dataset.map(tokenize, batched=True), tokenizer

# -----------------------------
# lora_finetune.py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from dataset_loader import load_and_tokenize_dataset

model_name = "gpt2"
dataset, tokenizer = load_and_tokenize_dataset(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./lora-gpt2",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=1e-4,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# -----------------------------
# qlora_finetune.py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from dataset_loader import load_and_tokenize_dataset

model_name = "meta-llama/Llama-2-7b-hf"
dataset, tokenizer = load_and_tokenize_dataset(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./qlora-llama2",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
