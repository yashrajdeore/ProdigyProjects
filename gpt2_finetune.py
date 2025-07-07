
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

# 1. Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. Load and tokenize dataset
dataset = load_dataset('text', data_files='data.txt')

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Group texts
block_size = 128
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size

    result = {}
    for k in concatenated.keys():
        result[k] = [
            concatenated[k][i : i + block_size]
            for i in range(0, total_length, block_size)
        ]

    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# 4. Training
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"]
)

trainer.train()

# 5. Save model
model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")

# 6. Generate text
generator = pipeline("text-generation", model="gpt2-finetuned", tokenizer="gpt2-finetuned")
output = generator("In the future,", max_length=50, num_return_sequences=1)
print(output[0]['generated_text'])
