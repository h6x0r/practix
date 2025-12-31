import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-peft-training-loop',
	title: 'PEFT Training Loop',
	difficulty: 'hard',
	tags: ['training', 'peft', 'huggingface'],
	estimatedTime: '30m',
	isPremium: true,
	order: 4,
	description: `# PEFT Training Loop

Implement a complete training loop for PEFT models using HuggingFace Trainer.

## Task

Create a training pipeline that:
- Prepares dataset for instruction fine-tuning
- Configures training arguments
- Uses HuggingFace Trainer with PEFT
- Implements proper evaluation

## Dataset Format

\`\`\`json
{
  "instruction": "Write a poem about AI",
  "input": "",
  "output": "Silicon minds..."
}
\`\`\`

## Example

\`\`\`python
trainer = create_trainer(
    model=peft_model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    output_dir="./output"
)

trainer.train()
\`\`\``,

	initialCode: `import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

def format_instruction(sample: dict) -> str:
    """Format a sample into instruction format."""
    # Your code here
    pass

def prepare_dataset(data: list, tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for training."""
    # Your code here
    pass

def create_training_args(output_dir: str, epochs: int = 3,
                         batch_size: int = 4,
                         learning_rate: float = 2e-4) -> TrainingArguments:
    """Create training arguments."""
    # Your code here
    pass

def create_trainer(model, tokenizer, train_dataset, eval_dataset,
                   training_args) -> Trainer:
    """Create a Trainer for PEFT model."""
    # Your code here
    pass
`,

	solutionCode: `import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

def format_instruction(sample: dict) -> str:
    """Format a sample into instruction format."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return prompt

def prepare_dataset(data: list, tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for training."""
    def tokenize_function(examples):
        # Format instructions
        texts = [format_instruction({"instruction": inst, "input": inp, "output": out})
                 for inst, inp, out in zip(
                     examples["instruction"],
                     examples["input"],
                     examples["output"]
                 )]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # Labels are same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

def create_training_args(output_dir: str, epochs: int = 3,
                         batch_size: int = 4,
                         learning_rate: float = 2e-4) -> TrainingArguments:
    """Create training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

def create_trainer(model, tokenizer, train_dataset, eval_dataset,
                   training_args) -> Trainer:
    """Create a Trainer for PEFT model."""
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    return trainer
`,

	testCode: `import unittest
from unittest.mock import MagicMock, patch

class TestPEFTTraining(unittest.TestCase):
    def test_format_instruction_basic(self):
        sample = {
            "instruction": "Translate to French",
            "input": "",
            "output": "Bonjour"
        }
        result = format_instruction(sample)
        self.assertIn("Translate to French", result)
        self.assertIn("Bonjour", result)

    def test_format_instruction_with_input(self):
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        }
        result = format_instruction(sample)
        self.assertIn("### Input:", result)
        self.assertIn("Hello", result)

    def test_create_training_args(self):
        args = create_training_args("./output", epochs=5, batch_size=8)
        self.assertEqual(args.num_train_epochs, 5)
        self.assertEqual(args.per_device_train_batch_size, 8)

    def test_create_training_args_learning_rate(self):
        args = create_training_args("./output", learning_rate=1e-4)
        self.assertEqual(args.learning_rate, 1e-4)

    def test_format_instruction_returns_string(self):
        sample = {"instruction": "Test", "input": "", "output": "Out"}
        result = format_instruction(sample)
        self.assertIsInstance(result, str)

    def test_format_includes_response_marker(self):
        sample = {"instruction": "Test", "input": "", "output": "Out"}
        result = format_instruction(sample)
        self.assertIn("### Response:", result)

    def test_format_includes_instruction_marker(self):
        sample = {"instruction": "Test", "input": "", "output": "Out"}
        result = format_instruction(sample)
        self.assertIn("### Instruction:", result)

    def test_training_args_has_fp16(self):
        args = create_training_args("./output")
        self.assertTrue(args.fp16)

    def test_training_args_has_output_dir(self):
        args = create_training_args("./my_output")
        self.assertEqual(args.output_dir, "./my_output")

    def test_training_args_returns_training_arguments(self):
        args = create_training_args("./output")
        self.assertIsInstance(args, TrainingArguments)
`,

	hint1: 'Use gradient_accumulation_steps to simulate larger batch sizes',
	hint2: 'Labels should equal input_ids for causal language modeling',

	whyItMatters: `Proper training setup is critical for success:

- **Instruction format**: Consistent prompts improve learning
- **Gradient accumulation**: Train with larger effective batch size
- **Mixed precision**: 2x speedup with fp16
- **Checkpointing**: Resume from failures

These techniques are standard in production fine-tuning.`,

	translations: {
		ru: {
			title: 'Цикл обучения PEFT',
			description: `# Цикл обучения PEFT

Реализуйте полный цикл обучения для PEFT моделей с использованием HuggingFace Trainer.

## Задача

Создайте pipeline обучения:
- Подготовка датасета для instruction fine-tuning
- Конфигурация аргументов обучения
- Использование HuggingFace Trainer с PEFT
- Правильная evaluation

## Пример

\`\`\`python
trainer = create_trainer(
    model=peft_model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    output_dir="./output"
)

trainer.train()
\`\`\``,
			hint1: 'Используйте gradient_accumulation_steps для симуляции больших батчей',
			hint2: 'Labels должны равняться input_ids для causal language modeling',
			whyItMatters: `Правильная настройка обучения критична для успеха:

- **Instruction формат**: Консистентные промпты улучшают обучение
- **Gradient accumulation**: Обучение с большим эффективным batch size
- **Mixed precision**: 2x ускорение с fp16
- **Checkpointing**: Продолжение после сбоев`,
		},
		uz: {
			title: "PEFT o'qitish sikli",
			description: `# PEFT o'qitish sikli

HuggingFace Trainer yordamida PEFT modellar uchun to'liq o'qitish siklini amalga oshiring.

## Topshiriq

O'qitish pipeline yarating:
- Instruction fine-tuning uchun datasetni tayyorlash
- O'qitish argumentlarini sozlash
- PEFT bilan HuggingFace Trainer dan foydalanish
- To'g'ri evaluation

## Misol

\`\`\`python
trainer = create_trainer(
    model=peft_model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    output_dir="./output"
)

trainer.train()
\`\`\``,
			hint1: "Kattaroq batch hajmlarini simulyatsiya qilish uchun gradient_accumulation_steps dan foydalaning",
			hint2: "Causal language modeling uchun Labels input_ids ga teng bo'lishi kerak",
			whyItMatters: `To'g'ri o'qitish sozlamalari muvaffaqiyat uchun muhim:

- **Instruction formati**: Izchil promptlar o'rganishni yaxshilaydi
- **Gradient accumulation**: Katta samarali batch hajmi bilan o'qitish
- **Mixed precision**: fp16 bilan 2x tezlashtirish
- **Checkpointing**: Nosozliklardan davom ettirish`,
		},
	},
};

export default task;
