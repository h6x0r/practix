import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-qlora-quantization',
	title: 'QLoRA Training',
	difficulty: 'hard',
	tags: ['qlora', 'quantization', 'bitsandbytes'],
	estimatedTime: '25m',
	isPremium: true,
	order: 3,
	description: `# QLoRA Training

Implement QLoRA for fine-tuning large models with limited GPU memory.

## What is QLoRA?

QLoRA combines:
- **4-bit quantization**: Reduce model size 4x
- **LoRA adapters**: Train small adapters in full precision
- **Paged optimizers**: Handle memory spikes

## Benefits

- Fine-tune 65B models on single 48GB GPU
- Fine-tune 7B models on consumer GPUs (16GB)
- No quality degradation vs full fine-tuning

## Example

\`\`\`python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
\`\`\``,

	initialCode: `import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def create_bnb_config(load_in_4bit: bool = True,
                      compute_dtype=torch.float16) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    # Your code here
    pass

def load_quantized_model(model_name: str, bnb_config: BitsAndBytesConfig):
    """Load a model with quantization."""
    # Your code here
    pass

def prepare_for_training(model):
    """Prepare quantized model for training."""
    # Your code here
    pass

def create_qlora_model(model_name: str, lora_rank: int = 16,
                       lora_alpha: int = 32) -> tuple:
    """Create a complete QLoRA model ready for training."""
    # Your code here
    pass
`,

	solutionCode: `import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def create_bnb_config(load_in_4bit: bool = True,
                      compute_dtype=torch.float16) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",  # normalized float 4
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True  # nested quantization
    )

def load_quantized_model(model_name: str, bnb_config: BitsAndBytesConfig):
    """Load a model with quantization."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model

def prepare_for_training(model):
    """Prepare quantized model for training."""
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model

def create_qlora_model(model_name: str, lora_rank: int = 16,
                       lora_alpha: int = 32) -> tuple:
    """Create a complete QLoRA model ready for training."""
    # Create quantization config
    bnb_config = create_bnb_config()

    # Load quantized model
    model = load_quantized_model(model_name, bnb_config)

    # Prepare for training
    model = prepare_for_training(model)

    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    return model, lora_config
`,

	testCode: `import torch
import unittest
from unittest.mock import MagicMock, patch

class TestQLoRA(unittest.TestCase):
    def test_create_bnb_config(self):
        config = create_bnb_config()
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")

    def test_create_bnb_config_custom_dtype(self):
        config = create_bnb_config(compute_dtype=torch.bfloat16)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.bfloat16)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_quantized_model(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        config = create_bnb_config()
        model = load_quantized_model("test-model", config)
        mock_from_pretrained.assert_called_once()
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["quantization_config"], config)

    @patch('peft.prepare_model_for_kbit_training')
    def test_prepare_for_training(self, mock_prepare):
        mock_model = MagicMock()
        mock_prepare.return_value = mock_model
        result = prepare_for_training(mock_model)
        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_bnb_config_returns_config_object(self):
        config = create_bnb_config()
        self.assertIsInstance(config, BitsAndBytesConfig)

    def test_bnb_config_has_double_quant(self):
        config = create_bnb_config()
        self.assertTrue(config.bnb_4bit_use_double_quant)

    def test_bnb_config_default_compute_dtype(self):
        config = create_bnb_config()
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)

    def test_bnb_config_8bit_option(self):
        config = create_bnb_config(load_in_4bit=False)
        self.assertFalse(config.load_in_4bit)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_quantized_uses_auto_device_map(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        config = create_bnb_config()
        load_quantized_model("test", config)
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["device_map"], "auto")

    @patch('peft.prepare_model_for_kbit_training')
    def test_prepare_calls_kbit_training(self, mock_prepare):
        mock_model = MagicMock()
        mock_prepare.return_value = mock_model
        prepare_for_training(mock_model)
        mock_prepare.assert_called_once()
`,

	hint1: 'Use nf4 quantization type for best quality',
	hint2: 'Enable double quantization for additional memory savings',

	whyItMatters: `QLoRA democratized LLM fine-tuning:

- **Consumer hardware**: Fine-tune 7B on RTX 3090
- **Same quality**: Matches 16-bit fine-tuning results
- **65B models**: Possible on single A100 80GB
- **Cost reduction**: 10x cheaper cloud training

Made LLM customization accessible to everyone.`,

	translations: {
		ru: {
			title: 'Обучение QLoRA',
			description: `# Обучение QLoRA

Реализуйте QLoRA для fine-tuning больших моделей с ограниченной GPU памятью.

## Что такое QLoRA?

QLoRA комбинирует:
- **4-bit квантизация**: Уменьшение размера модели в 4 раза
- **LoRA адаптеры**: Обучение маленьких адаптеров в полной точности
- **Paged optimizers**: Обработка пиков памяти

## Пример

\`\`\`python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
\`\`\``,
			hint1: 'Используйте nf4 тип квантизации для лучшего качества',
			hint2: 'Включите double quantization для дополнительной экономии памяти',
			whyItMatters: `QLoRA демократизировал fine-tuning LLM:

- **Потребительское железо**: Fine-tune 7B на RTX 3090
- **Такое же качество**: Соответствует 16-bit fine-tuning
- **Модели 65B**: Возможно на одном A100 80GB
- **Снижение стоимости**: 10x дешевле облачное обучение`,
		},
		uz: {
			title: "QLoRA o'qitish",
			description: `# QLoRA o'qitish

Cheklangan GPU xotirasi bilan katta modellarni fine-tuning qilish uchun QLoRA ni amalga oshiring.

## QLoRA nima?

QLoRA birlashtiradi:
- **4-bit kvantizatsiya**: Model hajmini 4 barobar kamaytirish
- **LoRA adapterlar**: Kichik adapterlarni to'liq aniqlikda o'qitish
- **Paged optimizers**: Xotira cho'qqilarini boshqarish

## Misol

\`\`\`python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
\`\`\``,
			hint1: "Eng yaxshi sifat uchun nf4 kvantizatsiya turidan foydalaning",
			hint2: "Qo'shimcha xotira tejash uchun double quantization ni yoqing",
			whyItMatters: `QLoRA LLM fine-tuning ni demokratlashtirdi:

- **Iste'molchi uskunasi**: RTX 3090 da 7B ni fine-tune qilish
- **Bir xil sifat**: 16-bit fine-tuning natijalariga mos
- **65B modellar**: Bitta A100 80GB da mumkin
- **Xarajat kamayishi**: 10x arzonroq bulutli o'qitish`,
		},
	},
};

export default task;
