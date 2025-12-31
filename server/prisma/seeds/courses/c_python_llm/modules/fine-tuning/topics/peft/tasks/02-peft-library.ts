import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-peft-library-usage',
	title: 'PEFT Library',
	difficulty: 'medium',
	tags: ['peft', 'huggingface', 'lora'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# PEFT Library

Learn to use HuggingFace's PEFT library for parameter-efficient fine-tuning.

## Task

Implement functions to:
- Configure LoRA with PEFT
- Create a PEFT model from base model
- Save and load PEFT adapters
- Merge adapters for inference

## Example

\`\`\`python
from peft import LoraConfig, get_peft_model, PeftModel

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Create PEFT model
peft_model = get_peft_model(base_model, config)
print(f"Trainable: {peft_model.print_trainable_parameters()}")
\`\`\``,

	initialCode: `from peft import LoraConfig, get_peft_model, PeftModel, TaskType

def create_lora_config(rank: int = 16, alpha: int = 32,
                       target_modules: list = None,
                       task_type: str = "CAUSAL_LM") -> LoraConfig:
    """Create a LoRA configuration."""
    # Your code here
    pass

def apply_lora_to_model(model, config: LoraConfig):
    """Apply LoRA to a base model."""
    # Your code here
    pass

def get_trainable_params(model) -> dict:
    """Get count of trainable vs total parameters."""
    # Your code here
    pass

def save_peft_model(model, path: str):
    """Save only the PEFT adapter weights."""
    # Your code here
    pass

def load_peft_model(base_model, adapter_path: str):
    """Load PEFT adapter onto a base model."""
    # Your code here
    pass
`,

	solutionCode: `from peft import LoraConfig, get_peft_model, PeftModel, TaskType

def create_lora_config(rank: int = 16, alpha: int = 32,
                       target_modules: list = None,
                       task_type: str = "CAUSAL_LM") -> LoraConfig:
    """Create a LoRA configuration."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    task_type_enum = getattr(TaskType, task_type)

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=task_type_enum
    )

def apply_lora_to_model(model, config: LoraConfig):
    """Apply LoRA to a base model."""
    peft_model = get_peft_model(model, config)
    return peft_model

def get_trainable_params(model) -> dict:
    """Get count of trainable vs total parameters."""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return {
        "trainable": trainable_params,
        "total": all_params,
        "percentage": 100 * trainable_params / all_params
    }

def save_peft_model(model, path: str):
    """Save only the PEFT adapter weights."""
    model.save_pretrained(path)

def load_peft_model(base_model, adapter_path: str):
    """Load PEFT adapter onto a base model."""
    return PeftModel.from_pretrained(base_model, adapter_path)
`,

	testCode: `import unittest
from unittest.mock import MagicMock, patch

class TestPEFT(unittest.TestCase):
    def test_create_lora_config(self):
        config = create_lora_config(rank=8, alpha=16)
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)

    def test_create_lora_config_custom_modules(self):
        config = create_lora_config(target_modules=["query", "value"])
        self.assertEqual(config.target_modules, ["query", "value"])

    def test_get_trainable_params(self):
        mock_model = MagicMock()
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        param2 = MagicMock()
        param2.numel.return_value = 9000
        param2.requires_grad = False
        mock_model.named_parameters.return_value = [("p1", param1), ("p2", param2)]

        result = get_trainable_params(mock_model)
        self.assertEqual(result["trainable"], 1000)
        self.assertEqual(result["total"], 10000)
        self.assertAlmostEqual(result["percentage"], 10.0)

    @patch('peft.get_peft_model')
    def test_apply_lora_to_model(self, mock_get_peft):
        mock_get_peft.return_value = MagicMock()
        config = create_lora_config()
        mock_model = MagicMock()
        result = apply_lora_to_model(mock_model, config)
        mock_get_peft.assert_called_once()

    def test_create_config_returns_lora_config(self):
        config = create_lora_config()
        self.assertIsInstance(config, LoraConfig)

    def test_create_config_default_modules(self):
        config = create_lora_config()
        self.assertIsNotNone(config.target_modules)

    def test_trainable_params_returns_dict(self):
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        result = get_trainable_params(mock_model)
        self.assertIsInstance(result, dict)

    def test_trainable_params_has_percentage(self):
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 100
        param.requires_grad = True
        mock_model.named_parameters.return_value = [("p", param)]
        result = get_trainable_params(mock_model)
        self.assertIn("percentage", result)

    def test_create_config_has_dropout(self):
        config = create_lora_config()
        self.assertIsNotNone(config.lora_dropout)

    def test_create_config_has_bias_setting(self):
        config = create_lora_config()
        self.assertEqual(config.bias, "none")
`,

	hint1: 'Use TaskType enum for proper task configuration',
	hint2: 'PEFT models save only adapter weights, not the full model',

	whyItMatters: `PEFT makes fine-tuning accessible:

- **Standard interface**: Same API for LoRA, Prefix Tuning, etc.
- **HuggingFace integration**: Works with Trainer, accelerate
- **Small checkpoints**: Save only adapter weights (few MB)
- **Easy switching**: Load different adapters on same base

Essential for practical LLM customization.`,

	translations: {
		ru: {
			title: 'Библиотека PEFT',
			description: `# Библиотека PEFT

Научитесь использовать библиотеку PEFT от HuggingFace для parameter-efficient fine-tuning.

## Задача

Реализуйте функции для:
- Конфигурации LoRA с PEFT
- Создания PEFT модели из базовой модели
- Сохранения и загрузки PEFT адаптеров
- Слияния адаптеров для инференса

## Пример

\`\`\`python
from peft import LoraConfig, get_peft_model, PeftModel

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Create PEFT model
peft_model = get_peft_model(base_model, config)
print(f"Trainable: {peft_model.print_trainable_parameters()}")
\`\`\``,
			hint1: 'Используйте TaskType enum для правильной конфигурации задачи',
			hint2: 'PEFT модели сохраняют только веса адаптера, не всю модель',
			whyItMatters: `PEFT делает fine-tuning доступным:

- **Стандартный интерфейс**: Один API для LoRA, Prefix Tuning и др.
- **Интеграция с HuggingFace**: Работает с Trainer, accelerate
- **Маленькие чекпоинты**: Сохранение только весов адаптера (несколько МБ)
- **Легкое переключение**: Загрузка разных адаптеров на одну базу`,
		},
		uz: {
			title: 'PEFT kutubxonasi',
			description: `# PEFT kutubxonasi

Parameter-efficient fine-tuning uchun HuggingFace ning PEFT kutubxonasidan foydalanishni o'rganing.

## Topshiriq

Funksiyalarni amalga oshiring:
- PEFT bilan LoRA ni sozlash
- Bazaviy modeldan PEFT model yaratish
- PEFT adapterlarni saqlash va yuklash
- Inference uchun adapterlarni birlashtirish

## Misol

\`\`\`python
from peft import LoraConfig, get_peft_model, PeftModel

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Create PEFT model
peft_model = get_peft_model(base_model, config)
print(f"Trainable: {peft_model.print_trainable_parameters()}")
\`\`\``,
			hint1: "To'g'ri vazifa konfiguratsiyasi uchun TaskType enum dan foydalaning",
			hint2: "PEFT modellar faqat adapter og'irliklarini saqlaydi, butun modelni emas",
			whyItMatters: `PEFT fine-tuning ni oson qiladi:

- **Standart interfeys**: LoRA, Prefix Tuning va boshqalar uchun bir API
- **HuggingFace integratsiyasi**: Trainer, accelerate bilan ishlaydi
- **Kichik checkpointlar**: Faqat adapter og'irliklarini saqlash (bir necha MB)
- **Oson almashtirish**: Bir bazaga turli adapterlarni yuklash`,
		},
	},
};

export default task;
