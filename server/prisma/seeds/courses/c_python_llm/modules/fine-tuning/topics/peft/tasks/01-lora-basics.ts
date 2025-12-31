import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-lora-basics',
	title: 'LoRA Fundamentals',
	difficulty: 'medium',
	tags: ['lora', 'peft', 'fine-tuning'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# LoRA Fundamentals

Understand and implement Low-Rank Adaptation (LoRA) for efficient fine-tuning.

## What is LoRA?

LoRA adds small trainable matrices to frozen pretrained weights:
- Original: W (frozen)
- LoRA: W + BA where B and A are low-rank matrices
- Only train B and A (~0.1% of parameters)

## Key Concepts

- **Rank (r)**: Size of low-rank matrices (4, 8, 16, 32)
- **Alpha**: Scaling factor for LoRA weights
- **Target modules**: Which layers to adapt (query, value, etc.)

## Example

\`\`\`python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        # Your code here
        pass

class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with original + LoRA weights."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        # x @ A.T @ B.T = x @ (B @ A).T
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scaling

class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha
        )

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with original + LoRA weights."""
        return self.linear(x) + self.lora(x)

    def merge_weights(self):
        """Merge LoRA weights into original for inference."""
        with torch.no_grad():
            merged = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            self.linear.weight.data += merged.data
        return self.linear
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestLoRA(unittest.TestCase):
    def test_lora_layer_shape(self):
        lora = LoRALayer(in_features=768, out_features=768, rank=8)
        x = torch.randn(2, 10, 768)
        output = lora(x)
        self.assertEqual(output.shape, (2, 10, 768))

    def test_lora_initialization(self):
        lora = LoRALayer(in_features=256, out_features=256, rank=4)
        # B should be initialized to zeros
        self.assertTrue(torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B)))

    def test_linear_with_lora(self):
        linear = nn.Linear(512, 512)
        lora_linear = LinearWithLoRA(linear, rank=8)
        x = torch.randn(2, 512)
        output = lora_linear(x)
        self.assertEqual(output.shape, (2, 512))

    def test_frozen_weights(self):
        linear = nn.Linear(256, 256)
        lora_linear = LinearWithLoRA(linear, rank=4)
        for param in lora_linear.linear.parameters():
            self.assertFalse(param.requires_grad)

    def test_lora_layer_is_nn_module(self):
        lora = LoRALayer(64, 64)
        self.assertIsInstance(lora, nn.Module)

    def test_linear_with_lora_is_nn_module(self):
        linear = nn.Linear(64, 64)
        lora_linear = LinearWithLoRA(linear)
        self.assertIsInstance(lora_linear, nn.Module)

    def test_lora_has_scaling(self):
        lora = LoRALayer(64, 64, rank=4, alpha=8)
        self.assertEqual(lora.scaling, 2.0)

    def test_lora_has_rank(self):
        lora = LoRALayer(64, 64, rank=16)
        self.assertEqual(lora.rank, 16)

    def test_lora_output_not_nan(self):
        lora = LoRALayer(128, 128, rank=8)
        x = torch.randn(2, 128)
        output = lora(x)
        self.assertFalse(torch.isnan(output).any())

    def test_linear_with_lora_has_lora_attribute(self):
        linear = nn.Linear(64, 64)
        lora_linear = LinearWithLoRA(linear)
        self.assertTrue(hasattr(lora_linear, 'lora'))
`,

	hint1: 'Initialize B with zeros so LoRA starts as identity',
	hint2: 'Scaling factor is alpha/rank to normalize the contribution',

	whyItMatters: `LoRA revolutionized LLM fine-tuning:

- **Memory efficient**: Train ~0.1% of parameters
- **No quality loss**: Matches full fine-tuning performance
- **Modular**: Swap LoRA adapters for different tasks
- **Composable**: Combine multiple LoRAs

Used in Stable Diffusion, LLaMA fine-tuning, and production LLMs.`,

	translations: {
		ru: {
			title: 'Основы LoRA',
			description: `# Основы LoRA

Понимание и реализация Low-Rank Adaptation (LoRA) для эффективного fine-tuning.

## Что такое LoRA?

LoRA добавляет маленькие обучаемые матрицы к замороженным весам:
- Оригинал: W (заморожен)
- LoRA: W + BA где B и A - низкоранговые матрицы
- Обучаем только B и A (~0.1% параметров)

## Пример

\`\`\`python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
\`\`\``,
			hint1: 'Инициализируйте B нулями, чтобы LoRA начинался как identity',
			hint2: 'Scaling factor = alpha/rank для нормализации вклада',
			whyItMatters: `LoRA революционизировал fine-tuning LLM:

- **Эффективность памяти**: Обучение ~0.1% параметров
- **Без потери качества**: Соответствует полному fine-tuning
- **Модульность**: Меняйте LoRA адаптеры для разных задач
- **Композируемость**: Комбинируйте несколько LoRA`,
		},
		uz: {
			title: 'LoRA asoslari',
			description: `# LoRA asoslari

Samarali fine-tuning uchun Low-Rank Adaptation (LoRA) ni tushunish va amalga oshirish.

## LoRA nima?

LoRA muzlatilgan og'irliklarga kichik o'qitiladigan matritsalar qo'shadi:
- Asl: W (muzlatilgan)
- LoRA: W + BA bu yerda B va A past-rang matritsalar
- Faqat B va A ni o'qitamiz (~0.1% parametrlar)

## Misol

\`\`\`python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
\`\`\``,
			hint1: "LoRA identity sifatida boshlashi uchun B ni nollar bilan ishga tushiring",
			hint2: "Scaling factor = alpha/rank hissani normallash uchun",
			whyItMatters: `LoRA LLM fine-tuning ni inqilob qildi:

- **Xotira samaradorligi**: ~0.1% parametrlarni o'qitish
- **Sifat yo'qotilmaydi**: To'liq fine-tuning ga mos
- **Modulli**: Turli vazifalar uchun LoRA adapterlarni almashtirish
- **Kompozitsiya**: Bir nechta LoRA larni birlashtirish`,
		},
	},
};

export default task;
