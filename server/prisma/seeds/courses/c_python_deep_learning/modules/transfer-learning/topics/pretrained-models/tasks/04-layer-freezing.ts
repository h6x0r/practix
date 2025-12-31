import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-layer-freezing',
	title: 'Layer Freezing Strategies',
	difficulty: 'medium',
	tags: ['pytorch', 'transfer-learning', 'freezing'],
	estimatedTime: '15m',
	isPremium: true,
	order: 4,
	description: `# Layer Freezing Strategies

Implement different layer freezing strategies for transfer learning.

## Task

Implement three functions:
1. \`freeze_all_but_last_n\` - Freeze all layers except last N
2. \`freeze_by_name\` - Freeze layers matching pattern
3. \`get_trainable_params\` - Return count of trainable parameters

## Example

\`\`\`python
model = models.resnet18()

freeze_all_but_last_n(model, n=2)
# Only last 2 layers are trainable

freeze_by_name(model, patterns=['layer4', 'fc'])
# Only layer4 and fc are trainable

count = get_trainable_params(model)
# Returns number of trainable parameters
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from typing import List

def freeze_all_but_last_n(model: nn.Module, n: int = 1):
    """Freeze all layers except the last n layers."""
    # Your code here
    pass

def freeze_by_name(model: nn.Module, patterns: List[str]):
    """Unfreeze only layers whose names contain any of the patterns."""
    # Your code here
    pass

def get_trainable_params(model: nn.Module) -> int:
    """Return count of trainable parameters."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
from typing import List

def freeze_all_but_last_n(model: nn.Module, n: int = 1):
    """Freeze all layers except the last n layers."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Get all named children
    children = list(model.named_children())

    # Unfreeze last n
    for name, child in children[-n:]:
        for param in child.parameters():
            param.requires_grad = True

def freeze_by_name(model: nn.Module, patterns: List[str]):
    """Unfreeze only layers whose names contain any of the patterns."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze matching layers
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True

def get_trainable_params(model: nn.Module) -> int:
    """Return count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
`,

	testCode: `import torch
import torch.nn as nn
from torchvision import models
import unittest

class TestFreezing(unittest.TestCase):
    def test_freeze_all_but_last_n(self):
        model = models.resnet18(weights=None)
        freeze_all_but_last_n(model, n=1)
        # Only fc should be trainable
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertFalse(model.conv1.weight.requires_grad)

    def test_freeze_by_name(self):
        model = models.resnet18(weights=None)
        freeze_by_name(model, ['fc'])
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertFalse(model.conv1.weight.requires_grad)

    def test_get_trainable_params(self):
        model = models.resnet18(weights=None)
        total = get_trainable_params(model)
        freeze_all_but_last_n(model, n=1)
        after_freeze = get_trainable_params(model)
        self.assertLess(after_freeze, total)

    def test_trainable_params_positive(self):
        model = models.resnet18(weights=None)
        count = get_trainable_params(model)
        self.assertGreater(count, 0)

    def test_freeze_all_but_last_2(self):
        model = models.resnet18(weights=None)
        freeze_all_but_last_n(model, n=2)
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertTrue(model.layer4[1].conv2.weight.requires_grad)

    def test_freeze_by_multiple_patterns(self):
        model = models.resnet18(weights=None)
        freeze_by_name(model, ['layer4', 'fc'])
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertFalse(model.layer3[0].conv1.weight.requires_grad)

    def test_freeze_reduces_trainable(self):
        model = models.resnet18(weights=None)
        before = get_trainable_params(model)
        freeze_by_name(model, ['fc'])
        after = get_trainable_params(model)
        self.assertLess(after, before)

    def test_returns_integer(self):
        model = models.resnet18(weights=None)
        count = get_trainable_params(model)
        self.assertIsInstance(count, int)

    def test_freeze_all_but_last_n_layer4(self):
        model = models.resnet18(weights=None)
        freeze_all_but_last_n(model, n=2)
        self.assertFalse(model.layer2[0].conv1.weight.requires_grad)

    def test_freeze_by_empty_patterns(self):
        model = models.resnet18(weights=None)
        freeze_by_name(model, [])
        count = get_trainable_params(model)
        self.assertEqual(count, 0)
`,

	hint1: 'Use model.named_children() to iterate over top-level modules',
	hint2: 'Use model.named_parameters() to freeze/unfreeze by name',

	whyItMatters: `Strategic layer freezing is key to effective transfer learning:

- **Lower layers**: Generic features (edges, textures) - usually keep frozen
- **Higher layers**: Task-specific features - often fine-tune
- **Gradual unfreezing**: Start frozen, unfreeze progressively
- **Computation savings**: Fewer trainable params = faster training

Different freezing strategies work better for different dataset sizes.`,

	translations: {
		ru: {
			title: 'Стратегии заморозки слоев',
			description: `# Стратегии заморозки слоев

Реализуйте различные стратегии заморозки слоев для transfer learning.

## Задача

Реализуйте три функции:
1. \`freeze_all_but_last_n\` - Заморозить все кроме последних N слоев
2. \`freeze_by_name\` - Заморозить слои по паттерну имени
3. \`get_trainable_params\` - Вернуть число обучаемых параметров

## Пример

\`\`\`python
model = models.resnet18()

freeze_all_but_last_n(model, n=2)
# Only last 2 layers are trainable

freeze_by_name(model, patterns=['layer4', 'fc'])
# Only layer4 and fc are trainable

count = get_trainable_params(model)
# Returns number of trainable parameters
\`\`\``,
			hint1: 'Используйте model.named_children() для итерации по модулям',
			hint2: 'Используйте model.named_parameters() для заморозки по имени',
			whyItMatters: `Стратегическая заморозка ключевая для transfer learning:

- **Нижние слои**: Общие признаки (края, текстуры) - обычно заморожены
- **Верхние слои**: Специфичные признаки - часто дообучаются
- **Постепенная разморозка**: Начать замороженным, размораживать постепенно
- **Экономия вычислений**: Меньше параметров = быстрее обучение`,
		},
		uz: {
			title: "Qatlamlarni muzlatish strategiyalari",
			description: `# Qatlamlarni muzlatish strategiyalari

Transfer learning uchun turli qatlam muzlatish strategiyalarini amalga oshiring.

## Topshiriq

Uchta funksiya amalga oshiring:
1. \`freeze_all_but_last_n\` - Oxirgi N qatlamdan tashqari hammasini muzlatish
2. \`freeze_by_name\` - Naqshga mos qatlamlarni muzlatish
3. \`get_trainable_params\` - O'rgatish mumkin bo'lgan parametrlar sonini qaytarish

## Misol

\`\`\`python
model = models.resnet18()

freeze_all_but_last_n(model, n=2)
# Only last 2 layers are trainable

freeze_by_name(model, patterns=['layer4', 'fc'])
# Only layer4 and fc are trainable

count = get_trainable_params(model)
# Returns number of trainable parameters
\`\`\``,
			hint1: "Yuqori darajadagi modullar ustida iteratsiya uchun model.named_children() dan foydalaning",
			hint2: "Nom bo'yicha muzlatish uchun model.named_parameters() dan foydalaning",
			whyItMatters: `Strategik qatlam muzlatish samarali transfer learning uchun kalit:

- **Pastki qatlamlar**: Umumiy xususiyatlar (qirralar, teksturalar) - odatda muzlatilgan
- **Yuqori qatlamlar**: Vazifaga xos xususiyatlar - ko'pincha nozik sozlanadi
- **Bosqichma-bosqich muzdan chiqarish**: Muzlatilgan holda boshlash, asta-sekin chiqarish
- **Hisoblash tejash**: Kamroq parametrlar = tezroq o'qitish`,
		},
	},
};

export default task;
