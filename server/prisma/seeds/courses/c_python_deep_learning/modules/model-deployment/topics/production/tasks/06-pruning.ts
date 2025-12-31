import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-model-pruning',
	title: 'Model Pruning',
	difficulty: 'hard',
	tags: ['pytorch', 'pruning', 'optimization'],
	estimatedTime: '18m',
	isPremium: true,
	order: 6,
	description: `# Model Pruning

Remove unnecessary weights to create smaller, faster models.

## Task

Implement functions for model pruning:
1. \`prune_model\` - Apply magnitude-based pruning
2. \`get_sparsity\` - Calculate model sparsity (% of zero weights)
3. \`remove_pruning\` - Make pruning permanent

## Example

\`\`\`python
model = SimpleModel()

# Prune 50% of weights
pruned = prune_model(model, amount=0.5)
sparsity = get_sparsity(pruned)  # ~0.5

# Make permanent
finalized = remove_pruning(pruned)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply magnitude-based pruning to all Linear layers."""
    # Your code here
    pass

def get_sparsity(model: nn.Module) -> float:
    """Calculate overall model sparsity (fraction of zero weights)."""
    # Your code here
    pass

def remove_pruning(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing reparametrization."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply magnitude-based pruning to all Linear layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def get_sparsity(model: nn.Module) -> float:
    """Calculate overall model sparsity (fraction of zero weights)."""
    total_zeros = 0
    total_elements = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            total_zeros += (param == 0).sum().item()
            total_elements += param.numel()

    return total_zeros / total_elements if total_elements > 0 else 0.0

def remove_pruning(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing reparametrization."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                # Not pruned, skip
                pass
    return model
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class TestPruning(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()

    def test_prune_model(self):
        pruned = prune_model(self.model, amount=0.5)
        self.assertIsNotNone(pruned)

    def test_sparsity_after_pruning(self):
        pruned = prune_model(self.model, amount=0.5)
        sparsity = get_sparsity(pruned)
        self.assertGreater(sparsity, 0.3)  # Should be around 0.5

    def test_remove_pruning(self):
        pruned = prune_model(self.model, amount=0.5)
        finalized = remove_pruning(pruned)
        # Should still work
        x = torch.randn(4, 100)
        out = finalized(x)
        self.assertEqual(out.shape, (4, 10))

    def test_prune_returns_module(self):
        pruned = prune_model(self.model, amount=0.3)
        self.assertIsInstance(pruned, nn.Module)

    def test_sparsity_returns_float(self):
        pruned = prune_model(self.model, amount=0.5)
        sparsity = get_sparsity(pruned)
        self.assertIsInstance(sparsity, float)

    def test_sparsity_in_range(self):
        pruned = prune_model(self.model, amount=0.5)
        sparsity = get_sparsity(pruned)
        self.assertTrue(0 <= sparsity <= 1)

    def test_different_prune_amounts(self):
        for amount in [0.2, 0.4, 0.6]:
            model = SimpleModel()
            pruned = prune_model(model, amount=amount)
            sparsity = get_sparsity(pruned)
            self.assertGreater(sparsity, 0)

    def test_remove_preserves_sparsity(self):
        pruned = prune_model(self.model, amount=0.5)
        sparsity_before = get_sparsity(pruned)
        finalized = remove_pruning(pruned)
        sparsity_after = get_sparsity(finalized)
        self.assertAlmostEqual(sparsity_before, sparsity_after, places=2)

    def test_model_still_trainable(self):
        pruned = prune_model(self.model, amount=0.3)
        x = torch.randn(4, 100)
        out = pruned(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(self.model.fc1.weight.grad)

    def test_zero_pruning(self):
        model = SimpleModel()
        initial_sparsity = get_sparsity(model)
        self.assertAlmostEqual(initial_sparsity, 0.0, places=2)
`,

	hint1: 'Use prune.l1_unstructured for magnitude-based pruning',
	hint2: 'Count zeros with (param == 0).sum()',

	whyItMatters: `Pruning creates efficient models:

- **Smaller models**: Remove redundant weights
- **Faster inference**: Sparse operations can be optimized
- **Lottery ticket**: Find smaller equally-accurate networks
- **Hardware support**: Specialized sparse accelerators

Pruning is key to deploying large models efficiently.`,

	translations: {
		ru: {
			title: 'Прунинг модели',
			description: `# Прунинг модели

Удалите ненужные веса для создания меньших и быстрых моделей.

## Задача

Реализуйте функции для прунинга модели:
1. \`prune_model\` - Применение прунинга по величине
2. \`get_sparsity\` - Расчет разреженности модели (% нулевых весов)
3. \`remove_pruning\` - Сделать прунинг постоянным

## Пример

\`\`\`python
model = SimpleModel()

# Prune 50% of weights
pruned = prune_model(model, amount=0.5)
sparsity = get_sparsity(pruned)  # ~0.5

# Make permanent
finalized = remove_pruning(pruned)
\`\`\``,
			hint1: 'Используйте prune.l1_unstructured для прунинга по величине',
			hint2: 'Подсчитайте нули с (param == 0).sum()',
			whyItMatters: `Прунинг создает эффективные модели:

- **Меньшие модели**: Удаление избыточных весов
- **Быстрый инференс**: Разреженные операции можно оптимизировать
- **Lottery ticket**: Поиск меньших равноточных сетей
- **Аппаратная поддержка**: Специализированные разреженные ускорители`,
		},
		uz: {
			title: 'Model pruning',
			description: `# Model pruning

Kichikroq va tezroq modellar yaratish uchun keraksiz vaznlarni olib tashlang.

## Topshiriq

Model pruning uchun funksiyalarni amalga oshiring:
1. \`prune_model\` - Kattalikga asoslangan pruningni qo'llash
2. \`get_sparsity\` - Model siyrakligini hisoblash (nol vaznlar %)
3. \`remove_pruning\` - Pruningni doimiy qilish

## Misol

\`\`\`python
model = SimpleModel()

# Prune 50% of weights
pruned = prune_model(model, amount=0.5)
sparsity = get_sparsity(pruned)  # ~0.5

# Make permanent
finalized = remove_pruning(pruned)
\`\`\``,
			hint1: "Kattalikka asoslangan pruning uchun prune.l1_unstructured dan foydalaning",
			hint2: "Nollarni (param == 0).sum() bilan hisoblang",
			whyItMatters: `Pruning samarali modellar yaratadi:

- **Kichikroq modellar**: Ortiqcha vaznlarni olib tashlash
- **Tezroq inference**: Siyrak operatsiyalarni optimallashtirish mumkin
- **Lottery ticket**: Kichikroq bir xil aniqlikdagi tarmoqlarni topish
- **Apparat qo'llab-quvvatlash**: Maxsus siyrak tezlatgichlar`,
		},
	},
};

export default task;
