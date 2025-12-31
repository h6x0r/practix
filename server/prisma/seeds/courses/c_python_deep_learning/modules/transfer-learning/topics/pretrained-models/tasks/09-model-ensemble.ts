import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-model-ensemble',
	title: 'Model Ensembling',
	difficulty: 'medium',
	tags: ['pytorch', 'transfer-learning', 'ensemble'],
	estimatedTime: '15m',
	isPremium: false,
	order: 9,
	description: `# Model Ensembling

Combine multiple models for better predictions.

## Task

Implement an \`EnsembleModel\` class that:
- Combines predictions from multiple models
- Supports voting and averaging strategies
- Works with models of different architectures

## Example

\`\`\`python
models = [resnet18, resnet34, vgg16]
ensemble = EnsembleModel(models, strategy='average')

x = torch.randn(4, 3, 224, 224)
output = ensemble(x)
# Averaged predictions from all models
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""

    def __init__(self, models: List[nn.Module], strategy: str = 'average'):
        super().__init__()
        # TODO: Store models in nn.ModuleList and save strategy
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine predictions from all models."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""

    def __init__(self, models: List[nn.Module], strategy: str = 'average'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine predictions from all models."""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        # Stack predictions
        stacked = torch.stack(predictions, dim=0)

        if self.strategy == 'average':
            # Average logits
            output = stacked.mean(dim=0)
        elif self.strategy == 'vote':
            # Majority voting
            probs = F.softmax(stacked, dim=-1)
            votes = probs.argmax(dim=-1)  # (n_models, batch)
            # Count votes for each class
            output = torch.zeros_like(predictions[0])
            for i in range(x.size(0)):
                vote_counts = votes[:, i].bincount(minlength=output.size(-1))
                output[i] = vote_counts.float()
        elif self.strategy == 'soft_vote':
            # Average probabilities
            probs = F.softmax(stacked, dim=-1)
            output = probs.mean(dim=0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return output
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleModel(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.offset = offset
    def forward(self, x):
        return self.fc(x) + self.offset

class TestEnsemble(unittest.TestCase):
    def test_average_strategy(self):
        models = [SimpleModel(0), SimpleModel(1), SimpleModel(2)]
        ensemble = EnsembleModel(models, strategy='average')
        x = torch.randn(4, 10)
        out = ensemble(x)
        self.assertEqual(out.shape, (4, 5))

    def test_vote_strategy(self):
        models = [SimpleModel(i) for i in range(3)]
        ensemble = EnsembleModel(models, strategy='vote')
        x = torch.randn(2, 10)
        out = ensemble(x)
        self.assertEqual(out.shape, (2, 5))

    def test_soft_vote(self):
        models = [SimpleModel(i) for i in range(3)]
        ensemble = EnsembleModel(models, strategy='soft_vote')
        x = torch.randn(2, 10)
        out = ensemble(x)
        # Soft vote probabilities should sum to 1
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_is_nn_module(self):
        models = [SimpleModel(0)]
        ensemble = EnsembleModel(models)
        self.assertIsInstance(ensemble, nn.Module)

    def test_has_models(self):
        models = [SimpleModel(i) for i in range(3)]
        ensemble = EnsembleModel(models)
        self.assertTrue(hasattr(ensemble, 'models'))
        self.assertEqual(len(ensemble.models), 3)

    def test_single_model(self):
        models = [SimpleModel(0)]
        ensemble = EnsembleModel(models)
        x = torch.randn(2, 10)
        out = ensemble(x)
        self.assertEqual(out.shape, (2, 5))

    def test_output_not_nan(self):
        models = [SimpleModel(i) for i in range(3)]
        ensemble = EnsembleModel(models)
        x = torch.randn(2, 10)
        out = ensemble(x)
        self.assertFalse(torch.isnan(out).any())

    def test_strategy_stored(self):
        models = [SimpleModel(0)]
        ensemble = EnsembleModel(models, strategy='vote')
        self.assertEqual(ensemble.strategy, 'vote')

    def test_different_batch_sizes(self):
        models = [SimpleModel(i) for i in range(3)]
        ensemble = EnsembleModel(models)
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10)
            out = ensemble(x)
            self.assertEqual(out.shape[0], batch_size)

    def test_models_is_module_list(self):
        models_list = [SimpleModel(i) for i in range(3)]
        ensemble = EnsembleModel(models_list)
        self.assertIsInstance(ensemble.models, nn.ModuleList)
`,

	hint1: 'Use nn.ModuleList to properly register models',
	hint2: 'Soft voting: average softmax probabilities across models',

	whyItMatters: `Ensembling improves model performance:

- **Reduced variance**: Multiple models average out errors
- **Better generalization**: Different models capture different patterns
- **Competition winning**: Top solutions often use ensembles
- **Robustness**: Less sensitive to individual model failures

Ensembles are a reliable way to boost accuracy.`,

	translations: {
		ru: {
			title: 'Ансамблирование моделей',
			description: `# Ансамблирование моделей

Комбинируйте несколько моделей для лучших предсказаний.

## Задача

Реализуйте класс \`EnsembleModel\`, который:
- Комбинирует предсказания нескольких моделей
- Поддерживает стратегии голосования и усреднения
- Работает с моделями разных архитектур

## Пример

\`\`\`python
models = [resnet18, resnet34, vgg16]
ensemble = EnsembleModel(models, strategy='average')

x = torch.randn(4, 3, 224, 224)
output = ensemble(x)
# Averaged predictions from all models
\`\`\``,
			hint1: 'Используйте nn.ModuleList для правильной регистрации моделей',
			hint2: 'Soft voting: усреднение softmax вероятностей между моделями',
			whyItMatters: `Ансамблирование улучшает производительность моделей:

- **Уменьшение дисперсии**: Несколько моделей усредняют ошибки
- **Лучшее обобщение**: Разные модели захватывают разные паттерны
- **Победы в соревнованиях**: Топ решения часто используют ансамбли
- **Робастность**: Меньше чувствительность к сбоям отдельных моделей`,
		},
		uz: {
			title: 'Model ansambllashtirish',
			description: `# Model ansambllashtirish

Yaxshiroq bashoratlar uchun bir nechta modellarni birlashtiring.

## Topshiriq

\`EnsembleModel\` sinfini amalga oshiring:
- Bir nechta modellarning bashoratlarini birlashtiradi
- Ovoz berish va o'rtacha olish strategiyalarini qo'llab-quvvatlaydi
- Turli arxitekturadagi modellar bilan ishlaydi

## Misol

\`\`\`python
models = [resnet18, resnet34, vgg16]
ensemble = EnsembleModel(models, strategy='average')

x = torch.randn(4, 3, 224, 224)
output = ensemble(x)
# Averaged predictions from all models
\`\`\``,
			hint1: "Modellarni to'g'ri ro'yxatdan o'tkazish uchun nn.ModuleList dan foydalaning",
			hint2: "Soft voting: modellar bo'ylab softmax ehtimolliklarini o'rtacha olish",
			whyItMatters: `Ansambllashtirish model ish faoliyatini yaxshilaydi:

- **Kamaytirilgan dispersiya**: Bir nechta modellar xatolarni o'rtacha oladi
- **Yaxshiroq umumlashtirish**: Turli modellar turli naqshlarni ushlaydi
- **Musobaqa g'alabasi**: Top yechimlar ko'pincha ansambllardan foydalanadi
- **Mustahkamlik**: Alohida model nosozliklariga kamroq sezgir`,
		},
	},
};

export default task;
