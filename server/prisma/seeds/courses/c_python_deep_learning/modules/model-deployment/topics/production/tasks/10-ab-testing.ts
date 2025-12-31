import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-ab-testing',
	title: 'A/B Testing Models',
	difficulty: 'medium',
	tags: ['pytorch', 'ab-testing', 'mlops'],
	estimatedTime: '15m',
	isPremium: true,
	order: 10,
	description: `# A/B Testing Models

Implement A/B testing to compare model versions in production.

## Task

Implement an \`ABTestManager\` class that:
- Routes traffic between model versions
- Tracks metrics per variant
- Determines winner based on statistical significance

## Example

\`\`\`python
ab_test = ABTestManager(
    model_a=model_v1,
    model_b=model_v2,
    traffic_split=0.5  # 50/50 split
)

# Route request
variant, prediction = ab_test.predict(features)
ab_test.log_outcome(variant, success=True)

# Get results
results = ab_test.get_results()
# {'variant_a': {'count': 500, 'success_rate': 0.92}, ...}
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import random

class ABTestManager:
    """Manage A/B testing between two model versions."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module,
                 traffic_split: float = 0.5):
        # Your code here
        pass

    def predict(self, features: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Route to a model and return (variant, prediction)."""
        # Your code here
        pass

    def log_outcome(self, variant: str, success: bool):
        """Log outcome for a variant."""
        # Your code here
        pass

    def get_results(self) -> Dict:
        """Get current A/B test results."""
        # Your code here
        pass

    def get_winner(self, confidence: float = 0.95) -> str:
        """Determine winner with statistical significance."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import random
from scipy import stats

class ABTestManager:
    """Manage A/B testing between two model versions."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module,
                 traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split

        self.model_a.eval()
        self.model_b.eval()

        self.results = {
            'A': {'count': 0, 'successes': 0},
            'B': {'count': 0, 'successes': 0}
        }

    def predict(self, features: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Route to a model and return (variant, prediction)."""
        with torch.no_grad():
            if random.random() < self.traffic_split:
                variant = 'A'
                prediction = self.model_a(features)
            else:
                variant = 'B'
                prediction = self.model_b(features)

        self.results[variant]['count'] += 1
        return variant, prediction

    def log_outcome(self, variant: str, success: bool):
        """Log outcome for a variant."""
        if success:
            self.results[variant]['successes'] += 1

    def get_results(self) -> Dict:
        """Get current A/B test results."""
        results = {}
        for variant in ['A', 'B']:
            count = self.results[variant]['count']
            successes = self.results[variant]['successes']
            results[f'variant_{variant.lower()}'] = {
                'count': count,
                'successes': successes,
                'success_rate': successes / count if count > 0 else 0
            }
        return results

    def get_winner(self, confidence: float = 0.95) -> str:
        """Determine winner with statistical significance."""
        a = self.results['A']
        b = self.results['B']

        if a['count'] < 30 or b['count'] < 30:
            return 'insufficient_data'

        # Two-proportion z-test
        p_a = a['successes'] / a['count']
        p_b = b['successes'] / b['count']

        p_pooled = (a['successes'] + b['successes']) / (a['count'] + b['count'])
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/a['count'] + 1/b['count']))

        if se == 0:
            return 'no_difference'

        z = (p_a - p_b) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        if p_value < (1 - confidence):
            return 'A' if p_a > p_b else 'B'
        return 'no_significant_difference'
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleModel(nn.Module):
    def __init__(self, bias=0):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.bias = bias
    def forward(self, x):
        return self.fc(x) + self.bias

class TestABTesting(unittest.TestCase):
    def setUp(self):
        self.model_a = SimpleModel(0)
        self.model_b = SimpleModel(1)
        self.ab_test = ABTestManager(self.model_a, self.model_b)

    def test_predict_returns_variant(self):
        features = torch.randn(1, 10)
        variant, pred = self.ab_test.predict(features)
        self.assertIn(variant, ['A', 'B'])

    def test_log_outcome(self):
        features = torch.randn(1, 10)
        variant, _ = self.ab_test.predict(features)
        self.ab_test.log_outcome(variant, success=True)
        self.assertEqual(self.ab_test.results[variant]['successes'], 1)

    def test_get_results(self):
        for _ in range(10):
            features = torch.randn(1, 10)
            variant, _ = self.ab_test.predict(features)
            self.ab_test.log_outcome(variant, success=True)
        results = self.ab_test.get_results()
        self.assertIn('variant_a', results)
        self.assertIn('variant_b', results)

    def test_has_models(self):
        self.assertTrue(hasattr(self.ab_test, 'model_a'))
        self.assertTrue(hasattr(self.ab_test, 'model_b'))

    def test_has_traffic_split(self):
        self.assertEqual(self.ab_test.traffic_split, 0.5)

    def test_has_results(self):
        self.assertTrue(hasattr(self.ab_test, 'results'))
        self.assertIn('A', self.ab_test.results)
        self.assertIn('B', self.ab_test.results)

    def test_prediction_returns_tensor(self):
        features = torch.randn(1, 10)
        variant, pred = self.ab_test.predict(features)
        self.assertIsInstance(pred, torch.Tensor)

    def test_get_winner_insufficient(self):
        winner = self.ab_test.get_winner()
        self.assertEqual(winner, 'insufficient_data')

    def test_results_have_success_rate(self):
        for _ in range(10):
            features = torch.randn(1, 10)
            variant, _ = self.ab_test.predict(features)
            self.ab_test.log_outcome(variant, success=True)
        results = self.ab_test.get_results()
        self.assertIn('success_rate', results['variant_a'])

    def test_models_in_eval_mode(self):
        self.assertFalse(self.ab_test.model_a.training)
        self.assertFalse(self.ab_test.model_b.training)
`,

	hint1: 'Use random.random() < split for traffic routing',
	hint2: 'Use two-proportion z-test for statistical significance',

	whyItMatters: `A/B testing is critical for safe model deployment:

- **Risk mitigation**: Test new models on subset of traffic
- **Data-driven decisions**: Statistical proof of improvement
- **Continuous improvement**: Always be testing
- **Rollback safety**: Easy to route all traffic back

A/B testing is how top companies deploy ML safely.`,

	translations: {
		ru: {
			title: 'A/B тестирование моделей',
			description: `# A/B тестирование моделей

Реализуйте A/B тестирование для сравнения версий моделей в production.

## Задача

Реализуйте класс \`ABTestManager\`, который:
- Распределяет трафик между версиями моделей
- Отслеживает метрики по вариантам
- Определяет победителя на основе статистической значимости

## Пример

\`\`\`python
ab_test = ABTestManager(
    model_a=model_v1,
    model_b=model_v2,
    traffic_split=0.5  # 50/50 split
)

# Route request
variant, prediction = ab_test.predict(features)
ab_test.log_outcome(variant, success=True)

# Get results
results = ab_test.get_results()
# {'variant_a': {'count': 500, 'success_rate': 0.92}, ...}
\`\`\``,
			hint1: 'Используйте random.random() < split для маршрутизации трафика',
			hint2: 'Используйте z-тест для двух пропорций для статистической значимости',
			whyItMatters: `A/B тестирование критично для безопасного развертывания:

- **Снижение рисков**: Тест новых моделей на части трафика
- **Решения на основе данных**: Статистическое доказательство улучшения
- **Непрерывное улучшение**: Всегда тестировать
- **Безопасный откат**: Легко вернуть весь трафик назад`,
		},
		uz: {
			title: 'A/B modellarni test qilish',
			description: `# A/B modellarni test qilish

Production da model versiyalarini solishtirish uchun A/B testni amalga oshiring.

## Topshiriq

\`ABTestManager\` sinfini amalga oshiring:
- Model versiyalari o'rtasida trafikni yo'naltiradi
- Variant bo'yicha metrikalarni kuzatadi
- Statistik ahamiyatga asoslangan g'olibni aniqlaydi

## Misol

\`\`\`python
ab_test = ABTestManager(
    model_a=model_v1,
    model_b=model_v2,
    traffic_split=0.5  # 50/50 split
)

# Route request
variant, prediction = ab_test.predict(features)
ab_test.log_outcome(variant, success=True)

# Get results
results = ab_test.get_results()
# {'variant_a': {'count': 500, 'success_rate': 0.92}, ...}
\`\`\``,
			hint1: "Trafikni yo'naltirish uchun random.random() < split dan foydalaning",
			hint2: "Statistik ahamiyat uchun ikki nisbat z-testidan foydalaning",
			whyItMatters: `A/B test xavfsiz model joylashtirish uchun muhim:

- **Xavfni kamaytirish**: Yangi modellarni trafik qismida test qilish
- **Ma'lumotlarga asoslangan qarorlar**: Yaxshilanishning statistik isboti
- **Doimiy yaxshilash**: Har doim test qilish
- **Xavfsiz orqaga qaytarish**: Barcha trafikni osongina qaytarish`,
		},
	},
};

export default task;
