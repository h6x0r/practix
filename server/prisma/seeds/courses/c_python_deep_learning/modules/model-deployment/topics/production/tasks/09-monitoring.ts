import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-model-monitoring',
	title: 'Model Monitoring',
	difficulty: 'medium',
	tags: ['pytorch', 'monitoring', 'mlops'],
	estimatedTime: '15m',
	isPremium: true,
	order: 9,
	description: `# Model Monitoring

Monitor model performance and detect issues in production.

## Task

Implement a \`ModelMonitor\` class that:
- Tracks prediction latency
- Monitors prediction distribution
- Detects data drift

## Example

\`\`\`python
monitor = ModelMonitor()

# Log predictions
monitor.log_prediction(input_features, prediction, latency_ms=15.2)

# Get metrics
metrics = monitor.get_metrics()
# {'avg_latency': 12.5, 'predictions_count': 1000, 'class_distribution': {...}}

# Check for drift
drift_detected = monitor.check_drift(new_data)
\`\`\``,

	initialCode: `import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List
import time

class ModelMonitor:
    """Monitor model predictions in production."""

    def __init__(self, window_size: int = 1000):
        # Your code here
        pass

    def log_prediction(self, features: np.ndarray, prediction: int,
                       latency_ms: float = None):
        """Log a single prediction."""
        # Your code here
        pass

    def get_metrics(self) -> Dict:
        """Get current monitoring metrics."""
        # Your code here
        pass

    def check_drift(self, reference_distribution: Dict,
                    threshold: float = 0.1) -> bool:
        """Check if prediction distribution has drifted."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List
import time

class ModelMonitor:
    """Monitor model predictions in production."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.features_mean = None
        self.features_std = None
        self.prediction_counts = defaultdict(int)
        self.total_predictions = 0

    def log_prediction(self, features: np.ndarray, prediction: int,
                       latency_ms: float = None):
        """Log a single prediction."""
        if latency_ms is not None:
            self.latencies.append(latency_ms)

        self.predictions.append(prediction)
        self.prediction_counts[prediction] += 1
        self.total_predictions += 1

        # Update feature statistics
        if self.features_mean is None:
            self.features_mean = features.copy()
            self.features_std = np.zeros_like(features)
        else:
            # Online mean/std update
            delta = features - self.features_mean
            self.features_mean += delta / self.total_predictions

    def get_metrics(self) -> Dict:
        """Get current monitoring metrics."""
        metrics = {
            'total_predictions': self.total_predictions,
            'window_size': len(self.predictions),
        }

        if self.latencies:
            latencies = list(self.latencies)
            metrics['avg_latency_ms'] = np.mean(latencies)
            metrics['p50_latency_ms'] = np.percentile(latencies, 50)
            metrics['p95_latency_ms'] = np.percentile(latencies, 95)
            metrics['p99_latency_ms'] = np.percentile(latencies, 99)

        if self.predictions:
            preds = list(self.predictions)
            unique, counts = np.unique(preds, return_counts=True)
            metrics['class_distribution'] = {
                int(k): int(v) / len(preds) for k, v in zip(unique, counts)
            }

        return metrics

    def check_drift(self, reference_distribution: Dict,
                    threshold: float = 0.1) -> bool:
        """Check if prediction distribution has drifted."""
        current_metrics = self.get_metrics()
        if 'class_distribution' not in current_metrics:
            return False

        current_dist = current_metrics['class_distribution']

        # Calculate distribution difference
        all_classes = set(reference_distribution.keys()) | set(current_dist.keys())
        total_diff = 0

        for cls in all_classes:
            ref_val = reference_distribution.get(cls, 0)
            cur_val = current_dist.get(cls, 0)
            total_diff += abs(ref_val - cur_val)

        return (total_diff / 2) > threshold
`,

	testCode: `import numpy as np
import unittest

class TestModelMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = ModelMonitor(window_size=100)

    def test_log_prediction(self):
        features = np.random.randn(10)
        self.monitor.log_prediction(features, 0, latency_ms=10.0)
        self.assertEqual(self.monitor.total_predictions, 1)

    def test_get_metrics(self):
        for i in range(50):
            features = np.random.randn(10)
            self.monitor.log_prediction(features, i % 5, latency_ms=10.0 + i)
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics['total_predictions'], 50)
        self.assertIn('avg_latency_ms', metrics)

    def test_check_drift(self):
        for _ in range(100):
            self.monitor.log_prediction(np.random.randn(10), 0)
        # Reference was balanced, now it's all class 0
        reference = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        self.assertTrue(self.monitor.check_drift(reference, threshold=0.5))

    def test_has_window_size(self):
        self.assertEqual(self.monitor.window_size, 100)

    def test_has_latencies(self):
        self.assertTrue(hasattr(self.monitor, 'latencies'))

    def test_has_predictions(self):
        self.assertTrue(hasattr(self.monitor, 'predictions'))

    def test_metrics_has_percentiles(self):
        for i in range(50):
            self.monitor.log_prediction(np.random.randn(10), 0, latency_ms=10+i)
        metrics = self.monitor.get_metrics()
        self.assertIn('p50_latency_ms', metrics)
        self.assertIn('p95_latency_ms', metrics)

    def test_no_drift_with_matching_distribution(self):
        for i in range(100):
            self.monitor.log_prediction(np.random.randn(10), i % 5)
        reference = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        self.assertFalse(self.monitor.check_drift(reference, threshold=0.1))

    def test_empty_metrics(self):
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics['total_predictions'], 0)

    def test_has_prediction_counts(self):
        self.assertTrue(hasattr(self.monitor, 'prediction_counts'))
`,

	hint1: 'Use deque with maxlen for sliding window',
	hint2: 'Track latency percentiles (p50, p95, p99) not just average',

	whyItMatters: `Monitoring is essential for production ML:

- **Performance tracking**: Catch degradation early
- **Drift detection**: Alert when data changes
- **SLA compliance**: Monitor latency guarantees
- **Debugging**: Understand model behavior

Without monitoring, you're flying blind in production.`,

	translations: {
		ru: {
			title: 'Мониторинг модели',
			description: `# Мониторинг модели

Мониторьте производительность модели и обнаруживайте проблемы в production.

## Задача

Реализуйте класс \`ModelMonitor\`, который:
- Отслеживает задержку предсказаний
- Мониторит распределение предсказаний
- Обнаруживает дрифт данных

## Пример

\`\`\`python
monitor = ModelMonitor()

# Log predictions
monitor.log_prediction(input_features, prediction, latency_ms=15.2)

# Get metrics
metrics = monitor.get_metrics()
# {'avg_latency': 12.5, 'predictions_count': 1000, 'class_distribution': {...}}

# Check for drift
drift_detected = monitor.check_drift(new_data)
\`\`\``,
			hint1: 'Используйте deque с maxlen для скользящего окна',
			hint2: 'Отслеживайте перцентили задержки (p50, p95, p99), а не только среднее',
			whyItMatters: `Мониторинг необходим для production ML:

- **Отслеживание производительности**: Раннее обнаружение деградации
- **Детекция дрифта**: Оповещение при изменении данных
- **Соответствие SLA**: Мониторинг гарантий задержки
- **Отладка**: Понимание поведения модели`,
		},
		uz: {
			title: 'Model monitoringi',
			description: `# Model monitoringi

Production da model ishlashini kuzating va muammolarni aniqlang.

## Topshiriq

\`ModelMonitor\` sinfini amalga oshiring:
- Bashorat kechikishini kuzatadi
- Bashorat taqsimotini monitoring qiladi
- Ma'lumotlar driftini aniqlaydi

## Misol

\`\`\`python
monitor = ModelMonitor()

# Log predictions
monitor.log_prediction(input_features, prediction, latency_ms=15.2)

# Get metrics
metrics = monitor.get_metrics()
# {'avg_latency': 12.5, 'predictions_count': 1000, 'class_distribution': {...}}

# Check for drift
drift_detected = monitor.check_drift(new_data)
\`\`\``,
			hint1: "Siljuvchi oyna uchun maxlen bilan deque dan foydalaning",
			hint2: "Faqat o'rtacha emas, kechikish persentillarini (p50, p95, p99) kuzating",
			whyItMatters: `Monitoring production ML uchun muhim:

- **Ishlash kuzatuvi**: Yomonlashuvni erta aniqlash
- **Drift aniqlash**: Ma'lumotlar o'zgarganda ogohlantirish
- **SLA muvofiqligi**: Kechikish kafolatlarini monitoring qilish
- **Debugging**: Model xatti-harakatini tushunish`,
		},
	},
};

export default task;
