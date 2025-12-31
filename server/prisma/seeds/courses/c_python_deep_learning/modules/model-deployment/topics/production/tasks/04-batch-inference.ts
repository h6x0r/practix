import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-batch-inference',
	title: 'Batch Inference',
	difficulty: 'medium',
	tags: ['pytorch', 'inference', 'batch'],
	estimatedTime: '15m',
	isPremium: true,
	order: 4,
	description: `# Batch Inference

Implement efficient batch inference for high-throughput predictions.

## Task

Implement a \`BatchPredictor\` class that:
- Processes large datasets in batches
- Uses DataLoader for efficient loading
- Supports GPU acceleration

## Example

\`\`\`python
predictor = BatchPredictor(model, batch_size=64, device='cuda')

# Predict on large dataset
dataset = TensorDataset(torch.randn(10000, 10))
predictions = predictor.predict_dataset(dataset)
# predictions.shape = (10000,)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List
import numpy as np

class BatchPredictor:
    """Efficient batch inference for large datasets."""

    def __init__(self, model: nn.Module, batch_size: int = 64,
                 device: str = 'cpu', num_workers: int = 4):
        # Your code here
        pass

    def predict_dataset(self, dataset: Dataset) -> np.ndarray:
        """Predict on entire dataset. Return numpy array of predictions."""
        # Your code here
        pass

    def predict_with_probs(self, dataset: Dataset) -> tuple:
        """Return (predictions, probabilities) for dataset."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List
import numpy as np

class BatchPredictor:
    """Efficient batch inference for large datasets."""

    def __init__(self, model: nn.Module, batch_size: int = 64,
                 device: str = 'cpu', num_workers: int = 4):
        self.model = model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers

    def predict_dataset(self, dataset: Dataset) -> np.ndarray:
        """Predict on entire dataset. Return numpy array of predictions."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )

        predictions = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(dim=-1)
                predictions.append(preds.cpu().numpy())

        return np.concatenate(predictions)

    def predict_with_probs(self, dataset: Dataset) -> tuple:
        """Return (predictions, probabilities) for dataset."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                outputs = self.model(x)
                probs = F.softmax(outputs, dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_preds), np.concatenate(all_probs)
`,

	testCode: `import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

class TestBatchPredictor(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.dataset = TensorDataset(torch.randn(100, 10))

    def test_predict_dataset(self):
        predictor = BatchPredictor(self.model, batch_size=16, num_workers=0)
        preds = predictor.predict_dataset(self.dataset)
        self.assertEqual(len(preds), 100)

    def test_predict_with_probs(self):
        predictor = BatchPredictor(self.model, batch_size=32, num_workers=0)
        preds, probs = predictor.predict_with_probs(self.dataset)
        self.assertEqual(len(preds), 100)
        self.assertEqual(probs.shape, (100, 5))
        # Probs should sum to 1
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_predictions_are_integers(self):
        predictor = BatchPredictor(self.model, batch_size=16, num_workers=0)
        preds = predictor.predict_dataset(self.dataset)
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in preds))

    def test_predictions_in_valid_range(self):
        predictor = BatchPredictor(self.model, batch_size=16, num_workers=0)
        preds = predictor.predict_dataset(self.dataset)
        self.assertTrue(all(0 <= p < 5 for p in preds))

    def test_different_batch_sizes(self):
        for bs in [8, 16, 32]:
            predictor = BatchPredictor(self.model, batch_size=bs, num_workers=0)
            preds = predictor.predict_dataset(self.dataset)
            self.assertEqual(len(preds), 100)

    def test_probs_range(self):
        predictor = BatchPredictor(self.model, batch_size=32, num_workers=0)
        _, probs = predictor.predict_with_probs(self.dataset)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

    def test_small_dataset(self):
        small_ds = TensorDataset(torch.randn(5, 10))
        predictor = BatchPredictor(self.model, batch_size=16, num_workers=0)
        preds = predictor.predict_dataset(small_ds)
        self.assertEqual(len(preds), 5)

    def test_predictor_stores_model(self):
        predictor = BatchPredictor(self.model, batch_size=16, num_workers=0)
        self.assertIsNotNone(predictor.model)

    def test_predictor_has_batch_size(self):
        predictor = BatchPredictor(self.model, batch_size=64, num_workers=0)
        self.assertEqual(predictor.batch_size, 64)

    def test_returns_numpy_array(self):
        predictor = BatchPredictor(self.model, batch_size=16, num_workers=0)
        preds = predictor.predict_dataset(self.dataset)
        self.assertIsInstance(preds, np.ndarray)
`,

	hint1: 'Use pin_memory=True with CUDA for faster transfers',
	hint2: 'Handle both labeled (x, y) and unlabeled (x,) datasets',

	whyItMatters: `Batch inference is essential for production:

- **Throughput**: Process millions of samples efficiently
- **GPU utilization**: Batch operations maximize parallelism
- **Memory management**: Process in chunks to fit in memory
- **Cost efficiency**: Reduce cloud compute costs

Batch inference is critical for offline prediction pipelines.`,

	translations: {
		ru: {
			title: 'Батч инференс',
			description: `# Батч инференс

Реализуйте эффективный batch inference для высокопропускных предсказаний.

## Задача

Реализуйте класс \`BatchPredictor\`, который:
- Обрабатывает большие датасеты батчами
- Использует DataLoader для эффективной загрузки
- Поддерживает GPU ускорение

## Пример

\`\`\`python
predictor = BatchPredictor(model, batch_size=64, device='cuda')

# Predict on large dataset
dataset = TensorDataset(torch.randn(10000, 10))
predictions = predictor.predict_dataset(dataset)
# predictions.shape = (10000,)
\`\`\``,
			hint1: 'Используйте pin_memory=True с CUDA для быстрых трансферов',
			hint2: 'Обрабатывайте как размеченные (x, y), так и неразмеченные (x,) датасеты',
			whyItMatters: `Batch inference необходим для production:

- **Пропускная способность**: Эффективная обработка миллионов примеров
- **Утилизация GPU**: Батч операции максимизируют параллелизм
- **Управление памятью**: Обработка чанками вмещается в память
- **Экономия**: Снижение затрат на облачные вычисления`,
		},
		uz: {
			title: 'Batch inference',
			description: `# Batch inference

Yuqori o'tkazuvchanlikdagi bashoratlar uchun samarali batch inference ni amalga oshiring.

## Topshiriq

\`BatchPredictor\` sinfini amalga oshiring:
- Katta ma'lumotlar to'plamlarini batchlarda qayta ishlaydi
- Samarali yuklash uchun DataLoader dan foydalanadi
- GPU tezlashtirishni qo'llab-quvvatlaydi

## Misol

\`\`\`python
predictor = BatchPredictor(model, batch_size=64, device='cuda')

# Predict on large dataset
dataset = TensorDataset(torch.randn(10000, 10))
predictions = predictor.predict_dataset(dataset)
# predictions.shape = (10000,)
\`\`\``,
			hint1: "Tezroq uzatishlar uchun CUDA bilan pin_memory=True dan foydalaning",
			hint2: "Belgilangan (x, y) va belgilanmagan (x,) ma'lumotlar to'plamlarini boshqaring",
			whyItMatters: `Batch inference production uchun muhim:

- **O'tkazuvchanlik**: Millionlab namunalarni samarali qayta ishlash
- **GPU foydalanish**: Batch operatsiyalar parallelizmni maksimal qiladi
- **Xotira boshqaruvi**: Xotiraga sig'ish uchun bo'laklarda qayta ishlash
- **Xarajat samaradorligi**: Bulutli hisoblash xarajatlarini kamaytirish`,
		},
	},
};

export default task;
