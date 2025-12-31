import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-model-evaluation',
	title: 'Model Evaluation',
	difficulty: 'easy',
	tags: ['pytorch', 'evaluation', 'metrics'],
	estimatedTime: '10m',
	isPremium: false,
	order: 11,
	description: `# Model Evaluation

Learn to properly evaluate PyTorch models.

## Task

Implement three functions:
1. \`evaluate_classification(model, dataloader, device)\` - Return accuracy and loss
2. \`get_predictions(model, dataloader, device)\` - Get all predictions
3. \`confusion_matrix_data(y_true, y_pred, num_classes)\` - Compute confusion matrix

## Example

\`\`\`python
model.eval()  # Set to evaluation mode

accuracy, loss = evaluate_classification(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.2%}, Loss: {loss:.4f}")

predictions = get_predictions(model, test_loader, device)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import numpy as np

def evaluate_classification(model: nn.Module, dataloader, device: str) -> tuple:
    """Evaluate model on dataloader. Return (accuracy, avg_loss)."""
    # Your code here
    pass

def get_predictions(model: nn.Module, dataloader, device: str) -> tuple:
    """Get predictions. Return (all_preds, all_labels) as numpy arrays."""
    # Your code here
    pass

def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix. Return (num_classes, num_classes) array."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import numpy as np

def evaluate_classification(model: nn.Module, dataloader, device: str) -> tuple:
    """Evaluate model on dataloader. Return (accuracy, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

def get_predictions(model: nn.Module, dataloader, device: str) -> tuple:
    """Get predictions. Return (all_preds, all_labels) as numpy arrays."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)

def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix. Return (num_classes, num_classes) array."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm
`,

	testCode: `import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import unittest

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.model = SimpleMLP()
        X = torch.randn(40, 10)
        y = torch.randint(0, 3, (40,))
        dataset = TensorDataset(X, y)
        self.loader = DataLoader(dataset, batch_size=10)

    def test_evaluate_returns_tuple(self):
        acc, loss = evaluate_classification(self.model, self.loader, 'cpu')
        self.assertIsInstance(acc, float)
        self.assertIsInstance(loss, float)

    def test_accuracy_range(self):
        acc, _ = evaluate_classification(self.model, self.loader, 'cpu')
        self.assertTrue(0 <= acc <= 1)

    def test_get_predictions_shape(self):
        preds, labels = get_predictions(self.model, self.loader, 'cpu')
        self.assertEqual(len(preds), 40)
        self.assertEqual(len(labels), 40)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        cm = confusion_matrix_data(y_true, y_pred, 3)
        self.assertEqual(cm.shape, (3, 3))

    def test_confusion_matrix_diagonal(self):
        y = np.array([0, 1, 2])
        cm = confusion_matrix_data(y, y, 3)
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[1, 1], 1)

    def test_loss_positive(self):
        _, loss = evaluate_classification(self.model, self.loader, 'cpu')
        self.assertGreater(loss, 0)

    def test_predictions_numpy(self):
        preds, labels = get_predictions(self.model, self.loader, 'cpu')
        self.assertIsInstance(preds, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_confusion_matrix_sum(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        cm = confusion_matrix_data(y_true, y_pred, 3)
        self.assertEqual(cm.sum(), 5)

    def test_evaluate_uses_no_grad(self):
        acc, loss = evaluate_classification(self.model, self.loader, 'cpu')
        # Should complete without gradient tracking errors
        self.assertIsInstance(acc, float)

    def test_predictions_in_valid_range(self):
        preds, _ = get_predictions(self.model, self.loader, 'cpu')
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < 3))
`,

	hint1: 'Use model.eval() and torch.no_grad() during evaluation',
	hint2: 'outputs.max(1) returns (values, indices) - indices are predictions',

	whyItMatters: `Proper model evaluation is essential:

- **No gradients**: Use torch.no_grad() for efficiency
- **Eval mode**: Disables dropout and uses running stats for batchnorm
- **Metrics**: Accuracy alone isn't enough - use confusion matrix
- **Best practices**: Separate train and eval behavior

Always evaluate on held-out test data.`,

	translations: {
		ru: {
			title: 'Оценка модели',
			description: `# Оценка модели

Научитесь правильно оценивать модели PyTorch.

## Задача

Реализуйте три функции:
1. \`evaluate_classification(model, dataloader, device)\` - Вернуть accuracy и loss
2. \`get_predictions(model, dataloader, device)\` - Получить все предсказания
3. \`confusion_matrix_data(y_true, y_pred, num_classes)\` - Матрица ошибок

## Пример

\`\`\`python
model.eval()  # Set to evaluation mode

accuracy, loss = evaluate_classification(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.2%}, Loss: {loss:.4f}")

predictions = get_predictions(model, test_loader, device)
\`\`\``,
			hint1: 'Используйте model.eval() и torch.no_grad() при оценке',
			hint2: 'outputs.max(1) возвращает (values, indices) - indices это предсказания',
			whyItMatters: `Правильная оценка модели важна:

- **Без градиентов**: Используйте torch.no_grad() для эффективности
- **Режим eval**: Отключает dropout и использует running stats для batchnorm
- **Метрики**: Одной accuracy недостаточно - используйте confusion matrix`,
		},
		uz: {
			title: 'Model baholash',
			description: `# Model baholash

PyTorch modellarini to'g'ri baholashni o'rganing.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`evaluate_classification(model, dataloader, device)\` - accuracy va loss qaytarish
2. \`get_predictions(model, dataloader, device)\` - Barcha bashoratlarni olish
3. \`confusion_matrix_data(y_true, y_pred, num_classes)\` - Chalkashlik matritsasi

## Misol

\`\`\`python
model.eval()  # Set to evaluation mode

accuracy, loss = evaluate_classification(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.2%}, Loss: {loss:.4f}")

predictions = get_predictions(model, test_loader, device)
\`\`\``,
			hint1: "Baholash paytida model.eval() va torch.no_grad() dan foydalaning",
			hint2: "outputs.max(1) (qiymatlar, indekslar) qaytaradi - indekslar bashoratlar",
			whyItMatters: `To'g'ri model baholash muhim:

- **Gradientsiz**: Samaradorlik uchun torch.no_grad() dan foydalaning
- **Eval rejimi**: Dropout ni o'chiradi va batchnorm uchun running stats ishlatadi
- **Metrikalar**: Faqat accuracy yetarli emas - confusion matrix ishlating`,
		},
	},
};

export default task;
