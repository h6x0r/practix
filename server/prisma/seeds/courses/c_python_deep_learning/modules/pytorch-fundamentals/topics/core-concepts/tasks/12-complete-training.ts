import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-complete-training',
	title: 'Complete Training Pipeline',
	difficulty: 'hard',
	tags: ['pytorch', 'training', 'pipeline'],
	estimatedTime: '20m',
	isPremium: true,
	order: 12,
	description: `# Complete Training Pipeline

Build a complete PyTorch training pipeline with all best practices.

## Task

Implement one class:
\`Trainer\` - Complete training pipeline with:
- Training loop with progress tracking
- Validation after each epoch
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Logging metrics

## Example

\`\`\`python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    patience=5
)

history = trainer.fit(epochs=50)
# Returns dict with train_loss, val_loss, val_acc per epoch
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from typing import Dict, List
import copy

class Trainer:
    """Complete training pipeline with best practices."""

    def __init__(self, model: nn.Module, train_loader, val_loader,
                 optimizer, criterion, device: str = 'cpu', patience: int = 5):
        # Your code here
        pass

    def train_epoch(self) -> float:
        """Train for one epoch. Return average loss."""
        # Your code here
        pass

    def validate(self) -> tuple:
        """Validate model. Return (val_loss, val_accuracy)."""
        # Your code here
        pass

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        """Full training loop. Return history dict."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
from typing import Dict, List
import copy

class Trainer:
    """Complete training pipeline with best practices."""

    def __init__(self, model: nn.Module, train_loader, val_loader,
                 optimizer, criterion, device: str = 'cpu', patience: int = 5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.best_loss = float('inf')
        self.best_model_state = None
        self.early_stop_counter = 0

    def train_epoch(self) -> float:
        """Train for one epoch. Return average loss."""
        self.model.train()
        total_loss = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> tuple:
        """Validate model. Return (val_loss, val_accuracy)."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        """Full training loop. Return history dict."""
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history
`,

	testCode: `import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import unittest

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model = SimpleMLP()
        X = torch.randn(80, 10)
        y = torch.randint(0, 3, (80,))
        train_set = TensorDataset(X[:60], y[:60])
        val_set = TensorDataset(X[60:], y[60:])
        self.train_loader = DataLoader(train_set, batch_size=10)
        self.val_loader = DataLoader(val_set, batch_size=10)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def test_trainer_creation(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion)
        self.assertIsNotNone(trainer)

    def test_train_epoch(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion)
        loss = trainer.train_epoch()
        self.assertIsInstance(loss, float)

    def test_validate(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion)
        val_loss, val_acc = trainer.validate()
        self.assertTrue(0 <= val_acc <= 1)

    def test_fit_returns_history(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion, patience=2)
        history = trainer.fit(epochs=5)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_acc', history)

    def test_train_epoch_positive(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion)
        loss = trainer.train_epoch()
        self.assertGreater(loss, 0)

    def test_validate_returns_tuple(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion)
        result = trainer.validate()
        self.assertEqual(len(result), 2)

    def test_validate_loss_positive(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion)
        val_loss, _ = trainer.validate()
        self.assertGreater(val_loss, 0)

    def test_history_lists_same_length(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion, patience=10)
        history = trainer.fit(epochs=3)
        self.assertEqual(len(history['train_loss']), len(history['val_loss']))

    def test_patience_stored(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion, patience=7)
        self.assertEqual(trainer.patience, 7)

    def test_device_stored(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader,
                         self.optimizer, self.criterion, device='cpu')
        self.assertEqual(trainer.device, 'cpu')
`,

	hint1: 'Use model.train() before training, model.eval() before validation',
	hint2: 'Track best model and restore it at the end if early stopping triggered',

	whyItMatters: `A complete training pipeline includes:

- **Training loop**: Forward, loss, backward, optimize
- **Validation**: Monitor generalization performance
- **Early stopping**: Prevent overfitting
- **Checkpointing**: Save best model
- **Logging**: Track metrics for debugging

This is the standard structure for all PyTorch training.`,

	translations: {
		ru: {
			title: 'Полный пайплайн обучения',
			description: `# Полный пайплайн обучения

Создайте полный пайплайн обучения PyTorch со всеми лучшими практиками.

## Задача

Реализуйте класс \`Trainer\` - полный пайплайн обучения с:
- Цикл обучения с отслеживанием прогресса
- Валидация после каждой эпохи
- Ранняя остановка
- Чекпоинтинг модели
- Логирование метрик

## Пример

\`\`\`python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    patience=5
)

history = trainer.fit(epochs=50)
# Returns dict with train_loss, val_loss, val_acc per epoch
\`\`\``,
			hint1: 'Используйте model.train() перед обучением, model.eval() перед валидацией',
			hint2: 'Отслеживайте лучшую модель и восстановите её если сработала ранняя остановка',
			whyItMatters: `Полный пайплайн обучения включает:

- **Цикл обучения**: Forward, loss, backward, optimize
- **Валидация**: Мониторинг производительности обобщения
- **Ранняя остановка**: Предотвращение переобучения
- **Чекпоинтинг**: Сохранение лучшей модели`,
		},
		uz: {
			title: "To'liq o'qitish pipeline",
			description: `# To'liq o'qitish pipeline

Barcha eng yaxshi amaliyotlar bilan to'liq PyTorch o'qitish pipeline ni yarating.

## Topshiriq

\`Trainer\` sinfini amalga oshiring - to'liq o'qitish pipeline:
- Progress kuzatuv bilan o'qitish sikli
- Har bir davrdan keyin validatsiya
- Erta to'xtatish
- Model checkpointing
- Metrikalarni loglash

## Misol

\`\`\`python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    patience=5
)

history = trainer.fit(epochs=50)
# Returns dict with train_loss, val_loss, val_acc per epoch
\`\`\``,
			hint1: "O'qitishdan oldin model.train(), validatsiyadan oldin model.eval() dan foydalaning",
			hint2: "Eng yaxshi modelni kuzating va erta to'xtatish ishlasa oxirida tiklang",
			whyItMatters: `To'liq o'qitish pipeline quyidagilarni o'z ichiga oladi:

- **O'qitish sikli**: Forward, loss, backward, optimize
- **Validatsiya**: Umumlashtirish samaradorligini kuzatish
- **Erta to'xtatish**: Ortiqcha moslanishni oldini olish
- **Checkpointing**: Eng yaxshi modelni saqlash`,
		},
	},
};

export default task;
