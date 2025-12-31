import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-early-stopping',
	title: 'Early Stopping',
	difficulty: 'medium',
	tags: ['pytorch', 'training', 'regularization'],
	estimatedTime: '12m',
	isPremium: false,
	order: 10,
	description: `# Early Stopping

Implement early stopping to prevent overfitting.

## Task

Implement one class:
\`EarlyStopping\` - Monitors validation loss and stops training when it stops improving.

Methods:
- \`__init__(patience, min_delta)\` - Initialize with patience and minimum improvement
- \`__call__(val_loss)\` - Check if should stop, return True to stop
- \`save_checkpoint(model)\` - Save best model
- \`load_best_model(model)\` - Load best model weights

## Example

\`\`\`python
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break

    if early_stopping.counter == 0:  # Improved
        early_stopping.save_checkpoint(model)

early_stopping.load_best_model(model)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import copy

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        # TODO: Initialize patience, min_delta, counter, best_loss, best_model_state
        pass

    def __call__(self, val_loss: float) -> bool:
        """Check if should stop. Return True if should stop training."""
        # Your code here
        pass

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save current model as best model."""
        # Your code here
        pass

    def load_best_model(self, model: nn.Module) -> None:
        """Load best model weights into model."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import copy

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None

    def __call__(self, val_loss: float) -> bool:
        """Check if should stop. Return True if should stop training."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save current model as best model."""
        self.best_model_state = copy.deepcopy(model.state_dict())

    def load_best_model(self, model: nn.Module) -> None:
        """Load best model weights into model."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)

class TestEarlyStopping(unittest.TestCase):
    def test_no_stop_improving(self):
        es = EarlyStopping(patience=3)
        self.assertFalse(es(1.0))
        self.assertFalse(es(0.9))
        self.assertFalse(es(0.8))

    def test_stop_after_patience(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # Best loss
        es(1.1)  # Worse, counter=1
        es(1.2)  # Worse, counter=2
        result = es(1.3)  # Worse, counter=3, should stop
        self.assertTrue(result)

    def test_reset_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)
        es(1.2)
        es(0.5)  # Better! Reset counter
        self.assertEqual(es.counter, 0)

    def test_save_load_model(self):
        model = SimpleMLP()
        original_weight = model.fc.weight.clone()

        es = EarlyStopping(patience=3)
        es.save_checkpoint(model)

        # Modify model
        with torch.no_grad():
            model.fc.weight.fill_(0)

        es.load_best_model(model)
        self.assertTrue(torch.allclose(model.fc.weight, original_weight))

    def test_init_patience(self):
        es = EarlyStopping(patience=10)
        self.assertEqual(es.patience, 10)

    def test_min_delta(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)
        es(0.95)  # Not enough improvement (< min_delta)
        self.assertEqual(es.counter, 1)

    def test_counter_increments(self):
        es = EarlyStopping(patience=5)
        es(1.0)
        es(1.1)
        self.assertEqual(es.counter, 1)
        es(1.2)
        self.assertEqual(es.counter, 2)

    def test_best_loss_updated(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        self.assertEqual(es.best_loss, 1.0)
        es(0.5)
        self.assertEqual(es.best_loss, 0.5)

    def test_first_call_no_stop(self):
        es = EarlyStopping(patience=1)
        result = es(1.0)
        self.assertFalse(result)

    def test_save_checkpoint_stores_state(self):
        model = SimpleMLP()
        es = EarlyStopping(patience=3)
        es.save_checkpoint(model)
        self.assertIsNotNone(es.best_model_state)
`,

	hint1: 'Track best_loss and counter. Increment counter if no improvement.',
	hint2: 'Use copy.deepcopy(model.state_dict()) to save model state',

	whyItMatters: `Early stopping prevents overfitting:

- **Regularization**: Implicit regularization by stopping at right time
- **Resource saving**: Don't waste compute on overfit epochs
- **Best model**: Save and restore the best performing model
- **Standard practice**: Used in almost all training pipelines

Simple but effective technique for better generalization.`,

	translations: {
		ru: {
			title: 'Ранняя остановка',
			description: `# Ранняя остановка

Реализуйте раннюю остановку для предотвращения переобучения.

## Задача

Реализуйте класс \`EarlyStopping\` - мониторит валидационную потерю и останавливает обучение когда улучшение прекращается.

Методы:
- \`__init__(patience, min_delta)\` - Инициализация
- \`__call__(val_loss)\` - Проверка, вернуть True для остановки
- \`save_checkpoint(model)\` - Сохранить лучшую модель
- \`load_best_model(model)\` - Загрузить лучшую модель

## Пример

\`\`\`python
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break

    if early_stopping.counter == 0:  # Improved
        early_stopping.save_checkpoint(model)

early_stopping.load_best_model(model)
\`\`\``,
			hint1: 'Отслеживайте best_loss и counter. Увеличивайте counter если нет улучшения.',
			hint2: 'Используйте copy.deepcopy(model.state_dict()) для сохранения',
			whyItMatters: `Ранняя остановка предотвращает переобучение:

- **Регуляризация**: Неявная регуляризация остановкой в правильный момент
- **Экономия ресурсов**: Не тратить вычисления на переобученные эпохи
- **Лучшая модель**: Сохранение и восстановление лучшей модели`,
		},
		uz: {
			title: "Erta to'xtatish",
			description: `# Erta to'xtatish

Ortiqcha moslanishni oldini olish uchun erta to'xtatishni amalga oshiring.

## Topshiriq

\`EarlyStopping\` sinfini amalga oshiring - validatsiya yo'qotishini kuzatadi va yaxshilanish to'xtaganda o'qitishni to'xtatadi.

Metodlar:
- \`__init__(patience, min_delta)\` - Sabr va minimal yaxshilanish bilan ishga tushirish
- \`__call__(val_loss)\` - To'xtatish kerakligini tekshirish
- \`save_checkpoint(model)\` - Eng yaxshi modelni saqlash
- \`load_best_model(model)\` - Eng yaxshi model og'irliklarini yuklash

## Misol

\`\`\`python
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break

    if early_stopping.counter == 0:  # Improved
        early_stopping.save_checkpoint(model)

early_stopping.load_best_model(model)
\`\`\``,
			hint1: "best_loss va counter ni kuzating. Yaxshilanish bo'lmasa counter ni oshiring.",
			hint2: "Model holatini saqlash uchun copy.deepcopy(model.state_dict()) dan foydalaning",
			whyItMatters: `Erta to'xtatish ortiqcha moslanishni oldini oladi:

- **Regulyarizatsiya**: To'g'ri vaqtda to'xtatish bilan yashirin regulyarizatsiya
- **Resurs tejash**: Ortiqcha moslangan davrlar uchun hisoblashni behuda sarflamang
- **Eng yaxshi model**: Eng yaxshi ishlaydigan modelni saqlash va tiklash`,
		},
	},
};

export default task;
