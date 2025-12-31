import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-loss-optimizer',
	title: 'Loss Functions and Optimizers',
	difficulty: 'easy',
	tags: ['pytorch', 'loss', 'optimizer'],
	estimatedTime: '12m',
	isPremium: false,
	order: 6,
	description: `# Loss Functions and Optimizers

Learn to use PyTorch's built-in loss functions and optimizers.

## Task

Implement four functions:
1. \`compute_ce_loss(predictions, targets)\` - Cross-entropy loss
2. \`compute_mse_loss(predictions, targets)\` - MSE loss
3. \`create_optimizer(model, lr, optimizer_type)\` - Create Adam or SGD
4. \`training_step(model, x, y, criterion, optimizer)\` - One training step

## Example

\`\`\`python
import torch.nn as nn
import torch.optim as optim

model = SimpleMLP(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(model, lr=0.001, optimizer_type='adam')

loss = training_step(model, x_batch, y_batch, criterion, optimizer)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.optim as optim

def compute_ce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss. predictions: logits, targets: class indices."""
    # Your code here
    pass

def compute_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss."""
    # Your code here
    pass

def create_optimizer(model: nn.Module, lr: float, optimizer_type: str):
    """Create optimizer. optimizer_type: 'adam' or 'sgd'."""
    # Your code here
    pass

def training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                  criterion, optimizer) -> float:
    """One training step: forward, loss, backward, update. Return loss value."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.optim as optim

def compute_ce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss. predictions: logits, targets: class indices."""
    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, targets)

def compute_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss."""
    criterion = nn.MSELoss()
    return criterion(predictions, targets)

def create_optimizer(model: nn.Module, lr: float, optimizer_type: str):
    """Create optimizer. optimizer_type: 'adam' or 'sgd'."""
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                  criterion, optimizer) -> float:
    """One training step: forward, loss, backward, update. Return loss value."""
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    return loss.item()
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

class TestLossOptimizer(unittest.TestCase):
    def test_ce_loss(self):
        preds = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 2, 0, 1])
        loss = compute_ce_loss(preds, targets)
        self.assertGreater(loss.item(), 0)

    def test_mse_loss(self):
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        loss = compute_mse_loss(preds, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_create_adam(self):
        model = SimpleMLP()
        opt = create_optimizer(model, 0.001, 'adam')
        self.assertIsInstance(opt, torch.optim.Adam)

    def test_create_sgd(self):
        model = SimpleMLP()
        opt = create_optimizer(model, 0.01, 'sgd')
        self.assertIsInstance(opt, torch.optim.SGD)

    def test_training_step(self):
        model = SimpleMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, 0.01, 'adam')
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        loss = training_step(model, x, y, criterion, optimizer)
        self.assertIsInstance(loss, float)

    def test_ce_loss_returns_tensor(self):
        preds = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 2, 0, 1])
        loss = compute_ce_loss(preds, targets)
        self.assertIsInstance(loss, torch.Tensor)

    def test_mse_loss_positive(self):
        preds = torch.tensor([1.0, 2.0])
        targets = torch.tensor([2.0, 3.0])
        loss = compute_mse_loss(preds, targets)
        self.assertGreater(loss.item(), 0)

    def test_training_step_positive_loss(self):
        model = SimpleMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, 0.01, 'adam')
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        loss = training_step(model, x, y, criterion, optimizer)
        self.assertGreater(loss, 0)

    def test_optimizer_has_params(self):
        model = SimpleMLP()
        opt = create_optimizer(model, 0.001, 'adam')
        self.assertGreater(len(opt.param_groups), 0)

    def test_training_step_updates_weights(self):
        model = SimpleMLP()
        w_before = model.fc.weight.clone()
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, 0.1, 'sgd')
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        training_step(model, x, y, criterion, optimizer)
        self.assertFalse(torch.allclose(model.fc.weight, w_before))
`,

	hint1: 'nn.CrossEntropyLoss() expects raw logits, not softmax output',
	hint2: 'Training step: zero_grad() -> forward -> loss -> backward() -> step()',

	whyItMatters: `Loss functions and optimizers drive training:

- **Loss functions**: Define what the model optimizes
- **Optimizers**: Implement gradient descent variants
- **Training step**: The core loop of neural network training
- **Built-in options**: PyTorch provides many optimized implementations

Understanding this is essential for training any neural network.`,

	translations: {
		ru: {
			title: 'Функции потерь и оптимизаторы',
			description: `# Функции потерь и оптимизаторы

Научитесь использовать встроенные функции потерь и оптимизаторы PyTorch.

## Задача

Реализуйте четыре функции:
1. \`compute_ce_loss(predictions, targets)\` - Кросс-энтропия
2. \`compute_mse_loss(predictions, targets)\` - MSE
3. \`create_optimizer(model, lr, optimizer_type)\` - Создать Adam или SGD
4. \`training_step(model, x, y, criterion, optimizer)\` - Один шаг обучения

## Пример

\`\`\`python
import torch.nn as nn
import torch.optim as optim

model = SimpleMLP(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(model, lr=0.001, optimizer_type='adam')

loss = training_step(model, x_batch, y_batch, criterion, optimizer)
\`\`\``,
			hint1: 'nn.CrossEntropyLoss() ожидает сырые логиты, не softmax',
			hint2: 'Шаг обучения: zero_grad() -> forward -> loss -> backward() -> step()',
			whyItMatters: `Функции потерь и оптимизаторы управляют обучением:

- **Функции потерь**: Определяют что оптимизирует модель
- **Оптимизаторы**: Реализуют варианты градиентного спуска
- **Шаг обучения**: Основной цикл обучения нейросети`,
		},
		uz: {
			title: "Yo'qotish funksiyalari va optimallashtiruvchilar",
			description: `# Yo'qotish funksiyalari va optimallashtiruvchilar

PyTorch ning o'rnatilgan yo'qotish funksiyalari va optimallashtiruvchilaridan foydalanishni o'rganing.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`compute_ce_loss(predictions, targets)\` - Kross-entropiya yo'qotishi
2. \`compute_mse_loss(predictions, targets)\` - MSE yo'qotishi
3. \`create_optimizer(model, lr, optimizer_type)\` - Adam yoki SGD yaratish
4. \`training_step(model, x, y, criterion, optimizer)\` - Bitta o'qitish qadami

## Misol

\`\`\`python
import torch.nn as nn
import torch.optim as optim

model = SimpleMLP(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(model, lr=0.001, optimizer_type='adam')

loss = training_step(model, x_batch, y_batch, criterion, optimizer)
\`\`\``,
			hint1: "nn.CrossEntropyLoss() xom logitlarni kutadi, softmax chiqishini emas",
			hint2: "O'qitish qadami: zero_grad() -> forward -> loss -> backward() -> step()",
			whyItMatters: `Yo'qotish funksiyalari va optimallashtiruvchilar o'qitishni boshqaradi:

- **Yo'qotish funksiyalari**: Model nimani optimallashtirganini aniqlaydi
- **Optimallashtiruvchilar**: Gradient tushish variantlarini amalga oshiradi
- **O'qitish qadami**: Neyrosetka o'qitishning asosiy sikli`,
		},
	},
};

export default task;
