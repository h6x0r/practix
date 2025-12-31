import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-lr-scheduler',
	title: 'Learning Rate Schedulers',
	difficulty: 'medium',
	tags: ['pytorch', 'scheduler', 'training'],
	estimatedTime: '12m',
	isPremium: true,
	order: 9,
	description: `# Learning Rate Schedulers

Learn to adjust learning rate during training.

## Task

Implement three functions:
1. \`create_step_scheduler(optimizer, step_size, gamma)\` - StepLR scheduler
2. \`create_cosine_scheduler(optimizer, T_max)\` - CosineAnnealingLR
3. \`train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, epochs)\` - Training with scheduler

## Example

\`\`\`python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = create_step_scheduler(optimizer, step_size=10, gamma=0.1)

# After each epoch
scheduler.step()  # Reduces LR by 0.1x every 10 epochs
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def create_step_scheduler(optimizer, step_size: int, gamma: float):
    """Create StepLR scheduler that reduces LR by gamma every step_size epochs."""
    # Your code here
    pass

def create_cosine_scheduler(optimizer, T_max: int):
    """Create CosineAnnealingLR scheduler with T_max iterations."""
    # Your code here
    pass

def train_with_scheduler(model: nn.Module, dataloader, criterion, optimizer,
                         scheduler, epochs: int) -> list:
    """Train with scheduler. Return list of (loss, lr) per epoch."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def create_step_scheduler(optimizer, step_size: int, gamma: float):
    """Create StepLR scheduler that reduces LR by gamma every step_size epochs."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

def create_cosine_scheduler(optimizer, T_max: int):
    """Create CosineAnnealingLR scheduler with T_max iterations."""
    return CosineAnnealingLR(optimizer, T_max=T_max)

def train_with_scheduler(model: nn.Module, dataloader, criterion, optimizer,
                         scheduler, epochs: int) -> list:
    """Train with scheduler. Return list of (loss, lr) per epoch."""
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        history.append((avg_loss, current_lr))
        scheduler.step()

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

class TestLRScheduler(unittest.TestCase):
    def test_step_scheduler(self):
        model = SimpleMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_step_scheduler(optimizer, step_size=2, gamma=0.1)

        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1)
        scheduler.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.01, places=5)

    def test_cosine_scheduler(self):
        model = SimpleMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_cosine_scheduler(optimizer, T_max=10)
        self.assertIsNotNone(scheduler)

    def test_train_with_scheduler(self):
        model = SimpleMLP()
        X = torch.randn(40, 10)
        y = torch.randint(0, 3, (40,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_step_scheduler(optimizer, step_size=2, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        history = train_with_scheduler(model, loader, criterion, optimizer, scheduler, 4)
        self.assertEqual(len(history), 4)

    def test_step_scheduler_returns_object(self):
        model = SimpleMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_step_scheduler(optimizer, step_size=5, gamma=0.1)
        self.assertIsNotNone(scheduler)

    def test_cosine_scheduler_lr_changes(self):
        model = SimpleMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_cosine_scheduler(optimizer, T_max=10)
        lr_before = optimizer.param_groups[0]['lr']
        for _ in range(5):
            scheduler.step()
        lr_after = optimizer.param_groups[0]['lr']
        self.assertNotEqual(lr_before, lr_after)

    def test_history_contains_tuples(self):
        model = SimpleMLP()
        X = torch.randn(40, 10)
        y = torch.randint(0, 3, (40,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_step_scheduler(optimizer, step_size=2, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        history = train_with_scheduler(model, loader, criterion, optimizer, scheduler, 2)
        self.assertEqual(len(history[0]), 2)

    def test_lr_decreases_with_step_scheduler(self):
        model = SimpleMLP()
        X = torch.randn(40, 10)
        y = torch.randint(0, 3, (40,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_step_scheduler(optimizer, step_size=1, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        history = train_with_scheduler(model, loader, criterion, optimizer, scheduler, 3)
        self.assertLess(history[-1][1], history[0][1])

    def test_step_scheduler_gamma_effect(self):
        model = SimpleMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = create_step_scheduler(optimizer, step_size=1, gamma=0.5)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.5, places=5)

    def test_cosine_scheduler_created(self):
        model = SimpleMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = create_cosine_scheduler(optimizer, T_max=20)
        self.assertTrue(hasattr(scheduler, 'step'))
`,

	hint1: 'StepLR(optimizer, step_size=10, gamma=0.1) reduces LR every 10 epochs',
	hint2: 'Call scheduler.step() after each epoch, not after each batch',

	whyItMatters: `Learning rate scheduling improves training:

- **Faster convergence**: High LR early, lower later
- **Better optima**: Avoid overshooting with decreasing LR
- **Warmup**: Gradually increase LR at start
- **Cosine annealing**: Smooth LR decay often works best

Almost all modern training uses some form of LR scheduling.`,

	translations: {
		ru: {
			title: 'Планировщики скорости обучения',
			description: `# Планировщики скорости обучения

Научитесь регулировать learning rate во время обучения.

## Задача

Реализуйте три функции:
1. \`create_step_scheduler(optimizer, step_size, gamma)\` - StepLR
2. \`create_cosine_scheduler(optimizer, T_max)\` - CosineAnnealingLR
3. \`train_with_scheduler(...)\` - Обучение с планировщиком

## Пример

\`\`\`python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = create_step_scheduler(optimizer, step_size=10, gamma=0.1)

# After each epoch
scheduler.step()  # Reduces LR by 0.1x every 10 epochs
\`\`\``,
			hint1: 'StepLR(optimizer, step_size=10, gamma=0.1) уменьшает LR каждые 10 эпох',
			hint2: 'Вызывайте scheduler.step() после каждой эпохи, не батча',
			whyItMatters: `Планирование learning rate улучшает обучение:

- **Быстрая сходимость**: Высокий LR вначале, ниже позже
- **Лучшие оптимумы**: Избежание перескока с уменьшением LR
- **Cosine annealing**: Плавное затухание часто работает лучше`,
		},
		uz: {
			title: "O'rganish tezligi rejalashtiruvchilari",
			description: `# O'rganish tezligi rejalashtiruvchilari

O'qitish paytida o'rganish tezligini sozlashni o'rganing.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`create_step_scheduler(optimizer, step_size, gamma)\` - StepLR rejalashtiruvchi
2. \`create_cosine_scheduler(optimizer, T_max)\` - CosineAnnealingLR
3. \`train_with_scheduler(...)\` - Rejalashtiruvchi bilan o'qitish

## Misol

\`\`\`python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = create_step_scheduler(optimizer, step_size=10, gamma=0.1)

# After each epoch
scheduler.step()  # Reduces LR by 0.1x every 10 epochs
\`\`\``,
			hint1: "StepLR(optimizer, step_size=10, gamma=0.1) har 10 davrda LR ni kamaytiradi",
			hint2: "scheduler.step() ni har bir davrdan keyin, har bir batchdan emas chaqiring",
			whyItMatters: `O'rganish tezligini rejalashtirish o'qitishni yaxshilaydi:

- **Tezroq yaqinlashish**: Boshida yuqori LR, keyin past
- **Yaxshiroq optimallar**: Kamayuvchi LR bilan oshib ketishdan qochish
- **Cosine annealing**: Silliq LR so'nishi ko'pincha eng yaxshi ishlaydi`,
		},
	},
};

export default task;
