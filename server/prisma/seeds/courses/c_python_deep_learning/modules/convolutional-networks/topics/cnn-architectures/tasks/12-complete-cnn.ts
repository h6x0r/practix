import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-complete-cnn',
	title: 'Complete CNN Project',
	difficulty: 'hard',
	tags: ['pytorch', 'cnn', 'project'],
	estimatedTime: '25m',
	isPremium: true,
	order: 12,
	description: `# Complete CNN Project

Build a production-ready CNN with all best practices.

## Task

Implement a \`ProductionCNN\` class with:
- Multiple ConvBNReLU blocks with increasing channels
- Global average pooling
- Dropout for regularization
- Xavier weight initialization

## Example

\`\`\`python
model = ProductionCNN(num_classes=100, dropout=0.5)
x = torch.randn(4, 3, 64, 64)
output = model(x)  # Shape: (4, 100)

# Initialize weights
model.apply(init_weights)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def init_weights(m):
    """Initialize weights with Xavier/He initialization."""
    # Your code here
    pass

class ProductionCNN(nn.Module):
    """Production-ready CNN with best practices."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn

def init_weights(m):
    """Initialize weights with Xavier/He initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class ProductionCNN(nn.Module):
    """Production-ready CNN with best practices."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestProductionCNN(unittest.TestCase):
    def test_output_shape(self):
        model = ProductionCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_different_input_sizes(self):
        model = ProductionCNN(num_classes=10)
        for size in [32, 64, 128]:
            x = torch.randn(2, 3, size, size)
            out = model(x)
            self.assertEqual(out.shape, (2, 10))

    def test_different_num_classes(self):
        model = ProductionCNN(num_classes=100)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 100))

    def test_init_weights(self):
        model = ProductionCNN()
        # Check that weights are initialized
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.assertIsNotNone(m.weight)

    def test_is_module(self):
        model = ProductionCNN()
        self.assertIsInstance(model, nn.Module)

    def test_has_dropout(self):
        model = ProductionCNN(dropout=0.5)
        dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
        self.assertGreaterEqual(dropout_count, 1)

    def test_has_batchnorm(self):
        model = ProductionCNN()
        bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        self.assertGreater(bn_count, 0)

    def test_output_not_nan(self):
        model = ProductionCNN()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())

    def test_single_sample(self):
        model = ProductionCNN()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape[0], 1)

    def test_init_weights_function(self):
        conv = nn.Conv2d(3, 16, 3)
        init_weights(conv)
        self.assertIsNotNone(conv.weight)
`,

	hint1: 'Use nn.init.kaiming_normal_ for conv layers before ReLU',
	hint2: 'AdaptiveAvgPool2d(1) outputs 1x1 regardless of input size',

	whyItMatters: `Production CNNs combine all best practices:

- **Batch normalization**: Stable training, higher learning rates
- **Proper initialization**: Prevents vanishing/exploding gradients
- **Dropout**: Regularization to prevent overfitting
- **Global pooling**: Handles any input size, fewer parameters

This architecture pattern is used in real-world image classification systems.`,

	translations: {
		ru: {
			title: 'Полный проект CNN',
			description: `# Полный проект CNN

Создайте production-ready CNN со всеми лучшими практиками.

## Задача

Реализуйте класс \`ProductionCNN\` с:
- Несколько блоков ConvBNReLU с увеличивающимися каналами
- Global average pooling
- Dropout для регуляризации
- Xavier инициализация весов

## Пример

\`\`\`python
model = ProductionCNN(num_classes=100, dropout=0.5)
x = torch.randn(4, 3, 64, 64)
output = model(x)  # Shape: (4, 100)

# Initialize weights
model.apply(init_weights)
\`\`\``,
			hint1: 'Используйте nn.init.kaiming_normal_ для conv слоев перед ReLU',
			hint2: 'AdaptiveAvgPool2d(1) выдает 1x1 независимо от размера входа',
			whyItMatters: `Production CNN объединяет все лучшие практики:

- **Batch normalization**: Стабильное обучение, высокие learning rates
- **Правильная инициализация**: Предотвращает затухание/взрыв градиентов
- **Dropout**: Регуляризация против переобучения
- **Global pooling**: Работает с любым размером входа`,
		},
		uz: {
			title: "To'liq CNN loyihasi",
			description: `# To'liq CNN loyihasi

Barcha eng yaxshi amaliyotlar bilan production-ready CNN yarating.

## Topshiriq

\`ProductionCNN\` sinfini amalga oshiring:
- Ortib boruvchi kanallar bilan bir nechta ConvBNReLU bloklari
- Global average pooling
- Regularizatsiya uchun Dropout
- Xavier og'irlik initializatsiyasi

## Misol

\`\`\`python
model = ProductionCNN(num_classes=100, dropout=0.5)
x = torch.randn(4, 3, 64, 64)
output = model(x)  # Shape: (4, 100)

# Initialize weights
model.apply(init_weights)
\`\`\``,
			hint1: "ReLU dan oldin conv qatlamlar uchun nn.init.kaiming_normal_ dan foydalaning",
			hint2: "AdaptiveAvgPool2d(1) kirish o'lchamidan qat'i nazar 1x1 chiqaradi",
			whyItMatters: `Production CNN barcha eng yaxshi amaliyotlarni birlashtiradi:

- **Batch normalization**: Barqaror o'qitish, yuqori learning rates
- **To'g'ri initializatsiya**: Gradientlarning yo'qolishi/portlashini oldini oladi
- **Dropout**: Ortiqcha moslanishga qarshi regularizatsiya
- **Global pooling**: Har qanday kirish o'lchami bilan ishlaydi`,
		},
	},
};

export default task;
