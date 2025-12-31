import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pooling-layers',
	title: 'Pooling Layers',
	difficulty: 'easy',
	tags: ['pytorch', 'cnn', 'pooling'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,
	description: `# Pooling Layers

Learn about MaxPool and AvgPool for downsampling feature maps.

## Task

Implement two functions:
1. \`apply_max_pool\` - Apply max pooling to reduce spatial dimensions
2. \`apply_avg_pool\` - Apply average pooling to reduce spatial dimensions

## Example

\`\`\`python
x = torch.randn(1, 16, 32, 32)  # Batch, Channels, H, W

out = apply_max_pool(x, kernel_size=2, stride=2)
# out.shape = (1, 16, 16, 16)

out = apply_avg_pool(x, kernel_size=2, stride=2)
# out.shape = (1, 16, 16, 16)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_max_pool(x: torch.Tensor, kernel_size: int = 2,
                   stride: int = 2) -> torch.Tensor:
    """Apply max pooling to input tensor."""
    # Your code here
    pass

def apply_avg_pool(x: torch.Tensor, kernel_size: int = 2,
                   stride: int = 2) -> torch.Tensor:
    """Apply average pooling to input tensor."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_max_pool(x: torch.Tensor, kernel_size: int = 2,
                   stride: int = 2) -> torch.Tensor:
    """Apply max pooling to input tensor."""
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride)

def apply_avg_pool(x: torch.Tensor, kernel_size: int = 2,
                   stride: int = 2) -> torch.Tensor:
    """Apply average pooling to input tensor."""
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
`,

	testCode: `import torch
import unittest

class TestPooling(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 16, 32, 32)

    def test_max_pool_shape(self):
        out = apply_max_pool(self.x, 2, 2)
        self.assertEqual(out.shape, (2, 16, 16, 16))

    def test_avg_pool_shape(self):
        out = apply_avg_pool(self.x, 2, 2)
        self.assertEqual(out.shape, (2, 16, 16, 16))

    def test_max_pool_values(self):
        x = torch.tensor([[[[1., 2.], [3., 4.]]]])
        out = apply_max_pool(x, 2, 2)
        self.assertEqual(out.item(), 4.0)

    def test_avg_pool_values(self):
        x = torch.tensor([[[[1., 2.], [3., 4.]]]])
        out = apply_avg_pool(x, 2, 2)
        self.assertEqual(out.item(), 2.5)

    def test_max_pool_returns_tensor(self):
        out = apply_max_pool(self.x, 2, 2)
        self.assertIsInstance(out, torch.Tensor)

    def test_avg_pool_returns_tensor(self):
        out = apply_avg_pool(self.x, 2, 2)
        self.assertIsInstance(out, torch.Tensor)

    def test_different_stride(self):
        x = torch.randn(1, 8, 16, 16)
        out = apply_max_pool(x, 2, 1)
        self.assertEqual(out.shape, (1, 8, 15, 15))

    def test_larger_kernel(self):
        x = torch.randn(1, 8, 16, 16)
        out = apply_max_pool(x, 4, 4)
        self.assertEqual(out.shape, (1, 8, 4, 4))

    def test_preserves_channels(self):
        x = torch.randn(2, 64, 32, 32)
        out = apply_avg_pool(x, 2, 2)
        self.assertEqual(out.shape[1], 64)
`,

	hint1: 'Use F.max_pool2d() and F.avg_pool2d() from torch.nn.functional',
	hint2: 'Both functions take tensor, kernel_size, and stride as arguments',

	whyItMatters: `Pooling layers are essential in CNNs:

- **Dimensionality reduction**: Reduce computational cost
- **Translation invariance**: Features become position-independent
- **MaxPool**: Keeps strongest activations
- **AvgPool**: Smooths feature maps

Almost every CNN uses pooling to progressively reduce spatial dimensions.`,

	translations: {
		ru: {
			title: 'Слои пулинга',
			description: `# Слои пулинга

Изучите MaxPool и AvgPool для уменьшения размера карт признаков.

## Задача

Реализуйте две функции:
1. \`apply_max_pool\` - Применение max pooling для уменьшения размерности
2. \`apply_avg_pool\` - Применение average pooling для уменьшения размерности

## Пример

\`\`\`python
x = torch.randn(1, 16, 32, 32)  # Batch, Channels, H, W

out = apply_max_pool(x, kernel_size=2, stride=2)
# out.shape = (1, 16, 16, 16)

out = apply_avg_pool(x, kernel_size=2, stride=2)
# out.shape = (1, 16, 16, 16)
\`\`\``,
			hint1: 'Используйте F.max_pool2d() и F.avg_pool2d() из torch.nn.functional',
			hint2: 'Обе функции принимают тензор, kernel_size и stride как аргументы',
			whyItMatters: `Слои пулинга необходимы в CNN:

- **Уменьшение размерности**: Снижение вычислительных затрат
- **Инвариантность к сдвигу**: Признаки становятся независимыми от позиции
- **MaxPool**: Сохраняет сильнейшие активации
- **AvgPool**: Сглаживает карты признаков`,
		},
		uz: {
			title: 'Pooling qatlamlari',
			description: `# Pooling qatlamlari

Xususiyat xaritalarini kamaytirish uchun MaxPool va AvgPool haqida o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`apply_max_pool\` - O'lchamlarni kamaytirish uchun max pooling qo'llash
2. \`apply_avg_pool\` - O'lchamlarni kamaytirish uchun average pooling qo'llash

## Misol

\`\`\`python
x = torch.randn(1, 16, 32, 32)  # Batch, Channels, H, W

out = apply_max_pool(x, kernel_size=2, stride=2)
# out.shape = (1, 16, 16, 16)

out = apply_avg_pool(x, kernel_size=2, stride=2)
# out.shape = (1, 16, 16, 16)
\`\`\``,
			hint1: "torch.nn.functional dan F.max_pool2d() va F.avg_pool2d() dan foydalaning",
			hint2: "Ikkala funksiya tensor, kernel_size va stride ni argument sifatida qabul qiladi",
			whyItMatters: `Pooling qatlamlari CNN da muhim:

- **O'lcham kamaytirish**: Hisoblash xarajatlarini kamaytiradi
- **Siljishga barqarorlik**: Xususiyatlar pozitsiyadan mustaqil bo'ladi
- **MaxPool**: Eng kuchli aktivatsiyalarni saqlaydi
- **AvgPool**: Xususiyat xaritalarini tekislaydi`,
		},
	},
};

export default task;
