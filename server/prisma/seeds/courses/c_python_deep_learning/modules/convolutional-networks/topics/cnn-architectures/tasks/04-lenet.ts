import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-lenet-architecture',
	title: 'LeNet Architecture',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'lenet'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# LeNet Architecture

Implement the classic LeNet-5 architecture, one of the first successful CNNs.

## Task

Implement a \`LeNet\` class following the original architecture:
- Conv1: 1 -> 6 channels, 5x5 kernel
- Pool1: 2x2 average pooling
- Conv2: 6 -> 16 channels, 5x5 kernel
- Pool2: 2x2 average pooling
- FC1: 16*4*4 -> 120
- FC2: 120 -> 84
- FC3: 84 -> num_classes

## Example

\`\`\`python
model = LeNet(num_classes=10)
x = torch.randn(4, 1, 28, 28)
output = model(x)
# output.shape = (4, 10)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """LeNet-5 architecture for digit classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """LeNet-5 architecture for digit classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        # Conv block 2
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestLeNet(unittest.TestCase):
    def test_output_shape(self):
        model = LeNet(num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_layer_structure(self):
        model = LeNet()
        self.assertEqual(model.conv1.out_channels, 6)
        self.assertEqual(model.conv2.out_channels, 16)
        self.assertEqual(model.fc1.out_features, 120)
        self.assertEqual(model.fc2.out_features, 84)

    def test_different_classes(self):
        model = LeNet(num_classes=26)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

    def test_model_is_module(self):
        model = LeNet()
        self.assertIsInstance(model, nn.Module)

    def test_has_conv_layers(self):
        model = LeNet()
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'conv2'))

    def test_has_fc_layers(self):
        model = LeNet()
        self.assertTrue(hasattr(model, 'fc1'))
        self.assertTrue(hasattr(model, 'fc2'))
        self.assertTrue(hasattr(model, 'fc3'))

    def test_single_sample(self):
        model = LeNet()
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (1, 10))

    def test_large_batch(self):
        model = LeNet()
        x = torch.randn(32, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (32, 10))

    def test_fc1_input_size(self):
        model = LeNet()
        self.assertEqual(model.fc1.in_features, 16 * 4 * 4)

    def test_output_not_nan(self):
        model = LeNet()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())
`,

	hint1: 'LeNet uses average pooling, not max pooling',
	hint2: 'Input 28x28 -> conv(5) -> 24x24 -> pool(2) -> 12x12 -> conv(5) -> 8x8 -> pool(2) -> 4x4',

	whyItMatters: `LeNet-5 (1998) is historically significant:

- **First successful CNN**: Proved CNNs work for vision tasks
- **Established patterns**: Conv-Pool-Conv-Pool-FC structure
- **Used in production**: Read millions of checks for banks
- **Foundation**: Most modern CNNs follow similar principles

Understanding LeNet helps appreciate how CNN architectures evolved.`,

	translations: {
		ru: {
			title: 'Архитектура LeNet',
			description: `# Архитектура LeNet

Реализуйте классическую архитектуру LeNet-5, одну из первых успешных CNN.

## Задача

Реализуйте класс \`LeNet\` следуя оригинальной архитектуре:
- Conv1: 1 -> 6 каналов, ядро 5x5
- Pool1: 2x2 average pooling
- Conv2: 6 -> 16 каналов, ядро 5x5
- Pool2: 2x2 average pooling
- FC1: 16*4*4 -> 120
- FC2: 120 -> 84
- FC3: 84 -> num_classes

## Пример

\`\`\`python
model = LeNet(num_classes=10)
x = torch.randn(4, 1, 28, 28)
output = model(x)
# output.shape = (4, 10)
\`\`\``,
			hint1: 'LeNet использует average pooling, а не max pooling',
			hint2: 'Вход 28x28 -> conv(5) -> 24x24 -> pool(2) -> 12x12 -> conv(5) -> 8x8 -> pool(2) -> 4x4',
			whyItMatters: `LeNet-5 (1998) исторически значим:

- **Первая успешная CNN**: Доказала работоспособность CNN для задач зрения
- **Установил паттерны**: Структура Conv-Pool-Conv-Pool-FC
- **Использовался в продакшене**: Читал миллионы чеков для банков
- **Основа**: Большинство современных CNN следуют схожим принципам`,
		},
		uz: {
			title: 'LeNet arxitekturasi',
			description: `# LeNet arxitekturasi

Birinchi muvaffaqiyatli CNN lardan biri - klassik LeNet-5 arxitekturasini amalga oshiring.

## Topshiriq

Original arxitektura bo'yicha \`LeNet\` sinfini amalga oshiring:
- Conv1: 1 -> 6 kanal, 5x5 kernel
- Pool1: 2x2 average pooling
- Conv2: 6 -> 16 kanal, 5x5 kernel
- Pool2: 2x2 average pooling
- FC1: 16*4*4 -> 120
- FC2: 120 -> 84
- FC3: 84 -> num_classes

## Misol

\`\`\`python
model = LeNet(num_classes=10)
x = torch.randn(4, 1, 28, 28)
output = model(x)
# output.shape = (4, 10)
\`\`\``,
			hint1: 'LeNet max pooling emas, average pooling ishlatadi',
			hint2: 'Kirish 28x28 -> conv(5) -> 24x24 -> pool(2) -> 12x12 -> conv(5) -> 8x8 -> pool(2) -> 4x4',
			whyItMatters: `LeNet-5 (1998) tarixiy ahamiyatga ega:

- **Birinchi muvaffaqiyatli CNN**: CNN lar ko'rish vazifalari uchun ishlashini isbotladi
- **Naqshlarni o'rnatdi**: Conv-Pool-Conv-Pool-FC tuzilmasi
- **Ishlab chiqarishda ishlatilgan**: Banklar uchun millionlab cheklarni o'qigan
- **Asos**: Ko'pchilik zamonaviy CNN lar o'xshash printsiplarga amal qiladi`,
		},
	},
};

export default task;
