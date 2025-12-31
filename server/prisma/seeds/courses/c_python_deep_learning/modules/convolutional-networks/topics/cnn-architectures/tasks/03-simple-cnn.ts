import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-simple-cnn',
	title: 'Simple CNN Architecture',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'architecture'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Simple CNN Architecture

Build a basic CNN for image classification from scratch.

## Task

Implement a \`SimpleCNN\` class with:
- 2 convolutional layers with ReLU and MaxPool
- 2 fully connected layers
- Forward pass that handles 28x28 grayscale images

## Example

\`\`\`python
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
output = model(x)
# output.shape = (4, 10)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN for 28x28 grayscale image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Your code here - define layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN for 28x28 grayscale image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 2 pools: 28 -> 14 -> 7, so 7*7*64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestSimpleCNN(unittest.TestCase):
    def test_output_shape(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_different_num_classes(self):
        model = SimpleCNN(num_classes=5)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_has_conv_layers(self):
        model = SimpleCNN()
        conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        self.assertGreaterEqual(conv_count, 2)

    def test_has_fc_layers(self):
        model = SimpleCNN()
        fc_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        self.assertGreaterEqual(fc_count, 2)

    def test_is_module(self):
        model = SimpleCNN()
        self.assertIsInstance(model, nn.Module)

    def test_has_parameters(self):
        model = SimpleCNN()
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_single_sample(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (1, 10))

    def test_output_not_nan(self):
        model = SimpleCNN()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())

    def test_has_pool_layers(self):
        model = SimpleCNN()
        pool_count = sum(1 for m in model.modules() if isinstance(m, nn.MaxPool2d))
        self.assertGreaterEqual(pool_count, 1)

    def test_batch_size_preserved(self):
        model = SimpleCNN()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape[0], 8)
`,

	hint1: 'After each pool(2,2) the spatial size halves: 28 -> 14 -> 7',
	hint2: 'Use x.view(x.size(0), -1) to flatten before FC layers',

	whyItMatters: `Building a CNN from scratch teaches:

- **Layer composition**: How conv, relu, pool work together
- **Dimension tracking**: Following tensor shapes through the network
- **Feature extraction**: Conv layers extract, FC layers classify
- **Architecture patterns**: Conv blocks followed by FC head

This pattern is the foundation of all modern CNNs.`,

	translations: {
		ru: {
			title: 'Простая архитектура CNN',
			description: `# Простая архитектура CNN

Создайте базовую CNN для классификации изображений с нуля.

## Задача

Реализуйте класс \`SimpleCNN\` с:
- 2 сверточных слоя с ReLU и MaxPool
- 2 полносвязных слоя
- Forward pass для изображений 28x28 в оттенках серого

## Пример

\`\`\`python
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
output = model(x)
# output.shape = (4, 10)
\`\`\``,
			hint1: 'После каждого pool(2,2) пространственный размер уменьшается вдвое: 28 -> 14 -> 7',
			hint2: 'Используйте x.view(x.size(0), -1) для выравнивания перед FC слоями',
			whyItMatters: `Создание CNN с нуля учит:

- **Композиция слоев**: Как conv, relu, pool работают вместе
- **Отслеживание размерностей**: Следование формам тензоров через сеть
- **Извлечение признаков**: Conv слои извлекают, FC классифицируют
- **Паттерны архитектуры**: Conv блоки с FC головой`,
		},
		uz: {
			title: "Oddiy CNN arxitekturasi",
			description: `# Oddiy CNN arxitekturasi

Tasvirlarni tasniflash uchun asosiy CNN ni noldan yarating.

## Topshiriq

\`SimpleCNN\` sinfini amalga oshiring:
- ReLU va MaxPool bilan 2 ta konvolyutsion qatlam
- 2 ta to'liq bog'langan qatlam
- 28x28 kulrang tasvirlar uchun forward pass

## Misol

\`\`\`python
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
output = model(x)
# output.shape = (4, 10)
\`\`\``,
			hint1: "Har bir pool(2,2) dan keyin fazoviy o'lcham yarmiga kamayadi: 28 -> 14 -> 7",
			hint2: "FC qatlamlardan oldin tekislash uchun x.view(x.size(0), -1) dan foydalaning",
			whyItMatters: `CNN ni noldan yaratish o'rgatadi:

- **Qatlam kompozitsiyasi**: Conv, relu, pool qanday birga ishlaydi
- **O'lcham kuzatuvi**: Tarmoq bo'ylab tensor shakllarini kuzatish
- **Xususiyat ajratish**: Conv qatlamlar ajratadi, FC tasniflaydi
- **Arxitektura naqshlari**: FC boshi bilan conv bloklari`,
		},
	},
};

export default task;
