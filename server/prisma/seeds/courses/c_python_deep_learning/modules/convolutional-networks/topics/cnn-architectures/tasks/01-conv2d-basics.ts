import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-conv2d-basics',
	title: 'Conv2D Layer Basics',
	difficulty: 'easy',
	tags: ['pytorch', 'cnn', 'conv2d'],
	estimatedTime: '12m',
	isPremium: false,
	order: 1,
	description: `# Conv2D Layer Basics

Learn how 2D convolution layers work in PyTorch.

## Task

Implement two functions:
1. \`create_conv_layer\` - Create a Conv2d layer with specified parameters
2. \`compute_output_size\` - Calculate output dimensions after convolution

## Example

\`\`\`python
conv = create_conv_layer(in_channels=3, out_channels=16, kernel_size=3)
# Returns nn.Conv2d(3, 16, 3)

output_size = compute_output_size(input_size=32, kernel_size=3, stride=1, padding=0)
# Returns 30 (formula: (32 - 3 + 2*0) / 1 + 1 = 30)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def create_conv_layer(in_channels: int, out_channels: int,
                      kernel_size: int, stride: int = 1,
                      padding: int = 0) -> nn.Conv2d:
    """Create a Conv2d layer with given parameters."""
    # Your code here
    pass

def compute_output_size(input_size: int, kernel_size: int,
                        stride: int = 1, padding: int = 0) -> int:
    """Compute output spatial dimension after convolution."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

def create_conv_layer(in_channels: int, out_channels: int,
                      kernel_size: int, stride: int = 1,
                      padding: int = 0) -> nn.Conv2d:
    """Create a Conv2d layer with given parameters."""
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=padding)

def compute_output_size(input_size: int, kernel_size: int,
                        stride: int = 1, padding: int = 0) -> int:
    """Compute output spatial dimension after convolution."""
    return (input_size - kernel_size + 2 * padding) // stride + 1
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestConv2d(unittest.TestCase):
    def test_create_conv_layer(self):
        conv = create_conv_layer(3, 16, 3)
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 16)

    def test_create_conv_with_params(self):
        conv = create_conv_layer(1, 32, 5, stride=2, padding=2)
        self.assertEqual(conv.stride, (2, 2))
        self.assertEqual(conv.padding, (2, 2))

    def test_output_size_no_padding(self):
        size = compute_output_size(32, 3, 1, 0)
        self.assertEqual(size, 30)

    def test_output_size_with_padding(self):
        size = compute_output_size(32, 3, 1, 1)
        self.assertEqual(size, 32)

    def test_output_size_with_stride(self):
        size = compute_output_size(64, 3, 2, 1)
        self.assertEqual(size, 32)

    def test_create_conv_kernel_size(self):
        conv = create_conv_layer(3, 16, 5)
        self.assertEqual(conv.kernel_size, (5, 5))

    def test_output_size_large_kernel(self):
        size = compute_output_size(32, 7, 1, 3)
        self.assertEqual(size, 32)

    def test_conv_output_shape(self):
        conv = create_conv_layer(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = conv(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_output_size_small_input(self):
        size = compute_output_size(8, 3, 1, 0)
        self.assertEqual(size, 6)

    def test_conv_returns_tensor(self):
        conv = create_conv_layer(1, 8, 3)
        x = torch.randn(1, 1, 10, 10)
        out = conv(x)
        self.assertIsInstance(out, torch.Tensor)
`,

	hint1: 'Use nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)',
	hint2: 'Output size formula: (input - kernel + 2*padding) // stride + 1',

	whyItMatters: `Conv2D layers are the building blocks of CNNs:

- **Kernel size**: Controls the receptive field of each neuron
- **Stride**: Controls how much the filter moves
- **Padding**: Preserves spatial dimensions
- **Output formula**: Critical for designing architectures

Understanding these parameters is essential for building any CNN.`,

	translations: {
		ru: {
			title: 'Основы слоя Conv2D',
			description: `# Основы слоя Conv2D

Изучите работу 2D сверточных слоев в PyTorch.

## Задача

Реализуйте две функции:
1. \`create_conv_layer\` - Создание слоя Conv2d с указанными параметрами
2. \`compute_output_size\` - Вычисление выходных размерностей после свертки

## Пример

\`\`\`python
conv = create_conv_layer(in_channels=3, out_channels=16, kernel_size=3)
# Returns nn.Conv2d(3, 16, 3)

output_size = compute_output_size(input_size=32, kernel_size=3, stride=1, padding=0)
# Returns 30 (formula: (32 - 3 + 2*0) / 1 + 1 = 30)
\`\`\``,
			hint1: 'Используйте nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)',
			hint2: 'Формула выходного размера: (input - kernel + 2*padding) // stride + 1',
			whyItMatters: `Слои Conv2D - строительные блоки CNN:

- **Размер ядра**: Контролирует рецептивное поле нейрона
- **Шаг (stride)**: Контролирует перемещение фильтра
- **Паддинг**: Сохраняет пространственные размерности
- **Формула выхода**: Критична для проектирования архитектур`,
		},
		uz: {
			title: 'Conv2D qatlami asoslari',
			description: `# Conv2D qatlami asoslari

PyTorch da 2D konvolyutsion qatlamlar qanday ishlashini o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`create_conv_layer\` - Berilgan parametrlar bilan Conv2d qatlami yaratish
2. \`compute_output_size\` - Konvolyutsiyadan keyin chiqish o'lchamlarini hisoblash

## Misol

\`\`\`python
conv = create_conv_layer(in_channels=3, out_channels=16, kernel_size=3)
# Returns nn.Conv2d(3, 16, 3)

output_size = compute_output_size(input_size=32, kernel_size=3, stride=1, padding=0)
# Returns 30 (formula: (32 - 3 + 2*0) / 1 + 1 = 30)
\`\`\``,
			hint1: 'nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) dan foydalaning',
			hint2: "Chiqish o'lchami formulasi: (input - kernel + 2*padding) // stride + 1",
			whyItMatters: `Conv2D qatlamlari CNN ning asosiy bloklari:

- **Kernel o'lchami**: Har bir neyronning qabul maydonini boshqaradi
- **Stride**: Filtr qanchaga harakatlanishini boshqaradi
- **Padding**: Fazoviy o'lchamlarni saqlaydi
- **Chiqish formulasi**: Arxitektura loyihalash uchun muhim`,
		},
	},
};

export default task;
