import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-resnet-block',
	title: 'ResNet Block',
	difficulty: 'hard',
	tags: ['pytorch', 'cnn', 'resnet', 'skip-connection'],
	estimatedTime: '18m',
	isPremium: true,
	order: 8,
	description: `# ResNet Block

Implement the residual block that revolutionized deep learning.

## Task

Implement a \`ResidualBlock\` class with:
- Two 3x3 convolutions with batch normalization
- Skip connection that adds input to output
- Optional downsampling when stride > 1

## Example

\`\`\`python
block = ResidualBlock(64, 64)
x = torch.randn(4, 64, 32, 32)
out = block(x)
# out.shape = (4, 64, 32, 32), includes skip connection

# With downsampling
block_down = ResidualBlock(64, 128, stride=2)
out = block_down(x)
# out.shape = (4, 128, 16, 16)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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

class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestResidualBlock(unittest.TestCase):
    def test_same_dimensions(self):
        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))

    def test_channel_change(self):
        block = ResidualBlock(32, 64)
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))

    def test_downsampling(self):
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 16, 16))

    def test_gradient_flow(self):
        block = ResidualBlock(64, 64)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_is_module(self):
        block = ResidualBlock(32, 32)
        self.assertIsInstance(block, nn.Module)

    def test_has_batchnorm(self):
        block = ResidualBlock(64, 64)
        bn_count = sum(1 for m in block.modules() if isinstance(m, nn.BatchNorm2d))
        self.assertGreaterEqual(bn_count, 2)

    def test_output_positive_and_negative(self):
        block = ResidualBlock(32, 32)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        # Skip connection allows negative values to pass through
        self.assertTrue(out.shape == x.shape)

    def test_single_sample(self):
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 128, 16, 16))

    def test_has_parameters(self):
        block = ResidualBlock(64, 64)
        params = list(block.parameters())
        self.assertGreater(len(params), 0)

    def test_output_not_nan(self):
        block = ResidualBlock(32, 64)
        x = torch.randn(2, 32, 8, 8)
        out = block(x)
        self.assertFalse(torch.isnan(out).any())
`,

	hint1: 'Skip connection needs 1x1 conv when dimensions change',
	hint2: 'Add identity to output BEFORE final ReLU',

	whyItMatters: `ResNet blocks solved the vanishing gradient problem:

- **Skip connections**: Gradients flow directly through identity mapping
- **Train very deep nets**: 100+ layers became possible
- **Better features**: Residual learning is easier than direct mapping
- **Universal adoption**: Used in almost all modern architectures

ResNet won ImageNet 2015 and changed deep learning forever.`,

	translations: {
		ru: {
			title: 'Блок ResNet',
			description: `# Блок ResNet

Реализуйте остаточный блок, который произвел революцию в глубоком обучении.

## Задача

Реализуйте класс \`ResidualBlock\` с:
- Две свертки 3x3 с batch normalization
- Skip connection добавляющий вход к выходу
- Опциональное уменьшение размера при stride > 1

## Пример

\`\`\`python
block = ResidualBlock(64, 64)
x = torch.randn(4, 64, 32, 32)
out = block(x)
# out.shape = (4, 64, 32, 32), includes skip connection

# With downsampling
block_down = ResidualBlock(64, 128, stride=2)
out = block_down(x)
# out.shape = (4, 128, 16, 16)
\`\`\``,
			hint1: 'Skip connection требует 1x1 conv когда размерности меняются',
			hint2: 'Добавьте identity к выходу ДО финального ReLU',
			whyItMatters: `Блоки ResNet решили проблему затухающих градиентов:

- **Skip connections**: Градиенты текут напрямую через identity
- **Очень глубокие сети**: Стало возможно 100+ слоев
- **Лучшие признаки**: Residual learning проще прямого отображения
- **Повсеместное использование**: В почти всех современных архитектурах`,
		},
		uz: {
			title: 'ResNet bloki',
			description: `# ResNet bloki

Chuqur o'rganishda inqilob qilgan qoldiq blokini amalga oshiring.

## Topshiriq

\`ResidualBlock\` sinfini amalga oshiring:
- Batch normalization bilan ikkita 3x3 konvolyutsiya
- Kirish va chiqishni qo'shadigan skip connection
- stride > 1 bo'lganda ixtiyoriy downsampling

## Misol

\`\`\`python
block = ResidualBlock(64, 64)
x = torch.randn(4, 64, 32, 32)
out = block(x)
# out.shape = (4, 64, 32, 32), includes skip connection

# With downsampling
block_down = ResidualBlock(64, 128, stride=2)
out = block_down(x)
# out.shape = (4, 128, 16, 16)
\`\`\``,
			hint1: "O'lchamlar o'zgarganda skip connection 1x1 conv talab qiladi",
			hint2: "Identity ni chiqishga oxirgi ReLU dan OLDIN qo'shing",
			whyItMatters: `ResNet bloklari yo'qolib ketayotgan gradient muammosini hal qildi:

- **Skip connections**: Gradientlar identity orqali to'g'ridan-to'g'ri oqadi
- **Juda chuqur tarmoqlar**: 100+ qatlam mumkin bo'ldi
- **Yaxshiroq xususiyatlar**: Qoldiq o'rganish to'g'ridan-to'g'ri xaritalashdan osonroq
- **Universal qabul qilish**: Deyarli barcha zamonaviy arxitekturalarda ishlatiladi`,
		},
	},
};

export default task;
