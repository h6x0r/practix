import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-vgg-block',
	title: 'VGG Block',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'vgg'],
	estimatedTime: '12m',
	isPremium: false,
	order: 9,
	description: `# VGG Block

Implement VGG-style blocks with stacked 3x3 convolutions.

## Task

Implement two functions:
1. \`make_vgg_block\` - Create a VGG block with N conv layers
2. \`VGGNet\` - Simple VGG-style network with multiple blocks

## Example

\`\`\`python
# Create a block with 2 conv layers: 64 -> 128 channels
block = make_vgg_block(64, 128, num_convs=2)

# Simple VGG network
model = VGGNet(num_classes=10)
x = torch.randn(4, 3, 32, 32)
output = model(x)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def make_vgg_block(in_channels: int, out_channels: int,
                   num_convs: int = 2) -> nn.Sequential:
    """Create a VGG-style block with multiple 3x3 convs + max pool."""
    # Your code here
    pass

class VGGNet(nn.Module):
    """Simple VGG-style network for image classification."""

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

def make_vgg_block(in_channels: int, out_channels: int,
                   num_convs: int = 2) -> nn.Sequential:
    """Create a VGG-style block with multiple 3x3 convs + max pool."""
    layers = []
    for i in range(num_convs):
        in_ch = in_channels if i == 0 else out_channels
        layers.append(nn.Conv2d(in_ch, out_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    """Simple VGG-style network for image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3, 64, num_convs=2),
            make_vgg_block(64, 128, num_convs=2),
            make_vgg_block(128, 256, num_convs=3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestVGG(unittest.TestCase):
    def test_make_vgg_block(self):
        block = make_vgg_block(3, 64, num_convs=2)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 16, 16))

    def test_vgg_block_3_convs(self):
        block = make_vgg_block(64, 128, num_convs=3)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 8, 8))

    def test_vggnet_output(self):
        model = VGGNet(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_vggnet_different_classes(self):
        model = VGGNet(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 100))

    def test_block_returns_sequential(self):
        block = make_vgg_block(3, 64, num_convs=2)
        self.assertIsInstance(block, nn.Sequential)

    def test_vggnet_is_module(self):
        model = VGGNet()
        self.assertIsInstance(model, nn.Module)

    def test_block_has_maxpool(self):
        block = make_vgg_block(32, 64, num_convs=2)
        pool_count = sum(1 for m in block.modules() if isinstance(m, nn.MaxPool2d))
        self.assertEqual(pool_count, 1)

    def test_vggnet_has_parameters(self):
        model = VGGNet()
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_single_conv_block(self):
        block = make_vgg_block(16, 32, num_convs=1)
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        self.assertEqual(out.shape, (1, 32, 4, 4))

    def test_vggnet_output_not_nan(self):
        model = VGGNet()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())
`,

	hint1: 'Each VGG block: conv-relu pairs followed by max pool',
	hint2: 'First conv uses in_channels, rest use out_channels',

	whyItMatters: `VGG architecture introduced important principles:

- **Small 3x3 filters**: Two 3x3 = one 5x5 receptive field, fewer params
- **Deeper is better**: Depth over width for representation power
- **Block patterns**: Reusable building blocks simplify design
- **Simplicity**: Uniform architecture, easy to understand

VGG showed that simple, deep networks work remarkably well.`,

	translations: {
		ru: {
			title: 'Блок VGG',
			description: `# Блок VGG

Реализуйте блоки в стиле VGG с последовательными свертками 3x3.

## Задача

Реализуйте две функции:
1. \`make_vgg_block\` - Создание VGG блока с N сверточными слоями
2. \`VGGNet\` - Простая VGG-подобная сеть с несколькими блоками

## Пример

\`\`\`python
# Create a block with 2 conv layers: 64 -> 128 channels
block = make_vgg_block(64, 128, num_convs=2)

# Simple VGG network
model = VGGNet(num_classes=10)
x = torch.randn(4, 3, 32, 32)
output = model(x)
\`\`\``,
			hint1: 'Каждый VGG блок: пары conv-relu с max pool в конце',
			hint2: 'Первая conv использует in_channels, остальные out_channels',
			whyItMatters: `Архитектура VGG ввела важные принципы:

- **Малые фильтры 3x3**: Два 3x3 = одно рецептивное поле 5x5, меньше параметров
- **Глубже лучше**: Глубина важнее ширины для силы представления
- **Паттерны блоков**: Переиспользуемые блоки упрощают дизайн
- **Простота**: Единообразная архитектура, легко понять`,
		},
		uz: {
			title: 'VGG bloki',
			description: `# VGG bloki

Ketma-ket 3x3 konvolyutsiyalar bilan VGG uslubidagi bloklarni amalga oshiring.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`make_vgg_block\` - N ta konvolyutsion qatlam bilan VGG bloki yaratish
2. \`VGGNet\` - Bir nechta blokli oddiy VGG uslubidagi tarmoq

## Misol

\`\`\`python
# Create a block with 2 conv layers: 64 -> 128 channels
block = make_vgg_block(64, 128, num_convs=2)

# Simple VGG network
model = VGGNet(num_classes=10)
x = torch.randn(4, 3, 32, 32)
output = model(x)
\`\`\``,
			hint1: "Har bir VGG blok: oxirida max pool bilan conv-relu juftliklari",
			hint2: "Birinchi conv in_channels dan foydalanadi, qolganlari out_channels",
			whyItMatters: `VGG arxitekturasi muhim printsiplarni kiritdi:

- **Kichik 3x3 filtrlar**: Ikki 3x3 = bitta 5x5 qabul maydoni, kamroq parametrlar
- **Chuqurroq yaxshiroq**: Tasvirlash kuchi uchun kenglikdan ko'ra chuqurlik
- **Blok naqshlari**: Qayta foydalaniladigan bloklar dizaynni soddalashtiradi
- **Oddiylik**: Bir xil arxitektura, tushunish oson`,
		},
	},
};

export default task;
