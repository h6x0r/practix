import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-batch-normalization-cnn',
	title: 'Batch Normalization in CNNs',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'batch-norm'],
	estimatedTime: '12m',
	isPremium: false,
	order: 11,
	description: `# Batch Normalization in CNNs

Apply batch normalization to convolutional networks for faster training.

## Task

Implement a \`ConvBNReLU\` block that combines:
- Conv2d layer
- BatchNorm2d layer
- ReLU activation

Then create a CNN using these blocks.

## Example

\`\`\`python
block = ConvBNReLU(3, 64, kernel_size=3)
x = torch.randn(4, 3, 32, 32)
out = block(x)  # Shape: (4, 64, 32, 32)

model = CNNWithBatchNorm(num_classes=10)
output = model(x)  # Shape: (4, 10)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass

class CNNWithBatchNorm(nn.Module):
    """CNN using ConvBNReLU blocks."""

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

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNWithBatchNorm(nn.Module):
    """CNN using ConvBNReLU blocks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestBatchNormCNN(unittest.TestCase):
    def test_conv_bn_relu_shape(self):
        block = ConvBNReLU(3, 64)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))

    def test_conv_bn_relu_positive(self):
        block = ConvBNReLU(3, 64)
        x = torch.randn(2, 3, 16, 16)
        out = block(x)
        # ReLU outputs should be >= 0
        self.assertTrue((out >= 0).all())

    def test_cnn_output(self):
        model = CNNWithBatchNorm(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_cnn_different_input_sizes(self):
        model = CNNWithBatchNorm(num_classes=10)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_conv_bn_relu_is_module(self):
        block = ConvBNReLU(3, 64)
        self.assertIsInstance(block, nn.Module)

    def test_cnn_is_module(self):
        model = CNNWithBatchNorm()
        self.assertIsInstance(model, nn.Module)

    def test_conv_bn_relu_has_bn(self):
        block = ConvBNReLU(3, 64)
        bn_count = sum(1 for m in block.modules() if isinstance(m, nn.BatchNorm2d))
        self.assertEqual(bn_count, 1)

    def test_cnn_has_parameters(self):
        model = CNNWithBatchNorm()
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_cnn_different_classes(self):
        model = CNNWithBatchNorm(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 100))

    def test_conv_bn_relu_stride(self):
        block = ConvBNReLU(3, 64, stride=2, padding=1)
        x = torch.randn(1, 3, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 64, 16, 16))
`,

	hint1: 'Set bias=False in Conv2d when followed by BatchNorm',
	hint2: 'Use AdaptiveAvgPool2d(1) to handle variable input sizes',

	whyItMatters: `Batch normalization is critical for CNN training:

- **Faster training**: Allows higher learning rates
- **Regularization**: Acts as implicit regularizer
- **Stability**: Reduces internal covariate shift
- **Standard practice**: Used in almost all modern CNNs

ConvBNReLU is a common building block pattern in production networks.`,

	translations: {
		ru: {
			title: 'Batch Normalization в CNN',
			description: `# Batch Normalization в CNN

Примените batch normalization к сверточным сетям для ускорения обучения.

## Задача

Реализуйте блок \`ConvBNReLU\`, объединяющий:
- Слой Conv2d
- Слой BatchNorm2d
- Активацию ReLU

Затем создайте CNN используя эти блоки.

## Пример

\`\`\`python
block = ConvBNReLU(3, 64, kernel_size=3)
x = torch.randn(4, 3, 32, 32)
out = block(x)  # Shape: (4, 64, 32, 32)

model = CNNWithBatchNorm(num_classes=10)
output = model(x)  # Shape: (4, 10)
\`\`\``,
			hint1: 'Установите bias=False в Conv2d когда за ним следует BatchNorm',
			hint2: 'Используйте AdaptiveAvgPool2d(1) для работы с разными размерами входа',
			whyItMatters: `Batch normalization критична для обучения CNN:

- **Быстрое обучение**: Позволяет использовать высокие learning rate
- **Регуляризация**: Действует как неявный регуляризатор
- **Стабильность**: Уменьшает internal covariate shift
- **Стандартная практика**: Используется почти во всех современных CNN`,
		},
		uz: {
			title: "CNN da Batch Normalization",
			description: `# CNN da Batch Normalization

Tezroq o'qitish uchun konvolyutsion tarmoqlarga batch normalization qo'llang.

## Topshiriq

Quyidagilarni birlashtiradigan \`ConvBNReLU\` blokini amalga oshiring:
- Conv2d qatlami
- BatchNorm2d qatlami
- ReLU aktivatsiyasi

Keyin ushbu bloklardan foydalangan holda CNN yarating.

## Misol

\`\`\`python
block = ConvBNReLU(3, 64, kernel_size=3)
x = torch.randn(4, 3, 32, 32)
out = block(x)  # Shape: (4, 64, 32, 32)

model = CNNWithBatchNorm(num_classes=10)
output = model(x)  # Shape: (4, 10)
\`\`\``,
			hint1: "BatchNorm dan keyin kelganda Conv2d da bias=False qo'ying",
			hint2: "O'zgaruvchan kirish o'lchamlari bilan ishlash uchun AdaptiveAvgPool2d(1) dan foydalaning",
			whyItMatters: `Batch normalization CNN o'qitish uchun muhim:

- **Tez o'qitish**: Yuqori learning rate dan foydalanish imkonini beradi
- **Regularizatsiya**: Yashirin regularizator sifatida ishlaydi
- **Barqarorlik**: Internal covariate shift ni kamaytiradi
- **Standart amaliyot**: Deyarli barcha zamonaviy CNN larda ishlatiladi`,
		},
	},
};

export default task;
