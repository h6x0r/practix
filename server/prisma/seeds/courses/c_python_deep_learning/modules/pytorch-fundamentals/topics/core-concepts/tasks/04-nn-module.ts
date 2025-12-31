import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pytorch-nn-module',
	title: 'Building with nn.Module',
	difficulty: 'medium',
	tags: ['pytorch', 'nn.Module', 'neural-networks'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Building with nn.Module

Learn to build neural networks using nn.Module.

## Task

Implement three classes:
1. \`SimpleMLP\` - 2-layer MLP with ReLU activation
2. \`FlexibleMLP\` - MLP with configurable hidden sizes
3. \`ResidualBlock\` - Block with skip connection

## Example

\`\`\`python
import torch.nn as nn

# Simple 2-layer MLP
model = SimpleMLP(input_size=784, hidden_size=128, output_size=10)
output = model(x)  # x: (batch, 784) -> (batch, 10)

# Flexible MLP
model = FlexibleMLP(784, [256, 128, 64], 10)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """2-layer MLP: input -> hidden (ReLU) -> output."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass

class FlexibleMLP(nn.Module):
    """MLP with configurable hidden layers."""
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass

class ResidualBlock(nn.Module):
    """Block with skip connection: output = relu(linear(x)) + x."""
    def __init__(self, size: int):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """2-layer MLP: input -> hidden (ReLU) -> output."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FlexibleMLP(nn.Module):
    """MLP with configurable hidden layers."""
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            prev_size = hidden
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ResidualBlock(nn.Module):
    """Block with skip connection: output = relu(linear(x)) + x."""
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x)) + x
`,

	testCode: `import torch
import unittest

class TestNNModule(unittest.TestCase):
    def test_simple_mlp_output_shape(self):
        model = SimpleMLP(784, 128, 10)
        x = torch.randn(32, 784)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 10]))

    def test_simple_mlp_has_parameters(self):
        model = SimpleMLP(784, 128, 10)
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_flexible_mlp(self):
        model = FlexibleMLP(100, [64, 32], 10)
        x = torch.randn(16, 100)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([16, 10]))

    def test_residual_block_shape(self):
        block = ResidualBlock(64)
        x = torch.randn(8, 64)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_residual_block_skip(self):
        block = ResidualBlock(64)
        x = torch.zeros(1, 64)
        out = block(x)
        # With zero input and zero-initialized bias, output should be close to x
        self.assertEqual(out.shape, x.shape)

    def test_simple_mlp_is_module(self):
        model = SimpleMLP(100, 50, 10)
        self.assertIsInstance(model, torch.nn.Module)

    def test_flexible_mlp_deep(self):
        model = FlexibleMLP(100, [64, 32, 16], 10)
        x = torch.randn(8, 100)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([8, 10]))

    def test_simple_mlp_param_count(self):
        model = SimpleMLP(10, 5, 2)
        params = sum(p.numel() for p in model.parameters())
        self.assertGreater(params, 0)

    def test_residual_block_is_module(self):
        block = ResidualBlock(32)
        self.assertIsInstance(block, torch.nn.Module)

    def test_flexible_mlp_single_hidden(self):
        model = FlexibleMLP(50, [25], 5)
        x = torch.randn(4, 50)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([4, 5]))
`,

	hint1: 'Define layers in __init__, use them in forward()',
	hint2: 'nn.Sequential(*layers) creates a sequential container from a list',

	whyItMatters: `nn.Module is the building block of PyTorch models:

- **Encapsulation**: Bundle layers, parameters, and forward logic
- **Composability**: Modules can contain other modules
- **Parameter management**: Automatic tracking of trainable parameters
- **Portability**: Easy to save, load, and move to GPU

Every PyTorch model is an nn.Module.`,

	translations: {
		ru: {
			title: 'Построение с nn.Module',
			description: `# Построение с nn.Module

Научитесь строить нейросети используя nn.Module.

## Задача

Реализуйте три класса:
1. \`SimpleMLP\` - 2-слойный MLP с ReLU
2. \`FlexibleMLP\` - MLP с настраиваемыми скрытыми слоями
3. \`ResidualBlock\` - Блок со skip-соединением

## Пример

\`\`\`python
import torch.nn as nn

# Simple 2-layer MLP
model = SimpleMLP(input_size=784, hidden_size=128, output_size=10)
output = model(x)  # x: (batch, 784) -> (batch, 10)

# Flexible MLP
model = FlexibleMLP(784, [256, 128, 64], 10)
\`\`\``,
			hint1: 'Определите слои в __init__, используйте их в forward()',
			hint2: 'nn.Sequential(*layers) создаёт последовательный контейнер из списка',
			whyItMatters: `nn.Module - строительный блок моделей PyTorch:

- **Инкапсуляция**: Объединение слоёв, параметров и логики forward
- **Композиция**: Модули могут содержать другие модули
- **Управление параметрами**: Автоматическое отслеживание`,
		},
		uz: {
			title: 'nn.Module bilan qurish',
			description: `# nn.Module bilan qurish

nn.Module yordamida neyrosetka qurishni o'rganing.

## Topshiriq

Uchta sinfni amalga oshiring:
1. \`SimpleMLP\` - ReLU aktivatsiyasi bilan 2 qatlamli MLP
2. \`FlexibleMLP\` - Sozlanuvchi yashirin o'lchamli MLP
3. \`ResidualBlock\` - Skip ulanishi bilan blok

## Misol

\`\`\`python
import torch.nn as nn

# Simple 2-layer MLP
model = SimpleMLP(input_size=784, hidden_size=128, output_size=10)
output = model(x)  # x: (batch, 784) -> (batch, 10)

# Flexible MLP
model = FlexibleMLP(784, [256, 128, 64], 10)
\`\`\``,
			hint1: "__init__ da qatlamlarni aniqlang, forward() da foydalaning",
			hint2: "nn.Sequential(*layers) ro'yxatdan ketma-ket konteyner yaratadi",
			whyItMatters: `nn.Module PyTorch modellarining qurilish bloki:

- **Inkapsulyatsiya**: Qatlamlar, parametrlar va forward logikani birlashtirish
- **Kompozitsiya**: Modullar boshqa modullarni o'z ichiga olishi mumkin
- **Parametr boshqaruvi**: O'qitiladigan parametrlarni avtomatik kuzatish`,
		},
	},
};

export default task;
