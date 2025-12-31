import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-feature-maps',
	title: 'Visualizing Feature Maps',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'visualization'],
	estimatedTime: '15m',
	isPremium: true,
	order: 7,
	description: `# Visualizing Feature Maps

Learn to extract and visualize intermediate feature maps from CNN layers.

## Task

Implement two functions:
1. \`extract_feature_maps\` - Extract feature maps from a specific layer
2. \`normalize_feature_map\` - Normalize feature map for visualization

## Example

\`\`\`python
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)

# Get feature maps from conv1
feature_maps = extract_feature_maps(model, x, 'conv1')
# feature_maps.shape = (1, 32, 14, 14) for example

# Normalize for display
normalized = normalize_feature_map(feature_maps[0, 0])
# Values in range [0, 1]
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def extract_feature_maps(model: nn.Module, x: torch.Tensor,
                         layer_name: str) -> torch.Tensor:
    """Extract feature maps from a specific layer."""
    # Your code here
    pass

def normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Normalize feature map to [0, 1] for visualization."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

def extract_feature_maps(model: nn.Module, x: torch.Tensor,
                         layer_name: str) -> torch.Tensor:
    """Extract feature maps from a specific layer."""
    activation = None

    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()

    # Register hook on the specified layer
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(x)

    handle.remove()
    return activation

def normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Normalize feature map to [0, 1] for visualization."""
    fmin = feature_map.min()
    fmax = feature_map.max()
    if fmax - fmin > 0:
        return (feature_map - fmin) / (fmax - fmin)
    return torch.zeros_like(feature_map)
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TestFeatureMaps(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.x = torch.randn(1, 1, 28, 28)

    def test_extract_feature_maps(self):
        fm = extract_feature_maps(self.model, self.x, 'conv1')
        self.assertEqual(fm.shape[1], 8)  # 8 output channels

    def test_normalize_range(self):
        fm = torch.randn(14, 14) * 10 + 5
        normalized = normalize_feature_map(fm)
        self.assertGreaterEqual(normalized.min().item(), 0)
        self.assertLessEqual(normalized.max().item(), 1)

    def test_normalize_constant(self):
        fm = torch.ones(5, 5) * 3
        normalized = normalize_feature_map(fm)
        self.assertEqual(normalized.sum().item(), 0)

    def test_extract_from_conv2(self):
        fm = extract_feature_maps(self.model, self.x, 'conv2')
        self.assertEqual(fm.shape[1], 16)

    def test_feature_map_is_tensor(self):
        fm = extract_feature_maps(self.model, self.x, 'conv1')
        self.assertIsInstance(fm, torch.Tensor)

    def test_normalized_is_tensor(self):
        fm = torch.randn(10, 10)
        normalized = normalize_feature_map(fm)
        self.assertIsInstance(normalized, torch.Tensor)

    def test_normalize_preserves_shape(self):
        fm = torch.randn(14, 14)
        normalized = normalize_feature_map(fm)
        self.assertEqual(normalized.shape, fm.shape)

    def test_extract_batch_size(self):
        x = torch.randn(4, 1, 28, 28)
        fm = extract_feature_maps(self.model, x, 'conv1')
        self.assertEqual(fm.shape[0], 4)

    def test_normalize_negative_values(self):
        fm = torch.randn(5, 5) - 5  # Mostly negative
        normalized = normalize_feature_map(fm)
        self.assertGreaterEqual(normalized.min().item(), 0)

    def test_normalize_max_equals_one(self):
        fm = torch.randn(8, 8) * 100
        normalized = normalize_feature_map(fm)
        self.assertAlmostEqual(normalized.max().item(), 1.0, places=5)
`,

	hint1: 'Use register_forward_hook to capture intermediate outputs',
	hint2: 'Min-max normalization: (x - min) / (max - min)',

	whyItMatters: `Visualizing feature maps helps understand CNNs:

- **Debug models**: See what each layer learns
- **Interpret predictions**: Understand why model made a decision
- **Research**: Analyze filter responses
- **Hooks mechanism**: Powerful PyTorch feature for introspection

This technique is essential for explainable AI in computer vision.`,

	translations: {
		ru: {
			title: 'Визуализация карт признаков',
			description: `# Визуализация карт признаков

Научитесь извлекать и визуализировать промежуточные карты признаков из слоев CNN.

## Задача

Реализуйте две функции:
1. \`extract_feature_maps\` - Извлечение карт признаков из конкретного слоя
2. \`normalize_feature_map\` - Нормализация карты признаков для визуализации

## Пример

\`\`\`python
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)

# Get feature maps from conv1
feature_maps = extract_feature_maps(model, x, 'conv1')
# feature_maps.shape = (1, 32, 14, 14) for example

# Normalize for display
normalized = normalize_feature_map(feature_maps[0, 0])
# Values in range [0, 1]
\`\`\``,
			hint1: 'Используйте register_forward_hook для захвата промежуточных выходов',
			hint2: 'Min-max нормализация: (x - min) / (max - min)',
			whyItMatters: `Визуализация карт признаков помогает понять CNN:

- **Отладка моделей**: Видеть что учит каждый слой
- **Интерпретация предсказаний**: Понимать решения модели
- **Исследования**: Анализ откликов фильтров
- **Механизм хуков**: Мощная возможность PyTorch для интроспекции`,
		},
		uz: {
			title: 'Xususiyat xaritalarini vizualizatsiya qilish',
			description: `# Xususiyat xaritalarini vizualizatsiya qilish

CNN qatlamlaridan oraliq xususiyat xaritalarini ajratish va vizualizatsiya qilishni o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`extract_feature_maps\` - Ma'lum qatlamdan xususiyat xaritalarini ajratish
2. \`normalize_feature_map\` - Vizualizatsiya uchun xususiyat xaritasini normalizatsiya qilish

## Misol

\`\`\`python
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)

# Get feature maps from conv1
feature_maps = extract_feature_maps(model, x, 'conv1')
# feature_maps.shape = (1, 32, 14, 14) for example

# Normalize for display
normalized = normalize_feature_map(feature_maps[0, 0])
# Values in range [0, 1]
\`\`\``,
			hint1: "Oraliq chiqishlarni olish uchun register_forward_hook dan foydalaning",
			hint2: 'Min-max normalizatsiya: (x - min) / (max - min)',
			whyItMatters: `Xususiyat xaritalarini vizualizatsiya qilish CNN larni tushunishga yordam beradi:

- **Modellarni sozlash**: Har bir qatlam nimani o'rganayotganini ko'rish
- **Bashoratlarni talqin qilish**: Model qarorlarini tushunish
- **Tadqiqot**: Filtr javoblarini tahlil qilish
- **Hooklar mexanizmi**: Introspektsiya uchun kuchli PyTorch xususiyati`,
		},
	},
};

export default task;
