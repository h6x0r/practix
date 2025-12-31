import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-model-surgery',
	title: 'Model Surgery',
	difficulty: 'hard',
	tags: ['pytorch', 'transfer-learning', 'architecture'],
	estimatedTime: '18m',
	isPremium: true,
	order: 7,
	description: `# Model Surgery

Learn to modify pretrained model architectures for custom tasks.

## Task

Implement functions to modify model architecture:
1. \`add_dropout\` - Add dropout before classifier
2. \`replace_classifier\` - Replace with custom classifier head
3. \`add_auxiliary_head\` - Add second output head for multi-task learning

## Example

\`\`\`python
model = models.resnet18(pretrained=True)

model = add_dropout(model, p=0.5)
model = replace_classifier(model, num_classes=10, hidden_dim=256)
model = add_auxiliary_head(model, aux_classes=3)

x = torch.randn(4, 3, 224, 224)
main_out, aux_out = model(x)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torchvision import models

def add_dropout(model: nn.Module, p: float = 0.5) -> nn.Module:
    """Add dropout before the classifier."""
    # Your code here
    pass

def replace_classifier(model: nn.Module, num_classes: int,
                       hidden_dim: int = 256) -> nn.Module:
    """Replace classifier with custom head."""
    # Your code here
    pass

class MultiHeadModel(nn.Module):
    """Model with main and auxiliary classification heads."""

    def __init__(self, base_model: nn.Module, main_classes: int, aux_classes: int):
        super().__init__()
        # Your code here
        pass

    def forward(self, x):
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
from torchvision import models

def add_dropout(model: nn.Module, p: float = 0.5) -> nn.Module:
    """Add dropout before the classifier."""
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        out_features = model.fc.out_features
        model.fc = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, out_features)
        )
    return model

def replace_classifier(model: nn.Module, num_classes: int,
                       hidden_dim: int = 256) -> nn.Module:
    """Replace classifier with custom head."""
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    return model

class MultiHeadModel(nn.Module):
    """Model with main and auxiliary classification heads."""

    def __init__(self, base_model: nn.Module, main_classes: int, aux_classes: int):
        super().__init__()
        # Get feature extractor (everything except fc)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Get feature dimension
        if hasattr(base_model, 'fc'):
            feature_dim = base_model.fc.in_features
        else:
            feature_dim = 512

        # Two classification heads
        self.main_head = nn.Linear(feature_dim, main_classes)
        self.aux_head = nn.Linear(feature_dim, aux_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        main_out = self.main_head(features)
        aux_out = self.aux_head(features)

        return main_out, aux_out
`,

	testCode: `import torch
import torch.nn as nn
from torchvision import models
import unittest

class TestModelSurgery(unittest.TestCase):
    def test_add_dropout(self):
        model = models.resnet18(weights=None)
        model = add_dropout(model, p=0.5)
        self.assertIsInstance(model.fc[0], nn.Dropout)

    def test_replace_classifier(self):
        model = models.resnet18(weights=None)
        model = replace_classifier(model, num_classes=10, hidden_dim=128)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_multi_head(self):
        base = models.resnet18(weights=None)
        model = MultiHeadModel(base, main_classes=10, aux_classes=5)
        x = torch.randn(2, 3, 224, 224)
        main_out, aux_out = model(x)
        self.assertEqual(main_out.shape, (2, 10))
        self.assertEqual(aux_out.shape, (2, 5))

    def test_add_dropout_returns_model(self):
        model = models.resnet18(weights=None)
        result = add_dropout(model, p=0.5)
        self.assertIsInstance(result, nn.Module)

    def test_replace_classifier_hidden_dim(self):
        model = models.resnet18(weights=None)
        model = replace_classifier(model, num_classes=5, hidden_dim=64)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (1, 5))

    def test_multi_head_is_module(self):
        base = models.resnet18(weights=None)
        model = MultiHeadModel(base, main_classes=10, aux_classes=5)
        self.assertIsInstance(model, nn.Module)

    def test_multi_head_has_heads(self):
        base = models.resnet18(weights=None)
        model = MultiHeadModel(base, main_classes=10, aux_classes=5)
        self.assertTrue(hasattr(model, 'main_head'))
        self.assertTrue(hasattr(model, 'aux_head'))

    def test_dropout_probability(self):
        model = models.resnet18(weights=None)
        model = add_dropout(model, p=0.7)
        self.assertEqual(model.fc[0].p, 0.7)

    def test_multi_head_single_sample(self):
        base = models.resnet18(weights=None)
        model = MultiHeadModel(base, main_classes=10, aux_classes=5)
        x = torch.randn(1, 3, 224, 224)
        main_out, aux_out = model(x)
        self.assertEqual(main_out.shape, (1, 10))

    def test_replace_returns_model(self):
        model = models.resnet18(weights=None)
        result = replace_classifier(model, num_classes=5)
        self.assertIsInstance(result, nn.Module)
`,

	hint1: 'Access classifier with model.fc for ResNet',
	hint2: 'Multi-head: share features, have separate linear heads',

	whyItMatters: `Model surgery allows customizing architectures:

- **Custom heads**: Match your specific task
- **Regularization**: Add dropout where needed
- **Multi-task**: Share features between tasks
- **Architecture search**: Test different configurations

These skills are essential for adapting models to real-world problems.`,

	translations: {
		ru: {
			title: 'Хирургия моделей',
			description: `# Хирургия моделей

Научитесь модифицировать архитектуры предобученных моделей.

## Задача

Реализуйте функции для модификации архитектуры:
1. \`add_dropout\` - Добавление dropout перед классификатором
2. \`replace_classifier\` - Замена на пользовательский классификатор
3. \`add_auxiliary_head\` - Добавление второй головы для multi-task learning

## Пример

\`\`\`python
model = models.resnet18(pretrained=True)

model = add_dropout(model, p=0.5)
model = replace_classifier(model, num_classes=10, hidden_dim=256)
model = add_auxiliary_head(model, aux_classes=3)

x = torch.randn(4, 3, 224, 224)
main_out, aux_out = model(x)
\`\`\``,
			hint1: 'Доступ к классификатору через model.fc для ResNet',
			hint2: 'Multi-head: общие признаки, отдельные линейные головы',
			whyItMatters: `Хирургия моделей позволяет кастомизировать архитектуры:

- **Кастомные головы**: Соответствие конкретной задаче
- **Регуляризация**: Добавление dropout где нужно
- **Multi-task**: Общие признаки между задачами
- **Поиск архитектуры**: Тестирование разных конфигураций`,
		},
		uz: {
			title: 'Model jarrohlik',
			description: `# Model jarrohlik

Oldindan o'qitilgan model arxitekturalarini o'zgartirishni o'rganing.

## Topshiriq

Arxitekturani o'zgartirish uchun funksiyalarni amalga oshiring:
1. \`add_dropout\` - Klassifikatordan oldin dropout qo'shish
2. \`replace_classifier\` - Maxsus klassifikator boshi bilan almashtirish
3. \`add_auxiliary_head\` - Multi-task o'rganish uchun ikkinchi bosh qo'shish

## Misol

\`\`\`python
model = models.resnet18(pretrained=True)

model = add_dropout(model, p=0.5)
model = replace_classifier(model, num_classes=10, hidden_dim=256)
model = add_auxiliary_head(model, aux_classes=3)

x = torch.randn(4, 3, 224, 224)
main_out, aux_out = model(x)
\`\`\``,
			hint1: "ResNet uchun klassifikatorga model.fc orqali kirish",
			hint2: "Multi-head: umumiy xususiyatlar, alohida chiziqli boshlar",
			whyItMatters: `Model jarrohlik arxitekturalarni moslashtirish imkonini beradi:

- **Maxsus boshlar**: Sizning aniq vazifangizga mos
- **Regularizatsiya**: Kerak joyda dropout qo'shish
- **Multi-task**: Vazifalar o'rtasida xususiyatlarni ulashish
- **Arxitektura qidirish**: Turli konfiguratsiyalarni sinash`,
		},
	},
};

export default task;
