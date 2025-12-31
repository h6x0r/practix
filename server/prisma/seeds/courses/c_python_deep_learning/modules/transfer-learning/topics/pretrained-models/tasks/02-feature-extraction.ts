import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-feature-extraction',
	title: 'Feature Extraction',
	difficulty: 'medium',
	tags: ['pytorch', 'transfer-learning', 'features'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Feature Extraction

Use pretrained models as fixed feature extractors.

## Task

Implement a \`FeatureExtractor\` class that:
- Uses a pretrained model's convolutional layers
- Freezes all pretrained weights
- Extracts features from images

## Example

\`\`\`python
extractor = FeatureExtractor(model_name='resnet18')

images = torch.randn(4, 3, 224, 224)
features = extractor(images)
# features.shape = (4, 512) - 512 features from ResNet-18
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """Extract features using pretrained model."""

    def __init__(self, model_name: str = 'resnet18'):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """Extract features using pretrained model."""

    def __init__(self, model_name: str = 'resnet18'):
        super().__init__()

        # Load pretrained model
        if model_name == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove the classification head
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # Freeze all weights
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestFeatureExtractor(unittest.TestCase):
    def test_output_shape(self):
        extractor = FeatureExtractor('resnet18')
        x = torch.randn(4, 3, 224, 224)
        features = extractor(x)
        self.assertEqual(features.shape, (4, 512))

    def test_frozen_weights(self):
        extractor = FeatureExtractor('resnet18')
        for param in extractor.features.parameters():
            self.assertFalse(param.requires_grad)

    def test_feature_dim_attribute(self):
        extractor = FeatureExtractor('resnet18')
        self.assertEqual(extractor.feature_dim, 512)

    def test_is_nn_module(self):
        extractor = FeatureExtractor('resnet18')
        self.assertIsInstance(extractor, nn.Module)

    def test_single_image(self):
        extractor = FeatureExtractor('resnet18')
        x = torch.randn(1, 3, 224, 224)
        features = extractor(x)
        self.assertEqual(features.shape, (1, 512))

    def test_larger_batch(self):
        extractor = FeatureExtractor('resnet18')
        x = torch.randn(8, 3, 224, 224)
        features = extractor(x)
        self.assertEqual(features.shape, (8, 512))

    def test_features_are_tensors(self):
        extractor = FeatureExtractor('resnet18')
        x = torch.randn(2, 3, 224, 224)
        features = extractor(x)
        self.assertIsInstance(features, torch.Tensor)

    def test_no_grad_computation(self):
        extractor = FeatureExtractor('resnet18')
        extractor.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = extractor(x)
        self.assertEqual(features.shape, (2, 512))

    def test_features_not_all_zeros(self):
        extractor = FeatureExtractor('resnet18')
        x = torch.randn(2, 3, 224, 224)
        features = extractor(x)
        self.assertGreater(features.abs().sum().item(), 0)

    def test_has_features_attribute(self):
        extractor = FeatureExtractor('resnet18')
        self.assertTrue(hasattr(extractor, 'features'))
`,

	hint1: 'Remove last layer with list(model.children())[:-1]',
	hint2: 'Freeze with: for param in model.parameters(): param.requires_grad = False',

	whyItMatters: `Feature extraction is a powerful transfer learning technique:

- **Fixed features**: Don't need to train the backbone
- **Small datasets**: Works well with few examples
- **Fast training**: Only train a small classifier
- **Versatile**: Same features for multiple tasks

This approach is ideal when you have limited data or compute.`,

	translations: {
		ru: {
			title: 'Извлечение признаков',
			description: `# Извлечение признаков

Используйте предобученные модели как фиксированные экстракторы признаков.

## Задача

Реализуйте класс \`FeatureExtractor\`, который:
- Использует сверточные слои предобученной модели
- Замораживает все предобученные веса
- Извлекает признаки из изображений

## Пример

\`\`\`python
extractor = FeatureExtractor(model_name='resnet18')

images = torch.randn(4, 3, 224, 224)
features = extractor(images)
# features.shape = (4, 512) - 512 features from ResNet-18
\`\`\``,
			hint1: 'Удалите последний слой с list(model.children())[:-1]',
			hint2: 'Заморозьте: for param in model.parameters(): param.requires_grad = False',
			whyItMatters: `Извлечение признаков - мощная техника transfer learning:

- **Фиксированные признаки**: Не нужно обучать backbone
- **Малые датасеты**: Работает с небольшим числом примеров
- **Быстрое обучение**: Обучаем только маленький классификатор
- **Универсальность**: Одни признаки для разных задач`,
		},
		uz: {
			title: 'Xususiyat ajratish',
			description: `# Xususiyat ajratish

Oldindan o'qitilgan modellarni qat'iy xususiyat ajratuvchilar sifatida ishlating.

## Topshiriq

\`FeatureExtractor\` sinfini amalga oshiring:
- Oldindan o'qitilgan model ning konvolyutsion qatlamlaridan foydalanadi
- Barcha oldindan o'qitilgan vaznlarni muzlatadi
- Tasvirlardan xususiyatlarni ajratadi

## Misol

\`\`\`python
extractor = FeatureExtractor(model_name='resnet18')

images = torch.randn(4, 3, 224, 224)
features = extractor(images)
# features.shape = (4, 512) - 512 features from ResNet-18
\`\`\``,
			hint1: "Oxirgi qatlamni list(model.children())[:-1] bilan olib tashlang",
			hint2: "Muzlatish: for param in model.parameters(): param.requires_grad = False",
			whyItMatters: `Xususiyat ajratish kuchli transfer learning texnikasi:

- **Qat'iy xususiyatlar**: Backbone ni o'qitish kerak emas
- **Kichik ma'lumotlar**: Kam misollar bilan yaxshi ishlaydi
- **Tez o'qitish**: Faqat kichik klassifikatorni o'qitamiz
- **Universallik**: Turli vazifalar uchun bir xil xususiyatlar`,
		},
	},
};

export default task;
