import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-domain-adaptation',
	title: 'Domain Adaptation',
	difficulty: 'hard',
	tags: ['pytorch', 'transfer-learning', 'domain-adaptation'],
	estimatedTime: '18m',
	isPremium: true,
	order: 8,
	description: `# Domain Adaptation

Learn techniques for adapting models to new domains.

## Task

Implement a \`DomainAdaptiveModel\` with:
- Feature extractor (shared)
- Class predictor
- Domain discriminator for adversarial training

## Example

\`\`\`python
model = DomainAdaptiveModel(num_classes=10, feature_dim=512)

source_images = torch.randn(32, 3, 224, 224)
target_images = torch.randn(32, 3, 224, 224)

class_pred, domain_pred = model(source_images, alpha=0.5)
# class_pred: class predictions
# domain_pred: source vs target domain prediction
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    """Gradient reversal for adversarial domain adaptation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainAdaptiveModel(nn.Module):
    """Model with domain adaptation using gradient reversal."""

    def __init__(self, num_classes: int, feature_dim: int = 512):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """
        Returns:
            class_output: class predictions
            domain_output: domain predictions (source=0, target=1)
        """
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

class GradientReversalLayer(Function):
    """Gradient reversal for adversarial domain adaptation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class DomainAdaptiveModel(nn.Module):
    """Model with domain adaptation using gradient reversal."""

    def __init__(self, num_classes: int, feature_dim: int = 512):
        super().__init__()

        # Feature extractor
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Class predictor
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # source vs target
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        # Class prediction (normal gradient)
        class_output = self.class_classifier(features)

        # Domain prediction (reversed gradient)
        reversed_features = grad_reverse(features, alpha)
        domain_output = self.domain_classifier(reversed_features)

        return class_output, domain_output
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestDomainAdaptation(unittest.TestCase):
    def test_output_shapes(self):
        model = DomainAdaptiveModel(num_classes=10)
        x = torch.randn(4, 3, 224, 224)
        class_out, domain_out = model(x, alpha=1.0)
        self.assertEqual(class_out.shape, (4, 10))
        self.assertEqual(domain_out.shape, (4, 2))

    def test_gradient_reversal(self):
        model = DomainAdaptiveModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        class_out, domain_out = model(x, alpha=1.0)
        loss = class_out.sum() + domain_out.sum()
        loss.backward()
        # Should have gradients
        self.assertIsNotNone(x.grad)

    def test_different_alpha(self):
        model = DomainAdaptiveModel(num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        out1 = model(x, alpha=0.0)
        out2 = model(x, alpha=1.0)
        # Class predictions should be same
        self.assertTrue(torch.allclose(out1[0], out2[0]))

    def test_is_nn_module(self):
        model = DomainAdaptiveModel(num_classes=10)
        self.assertIsInstance(model, nn.Module)

    def test_has_classifiers(self):
        model = DomainAdaptiveModel(num_classes=10)
        self.assertTrue(hasattr(model, 'class_classifier'))
        self.assertTrue(hasattr(model, 'domain_classifier'))

    def test_has_features(self):
        model = DomainAdaptiveModel(num_classes=10)
        self.assertTrue(hasattr(model, 'features'))

    def test_single_sample(self):
        model = DomainAdaptiveModel(num_classes=5)
        x = torch.randn(1, 3, 224, 224)
        class_out, domain_out = model(x)
        self.assertEqual(class_out.shape, (1, 5))
        self.assertEqual(domain_out.shape, (1, 2))

    def test_class_output_not_nan(self):
        model = DomainAdaptiveModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        class_out, _ = model(x)
        self.assertFalse(torch.isnan(class_out).any())

    def test_domain_output_not_nan(self):
        model = DomainAdaptiveModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        _, domain_out = model(x)
        self.assertFalse(torch.isnan(domain_out).any())

    def test_large_batch(self):
        model = DomainAdaptiveModel(num_classes=10)
        x = torch.randn(8, 3, 224, 224)
        class_out, domain_out = model(x)
        self.assertEqual(class_out.shape, (8, 10))
`,

	hint1: 'Gradient reversal negates gradients during backward pass',
	hint2: 'Alpha controls the strength of gradient reversal',

	whyItMatters: `Domain adaptation handles dataset shift:

- **Different domains**: Train on one dataset, test on another
- **Adversarial training**: Learn domain-invariant features
- **Real-world problem**: Data distribution often changes
- **Unlabeled target**: Can adapt without target labels

This is crucial for models that must generalize to new data.`,

	translations: {
		ru: {
			title: 'Адаптация домена',
			description: `# Адаптация домена

Изучите техники адаптации моделей к новым доменам.

## Задача

Реализуйте \`DomainAdaptiveModel\` с:
- Экстрактор признаков (общий)
- Предиктор классов
- Дискриминатор домена для adversarial обучения

## Пример

\`\`\`python
model = DomainAdaptiveModel(num_classes=10, feature_dim=512)

source_images = torch.randn(32, 3, 224, 224)
target_images = torch.randn(32, 3, 224, 224)

class_pred, domain_pred = model(source_images, alpha=0.5)
# class_pred: class predictions
# domain_pred: source vs target domain prediction
\`\`\``,
			hint1: 'Gradient reversal инвертирует градиенты при обратном проходе',
			hint2: 'Alpha контролирует силу инверсии градиентов',
			whyItMatters: `Адаптация домена решает проблему смещения данных:

- **Разные домены**: Обучение на одних данных, тест на других
- **Adversarial обучение**: Обучение домен-инвариантных признаков
- **Реальная проблема**: Распределение данных часто меняется
- **Без меток целевого домена**: Можно адаптировать без меток`,
		},
		uz: {
			title: "Domen adaptatsiyasi",
			description: `# Domen adaptatsiyasi

Modellarni yangi domenlarga moslash texnikalarini o'rganing.

## Topshiriq

\`DomainAdaptiveModel\` ni amalga oshiring:
- Xususiyat ajratuvchi (umumiy)
- Sinf bashoratlovchi
- Adversarial o'qitish uchun domen diskriminatori

## Misol

\`\`\`python
model = DomainAdaptiveModel(num_classes=10, feature_dim=512)

source_images = torch.randn(32, 3, 224, 224)
target_images = torch.randn(32, 3, 224, 224)

class_pred, domain_pred = model(source_images, alpha=0.5)
# class_pred: class predictions
# domain_pred: source vs target domain prediction
\`\`\``,
			hint1: "Gradient reversal orqaga o'tishda gradientlarni invert qiladi",
			hint2: "Alpha gradient inversion kuchini boshqaradi",
			whyItMatters: `Domen adaptatsiyasi ma'lumotlar siljishi muammosini hal qiladi:

- **Turli domenlar**: Bitta ma'lumotlarda o'qitish, boshqasida test
- **Adversarial o'qitish**: Domenga bog'liq bo'lmagan xususiyatlarni o'rganish
- **Haqiqiy muammo**: Ma'lumotlar taqsimoti ko'pincha o'zgaradi
- **Maqsadli belgisiz**: Belgilarsiz moslashish mumkin`,
		},
	},
};

export default task;
