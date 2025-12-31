import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-fine-tuning',
	title: 'Fine-Tuning Models',
	difficulty: 'medium',
	tags: ['pytorch', 'transfer-learning', 'fine-tuning'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Fine-Tuning Models

Learn to fine-tune pretrained models on custom datasets.

## Task

Implement a \`FineTunedModel\` class that:
- Loads a pretrained model
- Replaces the classifier head for a new number of classes
- Supports partial freezing of layers

## Example

\`\`\`python
model = FineTunedModel(
    base_model='resnet18',
    num_classes=10,
    freeze_backbone=True  # Only train the new head
)

x = torch.randn(4, 3, 224, 224)
output = model(x)
# output.shape = (4, 10)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torchvision import models

class FineTunedModel(nn.Module):
    """Fine-tuned model for custom classification."""

    def __init__(self, base_model: str = 'resnet18',
                 num_classes: int = 10, freeze_backbone: bool = True):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for full fine-tuning."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
from torchvision import models

class FineTunedModel(nn.Module):
    """Fine-tuned model for custom classification."""

    def __init__(self, base_model: str = 'resnet18',
                 num_classes: int = 10, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained model
        if base_model == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
        elif base_model == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
        else:
            raise ValueError(f"Unknown model: {base_model}")

        # Replace classifier
        self.model.fc = nn.Linear(num_features, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestFineTuning(unittest.TestCase):
    def test_output_shape(self):
        model = FineTunedModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_frozen_backbone(self):
        model = FineTunedModel(freeze_backbone=True)
        # FC should be trainable
        self.assertTrue(model.model.fc.weight.requires_grad)
        # Conv1 should be frozen
        self.assertFalse(model.model.conv1.weight.requires_grad)

    def test_unfreeze(self):
        model = FineTunedModel(freeze_backbone=True)
        model.unfreeze_backbone()
        self.assertTrue(model.model.conv1.weight.requires_grad)

    def test_different_num_classes(self):
        for num_classes in [5, 20, 100]:
            model = FineTunedModel(num_classes=num_classes)
            x = torch.randn(2, 3, 224, 224)
            out = model(x)
            self.assertEqual(out.shape, (2, num_classes))

    def test_is_nn_module(self):
        model = FineTunedModel()
        self.assertIsInstance(model, nn.Module)

    def test_single_image(self):
        model = FineTunedModel(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (1, 10))

    def test_not_frozen_fc(self):
        model = FineTunedModel(freeze_backbone=True)
        self.assertTrue(model.model.fc.weight.requires_grad)

    def test_output_not_nan(self):
        model = FineTunedModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())

    def test_backbone_initially_frozen(self):
        model = FineTunedModel(freeze_backbone=True)
        self.assertFalse(model.model.layer1[0].conv1.weight.requires_grad)

    def test_unfreeze_all_layers(self):
        model = FineTunedModel(freeze_backbone=True)
        model.unfreeze_backbone()
        for param in model.model.parameters():
            self.assertTrue(param.requires_grad)
`,

	hint1: 'Replace model.fc = nn.Linear(in_features, num_classes)',
	hint2: 'Check "fc" in name when freezing to skip the classifier',

	whyItMatters: `Fine-tuning adapts pretrained models to your task:

- **Transfer knowledge**: Use ImageNet features for any task
- **Better accuracy**: Beats training from scratch
- **Less data needed**: Pretrained weights help generalize
- **Progressive unfreezing**: Start frozen, then unfreeze

Fine-tuning is the most common transfer learning approach.`,

	translations: {
		ru: {
			title: 'Тонкая настройка моделей',
			description: `# Тонкая настройка моделей

Научитесь дообучать предобученные модели на пользовательских данных.

## Задача

Реализуйте класс \`FineTunedModel\`, который:
- Загружает предобученную модель
- Заменяет классификатор для нового числа классов
- Поддерживает частичную заморозку слоев

## Пример

\`\`\`python
model = FineTunedModel(
    base_model='resnet18',
    num_classes=10,
    freeze_backbone=True  # Only train the new head
)

x = torch.randn(4, 3, 224, 224)
output = model(x)
# output.shape = (4, 10)
\`\`\``,
			hint1: 'Замените model.fc = nn.Linear(in_features, num_classes)',
			hint2: 'Проверяйте "fc" in name при заморозке, чтобы пропустить классификатор',
			whyItMatters: `Тонкая настройка адаптирует предобученные модели:

- **Передача знаний**: Использование признаков ImageNet для любой задачи
- **Лучшая точность**: Превосходит обучение с нуля
- **Меньше данных**: Предобученные веса помогают обобщать
- **Постепенная разморозка**: Начать замороженным, затем разморозить`,
		},
		uz: {
			title: "Modellarni nozik sozlash",
			description: `# Modellarni nozik sozlash

Oldindan o'qitilgan modellarni maxsus ma'lumotlarda nozik sozlashni o'rganing.

## Topshiriq

\`FineTunedModel\` sinfini amalga oshiring:
- Oldindan o'qitilgan modelni yuklaydi
- Yangi sinflar soni uchun klassifikator boshini almashtiradi
- Qatlamlarni qisman muzlatishni qo'llab-quvvatlaydi

## Misol

\`\`\`python
model = FineTunedModel(
    base_model='resnet18',
    num_classes=10,
    freeze_backbone=True  # Only train the new head
)

x = torch.randn(4, 3, 224, 224)
output = model(x)
# output.shape = (4, 10)
\`\`\``,
			hint1: "model.fc = nn.Linear(in_features, num_classes) bilan almashtiring",
			hint2: "Klassifikatorni o'tkazib yuborish uchun muzlatishda \"fc\" in name ni tekshiring",
			whyItMatters: `Nozik sozlash oldindan o'qitilgan modellarni vazifangizga moslaydi:

- **Bilim uzatish**: Har qanday vazifa uchun ImageNet xususiyatlaridan foydalanish
- **Yaxshiroq aniqlik**: Noldan o'qitishdan ustun
- **Kamroq ma'lumot kerak**: Oldindan o'qitilgan vaznlar umumlashtirishga yordam beradi
- **Bosqichma-bosqich muzdan chiqarish**: Muzlatilgan holda boshlash, keyin muzdan chiqarish`,
		},
	},
};

export default task;
