import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-load-pretrained',
	title: 'Loading Pretrained Models',
	difficulty: 'easy',
	tags: ['pytorch', 'transfer-learning', 'pretrained'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Loading Pretrained Models

Learn to load pretrained models from torchvision.

## Task

Implement two functions:
1. \`load_resnet\` - Load pretrained ResNet-18 model
2. \`get_model_info\` - Return model info (number of parameters, output size)

## Example

\`\`\`python
model = load_resnet(pretrained=True)
# ResNet18 with ImageNet weights

info = get_model_info(model)
# {'num_params': 11689512, 'output_size': 1000}
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torchvision import models

def load_resnet(pretrained: bool = True) -> nn.Module:
    """Load pretrained ResNet-18 model."""
    # Your code here
    pass

def get_model_info(model: nn.Module) -> dict:
    """Get model info: num_params and output_size."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
from torchvision import models

def load_resnet(pretrained: bool = True) -> nn.Module:
    """Load pretrained ResNet-18 model."""
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    return models.resnet18(weights=weights)

def get_model_info(model: nn.Module) -> dict:
    """Get model info: num_params and output_size."""
    num_params = sum(p.numel() for p in model.parameters())

    # Get output size from the last fc layer
    if hasattr(model, 'fc'):
        output_size = model.fc.out_features
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            output_size = model.classifier.out_features
        else:
            output_size = model.classifier[-1].out_features
    else:
        output_size = None

    return {'num_params': num_params, 'output_size': output_size}
`,

	testCode: `import torch
import torch.nn as nn
from torchvision import models
import unittest

class TestLoadPretrained(unittest.TestCase):
    def test_load_resnet(self):
        model = load_resnet(pretrained=False)
        self.assertIsInstance(model, nn.Module)

    def test_model_output(self):
        model = load_resnet(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1000))

    def test_get_model_info(self):
        model = load_resnet(pretrained=False)
        info = get_model_info(model)
        self.assertIn('num_params', info)
        self.assertIn('output_size', info)
        self.assertEqual(info['output_size'], 1000)

    def test_model_is_resnet(self):
        model = load_resnet(pretrained=False)
        self.assertTrue(hasattr(model, 'fc'))
        self.assertTrue(hasattr(model, 'layer4'))

    def test_num_params_positive(self):
        model = load_resnet(pretrained=False)
        info = get_model_info(model)
        self.assertGreater(info['num_params'], 0)

    def test_single_image(self):
        model = load_resnet(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (1, 1000))

    def test_different_batch_sizes(self):
        model = load_resnet(pretrained=False)
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            out = model(x)
            self.assertEqual(out.shape[0], batch_size)

    def test_output_not_nan(self):
        model = load_resnet(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())

    def test_model_in_eval_mode(self):
        model = load_resnet(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 1000))

    def test_info_returns_dict(self):
        model = load_resnet(pretrained=False)
        info = get_model_info(model)
        self.assertIsInstance(info, dict)
`,

	hint1: 'Use models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)',
	hint2: 'Count parameters with sum(p.numel() for p in model.parameters())',

	whyItMatters: `Pretrained models are the starting point for most deep learning projects:

- **ImageNet weights**: Trained on 1M+ images, 1000 classes
- **Feature extractors**: Good features for many vision tasks
- **Time savings**: Skip weeks of training
- **torchvision models**: ResNet, VGG, EfficientNet, etc.

Loading pretrained models is the first step in transfer learning.`,

	translations: {
		ru: {
			title: 'Загрузка предобученных моделей',
			description: `# Загрузка предобученных моделей

Научитесь загружать предобученные модели из torchvision.

## Задача

Реализуйте две функции:
1. \`load_resnet\` - Загрузка предобученной модели ResNet-18
2. \`get_model_info\` - Получение информации о модели (число параметров, размер выхода)

## Пример

\`\`\`python
model = load_resnet(pretrained=True)
# ResNet18 with ImageNet weights

info = get_model_info(model)
# {'num_params': 11689512, 'output_size': 1000}
\`\`\``,
			hint1: 'Используйте models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)',
			hint2: 'Подсчет параметров: sum(p.numel() for p in model.parameters())',
			whyItMatters: `Предобученные модели - начальная точка большинства проектов:

- **Веса ImageNet**: Обучены на 1M+ изображений, 1000 классов
- **Экстракторы признаков**: Хорошие признаки для многих задач
- **Экономия времени**: Пропуск недель обучения
- **Модели torchvision**: ResNet, VGG, EfficientNet и др.`,
		},
		uz: {
			title: "Oldindan o'qitilgan modellarni yuklash",
			description: `# Oldindan o'qitilgan modellarni yuklash

torchvision dan oldindan o'qitilgan modellarni yuklashni o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`load_resnet\` - Oldindan o'qitilgan ResNet-18 modelini yuklash
2. \`get_model_info\` - Model haqida ma'lumot olish (parametrlar soni, chiqish o'lchami)

## Misol

\`\`\`python
model = load_resnet(pretrained=True)
# ResNet18 with ImageNet weights

info = get_model_info(model)
# {'num_params': 11689512, 'output_size': 1000}
\`\`\``,
			hint1: 'models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) dan foydalaning',
			hint2: "Parametrlarni hisoblash: sum(p.numel() for p in model.parameters())",
			whyItMatters: `Oldindan o'qitilgan modellar ko'pchilik loyihalar uchun boshlang'ich nuqta:

- **ImageNet vaznlari**: 1M+ tasvirlar, 1000 sinflarda o'qitilgan
- **Xususiyat ajratuvchilar**: Ko'p vazifalar uchun yaxshi xususiyatlar
- **Vaqtni tejash**: Haftalik o'qitishni o'tkazib yuborish
- **torchvision modellari**: ResNet, VGG, EfficientNet va boshqalar`,
		},
	},
};

export default task;
