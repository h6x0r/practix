import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-differential-learning-rates',
	title: 'Differential Learning Rates',
	difficulty: 'medium',
	tags: ['pytorch', 'transfer-learning', 'learning-rate'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Differential Learning Rates

Use different learning rates for different parts of the network.

## Task

Implement a function that creates parameter groups with different learning rates:
- Backbone layers: lower learning rate
- New classifier: higher learning rate

## Example

\`\`\`python
model = FineTunedModel(num_classes=10)
param_groups = create_param_groups(model, backbone_lr=1e-4, head_lr=1e-2)

optimizer = torch.optim.Adam(param_groups)
# Backbone trains with lr=1e-4, head with lr=1e-2
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from typing import List, Dict

def create_param_groups(model: nn.Module,
                        backbone_lr: float = 1e-4,
                        head_lr: float = 1e-2) -> List[Dict]:
    """Create parameter groups with differential learning rates."""
    # Your code here
    pass

def create_layerwise_lr(model: nn.Module,
                        base_lr: float = 1e-4,
                        lr_mult: float = 1.5) -> List[Dict]:
    """Create layer-wise learning rates (higher for later layers)."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
from typing import List, Dict

def create_param_groups(model: nn.Module,
                        backbone_lr: float = 1e-4,
                        head_lr: float = 1e-2) -> List[Dict]:
    """Create parameter groups with differential learning rates."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'fc' in name or 'classifier' in name or 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]

def create_layerwise_lr(model: nn.Module,
                        base_lr: float = 1e-4,
                        lr_mult: float = 1.5) -> List[Dict]:
    """Create layer-wise learning rates (higher for later layers)."""
    param_groups = []
    children = list(model.named_children())
    num_layers = len(children)

    for i, (name, child) in enumerate(children):
        params = list(child.parameters())
        if params:
            lr = base_lr * (lr_mult ** i)
            param_groups.append({'params': params, 'lr': lr, 'name': name})

    return param_groups
`,

	testCode: `import torch
import torch.nn as nn
from torchvision import models
import unittest

class TestDifferentialLR(unittest.TestCase):
    def test_param_groups(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model, backbone_lr=1e-4, head_lr=1e-2)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['lr'], 1e-4)
        self.assertEqual(groups[1]['lr'], 1e-2)

    def test_head_params_included(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model)
        # Head group should have fc parameters
        self.assertTrue(len(groups[1]['params']) > 0)

    def test_layerwise_lr(self):
        model = models.resnet18(weights=None)
        groups = create_layerwise_lr(model, base_lr=0.001, lr_mult=2.0)
        # Later layers should have higher lr
        self.assertGreater(groups[-1]['lr'], groups[0]['lr'])

    def test_returns_list(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model)
        self.assertIsInstance(groups, list)

    def test_groups_have_params(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model)
        self.assertIn('params', groups[0])
        self.assertIn('lr', groups[0])

    def test_backbone_params_included(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model)
        self.assertTrue(len(groups[0]['params']) > 0)

    def test_layerwise_returns_list(self):
        model = models.resnet18(weights=None)
        groups = create_layerwise_lr(model)
        self.assertIsInstance(groups, list)

    def test_layerwise_has_multiple_groups(self):
        model = models.resnet18(weights=None)
        groups = create_layerwise_lr(model)
        self.assertGreater(len(groups), 1)

    def test_different_base_lr(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model, backbone_lr=0.01, head_lr=0.1)
        self.assertEqual(groups[0]['lr'], 0.01)
        self.assertEqual(groups[1]['lr'], 0.1)

    def test_optimizer_accepts_groups(self):
        model = models.resnet18(weights=None)
        groups = create_param_groups(model)
        optimizer = torch.optim.Adam(groups)
        self.assertIsNotNone(optimizer)
`,

	hint1: 'Check for "fc" or "classifier" in parameter name to identify head',
	hint2: 'Later layers get lr = base_lr * (lr_mult ** layer_index)',

	whyItMatters: `Differential learning rates improve transfer learning:

- **Pretrained weights**: Small LR to not destroy good features
- **New layers**: Large LR to learn quickly
- **Layer-wise decay**: Gradually increase LR for later layers
- **Faster convergence**: Different parts learn at optimal rates

This technique is used in most state-of-the-art transfer learning.`,

	translations: {
		ru: {
			title: 'Дифференциальные скорости обучения',
			description: `# Дифференциальные скорости обучения

Используйте разные learning rate для разных частей сети.

## Задача

Реализуйте функцию создания групп параметров с разными learning rates:
- Backbone слои: низкий learning rate
- Новый классификатор: высокий learning rate

## Пример

\`\`\`python
model = FineTunedModel(num_classes=10)
param_groups = create_param_groups(model, backbone_lr=1e-4, head_lr=1e-2)

optimizer = torch.optim.Adam(param_groups)
# Backbone trains with lr=1e-4, head with lr=1e-2
\`\`\``,
			hint1: 'Проверяйте "fc" или "classifier" в имени параметра для определения head',
			hint2: 'Поздние слои получают lr = base_lr * (lr_mult ** layer_index)',
			whyItMatters: `Дифференциальные learning rates улучшают transfer learning:

- **Предобученные веса**: Малый LR чтобы не испортить хорошие признаки
- **Новые слои**: Большой LR для быстрого обучения
- **Послойное затухание**: Постепенное увеличение LR для поздних слоев
- **Быстрая сходимость**: Разные части учатся с оптимальной скоростью`,
		},
		uz: {
			title: "Differentsial o'rganish tezliklari",
			description: `# Differentsial o'rganish tezliklari

Tarmoqning turli qismlari uchun turli learning rate dan foydalaning.

## Topshiriq

Turli learning rate larga ega parametr guruhlari yaratadigan funksiyani amalga oshiring:
- Backbone qatlamlar: past learning rate
- Yangi klassifikator: yuqori learning rate

## Misol

\`\`\`python
model = FineTunedModel(num_classes=10)
param_groups = create_param_groups(model, backbone_lr=1e-4, head_lr=1e-2)

optimizer = torch.optim.Adam(param_groups)
# Backbone trains with lr=1e-4, head with lr=1e-2
\`\`\``,
			hint1: "Bosh qismni aniqlash uchun parametr nomida \"fc\" yoki \"classifier\" ni tekshiring",
			hint2: "Keyingi qatlamlar lr = base_lr * (lr_mult ** layer_index) oladi",
			whyItMatters: `Differentsial learning rate lar transfer learning ni yaxshilaydi:

- **Oldindan o'qitilgan vaznlar**: Yaxshi xususiyatlarni buzmaslik uchun kichik LR
- **Yangi qatlamlar**: Tez o'rganish uchun katta LR
- **Qatlam bo'yicha pasayish**: Keyingi qatlamlar uchun LR ni asta-sekin oshirish
- **Tez konvergensiya**: Turli qismlar optimal tezlikda o'rganadi`,
		},
	},
};

export default task;
