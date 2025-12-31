import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-transfer-learning-project',
	title: 'Complete Transfer Learning Project',
	difficulty: 'hard',
	tags: ['pytorch', 'transfer-learning', 'project'],
	estimatedTime: '25m',
	isPremium: true,
	order: 10,
	description: `# Complete Transfer Learning Project

Build a complete transfer learning pipeline from data to deployment.

## Task

Implement a \`TransferLearningPipeline\` class with:
- Data loading and augmentation
- Model setup with fine-tuning
- Training with differential learning rates
- Evaluation and model saving

## Example

\`\`\`python
pipeline = TransferLearningPipeline(
    data_dir='data/',
    num_classes=10,
    base_model='resnet18'
)

history = pipeline.train(epochs=10)
accuracy = pipeline.evaluate()
pipeline.save('model.pth')
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from typing import Dict, List
import os

class TransferLearningPipeline:
    """Complete transfer learning pipeline."""

    def __init__(self, data_dir: str, num_classes: int,
                 base_model: str = 'resnet18', device: str = 'cpu'):
        self.device = device
        self.num_classes = num_classes
        # Your code here - setup model, data, optimizer
        pass

    def train(self, epochs: int = 10) -> Dict[str, List[float]]:
        """Train the model. Return history dict."""
        # Your code here
        pass

    def evaluate(self) -> float:
        """Evaluate on validation set. Return accuracy."""
        # Your code here
        pass

    def save(self, path: str):
        """Save model weights."""
        # Your code here
        pass

    def load(self, path: str):
        """Load model weights."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from typing import Dict, List
import os
import copy

class TransferLearningPipeline:
    """Complete transfer learning pipeline."""

    def __init__(self, data_dir: str, num_classes: int,
                 base_model: str = 'resnet18', device: str = 'cpu'):
        self.device = device
        self.num_classes = num_classes

        # Setup data
        self._setup_data(data_dir)

        # Setup model
        self._setup_model(base_model)

        # Setup optimizer with differential learning rates
        self._setup_optimizer()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _setup_data(self, data_dir: str):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
        val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    def _setup_model(self, base_model: str):
        if base_model == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        self.model = self.model.to(self.device)

    def _setup_optimizer(self):
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 1e-4},
            {'params': head_params, 'lr': 1e-3}
        ])

    def train(self, epochs: int = 10) -> Dict[str, List[float]]:
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0
        best_model = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            val_loss, val_acc = self._validate()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(self.model.state_dict())

        if best_model:
            self.model.load_state_dict(best_model)

        return history

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return val_loss / len(self.val_loader), correct / total

    def evaluate(self) -> float:
        _, acc = self._validate()
        return acc

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
`,

	testCode: `import torch
import torch.nn as nn
import tempfile
import os
from PIL import Image
import unittest

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        for split in ['train', 'val']:
            for cls in ['cat', 'dog']:
                path = os.path.join(self.temp_dir, split, cls)
                os.makedirs(path)
                for i in range(5):
                    img = Image.new('RGB', (100, 100))
                    img.save(os.path.join(path, f'img_{i}.jpg'))

    def test_pipeline_creation(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        self.assertIsNotNone(pipeline.model)

    def test_train(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        history = pipeline.train(epochs=1)
        self.assertIn('train_loss', history)
        self.assertIn('val_acc', history)

    def test_save_load(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        save_path = os.path.join(self.temp_dir, 'model.pth')
        pipeline.save(save_path)
        pipeline.load(save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_has_model(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        self.assertTrue(hasattr(pipeline, 'model'))

    def test_has_optimizer(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        self.assertTrue(hasattr(pipeline, 'optimizer'))

    def test_has_criterion(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        self.assertTrue(hasattr(pipeline, 'criterion'))

    def test_has_data_loaders(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        self.assertTrue(hasattr(pipeline, 'train_loader'))
        self.assertTrue(hasattr(pipeline, 'val_loader'))

    def test_evaluate(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        acc = pipeline.evaluate()
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_history_structure(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=2
        )
        history = pipeline.train(epochs=1)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 1)

    def test_num_classes_stored(self):
        pipeline = TransferLearningPipeline(
            self.temp_dir, num_classes=5
        )
        self.assertEqual(pipeline.num_classes, 5)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
`,

	hint1: 'Use differential learning rates: lower for backbone, higher for head',
	hint2: 'Save best model based on validation accuracy',

	whyItMatters: `A complete transfer learning pipeline includes:

- **Data preparation**: Loading, augmentation, normalization
- **Model setup**: Pretrained backbone, custom head
- **Training strategy**: Differential LR, checkpointing
- **Evaluation**: Validation metrics, best model selection

This is the complete workflow for any transfer learning project.`,

	translations: {
		ru: {
			title: 'Полный проект Transfer Learning',
			description: `# Полный проект Transfer Learning

Создайте полный пайплайн transfer learning от данных до деплоя.

## Задача

Реализуйте класс \`TransferLearningPipeline\` с:
- Загрузка данных и аугментация
- Настройка модели с fine-tuning
- Обучение с дифференциальными learning rates
- Оценка и сохранение модели

## Пример

\`\`\`python
pipeline = TransferLearningPipeline(
    data_dir='data/',
    num_classes=10,
    base_model='resnet18'
)

history = pipeline.train(epochs=10)
accuracy = pipeline.evaluate()
pipeline.save('model.pth')
\`\`\``,
			hint1: 'Используйте дифференциальные LR: ниже для backbone, выше для head',
			hint2: 'Сохраняйте лучшую модель по validation accuracy',
			whyItMatters: `Полный пайплайн transfer learning включает:

- **Подготовка данных**: Загрузка, аугментация, нормализация
- **Настройка модели**: Предобученный backbone, кастомный head
- **Стратегия обучения**: Дифференциальный LR, чекпоинтинг
- **Оценка**: Метрики валидации, выбор лучшей модели`,
		},
		uz: {
			title: "To'liq Transfer Learning loyihasi",
			description: `# To'liq Transfer Learning loyihasi

Ma'lumotlardan deployga qadar to'liq transfer learning pipeline yarating.

## Topshiriq

\`TransferLearningPipeline\` sinfini amalga oshiring:
- Ma'lumotlarni yuklash va kengaytirish
- Fine-tuning bilan model sozlash
- Differentsial learning rate lar bilan o'qitish
- Baholash va modelni saqlash

## Misol

\`\`\`python
pipeline = TransferLearningPipeline(
    data_dir='data/',
    num_classes=10,
    base_model='resnet18'
)

history = pipeline.train(epochs=10)
accuracy = pipeline.evaluate()
pipeline.save('model.pth')
\`\`\``,
			hint1: "Differentsial LR dan foydalaning: backbone uchun pastroq, head uchun yuqoriroq",
			hint2: "Validation accuracy bo'yicha eng yaxshi modelni saqlang",
			whyItMatters: `To'liq transfer learning pipeline quyidagilarni o'z ichiga oladi:

- **Ma'lumotlarni tayyorlash**: Yuklash, kengaytirish, normalizatsiya
- **Model sozlash**: Oldindan o'qitilgan backbone, maxsus head
- **O'qitish strategiyasi**: Differentsial LR, checkpointing
- **Baholash**: Validatsiya metrikalari, eng yaxshi modelni tanlash`,
		},
	},
};

export default task;
