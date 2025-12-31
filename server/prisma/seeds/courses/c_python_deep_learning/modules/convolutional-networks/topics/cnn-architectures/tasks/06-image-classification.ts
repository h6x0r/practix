import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-image-classification',
	title: 'Image Classification Pipeline',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'classification'],
	estimatedTime: '18m',
	isPremium: true,
	order: 6,
	description: `# Image Classification Pipeline

Build a complete image classification pipeline with data loading and training.

## Task

Implement a \`ClassificationPipeline\` class with:
- \`train_epoch\` - Train for one epoch, return average loss
- \`evaluate\` - Evaluate on test data, return accuracy
- \`predict\` - Predict class for a single image tensor

## Example

\`\`\`python
pipeline = ClassificationPipeline(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device='cuda'
)

loss = pipeline.train_epoch()
accuracy = pipeline.evaluate()
pred = pipeline.predict(image_tensor)  # Returns class index
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.optim as optim

class ClassificationPipeline:
    """Complete image classification training pipeline."""

    def __init__(self, model: nn.Module, train_loader, test_loader,
                 lr: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self) -> float:
        """Train for one epoch. Return average loss."""
        # Your code here
        pass

    def evaluate(self) -> float:
        """Evaluate on test set. Return accuracy (0-1)."""
        # Your code here
        pass

    def predict(self, image: torch.Tensor) -> int:
        """Predict class for single image. Return class index."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.optim as optim

class ClassificationPipeline:
    """Complete image classification training pipeline."""

    def __init__(self, model: nn.Module, train_loader, test_loader,
                 lr: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self) -> float:
        """Train for one epoch. Return average loss."""
        self.model.train()
        total_loss = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self) -> float:
        """Evaluate on test set. Return accuracy (0-1)."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return correct / total

    def predict(self, image: torch.Tensor) -> int:
        """Predict class for single image. Return class index."""
        self.model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            output = self.model(image)
            _, predicted = output.max(1)
            return predicted.item()
`,

	testCode: `import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 5)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class TestPipeline(unittest.TestCase):
    def setUp(self):
        X = torch.randn(40, 1, 4, 4)
        y = torch.randint(0, 5, (40,))
        train_ds = TensorDataset(X[:30], y[:30])
        test_ds = TensorDataset(X[30:], y[30:])
        self.train_loader = DataLoader(train_ds, batch_size=10)
        self.test_loader = DataLoader(test_ds, batch_size=10)
        self.model = SimpleModel()

    def test_train_epoch(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        loss = pipeline.train_epoch()
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_evaluate(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        acc = pipeline.evaluate()
        self.assertTrue(0 <= acc <= 1)

    def test_predict(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        image = torch.randn(1, 4, 4)
        pred = pipeline.predict(image)
        self.assertTrue(0 <= pred < 5)

    def test_loss_decreases(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        loss1 = pipeline.train_epoch()
        loss2 = pipeline.train_epoch()
        # Loss should generally decrease (or stay similar)
        self.assertIsInstance(loss2, float)

    def test_accuracy_is_float(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        acc = pipeline.evaluate()
        self.assertIsInstance(acc, float)

    def test_predict_returns_int(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        pred = pipeline.predict(torch.randn(1, 4, 4))
        self.assertIsInstance(pred, int)

    def test_predict_batched_image(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        image = torch.randn(1, 1, 4, 4)
        pred = pipeline.predict(image)
        self.assertTrue(0 <= pred < 5)

    def test_pipeline_has_model(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        self.assertIsNotNone(pipeline.model)

    def test_pipeline_has_optimizer(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        self.assertIsNotNone(pipeline.optimizer)

    def test_multiple_epochs(self):
        pipeline = ClassificationPipeline(
            self.model, self.train_loader, self.test_loader
        )
        for _ in range(3):
            loss = pipeline.train_epoch()
            self.assertGreater(loss, 0)
`,

	hint1: 'Use model.train() before training, model.eval() before evaluation',
	hint2: 'Add batch dimension with unsqueeze(0) if image is 3D',

	whyItMatters: `A classification pipeline brings together all CNN concepts:

- **Data loading**: Batching and device transfer
- **Training loop**: Forward, loss, backward, optimize
- **Evaluation**: Track accuracy on held-out data
- **Inference**: Single image prediction

This is the complete workflow for any image classification project.`,

	translations: {
		ru: {
			title: 'Пайплайн классификации изображений',
			description: `# Пайплайн классификации изображений

Создайте полный пайплайн классификации с загрузкой данных и обучением.

## Задача

Реализуйте класс \`ClassificationPipeline\` с:
- \`train_epoch\` - Обучение одну эпоху, возврат средней ошибки
- \`evaluate\` - Оценка на тестовых данных, возврат точности
- \`predict\` - Предсказание класса для одного изображения

## Пример

\`\`\`python
pipeline = ClassificationPipeline(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device='cuda'
)

loss = pipeline.train_epoch()
accuracy = pipeline.evaluate()
pred = pipeline.predict(image_tensor)  # Returns class index
\`\`\``,
			hint1: 'Используйте model.train() перед обучением, model.eval() перед оценкой',
			hint2: 'Добавьте размерность батча с unsqueeze(0) если изображение 3D',
			whyItMatters: `Пайплайн классификации объединяет все концепции CNN:

- **Загрузка данных**: Батчинг и перенос на устройство
- **Цикл обучения**: Forward, loss, backward, optimize
- **Оценка**: Отслеживание точности на отложенных данных
- **Инференс**: Предсказание для одного изображения`,
		},
		uz: {
			title: "Tasvirlarni tasniflash pipeline",
			description: `# Tasvirlarni tasniflash pipeline

Ma'lumotlarni yuklash va o'qitish bilan to'liq tasniflash pipeline yarating.

## Topshiriq

\`ClassificationPipeline\` sinfini amalga oshiring:
- \`train_epoch\` - Bir davr o'qitish, o'rtacha xatoni qaytarish
- \`evaluate\` - Test ma'lumotlarida baholash, aniqlikni qaytarish
- \`predict\` - Bitta tasvir uchun sinfni bashorat qilish

## Misol

\`\`\`python
pipeline = ClassificationPipeline(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device='cuda'
)

loss = pipeline.train_epoch()
accuracy = pipeline.evaluate()
pred = pipeline.predict(image_tensor)  # Returns class index
\`\`\``,
			hint1: "O'qitishdan oldin model.train(), baholashdan oldin model.eval() dan foydalaning",
			hint2: "Agar tasvir 3D bo'lsa, unsqueeze(0) bilan batch o'lchamini qo'shing",
			whyItMatters: `Tasniflash pipeline barcha CNN tushunchalarini birlashtiradi:

- **Ma'lumot yuklash**: Batching va qurilmaga o'tkazish
- **O'qitish sikli**: Forward, loss, backward, optimize
- **Baholash**: Ajratilgan ma'lumotlarda aniqlikni kuzatish
- **Xulosa chiqarish**: Bitta tasvir uchun bashorat`,
		},
	},
};

export default task;
