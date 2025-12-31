import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-save-load',
	title: 'Saving and Loading Models',
	difficulty: 'easy',
	tags: ['pytorch', 'save', 'load'],
	estimatedTime: '10m',
	isPremium: false,
	order: 8,
	description: `# Saving and Loading Models

Learn to save and load PyTorch models for deployment.

## Task

Implement four functions:
1. \`save_checkpoint(model, optimizer, epoch, path)\` - Save full checkpoint
2. \`load_checkpoint(model, optimizer, path)\` - Load checkpoint
3. \`save_model_only(model, path)\` - Save just the model weights
4. \`load_model_only(model, path)\` - Load just weights

## Example

\`\`\`python
# Save during training
save_checkpoint(model, optimizer, epoch=10, path='checkpoint.pt')

# Resume training
epoch = load_checkpoint(model, optimizer, 'checkpoint.pt')

# For inference only
save_model_only(model, 'model.pt')
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def save_checkpoint(model: nn.Module, optimizer, epoch: int, path: str) -> None:
    """Save model, optimizer state, and epoch."""
    # Your code here
    pass

def load_checkpoint(model: nn.Module, optimizer, path: str) -> int:
    """Load checkpoint and return epoch number."""
    # Your code here
    pass

def save_model_only(model: nn.Module, path: str) -> None:
    """Save only model weights (state_dict)."""
    # Your code here
    pass

def load_model_only(model: nn.Module, path: str) -> nn.Module:
    """Load weights into model. Return model."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

def save_checkpoint(model: nn.Module, optimizer, epoch: int, path: str) -> None:
    """Save model, optimizer state, and epoch."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

def load_checkpoint(model: nn.Module, optimizer, path: str) -> int:
    """Load checkpoint and return epoch number."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def save_model_only(model: nn.Module, path: str) -> None:
    """Save only model weights (state_dict)."""
    torch.save(model.state_dict(), path)

def load_model_only(model: nn.Module, path: str) -> nn.Module:
    """Load weights into model. Return model."""
    model.load_state_dict(torch.load(path))
    return model
`,

	testCode: `import torch
import torch.nn as nn
import tempfile
import os
import unittest

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)

class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_save_load_checkpoint(self):
        model = SimpleMLP()
        optimizer = torch.optim.Adam(model.parameters())
        path = os.path.join(self.temp_dir, 'ckpt.pt')

        save_checkpoint(model, optimizer, 5, path)
        self.assertTrue(os.path.exists(path))

        model2 = SimpleMLP()
        opt2 = torch.optim.Adam(model2.parameters())
        epoch = load_checkpoint(model2, opt2, path)
        self.assertEqual(epoch, 5)

    def test_save_load_model_only(self):
        model = SimpleMLP()
        path = os.path.join(self.temp_dir, 'model.pt')

        save_model_only(model, path)
        self.assertTrue(os.path.exists(path))

        model2 = SimpleMLP()
        model2 = load_model_only(model2, path)
        # Check weights are same
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_file_not_empty(self):
        model = SimpleMLP()
        optimizer = torch.optim.Adam(model.parameters())
        path = os.path.join(self.temp_dir, 'ckpt2.pt')
        save_checkpoint(model, optimizer, 10, path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_model_only_file_not_empty(self):
        model = SimpleMLP()
        path = os.path.join(self.temp_dir, 'model2.pt')
        save_model_only(model, path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_load_checkpoint_returns_int(self):
        model = SimpleMLP()
        optimizer = torch.optim.Adam(model.parameters())
        path = os.path.join(self.temp_dir, 'ckpt3.pt')
        save_checkpoint(model, optimizer, 15, path)
        model2 = SimpleMLP()
        opt2 = torch.optim.Adam(model2.parameters())
        epoch = load_checkpoint(model2, opt2, path)
        self.assertIsInstance(epoch, int)

    def test_load_model_only_returns_module(self):
        model = SimpleMLP()
        path = os.path.join(self.temp_dir, 'model3.pt')
        save_model_only(model, path)
        model2 = SimpleMLP()
        result = load_model_only(model2, path)
        self.assertIsInstance(result, nn.Module)

    def test_different_epochs(self):
        model = SimpleMLP()
        optimizer = torch.optim.Adam(model.parameters())
        for e in [1, 50, 100]:
            path = os.path.join(self.temp_dir, f'ckpt_{e}.pt')
            save_checkpoint(model, optimizer, e, path)
            model2 = SimpleMLP()
            opt2 = torch.optim.Adam(model2.parameters())
            loaded_epoch = load_checkpoint(model2, opt2, path)
            self.assertEqual(loaded_epoch, e)

    def test_model_predictions_same_after_save_load(self):
        model = SimpleMLP()
        x = torch.randn(2, 10)
        model.eval()
        with torch.no_grad():
            pred1 = model(x)
        path = os.path.join(self.temp_dir, 'model4.pt')
        save_model_only(model, path)
        model2 = SimpleMLP()
        load_model_only(model2, path)
        model2.eval()
        with torch.no_grad():
            pred2 = model2(x)
        self.assertTrue(torch.allclose(pred1, pred2))

    def test_optimizer_state_restored(self):
        model = SimpleMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Take a step
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()
        path = os.path.join(self.temp_dir, 'ckpt4.pt')
        save_checkpoint(model, optimizer, 1, path)
        model2 = SimpleMLP()
        opt2 = torch.optim.Adam(model2.parameters())
        load_checkpoint(model2, opt2, path)
        self.assertEqual(len(opt2.state), len(optimizer.state))

    def test_multiple_saves(self):
        model = SimpleMLP()
        for i in range(3):
            path = os.path.join(self.temp_dir, f'model_{i}.pt')
            save_model_only(model, path)
            self.assertTrue(os.path.exists(path))
`,

	hint1: 'torch.save(dict, path) saves, torch.load(path) loads',
	hint2: 'model.state_dict() gets weights, model.load_state_dict(d) loads them',

	whyItMatters: `Saving and loading models is essential for:

- **Checkpointing**: Resume training after interruption
- **Deployment**: Load trained models in production
- **Experimentation**: Compare different training runs
- **Sharing**: Distribute models to others

Always save checkpoints during long training runs.`,

	translations: {
		ru: {
			title: 'Сохранение и загрузка моделей',
			description: `# Сохранение и загрузка моделей

Научитесь сохранять и загружать модели PyTorch для деплоя.

## Задача

Реализуйте четыре функции:
1. \`save_checkpoint(model, optimizer, epoch, path)\` - Сохранить полный чекпоинт
2. \`load_checkpoint(model, optimizer, path)\` - Загрузить чекпоинт
3. \`save_model_only(model, path)\` - Сохранить только веса
4. \`load_model_only(model, path)\` - Загрузить только веса

## Пример

\`\`\`python
# Save during training
save_checkpoint(model, optimizer, epoch=10, path='checkpoint.pt')

# Resume training
epoch = load_checkpoint(model, optimizer, 'checkpoint.pt')

# For inference only
save_model_only(model, 'model.pt')
\`\`\``,
			hint1: 'torch.save(dict, path) сохраняет, torch.load(path) загружает',
			hint2: 'model.state_dict() получает веса, model.load_state_dict(d) загружает',
			whyItMatters: `Сохранение и загрузка моделей важны для:

- **Чекпоинтинг**: Возобновление обучения после прерывания
- **Деплой**: Загрузка обученных моделей в продакшен
- **Эксперименты**: Сравнение разных запусков обучения`,
		},
		uz: {
			title: 'Modellarni saqlash va yuklash',
			description: `# Modellarni saqlash va yuklash

Deployment uchun PyTorch modellarini saqlash va yuklashni o'rganing.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`save_checkpoint(model, optimizer, epoch, path)\` - To'liq checkpoint saqlash
2. \`load_checkpoint(model, optimizer, path)\` - Checkpoint yuklash
3. \`save_model_only(model, path)\` - Faqat model og'irliklarini saqlash
4. \`load_model_only(model, path)\` - Faqat og'irliklarni yuklash

## Misol

\`\`\`python
# Save during training
save_checkpoint(model, optimizer, epoch=10, path='checkpoint.pt')

# Resume training
epoch = load_checkpoint(model, optimizer, 'checkpoint.pt')

# For inference only
save_model_only(model, 'model.pt')
\`\`\``,
			hint1: "torch.save(dict, path) saqlaydi, torch.load(path) yuklaydi",
			hint2: "model.state_dict() og'irliklarni oladi, model.load_state_dict(d) yuklaydi",
			whyItMatters: `Modellarni saqlash va yuklash quyidagilar uchun muhim:

- **Checkpointing**: Uzilishdan keyin o'qitishni davom ettirish
- **Deployment**: O'qitilgan modellarni ishlab chiqarishda yuklash
- **Eksperimentlar**: Turli o'qitish ishlashlarini solishtirish`,
		},
	},
};

export default task;
