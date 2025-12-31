import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-gpu-training',
	title: 'GPU Training',
	difficulty: 'medium',
	tags: ['pytorch', 'gpu', 'cuda'],
	estimatedTime: '12m',
	isPremium: false,
	order: 7,
	description: `# GPU Training

Learn to train models on GPU for faster computation.

## Task

Implement four functions:
1. \`get_device()\` - Return 'cuda' if available, else 'cpu'
2. \`move_to_device(model, device)\` - Move model to device
3. \`move_data_to_device(x, y, device)\` - Move tensors to device
4. \`gpu_training_step(model, x, y, criterion, optimizer, device)\` - Train on GPU

## Example

\`\`\`python
device = get_device()  # 'cuda' or 'cpu'
model = move_to_device(model, device)

for x, y in dataloader:
    x, y = move_data_to_device(x, y, device)
    loss = gpu_training_step(model, x, y, criterion, optimizer, device)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    # Your code here
    pass

def move_to_device(model: nn.Module, device: str) -> nn.Module:
    """Move model to device. Return model."""
    # Your code here
    pass

def move_data_to_device(x: torch.Tensor, y: torch.Tensor, device: str) -> tuple:
    """Move x and y to device. Return (x, y)."""
    # Your code here
    pass

def gpu_training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      criterion, optimizer, device: str) -> float:
    """Training step with device handling. Return loss."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

def get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def move_to_device(model: nn.Module, device: str) -> nn.Module:
    """Move model to device. Return model."""
    return model.to(device)

def move_data_to_device(x: torch.Tensor, y: torch.Tensor, device: str) -> tuple:
    """Move x and y to device. Return (x, y)."""
    return x.to(device), y.to(device)

def gpu_training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      criterion, optimizer, device: str) -> float:
    """Training step with device handling. Return loss."""
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    return loss.item()
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)

class TestGPUTraining(unittest.TestCase):
    def test_get_device(self):
        device = get_device()
        self.assertIn(device, ['cuda', 'cpu'])

    def test_move_model(self):
        model = SimpleMLP()
        device = get_device()
        model = move_to_device(model, device)
        # Check first parameter is on correct device
        param_device = str(next(model.parameters()).device)
        self.assertTrue(param_device.startswith(device.split(':')[0]))

    def test_move_data(self):
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        device = get_device()
        x_d, y_d = move_data_to_device(x, y, device)
        self.assertTrue(str(x_d.device).startswith(device.split(':')[0]))

    def test_gpu_training_step(self):
        model = SimpleMLP()
        device = get_device()
        model = move_to_device(model, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        loss = gpu_training_step(model, x, y, criterion, optimizer, device)
        self.assertIsInstance(loss, float)

    def test_get_device_returns_string(self):
        device = get_device()
        self.assertIsInstance(device, str)

    def test_move_model_returns_module(self):
        model = SimpleMLP()
        device = get_device()
        result = move_to_device(model, device)
        self.assertIsInstance(result, nn.Module)

    def test_move_data_returns_tuple(self):
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        device = get_device()
        result = move_data_to_device(x, y, device)
        self.assertEqual(len(result), 2)

    def test_gpu_training_step_positive_loss(self):
        model = SimpleMLP()
        device = get_device()
        model = move_to_device(model, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        loss = gpu_training_step(model, x, y, criterion, optimizer, device)
        self.assertGreater(loss, 0)

    def test_move_data_same_device(self):
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 0])
        device = get_device()
        x_d, y_d = move_data_to_device(x, y, device)
        self.assertEqual(str(x_d.device).split(':')[0], str(y_d.device).split(':')[0])
`,

	hint1: 'torch.cuda.is_available() checks for GPU, model.to(device) moves model',
	hint2: 'tensor.to(device) moves tensor to CPU or GPU',

	whyItMatters: `GPU training is essential for:

- **Speed**: GPUs are 10-100x faster for neural networks
- **Scalability**: Train larger models with more data
- **Parallel computation**: GPUs have thousands of cores
- **Industry standard**: Production models train on GPUs

Moving to GPU is a single line in PyTorch - but crucial.`,

	translations: {
		ru: {
			title: 'Обучение на GPU',
			description: `# Обучение на GPU

Научитесь обучать модели на GPU для ускорения вычислений.

## Задача

Реализуйте четыре функции:
1. \`get_device()\` - Вернуть 'cuda' если доступен, иначе 'cpu'
2. \`move_to_device(model, device)\` - Переместить модель на устройство
3. \`move_data_to_device(x, y, device)\` - Переместить тензоры
4. \`gpu_training_step(model, x, y, criterion, optimizer, device)\` - Обучение на GPU

## Пример

\`\`\`python
device = get_device()  # 'cuda' or 'cpu'
model = move_to_device(model, device)

for x, y in dataloader:
    x, y = move_data_to_device(x, y, device)
    loss = gpu_training_step(model, x, y, criterion, optimizer, device)
\`\`\``,
			hint1: 'torch.cuda.is_available() проверяет GPU, model.to(device) перемещает',
			hint2: 'tensor.to(device) перемещает тензор на CPU или GPU',
			whyItMatters: `Обучение на GPU важно для:

- **Скорость**: GPU в 10-100 раз быстрее для нейросетей
- **Масштабируемость**: Обучение больших моделей
- **Параллельные вычисления**: Тысячи ядер GPU`,
		},
		uz: {
			title: "GPU da o'qitish",
			description: `# GPU da o'qitish

Tezroq hisoblash uchun modellarni GPU da o'qitishni o'rganing.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`get_device()\` - Mavjud bo'lsa 'cuda' qaytarish, aks holda 'cpu'
2. \`move_to_device(model, device)\` - Modelni qurilmaga ko'chirish
3. \`move_data_to_device(x, y, device)\` - Tensorlarni qurilmaga ko'chirish
4. \`gpu_training_step(model, x, y, criterion, optimizer, device)\` - GPU da o'qitish

## Misol

\`\`\`python
device = get_device()  # 'cuda' or 'cpu'
model = move_to_device(model, device)

for x, y in dataloader:
    x, y = move_data_to_device(x, y, device)
    loss = gpu_training_step(model, x, y, criterion, optimizer, device)
\`\`\``,
			hint1: "torch.cuda.is_available() GPU ni tekshiradi, model.to(device) ko'chiradi",
			hint2: "tensor.to(device) tensorni CPU yoki GPU ga ko'chiradi",
			whyItMatters: `GPU da o'qitish quyidagilar uchun muhim:

- **Tezlik**: GPU lar neyrosetka uchun 10-100 marta tezroq
- **Masshtablilik**: Katta modellarni ko'proq ma'lumotlar bilan o'qitish
- **Parallel hisoblash**: GPU larda minglab yadrolar bor`,
		},
	},
};

export default task;
