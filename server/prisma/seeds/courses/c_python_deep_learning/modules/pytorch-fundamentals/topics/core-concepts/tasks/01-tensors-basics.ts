import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-tensors-basics',
	title: 'Tensor Basics',
	difficulty: 'easy',
	tags: ['pytorch', 'tensors', 'basics'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Tensor Basics

Learn to create and manipulate PyTorch tensors.

## Task

Implement four functions:
1. \`create_tensor(data)\` - Create tensor from Python list
2. \`create_random_tensor(shape)\` - Create random tensor with given shape
3. \`tensor_info(t)\` - Return dict with shape, dtype, device
4. \`convert_numpy(t)\` - Convert tensor to numpy and back

## Example

\`\`\`python
import torch

t = create_tensor([[1, 2], [3, 4]])  # shape: (2, 2)
r = create_random_tensor((3, 4))     # shape: (3, 4)
info = tensor_info(t)  # {'shape': (2, 2), 'dtype': torch.int64, 'device': 'cpu'}
\`\`\``,

	initialCode: `import torch
import numpy as np

def create_tensor(data: list) -> torch.Tensor:
    """Create tensor from Python list."""
    # Your code here
    pass

def create_random_tensor(shape: tuple) -> torch.Tensor:
    """Create random tensor with shape, values in [0, 1)."""
    # Your code here
    pass

def tensor_info(t: torch.Tensor) -> dict:
    """Return dict with 'shape', 'dtype', 'device'."""
    # Your code here
    pass

def convert_numpy(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to numpy, multiply by 2, convert back to tensor."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import numpy as np

def create_tensor(data: list) -> torch.Tensor:
    """Create tensor from Python list."""
    return torch.tensor(data)

def create_random_tensor(shape: tuple) -> torch.Tensor:
    """Create random tensor with shape, values in [0, 1)."""
    return torch.rand(shape)

def tensor_info(t: torch.Tensor) -> dict:
    """Return dict with 'shape', 'dtype', 'device'."""
    return {
        'shape': tuple(t.shape),
        'dtype': t.dtype,
        'device': str(t.device)
    }

def convert_numpy(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to numpy, multiply by 2, convert back to tensor."""
    arr = t.numpy()
    arr = arr * 2
    return torch.from_numpy(arr)
`,

	testCode: `import torch
import numpy as np
import unittest

class TestTensorBasics(unittest.TestCase):
    def test_create_tensor(self):
        t = create_tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, torch.Size([2, 2]))

    def test_create_random_shape(self):
        t = create_random_tensor((3, 4, 5))
        self.assertEqual(t.shape, torch.Size([3, 4, 5]))

    def test_create_random_range(self):
        t = create_random_tensor((100,))
        self.assertTrue(torch.all(t >= 0))
        self.assertTrue(torch.all(t < 1))

    def test_tensor_info_keys(self):
        t = torch.tensor([1.0, 2.0])
        info = tensor_info(t)
        self.assertIn('shape', info)
        self.assertIn('dtype', info)
        self.assertIn('device', info)

    def test_convert_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = convert_numpy(t)
        expected = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(result, expected)

    def test_create_tensor_values(self):
        t = create_tensor([1, 2, 3])
        self.assertEqual(t[0].item(), 1)
        self.assertEqual(t[2].item(), 3)

    def test_create_tensor_is_tensor(self):
        t = create_tensor([[1, 2]])
        self.assertIsInstance(t, torch.Tensor)

    def test_tensor_info_shape(self):
        t = torch.zeros(2, 3, 4)
        info = tensor_info(t)
        self.assertEqual(info['shape'], (2, 3, 4))

    def test_tensor_info_device(self):
        t = torch.tensor([1.0])
        info = tensor_info(t)
        self.assertIn('cpu', info['device'])

    def test_convert_numpy_returns_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = convert_numpy(t)
        self.assertIsInstance(result, torch.Tensor)
`,

	hint1: 'torch.tensor(data) creates from list, torch.rand(shape) creates random',
	hint2: 't.numpy() converts to numpy, torch.from_numpy(arr) converts back',

	whyItMatters: `Tensors are the foundation of PyTorch:

- **Multi-dimensional arrays**: Like numpy but with GPU support
- **Automatic differentiation**: Track operations for gradients
- **Interoperability**: Easy conversion to/from numpy
- **Building blocks**: All neural network data flows through tensors

Everything in PyTorch starts with tensors.`,

	translations: {
		ru: {
			title: 'Основы тензоров',
			description: `# Основы тензоров

Научитесь создавать и манипулировать тензорами PyTorch.

## Задача

Реализуйте четыре функции:
1. \`create_tensor(data)\` - Создать тензор из списка Python
2. \`create_random_tensor(shape)\` - Создать случайный тензор
3. \`tensor_info(t)\` - Вернуть словарь с shape, dtype, device
4. \`convert_numpy(t)\` - Конвертировать в numpy и обратно

## Пример

\`\`\`python
import torch

t = create_tensor([[1, 2], [3, 4]])  # shape: (2, 2)
r = create_random_tensor((3, 4))     # shape: (3, 4)
info = tensor_info(t)  # {'shape': (2, 2), 'dtype': torch.int64, 'device': 'cpu'}
\`\`\``,
			hint1: 'torch.tensor(data) создаёт из списка, torch.rand(shape) создаёт случайный',
			hint2: 't.numpy() конвертирует в numpy, torch.from_numpy(arr) обратно',
			whyItMatters: `Тензоры - фундамент PyTorch:

- **Многомерные массивы**: Как numpy но с поддержкой GPU
- **Автоматическое дифференцирование**: Отслеживание операций для градиентов
- **Совместимость**: Простая конвертация в/из numpy`,
		},
		uz: {
			title: 'Tensor asoslari',
			description: `# Tensor asoslari

PyTorch tensorlarini yaratish va boshqarishni o'rganing.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`create_tensor(data)\` - Python ro'yxatidan tensor yaratish
2. \`create_random_tensor(shape)\` - Berilgan shakl bilan tasodifiy tensor
3. \`tensor_info(t)\` - shape, dtype, device bilan lug'at qaytarish
4. \`convert_numpy(t)\` - Tensorni numpy ga va orqaga aylantirish

## Misol

\`\`\`python
import torch

t = create_tensor([[1, 2], [3, 4]])  # shape: (2, 2)
r = create_random_tensor((3, 4))     # shape: (3, 4)
info = tensor_info(t)  # {'shape': (2, 2), 'dtype': torch.int64, 'device': 'cpu'}
\`\`\``,
			hint1: "torch.tensor(data) ro'yxatdan yaratadi, torch.rand(shape) tasodifiy yaratadi",
			hint2: "t.numpy() numpy ga aylantiradi, torch.from_numpy(arr) orqaga aylantiradi",
			whyItMatters: `Tensorlar PyTorch ning asosi:

- **Ko'p o'lchovli massivlar**: numpy kabi lekin GPU qo'llab-quvvatlashi bilan
- **Avtomatik differensiyalash**: Gradientlar uchun operatsiyalarni kuzatish
- **O'zaro muvofiqlik**: numpy ga/dan oson aylantirish`,
		},
	},
};

export default task;
