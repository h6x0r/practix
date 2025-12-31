import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-tensor-operations',
	title: 'Tensor Operations',
	difficulty: 'easy',
	tags: ['pytorch', 'tensors', 'operations'],
	estimatedTime: '12m',
	isPremium: false,
	order: 2,
	description: `# Tensor Operations

Master common tensor operations in PyTorch.

## Task

Implement five functions:
1. \`matrix_multiply(a, b)\` - Matrix multiplication
2. \`elementwise_ops(a, b)\` - Return dict with add, sub, mul, div results
3. \`reshape_tensor(t, new_shape)\` - Reshape tensor
4. \`concatenate_tensors(tensors, dim)\` - Concatenate along dimension
5. \`reduction_ops(t)\` - Return dict with sum, mean, max, min

## Example

\`\`\`python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

c = matrix_multiply(a, b)  # 2x2 matrix
ops = elementwise_ops(a, b)  # {'add': ..., 'sub': ..., ...}
\`\`\``,

	initialCode: `import torch

def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication: a @ b."""
    # Your code here
    pass

def elementwise_ops(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Return dict with 'add', 'sub', 'mul', 'div' results."""
    # Your code here
    pass

def reshape_tensor(t: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """Reshape tensor to new_shape."""
    # Your code here
    pass

def concatenate_tensors(tensors: list, dim: int) -> torch.Tensor:
    """Concatenate list of tensors along dimension."""
    # Your code here
    pass

def reduction_ops(t: torch.Tensor) -> dict:
    """Return dict with 'sum', 'mean', 'max', 'min' values."""
    # Your code here
    pass
`,

	solutionCode: `import torch

def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication: a @ b."""
    return torch.matmul(a, b)

def elementwise_ops(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Return dict with 'add', 'sub', 'mul', 'div' results."""
    return {
        'add': a + b,
        'sub': a - b,
        'mul': a * b,
        'div': a / b
    }

def reshape_tensor(t: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """Reshape tensor to new_shape."""
    return t.reshape(new_shape)

def concatenate_tensors(tensors: list, dim: int) -> torch.Tensor:
    """Concatenate list of tensors along dimension."""
    return torch.cat(tensors, dim=dim)

def reduction_ops(t: torch.Tensor) -> dict:
    """Return dict with 'sum', 'mean', 'max', 'min' values."""
    return {
        'sum': t.sum().item(),
        'mean': t.float().mean().item(),
        'max': t.max().item(),
        'min': t.min().item()
    }
`,

	testCode: `import torch
import unittest

class TestTensorOperations(unittest.TestCase):
    def test_matrix_multiply_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = matrix_multiply(a, b)
        self.assertEqual(c.shape, torch.Size([3, 5]))

    def test_elementwise_ops_keys(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = elementwise_ops(a, b)
        self.assertIn('add', result)
        self.assertIn('mul', result)

    def test_reshape(self):
        t = torch.randn(2, 3, 4)
        reshaped = reshape_tensor(t, (6, 4))
        self.assertEqual(reshaped.shape, torch.Size([6, 4]))

    def test_concatenate(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = concatenate_tensors([a, b], dim=0)
        self.assertEqual(c.shape, torch.Size([4, 3]))

    def test_reduction_ops(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = reduction_ops(t)
        self.assertAlmostEqual(result['sum'], 10.0)
        self.assertAlmostEqual(result['mean'], 2.5)

    def test_matrix_multiply_values(self):
        a = torch.eye(3)
        b = torch.ones(3, 3)
        c = matrix_multiply(a, b)
        torch.testing.assert_close(c, b)

    def test_elementwise_add(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = elementwise_ops(a, b)
        torch.testing.assert_close(result['add'], torch.tensor([4.0, 6.0]))

    def test_concatenate_dim1(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        c = concatenate_tensors([a, b], dim=1)
        self.assertEqual(c.shape, torch.Size([2, 7]))

    def test_reduction_max_min(self):
        t = torch.tensor([1.0, 5.0, 3.0])
        result = reduction_ops(t)
        self.assertEqual(result['max'], 5.0)
        self.assertEqual(result['min'], 1.0)

    def test_reshape_preserves_elements(self):
        t = torch.arange(24)
        reshaped = reshape_tensor(t, (4, 6))
        self.assertEqual(reshaped.numel(), 24)
`,

	hint1: 'torch.matmul(a, b) or a @ b for matrix multiplication',
	hint2: 'torch.cat(tensors, dim=0) concatenates, t.reshape(shape) reshapes',

	whyItMatters: `Tensor operations are essential for:

- **Forward pass**: Matrix multiplications for layers
- **Data preprocessing**: Reshaping, concatenating batches
- **Loss computation**: Reductions like mean, sum
- **Feature engineering**: Element-wise operations

Core operations you'll use in every neural network.`,

	translations: {
		ru: {
			title: 'Операции с тензорами',
			description: `# Операции с тензорами

Освойте распространённые операции с тензорами в PyTorch.

## Задача

Реализуйте пять функций:
1. \`matrix_multiply(a, b)\` - Матричное умножение
2. \`elementwise_ops(a, b)\` - Поэлементные операции
3. \`reshape_tensor(t, new_shape)\` - Изменить форму
4. \`concatenate_tensors(tensors, dim)\` - Конкатенация
5. \`reduction_ops(t)\` - Редукции (сумма, среднее, макс, мин)

## Пример

\`\`\`python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

c = matrix_multiply(a, b)  # 2x2 matrix
ops = elementwise_ops(a, b)  # {'add': ..., 'sub': ..., ...}
\`\`\``,
			hint1: 'torch.matmul(a, b) или a @ b для матричного умножения',
			hint2: 'torch.cat(tensors, dim=0) конкатенирует, t.reshape(shape) изменяет форму',
			whyItMatters: `Операции с тензорами важны для:

- **Прямой проход**: Матричные умножения для слоёв
- **Предобработка данных**: Изменение формы, конкатенация батчей
- **Вычисление потерь**: Редукции как mean, sum`,
		},
		uz: {
			title: 'Tensor operatsiyalari',
			description: `# Tensor operatsiyalari

PyTorch da umumiy tensor operatsiyalarini o'rganing.

## Topshiriq

Beshta funksiyani amalga oshiring:
1. \`matrix_multiply(a, b)\` - Matritsa ko'paytirish
2. \`elementwise_ops(a, b)\` - Elementwise operatsiyalar
3. \`reshape_tensor(t, new_shape)\` - Tensorni qayta shakllantirish
4. \`concatenate_tensors(tensors, dim)\` - O'lcham bo'ylab birlashtirish
5. \`reduction_ops(t)\` - sum, mean, max, min qisqartirishlari

## Misol

\`\`\`python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

c = matrix_multiply(a, b)  # 2x2 matrix
ops = elementwise_ops(a, b)  # {'add': ..., 'sub': ..., ...}
\`\`\``,
			hint1: "Matritsa ko'paytirish uchun torch.matmul(a, b) yoki a @ b",
			hint2: "torch.cat(tensors, dim=0) birlashtiradi, t.reshape(shape) qayta shakllantiradi",
			whyItMatters: `Tensor operatsiyalari quyidagilar uchun muhim:

- **Oldinga o'tish**: Qatlamlar uchun matritsa ko'paytirish
- **Ma'lumotlarni oldindan ishlov berish**: Qayta shakllantirish, batchlarni birlashtirish
- **Yo'qotishni hisoblash**: mean, sum kabi qisqartirishlar`,
		},
	},
};

export default task;
