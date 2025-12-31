import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-autograd',
	title: 'Automatic Differentiation',
	difficulty: 'medium',
	tags: ['pytorch', 'autograd', 'gradients'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Automatic Differentiation

Learn PyTorch's automatic differentiation with autograd.

## Task

Implement four functions:
1. \`simple_gradient(x)\` - Compute gradient of x^2
2. \`chain_rule_gradient(x)\` - Compute gradient of sin(x^2)
3. \`neural_net_gradient(x, w)\` - Gradient of (x @ w).sum()
4. \`no_grad_operation(x)\` - Perform operation without tracking gradients

## Example

\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
grad = simple_gradient(x)  # d(x^2)/dx = 2x = 4.0

w = torch.randn(3, 2, requires_grad=True)
grad_w = neural_net_gradient(x, w)  # Gradient w.r.t. w
\`\`\``,

	initialCode: `import torch

def simple_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of x^2 at x. Return x.grad."""
    # Your code here
    pass

def chain_rule_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of sin(x^2) at x. Return x.grad."""
    # Your code here
    pass

def neural_net_gradient(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute gradient of (x @ w).sum() w.r.t. w. Return w.grad."""
    # Your code here
    pass

def no_grad_operation(x: torch.Tensor) -> torch.Tensor:
    """Square x without tracking gradients. Return result."""
    # Your code here
    pass
`,

	solutionCode: `import torch

def simple_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of x^2 at x. Return x.grad."""
    y = x ** 2
    y.backward()
    return x.grad.clone()

def chain_rule_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of sin(x^2) at x. Return x.grad."""
    y = torch.sin(x ** 2)
    y.backward()
    return x.grad.clone()

def neural_net_gradient(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute gradient of (x @ w).sum() w.r.t. w. Return w.grad."""
    y = (x @ w).sum()
    y.backward()
    return w.grad.clone()

def no_grad_operation(x: torch.Tensor) -> torch.Tensor:
    """Square x without tracking gradients. Return result."""
    with torch.no_grad():
        return x ** 2
`,

	testCode: `import torch
import unittest

class TestAutograd(unittest.TestCase):
    def test_simple_gradient(self):
        x = torch.tensor([3.0], requires_grad=True)
        grad = simple_gradient(x)
        self.assertAlmostEqual(grad.item(), 6.0, places=4)

    def test_chain_rule(self):
        x = torch.tensor([0.0], requires_grad=True)
        grad = chain_rule_gradient(x)
        # d/dx sin(x^2) at x=0 is 2x*cos(x^2) = 0
        self.assertAlmostEqual(grad.item(), 0.0, places=4)

    def test_neural_net_gradient_shape(self):
        x = torch.randn(5, 3, requires_grad=True)
        w = torch.randn(3, 2, requires_grad=True)
        grad = neural_net_gradient(x, w)
        self.assertEqual(grad.shape, torch.Size([3, 2]))

    def test_no_grad(self):
        x = torch.tensor([2.0], requires_grad=True)
        result = no_grad_operation(x)
        self.assertFalse(result.requires_grad)
        self.assertAlmostEqual(result.item(), 4.0)

    def test_simple_gradient_returns_tensor(self):
        x = torch.tensor([1.0], requires_grad=True)
        grad = simple_gradient(x)
        self.assertIsInstance(grad, torch.Tensor)

    def test_chain_rule_nonzero(self):
        x = torch.tensor([1.0], requires_grad=True)
        grad = chain_rule_gradient(x)
        self.assertNotEqual(grad.item(), 0.0)

    def test_neural_net_gradient_not_none(self):
        x = torch.randn(5, 3, requires_grad=True)
        w = torch.randn(3, 2, requires_grad=True)
        grad = neural_net_gradient(x, w)
        self.assertIsNotNone(grad)

    def test_no_grad_preserves_value(self):
        x = torch.tensor([3.0], requires_grad=True)
        result = no_grad_operation(x)
        self.assertAlmostEqual(result.item(), 9.0, places=4)

    def test_simple_gradient_vector(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        expected = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(x.grad, expected)

    def test_neural_net_gradient_finite(self):
        x = torch.randn(5, 3, requires_grad=True)
        w = torch.randn(3, 2, requires_grad=True)
        grad = neural_net_gradient(x, w)
        self.assertTrue(torch.isfinite(grad).all())
`,

	hint1: 'Call y.backward() to compute gradients, then access x.grad',
	hint2: 'Use with torch.no_grad(): to disable gradient tracking',

	whyItMatters: `Autograd is the heart of PyTorch:

- **Automatic backpropagation**: No manual gradient calculation
- **Dynamic computation graphs**: Built on-the-fly
- **Debugging**: Inspect gradients at any point
- **Flexibility**: Works with any differentiable operations

Understanding autograd is key to understanding PyTorch.`,

	translations: {
		ru: {
			title: 'Автоматическое дифференцирование',
			description: `# Автоматическое дифференцирование

Изучите автоматическое дифференцирование PyTorch с autograd.

## Задача

Реализуйте четыре функции:
1. \`simple_gradient(x)\` - Вычислить градиент x^2
2. \`chain_rule_gradient(x)\` - Градиент sin(x^2)
3. \`neural_net_gradient(x, w)\` - Градиент (x @ w).sum()
4. \`no_grad_operation(x)\` - Операция без отслеживания градиентов

## Пример

\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
grad = simple_gradient(x)  # d(x^2)/dx = 2x = 4.0

w = torch.randn(3, 2, requires_grad=True)
grad_w = neural_net_gradient(x, w)  # Gradient w.r.t. w
\`\`\``,
			hint1: 'Вызовите y.backward() для вычисления градиентов, затем x.grad',
			hint2: 'Используйте with torch.no_grad(): для отключения отслеживания',
			whyItMatters: `Autograd - сердце PyTorch:

- **Автоматическое обратное распространение**: Без ручного вычисления градиентов
- **Динамические графы**: Строятся на лету
- **Отладка**: Проверка градиентов в любой точке`,
		},
		uz: {
			title: 'Avtomatik differensiyalash',
			description: `# Avtomatik differensiyalash

PyTorch ning autograd bilan avtomatik differensiyalashini o'rganing.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`simple_gradient(x)\` - x^2 ning gradientini hisoblash
2. \`chain_rule_gradient(x)\` - sin(x^2) ning gradienti
3. \`neural_net_gradient(x, w)\` - (x @ w).sum() ning w bo'yicha gradienti
4. \`no_grad_operation(x)\` - Gradientlarni kuzatmasdan operatsiya

## Misol

\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
grad = simple_gradient(x)  # d(x^2)/dx = 2x = 4.0

w = torch.randn(3, 2, requires_grad=True)
grad_w = neural_net_gradient(x, w)  # Gradient w.r.t. w
\`\`\``,
			hint1: "Gradientlarni hisoblash uchun y.backward() ni chaqiring, keyin x.grad ga kiring",
			hint2: "Gradient kuzatuvini o'chirish uchun with torch.no_grad(): dan foydalaning",
			whyItMatters: `Autograd PyTorch ning yuragi:

- **Avtomatik orqaga tarqalish**: Qo'lda gradient hisoblash yo'q
- **Dinamik hisoblash grafiklari**: Parvoz paytida quriladi
- **Nosozliklarni tuzatish**: Istalgan nuqtada gradientlarni tekshirish`,
		},
	},
};

export default task;
