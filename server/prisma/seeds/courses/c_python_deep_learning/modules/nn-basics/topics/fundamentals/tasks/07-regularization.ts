import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-regularization',
	title: 'Regularization Techniques',
	difficulty: 'medium',
	tags: ['numpy', 'regularization', 'neural-networks'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,
	description: `# Regularization Techniques

Implement regularization techniques to prevent overfitting.

## Task

Implement four functions:
1. \`l2_regularization(weights, lambda_)\` - L2 penalty term
2. \`l1_regularization(weights, lambda_)\` - L1 penalty term
3. \`dropout_forward(x, p, training)\` - Apply dropout during training
4. \`add_regularization_gradient(grads, weights, lambda_)\` - Add L2 gradient term

## Example

\`\`\`python
# L2 regularization loss
reg_loss = l2_regularization(weights, lambda_=0.01)
total_loss = cross_entropy_loss + reg_loss

# Dropout during training
h = relu(x @ W1)
h = dropout_forward(h, p=0.5, training=True)
\`\`\``,

	initialCode: `import numpy as np

def l2_regularization(weights: list, lambda_: float) -> float:
    """Compute L2 regularization term: (lambda/2) * sum(w^2)."""
    # Your code here
    pass

def l1_regularization(weights: list, lambda_: float) -> float:
    """Compute L1 regularization term: lambda * sum(|w|)."""
    # Your code here
    pass

def dropout_forward(x: np.ndarray, p: float, training: bool = True) -> tuple:
    """Apply dropout. Return (output, mask) during training, (x, None) during eval."""
    # Your code here
    pass

def add_regularization_gradient(grads: np.ndarray, weights: np.ndarray, lambda_: float) -> np.ndarray:
    """Add L2 regularization gradient: grad + lambda * weights."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def l2_regularization(weights: list, lambda_: float) -> float:
    """Compute L2 regularization term: (lambda/2) * sum(w^2)."""
    reg = 0.0
    for w in weights:
        reg += np.sum(w ** 2)
    return 0.5 * lambda_ * reg

def l1_regularization(weights: list, lambda_: float) -> float:
    """Compute L1 regularization term: lambda * sum(|w|)."""
    reg = 0.0
    for w in weights:
        reg += np.sum(np.abs(w))
    return lambda_ * reg

def dropout_forward(x: np.ndarray, p: float, training: bool = True) -> tuple:
    """Apply dropout. Return (output, mask) during training, (x, None) during eval."""
    if not training or p == 0:
        return x, None
    mask = (np.random.rand(*x.shape) > p).astype(float)
    out = x * mask / (1 - p)  # Inverted dropout
    return out, mask

def add_regularization_gradient(grads: np.ndarray, weights: np.ndarray, lambda_: float) -> np.ndarray:
    """Add L2 regularization gradient: grad + lambda * weights."""
    return grads + lambda_ * weights
`,

	testCode: `import numpy as np
import unittest

class TestRegularization(unittest.TestCase):
    def test_l2_positive(self):
        weights = [np.array([1.0, 2.0]), np.array([3.0])]
        reg = l2_regularization(weights, lambda_=0.1)
        self.assertGreater(reg, 0)

    def test_l1_positive(self):
        weights = [np.array([1.0, -2.0])]
        reg = l1_regularization(weights, lambda_=0.1)
        self.assertAlmostEqual(reg, 0.3, places=5)

    def test_dropout_training(self):
        np.random.seed(42)
        x = np.ones((10, 5))
        out, mask = dropout_forward(x, p=0.5, training=True)
        self.assertIsNotNone(mask)
        self.assertTrue(np.any(out == 0))

    def test_dropout_eval(self):
        x = np.ones((10, 5))
        out, mask = dropout_forward(x, p=0.5, training=False)
        self.assertIsNone(mask)
        np.testing.assert_array_equal(out, x)

    def test_reg_gradient(self):
        grads = np.array([1.0, 2.0])
        weights = np.array([0.5, 1.0])
        new_grads = add_regularization_gradient(grads, weights, lambda_=0.1)
        expected = np.array([1.05, 2.1])
        np.testing.assert_array_almost_equal(new_grads, expected)

    def test_l2_zero_weights(self):
        weights = [np.zeros((3, 3))]
        reg = l2_regularization(weights, lambda_=0.1)
        self.assertEqual(reg, 0.0)

    def test_l1_zero_weights(self):
        weights = [np.zeros((3, 3))]
        reg = l1_regularization(weights, lambda_=0.1)
        self.assertEqual(reg, 0.0)

    def test_l2_returns_float(self):
        weights = [np.random.randn(5, 5)]
        reg = l2_regularization(weights, lambda_=0.01)
        self.assertIsInstance(reg, (float, np.floating))

    def test_dropout_returns_tuple(self):
        x = np.ones((5, 5))
        result = dropout_forward(x, p=0.5, training=True)
        self.assertEqual(len(result), 2)

    def test_dropout_p_zero(self):
        x = np.ones((5, 5))
        out, mask = dropout_forward(x, p=0.0, training=True)
        np.testing.assert_array_equal(out, x)
`,

	hint1: 'L2: (lambda/2) * sum(w^2), L1: lambda * sum(|w|)',
	hint2: 'Inverted dropout: multiply by 1/(1-p) during training so no change at test time',

	whyItMatters: `Regularization prevents overfitting:

- **L2 (weight decay)**: Encourages small weights, smooth functions
- **L1 (sparsity)**: Encourages zero weights, feature selection
- **Dropout**: Prevents co-adaptation of neurons
- **Generalization**: Better performance on unseen data

Essential for training neural networks that generalize.`,

	translations: {
		ru: {
			title: 'Техники регуляризации',
			description: `# Техники регуляризации

Реализуйте техники регуляризации для предотвращения переобучения.

## Задача

Реализуйте четыре функции:
1. \`l2_regularization(weights, lambda_)\` - L2 штраф
2. \`l1_regularization(weights, lambda_)\` - L1 штраф
3. \`dropout_forward(x, p, training)\` - Dropout при обучении
4. \`add_regularization_gradient(grads, weights, lambda_)\` - Добавить L2 градиент

## Пример

\`\`\`python
# L2 regularization loss
reg_loss = l2_regularization(weights, lambda_=0.01)
total_loss = cross_entropy_loss + reg_loss

# Dropout during training
h = relu(x @ W1)
h = dropout_forward(h, p=0.5, training=True)
\`\`\``,
			hint1: 'L2: (lambda/2) * sum(w^2), L1: lambda * sum(|w|)',
			hint2: 'Инвертированный dropout: умножаем на 1/(1-p) при обучении',
			whyItMatters: `Регуляризация предотвращает переобучение:

- **L2 (weight decay)**: Поощряет малые веса, гладкие функции
- **L1 (разреженность)**: Поощряет нулевые веса, отбор признаков
- **Dropout**: Предотвращает коадаптацию нейронов`,
		},
		uz: {
			title: 'Regulyarizatsiya texnikalari',
			description: `# Regulyarizatsiya texnikalari

Ortiqcha moslanishni oldini olish uchun regulyarizatsiya texnikalarini amalga oshiring.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`l2_regularization(weights, lambda_)\` - L2 jarima atamasi
2. \`l1_regularization(weights, lambda_)\` - L1 jarima atamasi
3. \`dropout_forward(x, p, training)\` - O'qitish paytida dropout qo'llash
4. \`add_regularization_gradient(grads, weights, lambda_)\` - L2 gradient atamasini qo'shish

## Misol

\`\`\`python
# L2 regularization loss
reg_loss = l2_regularization(weights, lambda_=0.01)
total_loss = cross_entropy_loss + reg_loss

# Dropout during training
h = relu(x @ W1)
h = dropout_forward(h, p=0.5, training=True)
\`\`\``,
			hint1: "L2: (lambda/2) * sum(w^2), L1: lambda * sum(|w|)",
			hint2: "Teskari dropout: test vaqtida o'zgarish bo'lmasligi uchun o'qitishda 1/(1-p) ga ko'paytiring",
			whyItMatters: `Regulyarizatsiya ortiqcha moslanishni oldini oladi:

- **L2 (vazn pasayishi)**: Kichik og'irliklarni, silliq funksiyalarni rag'batlantiradi
- **L1 (siyraklik)**: Nol og'irliklarni, xususiyat tanlashni rag'batlantiradi
- **Dropout**: Neyronlarning birgalikda moslashishini oldini oladi`,
		},
	},
};

export default task;
