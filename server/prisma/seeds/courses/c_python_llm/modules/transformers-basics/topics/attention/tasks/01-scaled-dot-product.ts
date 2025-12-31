import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-scaled-dot-product-attention',
	title: 'Scaled Dot-Product Attention',
	difficulty: 'medium',
	tags: ['pytorch', 'attention', 'transformer'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Scaled Dot-Product Attention

Implement the core attention mechanism used in Transformers.

## Task

Implement the \`scaled_dot_product_attention\` function that:
- Computes attention scores from Q, K, V matrices
- Applies optional mask for causal attention
- Returns attention output and weights

## Formula

\\[\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V\\]

## Example

\`\`\`python
Q = torch.randn(2, 4, 64)  # batch, seq_len, d_k
K = torch.randn(2, 4, 64)
V = torch.randn(2, 4, 64)

output, weights = scaled_dot_product_attention(Q, K, V)
# output.shape = (2, 4, 64)
# weights.shape = (2, 4, 4)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple:
    # TODO: Compute attention scores (Q @ K.T / sqrt(d_k)), apply mask, softmax, return (output, weights)

    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple:
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch, seq_len, d_k)
        key: (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        mask: optional mask for attention scores

    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Compute attention scores: (batch, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Compute output: (batch, seq_len, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
`,

	testCode: `import torch
import math
import unittest

class TestScaledDotProductAttention(unittest.TestCase):
    def test_output_shape(self):
        Q = torch.randn(2, 4, 64)
        K = torch.randn(2, 4, 64)
        V = torch.randn(2, 4, 64)
        output, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(output.shape, (2, 4, 64))
        self.assertEqual(weights.shape, (2, 4, 4))

    def test_attention_weights_sum_to_one(self):
        Q = torch.randn(2, 4, 64)
        K = torch.randn(2, 4, 64)
        V = torch.randn(2, 4, 64)
        _, weights = scaled_dot_product_attention(Q, K, V)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_with_mask(self):
        Q = torch.randn(2, 4, 64)
        K = torch.randn(2, 4, 64)
        V = torch.randn(2, 4, 64)
        mask = torch.tril(torch.ones(4, 4))
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        # Upper triangle of weights should be 0
        self.assertTrue((weights[:, 0, 1:] == 0).all())

    def test_returns_tuple(self):
        Q = torch.randn(1, 4, 32)
        K = torch.randn(1, 4, 32)
        V = torch.randn(1, 4, 32)
        result = scaled_dot_product_attention(Q, K, V)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_output_not_nan(self):
        Q = torch.randn(2, 6, 64)
        K = torch.randn(2, 6, 64)
        V = torch.randn(2, 6, 64)
        output, _ = scaled_dot_product_attention(Q, K, V)
        self.assertFalse(torch.isnan(output).any())

    def test_weights_positive(self):
        Q = torch.randn(2, 4, 64)
        K = torch.randn(2, 4, 64)
        V = torch.randn(2, 4, 64)
        _, weights = scaled_dot_product_attention(Q, K, V)
        self.assertTrue((weights >= 0).all())

    def test_single_sample(self):
        Q = torch.randn(1, 5, 32)
        K = torch.randn(1, 5, 32)
        V = torch.randn(1, 5, 32)
        output, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(output.shape, (1, 5, 32))
        self.assertEqual(weights.shape, (1, 5, 5))

    def test_different_dimensions(self):
        for d_k in [32, 64, 128]:
            Q = torch.randn(2, 4, d_k)
            K = torch.randn(2, 4, d_k)
            V = torch.randn(2, 4, d_k)
            output, _ = scaled_dot_product_attention(Q, K, V)
            self.assertEqual(output.shape[-1], d_k)

    def test_different_seq_lengths(self):
        Q = torch.randn(2, 8, 64)
        K = torch.randn(2, 8, 64)
        V = torch.randn(2, 8, 64)
        output, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(weights.shape, (2, 8, 8))

    def test_weights_is_tensor(self):
        Q = torch.randn(1, 3, 32)
        K = torch.randn(1, 3, 32)
        V = torch.randn(1, 3, 32)
        _, weights = scaled_dot_product_attention(Q, K, V)
        self.assertIsInstance(weights, torch.Tensor)
`,

	hint1: 'Scale by sqrt(d_k) before softmax to prevent vanishing gradients',
	hint2: 'Use masked_fill with -inf before softmax to mask positions',

	whyItMatters: `Scaled dot-product attention is the foundation of transformers:

- **Self-attention**: Each position attends to all others
- **Scaling**: Prevents softmax saturation with large dimensions
- **Masking**: Enables causal/decoder attention
- **Parallelizable**: All positions computed simultaneously

This is the core operation in BERT, GPT, and all modern LLMs.`,

	translations: {
		ru: {
			title: 'Scaled Dot-Product Attention',
			description: `# Scaled Dot-Product Attention

Реализуйте основной механизм внимания, используемый в трансформерах.

## Задача

Реализуйте функцию \`scaled_dot_product_attention\`, которая:
- Вычисляет оценки внимания из матриц Q, K, V
- Применяет опциональную маску для causal attention
- Возвращает выход внимания и веса

## Пример

\`\`\`python
Q = torch.randn(2, 4, 64)  # batch, seq_len, d_k
K = torch.randn(2, 4, 64)
V = torch.randn(2, 4, 64)

output, weights = scaled_dot_product_attention(Q, K, V)
# output.shape = (2, 4, 64)
# weights.shape = (2, 4, 4)
\`\`\``,
			hint1: 'Масштабируйте на sqrt(d_k) перед softmax для предотвращения затухающих градиентов',
			hint2: 'Используйте masked_fill с -inf перед softmax для маскировки позиций',
			whyItMatters: `Scaled dot-product attention - основа трансформеров:

- **Self-attention**: Каждая позиция обращает внимание на все остальные
- **Масштабирование**: Предотвращает насыщение softmax при больших размерностях
- **Маскирование**: Позволяет causal/decoder attention
- **Параллелизуемость**: Все позиции вычисляются одновременно`,
		},
		uz: {
			title: 'Scaled Dot-Product Attention',
			description: `# Scaled Dot-Product Attention

Transformerlarda ishlatiladigan asosiy diqqat mexanizmini amalga oshiring.

## Topshiriq

\`scaled_dot_product_attention\` funksiyasini amalga oshiring:
- Q, K, V matrisalaridan diqqat ballarini hisoblaydi
- Causal attention uchun ixtiyoriy niqobni qo'llaydi
- Diqqat chiqishi va vaznlarini qaytaradi

## Misol

\`\`\`python
Q = torch.randn(2, 4, 64)  # batch, seq_len, d_k
K = torch.randn(2, 4, 64)
V = torch.randn(2, 4, 64)

output, weights = scaled_dot_product_attention(Q, K, V)
# output.shape = (2, 4, 64)
# weights.shape = (2, 4, 4)
\`\`\``,
			hint1: "Yo'qolib ketayotgan gradientlarni oldini olish uchun softmax dan oldin sqrt(d_k) ga bo'ling",
			hint2: "Pozitsiyalarni niqoblash uchun softmax dan oldin masked_fill ni -inf bilan ishlating",
			whyItMatters: `Scaled dot-product attention transformerlarning asosi:

- **Self-attention**: Har bir pozitsiya barchasiga e'tibor beradi
- **Masshtablash**: Katta o'lchamlarda softmax to'yinishini oldini oladi
- **Niqoblash**: Causal/decoder diqqatini yoqadi
- **Parallellanishi mumkin**: Barcha pozitsiyalar bir vaqtda hisoblanadi`,
		},
	},
};

export default task;
