import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-self-attention-layer',
	title: 'Self-Attention Layer',
	difficulty: 'medium',
	tags: ['pytorch', 'transformer', 'self-attention'],
	estimatedTime: '15m',
	isPremium: true,
	order: 4,
	description: `# Self-Attention Layer

Build a complete self-attention layer with layer normalization and residual connection.

## Task

Implement a \`SelfAttentionLayer\` class that:
- Uses multi-head self-attention
- Applies residual connection
- Uses layer normalization

## Example

\`\`\`python
layer = SelfAttentionLayer(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)
output = layer(x)
# output.shape = (2, 10, 512)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    """Self-attention layer with residual and layer norm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply self-attention with residual connection."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionLayer(nn.Module):
    """Self-attention layer with residual and layer norm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply self-attention with residual connection."""
        batch_size = x.size(0)
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Project Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        context = torch.matmul(attn, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(context)
        output = self.dropout(output)

        # Residual connection
        return residual + output
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestSelfAttentionLayer(unittest.TestCase):
    def test_output_shape(self):
        layer = SelfAttentionLayer(d_model=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        output = layer(x)
        self.assertEqual(output.shape, (2, 10, 512))

    def test_residual_connection(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4)
        x = torch.randn(2, 5, 256)
        output = layer(x)
        # Output should be different from input (due to attention)
        self.assertFalse(torch.allclose(x, output))

    def test_with_mask(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4)
        x = torch.randn(2, 5, 256)
        mask = torch.tril(torch.ones(5, 5))
        output = layer(x, mask)
        self.assertEqual(output.shape, (2, 5, 256))

    def test_is_nn_module(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4)
        self.assertIsInstance(layer, nn.Module)

    def test_has_layer_norm(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4)
        self.assertTrue(hasattr(layer, 'norm'))
        self.assertIsInstance(layer.norm, nn.LayerNorm)

    def test_has_projections(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4)
        self.assertTrue(hasattr(layer, 'W_q'))
        self.assertTrue(hasattr(layer, 'W_k'))
        self.assertTrue(hasattr(layer, 'W_v'))
        self.assertTrue(hasattr(layer, 'W_o'))

    def test_output_not_nan(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4)
        x = torch.randn(2, 8, 256)
        output = layer(x)
        self.assertFalse(torch.isnan(output).any())

    def test_single_sample(self):
        layer = SelfAttentionLayer(d_model=512, num_heads=8)
        x = torch.randn(1, 6, 512)
        output = layer(x)
        self.assertEqual(output.shape, (1, 6, 512))

    def test_has_dropout(self):
        layer = SelfAttentionLayer(d_model=256, num_heads=4, dropout=0.5)
        self.assertTrue(hasattr(layer, 'dropout'))

    def test_d_k_calculation(self):
        layer = SelfAttentionLayer(d_model=512, num_heads=8)
        self.assertEqual(layer.d_k, 64)
`,

	hint1: 'Apply layer norm before attention (pre-norm)',
	hint2: 'Add residual connection: output = x + attention(norm(x))',

	whyItMatters: `Self-attention layers are transformer building blocks:

- **Residual connections**: Enable deep networks, preserve gradients
- **Layer normalization**: Stabilize training
- **Pre-norm vs post-norm**: Pre-norm is more stable for deep models
- **Standard pattern**: Used in every transformer layer

This pattern is repeated in all transformer architectures.`,

	translations: {
		ru: {
			title: 'Слой Self-Attention',
			description: `# Слой Self-Attention

Создайте полный слой self-attention с layer normalization и residual connection.

## Задача

Реализуйте класс \`SelfAttentionLayer\`, который:
- Использует multi-head self-attention
- Применяет residual connection
- Использует layer normalization

## Пример

\`\`\`python
layer = SelfAttentionLayer(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)
output = layer(x)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: 'Применяйте layer norm перед attention (pre-norm)',
			hint2: 'Добавьте residual connection: output = x + attention(norm(x))',
			whyItMatters: `Слои self-attention - строительные блоки трансформеров:

- **Residual connections**: Позволяют глубокие сети, сохраняют градиенты
- **Layer normalization**: Стабилизирует обучение
- **Pre-norm vs post-norm**: Pre-norm стабильнее для глубоких моделей
- **Стандартный паттерн**: Используется в каждом слое трансформера`,
		},
		uz: {
			title: 'Self-Attention qatlami',
			description: `# Self-Attention qatlami

Layer normalization va residual connection bilan to'liq self-attention qatlamini yarating.

## Topshiriq

\`SelfAttentionLayer\` sinfini amalga oshiring:
- Multi-head self-attention dan foydalanadi
- Residual connection qo'llaydi
- Layer normalization dan foydalanadi

## Misol

\`\`\`python
layer = SelfAttentionLayer(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)
output = layer(x)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: "Attention dan oldin layer norm qo'llang (pre-norm)",
			hint2: "Residual connection qo'shing: output = x + attention(norm(x))",
			whyItMatters: `Self-attention qatlamlari transformer qurilish bloklari:

- **Residual connections**: Chuqur tarmoqlarni yoqadi, gradientlarni saqlaydi
- **Layer normalization**: O'qitishni barqarorlashtiradi
- **Pre-norm vs post-norm**: Pre-norm chuqur modellar uchun barqarorroq
- **Standart naqsh**: Har bir transformer qatlamida ishlatiladi`,
		},
	},
};

export default task;
