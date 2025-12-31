import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-transformer-encoder-block',
	title: 'Transformer Encoder Block',
	difficulty: 'medium',
	tags: ['pytorch', 'transformer', 'encoder'],
	estimatedTime: '20m',
	isPremium: true,
	order: 7,
	description: `# Transformer Encoder Block

Implement a complete transformer encoder block combining self-attention and feed-forward layers.

## Task

Implement an \`EncoderBlock\` class that combines:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections
- Dropout

## Architecture

\`\`\`
Input → LayerNorm → Self-Attention → Add → LayerNorm → FFN → Add → Output
         ↑___________________________↓       ↑________________↓
                  (residual)                     (residual)
\`\`\`

## Example

\`\`\`python
block = EncoderBlock(d_model=512, num_heads=8, d_ff=2048)

x = torch.randn(2, 10, 512)
output = block(x)
# output.shape = (2, 10, 512)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        # Your code here
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through encoder block."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Self-attention components
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Feed-forward components
        self.ff_linear1 = nn.Linear(d_model, d_ff)
        self.ff_linear2 = nn.Linear(d_ff, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def self_attention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.size(0)

        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context)

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_linear2(self.dropout(F.gelu(self.ff_linear1(x))))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm self-attention with residual
        attn_out = self.self_attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestEncoderBlock(unittest.TestCase):
    def test_output_shape(self):
        block = EncoderBlock(d_model=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        output = block(x)
        self.assertEqual(output.shape, (2, 10, 512))

    def test_with_mask(self):
        block = EncoderBlock(d_model=256, num_heads=4)
        x = torch.randn(2, 8, 256)
        mask = torch.ones(8, 8)
        output = block(x, mask)
        self.assertEqual(output.shape, (2, 8, 256))

    def test_custom_ff_dim(self):
        block = EncoderBlock(d_model=256, num_heads=4, d_ff=1024)
        self.assertEqual(block.ff_linear1.out_features, 1024)

    def test_residual_connection(self):
        block = EncoderBlock(d_model=128, num_heads=2)
        x = torch.randn(1, 5, 128)
        output = block(x)
        # Output should be different but have same shape
        self.assertFalse(torch.allclose(x, output))

    def test_is_nn_module(self):
        block = EncoderBlock(d_model=256, num_heads=4)
        self.assertIsInstance(block, nn.Module)

    def test_has_layer_norms(self):
        block = EncoderBlock(d_model=256, num_heads=4)
        self.assertTrue(hasattr(block, 'norm1'))
        self.assertTrue(hasattr(block, 'norm2'))

    def test_has_projections(self):
        block = EncoderBlock(d_model=256, num_heads=4)
        self.assertTrue(hasattr(block, 'W_q'))
        self.assertTrue(hasattr(block, 'W_k'))
        self.assertTrue(hasattr(block, 'W_v'))
        self.assertTrue(hasattr(block, 'W_o'))

    def test_output_not_nan(self):
        block = EncoderBlock(d_model=128, num_heads=2)
        x = torch.randn(2, 8, 128)
        output = block(x)
        self.assertFalse(torch.isnan(output).any())

    def test_single_sample(self):
        block = EncoderBlock(d_model=256, num_heads=4)
        x = torch.randn(1, 6, 256)
        output = block(x)
        self.assertEqual(output.shape, (1, 6, 256))

    def test_has_dropout(self):
        block = EncoderBlock(d_model=256, num_heads=4, dropout=0.2)
        self.assertTrue(hasattr(block, 'dropout'))
`,

	hint1: 'Use pre-norm: apply LayerNorm before attention and FFN',
	hint2: 'Residual: output = x + sublayer(norm(x))',

	whyItMatters: `Encoder blocks are the foundation of BERT-style models:

- **Pre-norm**: More stable training for deep models
- **Residual connections**: Enable gradient flow in deep networks
- **Stacked blocks**: 6-24 blocks create powerful representations
- **BERT architecture**: 12 encoder blocks in BERT-base

Each block refines the representations progressively.`,

	translations: {
		ru: {
			title: 'Блок энкодера трансформера',
			description: `# Блок энкодера трансформера

Реализуйте полный блок энкодера трансформера, комбинирующий self-attention и feed-forward слои.

## Задача

Реализуйте класс \`EncoderBlock\`, объединяющий:
- Multi-head self-attention
- Feed-forward сеть
- Layer normalization
- Residual connections
- Dropout

## Пример

\`\`\`python
block = EncoderBlock(d_model=512, num_heads=8, d_ff=2048)

x = torch.randn(2, 10, 512)
output = block(x)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: 'Используйте pre-norm: применяйте LayerNorm перед attention и FFN',
			hint2: 'Residual: output = x + sublayer(norm(x))',
			whyItMatters: `Блоки энкодера - основа моделей типа BERT:

- **Pre-norm**: Более стабильное обучение глубоких моделей
- **Residual connections**: Обеспечивают поток градиентов
- **Стек блоков**: 6-24 блока создают мощные представления
- **Архитектура BERT**: 12 блоков энкодера в BERT-base`,
		},
		uz: {
			title: 'Transformer encoder bloki',
			description: `# Transformer encoder bloki

Self-attention va feed-forward qatlamlarini birlashtiruvchi to'liq transformer encoder blokini amalga oshiring.

## Topshiriq

\`EncoderBlock\` sinfini amalga oshiring:
- Multi-head self-attention
- Feed-forward tarmoq
- Layer normalization
- Residual connections
- Dropout

## Misol

\`\`\`python
block = EncoderBlock(d_model=512, num_heads=8, d_ff=2048)

x = torch.randn(2, 10, 512)
output = block(x)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: "Pre-norm dan foydalaning: attention va FFN dan oldin LayerNorm qo'llang",
			hint2: 'Residual: output = x + sublayer(norm(x))',
			whyItMatters: `Encoder bloklari BERT uslubidagi modellarning asosi:

- **Pre-norm**: Chuqur modellarni barqaror o'qitish
- **Residual connections**: Chuqur tarmoqlarda gradient oqimini ta'minlaydi
- **Qatlamlangan bloklar**: 6-24 blok kuchli tasvirlar yaratadi
- **BERT arxitekturasi**: BERT-base da 12 encoder bloki`,
		},
	},
};

export default task;
