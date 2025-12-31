import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-transformer-decoder-block',
	title: 'Transformer Decoder Block',
	difficulty: 'hard',
	tags: ['pytorch', 'transformer', 'decoder'],
	estimatedTime: '25m',
	isPremium: true,
	order: 8,
	description: `# Transformer Decoder Block

Implement a transformer decoder block with both self-attention and cross-attention.

## Task

Implement a \`DecoderBlock\` class with:
- Masked self-attention (causal)
- Cross-attention to encoder output
- Feed-forward network
- Layer normalization and residual connections

## Architecture

\`\`\`
Input → Norm → Masked Self-Attn → Add → Norm → Cross-Attn → Add → Norm → FFN → Add
         ↑_______________________↓       ↑_______________↓       ↑__________↓
                                              encoder_output
\`\`\`

## Example

\`\`\`python
decoder = DecoderBlock(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)  # decoder input
encoder_out = torch.randn(2, 20, 512)  # encoder output
output = decoder(x, encoder_out)
# output.shape = (2, 10, 512)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    """Transformer decoder block with self and cross attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        # Your code here
        pass

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None) -> torch.Tensor:
        # TODO: Apply masked self-attn, cross-attn to encoder, FFN with residuals
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    """Transformer decoder block with self and cross attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Self-attention
        self.self_W_q = nn.Linear(d_model, d_model)
        self.self_W_k = nn.Linear(d_model, d_model)
        self.self_W_v = nn.Linear(d_model, d_model)
        self.self_W_o = nn.Linear(d_model, d_model)

        # Cross-attention
        self.cross_W_q = nn.Linear(d_model, d_model)
        self.cross_W_k = nn.Linear(d_model, d_model)
        self.cross_W_v = nn.Linear(d_model, d_model)
        self.cross_W_o = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ff_linear1 = nn.Linear(d_model, d_ff)
        self.ff_linear2 = nn.Linear(d_ff, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        return context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None) -> torch.Tensor:
        # Masked self-attention
        normed = self.norm1(x)
        Q = self.self_W_q(normed)
        K = self.self_W_k(normed)
        V = self.self_W_v(normed)
        self_attn = self.self_W_o(self.attention(Q, K, V, self_mask))
        x = x + self.dropout(self_attn)

        # Cross-attention
        normed = self.norm2(x)
        Q = self.cross_W_q(normed)
        K = self.cross_W_k(encoder_output)
        V = self.cross_W_v(encoder_output)
        cross_attn = self.cross_W_o(self.attention(Q, K, V, cross_mask))
        x = x + self.dropout(cross_attn)

        # Feed-forward
        normed = self.norm3(x)
        ff = self.ff_linear2(self.dropout(F.gelu(self.ff_linear1(normed))))
        x = x + self.dropout(ff)

        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestDecoderBlock(unittest.TestCase):
    def test_output_shape(self):
        decoder = DecoderBlock(d_model=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        enc_out = torch.randn(2, 20, 512)
        output = decoder(x, enc_out)
        self.assertEqual(output.shape, (2, 10, 512))

    def test_with_masks(self):
        decoder = DecoderBlock(d_model=256, num_heads=4)
        x = torch.randn(2, 5, 256)
        enc_out = torch.randn(2, 10, 256)
        self_mask = torch.tril(torch.ones(5, 5))
        output = decoder(x, enc_out, self_mask=self_mask)
        self.assertEqual(output.shape, (2, 5, 256))

    def test_different_seq_lengths(self):
        decoder = DecoderBlock(d_model=128, num_heads=2)
        x = torch.randn(1, 8, 128)  # target
        enc_out = torch.randn(1, 15, 128)  # source
        output = decoder(x, enc_out)
        self.assertEqual(output.shape, (1, 8, 128))

    def test_is_nn_module(self):
        decoder = DecoderBlock(d_model=256, num_heads=4)
        self.assertIsInstance(decoder, nn.Module)

    def test_has_layer_norms(self):
        decoder = DecoderBlock(d_model=256, num_heads=4)
        self.assertTrue(hasattr(decoder, 'norm1'))
        self.assertTrue(hasattr(decoder, 'norm2'))
        self.assertTrue(hasattr(decoder, 'norm3'))

    def test_has_self_attention(self):
        decoder = DecoderBlock(d_model=256, num_heads=4)
        self.assertTrue(hasattr(decoder, 'self_W_q'))
        self.assertTrue(hasattr(decoder, 'self_W_k'))

    def test_has_cross_attention(self):
        decoder = DecoderBlock(d_model=256, num_heads=4)
        self.assertTrue(hasattr(decoder, 'cross_W_q'))
        self.assertTrue(hasattr(decoder, 'cross_W_k'))

    def test_output_not_nan(self):
        decoder = DecoderBlock(d_model=128, num_heads=2)
        x = torch.randn(2, 5, 128)
        enc_out = torch.randn(2, 10, 128)
        output = decoder(x, enc_out)
        self.assertFalse(torch.isnan(output).any())

    def test_single_sample(self):
        decoder = DecoderBlock(d_model=256, num_heads=4)
        x = torch.randn(1, 6, 256)
        enc_out = torch.randn(1, 12, 256)
        output = decoder(x, enc_out)
        self.assertEqual(output.shape, (1, 6, 256))

    def test_has_dropout(self):
        decoder = DecoderBlock(d_model=256, num_heads=4, dropout=0.2)
        self.assertTrue(hasattr(decoder, 'dropout'))
`,

	hint1: 'Self-attention uses Q,K,V from decoder; cross-attention uses Q from decoder, K,V from encoder',
	hint2: 'Apply causal mask to self-attention to prevent looking at future tokens',

	whyItMatters: `Decoder blocks power autoregressive models like GPT:

- **Masked self-attention**: Only attend to previous positions
- **Cross-attention**: Attend to encoder output (for translation)
- **GPT-style**: Only self-attention, no cross-attention
- **Translation**: Uses both self and cross-attention

Understanding decoders is key for text generation.`,

	translations: {
		ru: {
			title: 'Блок декодера трансформера',
			description: `# Блок декодера трансформера

Реализуйте блок декодера трансформера с self-attention и cross-attention.

## Задача

Реализуйте класс \`DecoderBlock\` с:
- Masked self-attention (каузальный)
- Cross-attention к выходу энкодера
- Feed-forward сеть
- Layer normalization и residual connections

## Пример

\`\`\`python
decoder = DecoderBlock(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)  # decoder input
encoder_out = torch.randn(2, 20, 512)  # encoder output
output = decoder(x, encoder_out)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: 'Self-attention использует Q,K,V из декодера; cross-attention использует Q из декодера, K,V из энкодера',
			hint2: 'Применяйте каузальную маску к self-attention для предотвращения просмотра будущих токенов',
			whyItMatters: `Блоки декодера питают авторегрессивные модели типа GPT:

- **Masked self-attention**: Внимание только к предыдущим позициям
- **Cross-attention**: Внимание к выходу энкодера (для перевода)
- **GPT-style**: Только self-attention, без cross-attention
- **Перевод**: Использует оба типа attention`,
		},
		uz: {
			title: 'Transformer decoder bloki',
			description: `# Transformer decoder bloki

Self-attention va cross-attention bilan transformer decoder blokini amalga oshiring.

## Topshiriq

\`DecoderBlock\` sinfini amalga oshiring:
- Masked self-attention (kauzal)
- Encoder chiqishiga cross-attention
- Feed-forward tarmoq
- Layer normalization va residual connections

## Misol

\`\`\`python
decoder = DecoderBlock(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)  # decoder input
encoder_out = torch.randn(2, 20, 512)  # encoder output
output = decoder(x, encoder_out)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: "Self-attention decoder dan Q,K,V oladi; cross-attention decoder dan Q, encoder dan K,V oladi",
			hint2: "Kelajak tokenlarni ko'rishni oldini olish uchun self-attention ga kauzal niqob qo'llang",
			whyItMatters: `Decoder bloklari GPT kabi avtoregessiv modellarni quvvatlaydi:

- **Masked self-attention**: Faqat oldingi pozitsiyalarga e'tibor
- **Cross-attention**: Encoder chiqishiga e'tibor (tarjima uchun)
- **GPT uslubi**: Faqat self-attention, cross-attention yo'q
- **Tarjima**: Ikkala attention turidan foydalanadi`,
		},
	},
};

export default task;
