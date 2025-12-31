import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-transformer-encoder',
	title: 'Transformer Encoder',
	difficulty: 'hard',
	tags: ['pytorch', 'transformer', 'encoder'],
	estimatedTime: '25m',
	isPremium: false,
	order: 1,
	description: `# Transformer Encoder

Build a complete transformer encoder by stacking multiple encoder blocks.

## Task

Implement a \`TransformerEncoder\` class that:
- Stacks N encoder blocks
- Includes token and positional embeddings
- Applies final layer normalization

## Example

\`\`\`python
encoder = TransformerEncoder(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

tokens = torch.randint(0, 30000, (2, 50))
output = encoder(tokens)
# output.shape = (2, 50, 512)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    """Complete transformer encoder."""

    def __init__(self, vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # TODO: Embed tokens, add positions, pass through layers, return encoded
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch = x.size(0)
        normed = self.norm1(x)

        Q = self.W_q(normed).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(normed).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(normed).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        x = x + self.dropout(self.W_o(context))
        x = x + self.dropout(self.ff2(F.gelu(self.ff1(self.norm2(x)))))
        return x

class TransformerEncoder(nn.Module):
    """Complete transformer encoder."""

    def __init__(self, vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Embed tokens and positions
        x = self.token_emb(x) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        # Pass through encoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
`,

	testCode: `import torch
import unittest

class TestTransformerEncoder(unittest.TestCase):
    def test_output_shape(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=256, num_heads=4, num_layers=2)
        tokens = torch.randint(0, 1000, (2, 20))
        output = encoder(tokens)
        self.assertEqual(output.shape, (2, 20, 256))

    def test_different_seq_lengths(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        tokens1 = torch.randint(0, 1000, (1, 10))
        tokens2 = torch.randint(0, 1000, (1, 50))
        out1 = encoder(tokens1)
        out2 = encoder(tokens2)
        self.assertEqual(out1.shape, (1, 10, 128))
        self.assertEqual(out2.shape, (1, 50, 128))

    def test_with_mask(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        tokens = torch.randint(0, 1000, (2, 15))
        mask = torch.ones(15, 15)
        output = encoder(tokens, mask)
        self.assertEqual(output.shape, (2, 15, 128))

    def test_is_nn_module(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        self.assertIsInstance(encoder, torch.nn.Module)

    def test_has_token_embedding(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        self.assertTrue(hasattr(encoder, 'token_emb'))

    def test_has_pos_embedding(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        self.assertTrue(hasattr(encoder, 'pos_emb'))

    def test_has_layers(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=3)
        self.assertTrue(hasattr(encoder, 'layers'))
        self.assertEqual(len(encoder.layers), 3)

    def test_output_not_nan(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        tokens = torch.randint(0, 1000, (2, 10))
        output = encoder(tokens)
        self.assertFalse(torch.isnan(output).any())

    def test_single_sample(self):
        encoder = TransformerEncoder(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
        tokens = torch.randint(0, 1000, (1, 8))
        output = encoder(tokens)
        self.assertEqual(output.shape, (1, 8, 128))
`,

	hint1: 'Scale token embeddings by sqrt(d_model) before adding positions',
	hint2: 'Use nn.ModuleList to stack encoder blocks properly',

	whyItMatters: `The transformer encoder is the backbone of BERT:

- **Stacked layers**: Deep representations emerge from multiple blocks
- **Bidirectional**: Attends to entire sequence simultaneously
- **BERT-base**: 12 layers, 768 hidden, 12 heads
- **Applications**: Classification, NER, question answering

Understanding encoder architecture is essential for NLU tasks.`,

	translations: {
		ru: {
			title: 'Трансформер Encoder',
			description: `# Трансформер Encoder

Создайте полный encoder трансформера, объединяя несколько блоков encoder.

## Задача

Реализуйте класс \`TransformerEncoder\`:
- Объединяет N блоков encoder
- Включает token и positional embeddings
- Применяет финальную layer normalization

## Пример

\`\`\`python
encoder = TransformerEncoder(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

tokens = torch.randint(0, 30000, (2, 50))
output = encoder(tokens)
# output.shape = (2, 50, 512)
\`\`\``,
			hint1: 'Масштабируйте token embeddings на sqrt(d_model) перед добавлением позиций',
			hint2: 'Используйте nn.ModuleList для правильного объединения блоков encoder',
			whyItMatters: `Encoder трансформера - основа BERT:

- **Стек слоев**: Глубокие представления из множества блоков
- **Двунаправленный**: Внимание ко всей последовательности одновременно
- **BERT-base**: 12 слоев, 768 hidden, 12 голов
- **Применения**: Классификация, NER, ответы на вопросы`,
		},
		uz: {
			title: 'Transformer Encoder',
			description: `# Transformer Encoder

Bir nechta encoder bloklarini birlashtirish orqali to'liq transformer encoder yarating.

## Topshiriq

\`TransformerEncoder\` sinfini amalga oshiring:
- N ta encoder blokini birlashtiradi
- Token va positional embeddings ni o'z ichiga oladi
- Oxirgi layer normalization ni qo'llaydi

## Misol

\`\`\`python
encoder = TransformerEncoder(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

tokens = torch.randint(0, 30000, (2, 50))
output = encoder(tokens)
# output.shape = (2, 50, 512)
\`\`\``,
			hint1: "Pozitsiyalarni qo'shishdan oldin token embeddings ni sqrt(d_model) ga ko'paytiring",
			hint2: "Encoder bloklarini to'g'ri birlashtirish uchun nn.ModuleList dan foydalaning",
			whyItMatters: `Transformer encoder BERT ning asosi:

- **Qatlamlangan**: Chuqur tasvirlar bir nechta bloklardan paydo bo'ladi
- **Ikki yo'nalishli**: Butun ketma-ketlikka bir vaqtda e'tibor beradi
- **BERT-base**: 12 qatlam, 768 hidden, 12 bosh
- **Qo'llanilishi**: Klassifikatsiya, NER, savolga javob`,
		},
	},
};

export default task;
