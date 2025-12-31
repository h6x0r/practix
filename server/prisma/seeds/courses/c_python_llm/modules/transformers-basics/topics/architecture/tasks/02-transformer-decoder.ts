import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-transformer-decoder',
	title: 'Transformer Decoder',
	difficulty: 'hard',
	tags: ['pytorch', 'transformer', 'decoder', 'gpt'],
	estimatedTime: '30m',
	isPremium: true,
	order: 2,
	description: `# Transformer Decoder

Build a GPT-style decoder-only transformer for autoregressive generation.

## Task

Implement a \`TransformerDecoder\` class that:
- Uses causal (masked) self-attention
- Stacks N decoder blocks
- Includes language modeling head

## Example

\`\`\`python
decoder = TransformerDecoder(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    num_layers=12
)

tokens = torch.randint(0, 50000, (2, 100))
logits = decoder(tokens)
# logits.shape = (2, 100, 50000)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerDecoder(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(self, vocab_size: int, d_model: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 3072, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Embed, apply causal mask, pass through layers, return logits
        pass

    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive generation."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
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

    def forward(self, x, mask):
        batch, seq_len = x.size(0), x.size(1)
        normed = self.norm1(x)

        Q = self.W_q(normed).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(normed).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(normed).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        x = x + self.dropout(self.W_o(context))
        x = x + self.dropout(self.ff2(F.gelu(self.ff1(self.norm2(x)))))
        return x

class TransformerDecoder(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(self, vocab_size: int, d_model: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 3072, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Tie weights
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Embeddings
        x = self.token_emb(x) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        # Decoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.lm_head(x)

    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        tokens = prompt.clone()

        for _ in range(max_new_tokens):
            # Truncate if too long
            context = tokens[:, -self.max_len:]

            # Get predictions
            logits = self(context)
            logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
`,

	testCode: `import torch
import unittest

class TestTransformerDecoder(unittest.TestCase):
    def test_output_shape(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=128, num_heads=4, num_layers=2, d_ff=512)
        tokens = torch.randint(0, 1000, (2, 20))
        logits = decoder(tokens)
        self.assertEqual(logits.shape, (2, 20, 1000))

    def test_causal_mask_applied(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=1)
        tokens = torch.randint(0, 1000, (1, 10))
        logits = decoder(tokens)
        self.assertEqual(logits.shape, (1, 10, 1000))

    def test_generate(self):
        decoder = TransformerDecoder(vocab_size=100, d_model=64, num_heads=2, num_layers=1, max_len=50)
        prompt = torch.randint(0, 100, (1, 5))
        generated = decoder.generate(prompt, max_new_tokens=10)
        self.assertEqual(generated.shape[1], 15)  # 5 prompt + 10 new

    def test_is_nn_module(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=1)
        self.assertIsInstance(decoder, torch.nn.Module)

    def test_has_token_embedding(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=1)
        self.assertTrue(hasattr(decoder, 'token_emb'))

    def test_has_lm_head(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=1)
        self.assertTrue(hasattr(decoder, 'lm_head'))

    def test_has_layers(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=4)
        self.assertTrue(hasattr(decoder, 'layers'))
        self.assertEqual(len(decoder.layers), 4)

    def test_output_not_nan(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=1)
        tokens = torch.randint(0, 1000, (2, 10))
        logits = decoder(tokens)
        self.assertFalse(torch.isnan(logits).any())

    def test_single_sample(self):
        decoder = TransformerDecoder(vocab_size=1000, d_model=64, num_heads=2, num_layers=1)
        tokens = torch.randint(0, 1000, (1, 8))
        logits = decoder(tokens)
        self.assertEqual(logits.shape, (1, 8, 1000))
`,

	hint1: 'Use torch.tril to create causal mask for each forward pass',
	hint2: 'Tie lm_head weights to token embeddings for better generalization',

	whyItMatters: `Decoder-only transformers power modern LLMs:

- **GPT architecture**: Foundation of ChatGPT, GPT-4
- **Causal attention**: Each position only sees previous tokens
- **Language modeling**: Predict next token autoregressively
- **Weight tying**: Share embeddings with output layer

This is the architecture behind most conversational AI.`,

	translations: {
		ru: {
			title: 'Трансформер Decoder',
			description: `# Трансформер Decoder

Создайте GPT-style decoder-only трансформер для авторегрессивной генерации.

## Задача

Реализуйте класс \`TransformerDecoder\`:
- Использует каузальный (masked) self-attention
- Объединяет N блоков decoder
- Включает language modeling head

## Пример

\`\`\`python
decoder = TransformerDecoder(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    num_layers=12
)

tokens = torch.randint(0, 50000, (2, 100))
logits = decoder(tokens)
# logits.shape = (2, 100, 50000)
\`\`\``,
			hint1: 'Используйте torch.tril для создания каузальной маски при каждом forward pass',
			hint2: 'Свяжите веса lm_head с token embeddings для лучшего обобщения',
			whyItMatters: `Decoder-only трансформеры питают современные LLM:

- **Архитектура GPT**: Основа ChatGPT, GPT-4
- **Каузальное внимание**: Каждая позиция видит только предыдущие токены
- **Language modeling**: Авторегрессивное предсказание следующего токена
- **Weight tying**: Общие embeddings с выходным слоем`,
		},
		uz: {
			title: 'Transformer Decoder',
			description: `# Transformer Decoder

Avtoregessiv generatsiya uchun GPT uslubidagi decoder-only transformer yarating.

## Topshiriq

\`TransformerDecoder\` sinfini amalga oshiring:
- Kauzal (masked) self-attention dan foydalanadi
- N ta decoder blokini birlashtiradi
- Language modeling head ni o'z ichiga oladi

## Misol

\`\`\`python
decoder = TransformerDecoder(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    num_layers=12
)

tokens = torch.randint(0, 50000, (2, 100))
logits = decoder(tokens)
# logits.shape = (2, 100, 50000)
\`\`\``,
			hint1: 'Har bir forward pass uchun kauzal niqob yaratish uchun torch.tril dan foydalaning',
			hint2: "Yaxshiroq umumlashtirish uchun lm_head og'irliklarini token embeddings bilan bog'lang",
			whyItMatters: `Decoder-only transformerlar zamonaviy LLM larni quvvatlaydi:

- **GPT arxitekturasi**: ChatGPT, GPT-4 ning asosi
- **Kauzal diqqat**: Har bir pozitsiya faqat oldingi tokenlarni ko'radi
- **Language modeling**: Keyingi tokenni avtoregessiv bashorat qilish
- **Weight tying**: Chiqish qatlami bilan umumiy embeddings`,
		},
	},
};

export default task;
