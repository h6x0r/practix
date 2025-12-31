import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-encoder-decoder-transformer',
	title: 'Encoder-Decoder Transformer',
	difficulty: 'hard',
	tags: ['pytorch', 'transformer', 'seq2seq', 't5'],
	estimatedTime: '35m',
	isPremium: true,
	order: 3,
	description: `# Encoder-Decoder Transformer

Build a complete encoder-decoder transformer for sequence-to-sequence tasks.

## Task

Implement a \`Seq2SeqTransformer\` class that:
- Combines encoder and decoder
- Uses cross-attention for encoder-decoder connection
- Supports teacher forcing during training

## Applications

- Machine translation
- Summarization
- Question answering

## Example

\`\`\`python
model = Seq2SeqTransformer(
    src_vocab=32000,
    tgt_vocab=32000,
    d_model=512,
    num_layers=6
)

src = torch.randint(0, 32000, (2, 50))
tgt = torch.randint(0, 32000, (2, 30))
logits = model(src, tgt)
# logits.shape = (2, 30, 32000)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for seq2seq tasks."""

    def __init__(self, src_vocab: int, tgt_vocab: int,
                 d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # Your code here
        pass

    def encode(self, src: torch.Tensor,
               src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode source sequence."""
        # Your code here
        pass

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """Decode with encoder memory."""
        # Your code here
        pass

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for seq2seq tasks."""

    def __init__(self, src_vocab: int, tgt_vocab: int,
                 d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Use PyTorch's built-in transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def _get_positions(self, seq_len: int, device) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0)

    def _create_causal_mask(self, size: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()

    def encode(self, src: torch.Tensor,
               src_mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = src.size(1)
        positions = self._get_positions(seq_len, src.device)

        src_emb = self.src_emb(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_emb(positions)
        src_emb = self.dropout(src_emb)

        # PyTorch transformer encoder expects (seq, batch, d_model) if not batch_first
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = tgt.size(1)
        positions = self._get_positions(seq_len, tgt.device)

        tgt_emb = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_emb(positions)
        tgt_emb = self.dropout(tgt_emb)

        causal_mask = self._create_causal_mask(seq_len, tgt.device)

        output = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask
        )
        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        logits = self.output_proj(output)
        return logits
`,

	testCode: `import torch
import unittest

class TestSeq2SeqTransformer(unittest.TestCase):
    def test_output_shape(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=128, num_heads=4, num_layers=2)
        src = torch.randint(0, 1000, (2, 20))
        tgt = torch.randint(0, 1000, (2, 15))
        logits = model(src, tgt)
        self.assertEqual(logits.shape, (2, 15, 1000))

    def test_encode(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        src = torch.randint(0, 1000, (1, 10))
        memory = model.encode(src)
        self.assertEqual(memory.shape, (1, 10, 64))

    def test_decode(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        memory = torch.randn(1, 10, 64)
        tgt = torch.randint(0, 1000, (1, 5))
        output = model.decode(tgt, memory)
        self.assertEqual(output.shape, (1, 5, 64))

    def test_is_nn_module(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        self.assertIsInstance(model, torch.nn.Module)

    def test_has_embeddings(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        self.assertTrue(hasattr(model, 'src_emb'))
        self.assertTrue(hasattr(model, 'tgt_emb'))

    def test_has_output_proj(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        self.assertTrue(hasattr(model, 'output_proj'))

    def test_output_not_nan(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        src = torch.randint(0, 1000, (2, 10))
        tgt = torch.randint(0, 1000, (2, 8))
        logits = model(src, tgt)
        self.assertFalse(torch.isnan(logits).any())

    def test_single_sample(self):
        model = Seq2SeqTransformer(src_vocab=1000, tgt_vocab=1000, d_model=64, num_heads=2, num_layers=1)
        src = torch.randint(0, 1000, (1, 12))
        tgt = torch.randint(0, 1000, (1, 6))
        logits = model(src, tgt)
        self.assertEqual(logits.shape, (1, 6, 1000))

    def test_different_vocab_sizes(self):
        model = Seq2SeqTransformer(src_vocab=500, tgt_vocab=800, d_model=64, num_heads=2, num_layers=1)
        src = torch.randint(0, 500, (1, 10))
        tgt = torch.randint(0, 800, (1, 5))
        logits = model(src, tgt)
        self.assertEqual(logits.shape, (1, 5, 800))
`,

	hint1: 'Use nn.Transformer for the core - it handles all attention internally',
	hint2: 'Create causal mask with torch.triu for decoder self-attention',

	whyItMatters: `Encoder-decoder architecture enables translation and summarization:

- **T5**: Treats all NLP as text-to-text
- **BART**: Denoising autoencoder for generation
- **mT5**: Multilingual translation
- **Cross-attention**: Decoder attends to encoder output

This architecture excels at conditional generation.`,

	translations: {
		ru: {
			title: 'Encoder-Decoder трансформер',
			description: `# Encoder-Decoder трансформер

Создайте полный encoder-decoder трансформер для seq2seq задач.

## Задача

Реализуйте класс \`Seq2SeqTransformer\`:
- Комбинирует encoder и decoder
- Использует cross-attention для связи encoder-decoder
- Поддерживает teacher forcing при обучении

## Пример

\`\`\`python
model = Seq2SeqTransformer(
    src_vocab=32000,
    tgt_vocab=32000,
    d_model=512,
    num_layers=6
)

src = torch.randint(0, 32000, (2, 50))
tgt = torch.randint(0, 32000, (2, 30))
logits = model(src, tgt)
# logits.shape = (2, 30, 32000)
\`\`\``,
			hint1: 'Используйте nn.Transformer для ядра - он обрабатывает все attention внутри',
			hint2: 'Создайте каузальную маску с torch.triu для self-attention декодера',
			whyItMatters: `Архитектура encoder-decoder обеспечивает перевод и суммаризацию:

- **T5**: Рассматривает все NLP как text-to-text
- **BART**: Denoising autoencoder для генерации
- **mT5**: Многоязычный перевод
- **Cross-attention**: Decoder обращает внимание на выход encoder`,
		},
		uz: {
			title: 'Encoder-Decoder transformer',
			description: `# Encoder-Decoder transformer

Seq2seq topshiriqlari uchun to'liq encoder-decoder transformer yarating.

## Topshiriq

\`Seq2SeqTransformer\` sinfini amalga oshiring:
- Encoder va decoder ni birlashtiradi
- Encoder-decoder aloqasi uchun cross-attention dan foydalanadi
- O'qitish vaqtida teacher forcing ni qo'llab-quvvatlaydi

## Misol

\`\`\`python
model = Seq2SeqTransformer(
    src_vocab=32000,
    tgt_vocab=32000,
    d_model=512,
    num_layers=6
)

src = torch.randint(0, 32000, (2, 50))
tgt = torch.randint(0, 32000, (2, 30))
logits = model(src, tgt)
# logits.shape = (2, 30, 32000)
\`\`\``,
			hint1: "Yadro uchun nn.Transformer dan foydalaning - u barcha attention ni ichkarida boshqaradi",
			hint2: "Decoder self-attention uchun torch.triu bilan kauzal niqob yarating",
			whyItMatters: `Encoder-decoder arxitekturasi tarjima va xulosa chiqarishni ta'minlaydi:

- **T5**: Barcha NLP ni text-to-text sifatida ko'radi
- **BART**: Generatsiya uchun denoising autoencoder
- **mT5**: Ko'p tilli tarjima
- **Cross-attention**: Decoder encoder chiqishiga e'tibor beradi`,
		},
	},
};

export default task;
