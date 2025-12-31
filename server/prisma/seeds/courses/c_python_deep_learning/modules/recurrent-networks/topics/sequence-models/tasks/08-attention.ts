import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-attention-mechanism',
	title: 'Attention Mechanism',
	difficulty: 'hard',
	tags: ['pytorch', 'rnn', 'attention'],
	estimatedTime: '20m',
	isPremium: true,
	order: 8,
	description: `# Attention Mechanism

Implement the attention mechanism for sequence models.

## Task

Implement an \`Attention\` class that:
- Computes attention weights between query and keys
- Returns weighted sum of values (context vector)

## Example

\`\`\`python
attention = Attention(hidden_size=256)

# Query: decoder hidden state (batch, hidden)
# Keys/Values: encoder outputs (batch, seq_len, hidden)
query = torch.randn(4, 256)
encoder_outputs = torch.randn(4, 20, 256)

context, weights = attention(query, encoder_outputs)
# context.shape = (4, 256)
# weights.shape = (4, 20)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # Your code here
        pass

    def forward(self, query: torch.Tensor,
                keys: torch.Tensor) -> tuple:
        # TODO: Compute attention between query and keys, return (context, weights)
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor,
                keys: torch.Tensor) -> tuple:
        """
        Args:
            query: (batch, hidden) - decoder state
            keys: (batch, seq_len, hidden) - encoder outputs
        Returns:
            context: (batch, hidden) - weighted sum
            weights: (batch, seq_len) - attention weights
        """
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden)
        keys_proj = self.key_proj(keys)  # (batch, seq_len, hidden)

        # Compute attention scores
        scores = self.v(torch.tanh(query_proj + keys_proj))  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Compute weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Compute context
        context = torch.bmm(weights.unsqueeze(1), keys)  # (batch, 1, hidden)
        context = context.squeeze(1)  # (batch, hidden)

        return context, weights
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestAttention(unittest.TestCase):
    def test_output_shapes(self):
        attention = Attention(256)
        query = torch.randn(4, 256)
        keys = torch.randn(4, 20, 256)
        context, weights = attention(query, keys)
        self.assertEqual(context.shape, (4, 256))
        self.assertEqual(weights.shape, (4, 20))

    def test_weights_sum_to_one(self):
        attention = Attention(128)
        query = torch.randn(2, 128)
        keys = torch.randn(2, 10, 128)
        _, weights = attention(query, keys)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(2), atol=1e-5))

    def test_different_seq_len(self):
        attention = Attention(64)
        query = torch.randn(3, 64)
        keys = torch.randn(3, 50, 64)
        context, weights = attention(query, keys)
        self.assertEqual(weights.shape, (3, 50))

    def test_is_nn_module(self):
        attention = Attention(256)
        self.assertIsInstance(attention, nn.Module)

    def test_context_shape(self):
        attention = Attention(128)
        query = torch.randn(4, 128)
        keys = torch.randn(4, 30, 128)
        context, _ = attention(query, keys)
        self.assertEqual(context.shape, (4, 128))

    def test_single_sample(self):
        attention = Attention(64)
        query = torch.randn(1, 64)
        keys = torch.randn(1, 20, 64)
        context, weights = attention(query, keys)
        self.assertEqual(context.shape, (1, 64))
        self.assertEqual(weights.shape, (1, 20))

    def test_weights_positive(self):
        attention = Attention(128)
        query = torch.randn(2, 128)
        keys = torch.randn(2, 10, 128)
        _, weights = attention(query, keys)
        self.assertTrue((weights >= 0).all())

    def test_context_not_nan(self):
        attention = Attention(128)
        query = torch.randn(2, 128)
        keys = torch.randn(2, 10, 128)
        context, _ = attention(query, keys)
        self.assertFalse(torch.isnan(context).any())

    def test_has_projections(self):
        attention = Attention(128)
        self.assertTrue(hasattr(attention, 'query_proj'))
        self.assertTrue(hasattr(attention, 'key_proj'))

    def test_large_batch(self):
        attention = Attention(256)
        query = torch.randn(32, 256)
        keys = torch.randn(32, 50, 256)
        context, weights = attention(query, keys)
        self.assertEqual(context.shape, (32, 256))
`,

	hint1: 'Attention scores = v(tanh(W_q * query + W_k * keys))',
	hint2: 'Context = weighted sum of values (keys) by attention weights',

	whyItMatters: `Attention revolutionized sequence modeling:

- **Dynamic focus**: Attend to relevant parts of input
- **Long-range dependencies**: No information bottleneck
- **Interpretability**: Weights show what model focuses on
- **Foundation for Transformers**: Self-attention is attention everywhere

Attention is the most important concept in modern deep learning.`,

	translations: {
		ru: {
			title: 'Механизм внимания',
			description: `# Механизм внимания

Реализуйте механизм внимания для моделей последовательностей.

## Задача

Реализуйте класс \`Attention\`, который:
- Вычисляет веса внимания между query и keys
- Возвращает взвешенную сумму values (вектор контекста)

## Пример

\`\`\`python
attention = Attention(hidden_size=256)

# Query: decoder hidden state (batch, hidden)
# Keys/Values: encoder outputs (batch, seq_len, hidden)
query = torch.randn(4, 256)
encoder_outputs = torch.randn(4, 20, 256)

context, weights = attention(query, encoder_outputs)
# context.shape = (4, 256)
# weights.shape = (4, 20)
\`\`\``,
			hint1: 'Attention scores = v(tanh(W_q * query + W_k * keys))',
			hint2: 'Context = взвешенная сумма values (keys) по весам внимания',
			whyItMatters: `Attention произвел революцию в моделировании последовательностей:

- **Динамический фокус**: Внимание к релевантным частям входа
- **Долгосрочные зависимости**: Нет информационного узкого места
- **Интерпретируемость**: Веса показывают на что смотрит модель
- **Основа Transformers**: Self-attention - это внимание везде`,
		},
		uz: {
			title: "Diqqat mexanizmi",
			description: `# Diqqat mexanizmi

Ketma-ketlik modellari uchun diqqat mexanizmini amalga oshiring.

## Topshiriq

\`Attention\` sinfini amalga oshiring:
- Query va keys o'rtasida diqqat vaznlarini hisoblaydi
- Values ning og'irlikli yig'indisini qaytaradi (kontekst vektori)

## Misol

\`\`\`python
attention = Attention(hidden_size=256)

# Query: decoder hidden state (batch, hidden)
# Keys/Values: encoder outputs (batch, seq_len, hidden)
query = torch.randn(4, 256)
encoder_outputs = torch.randn(4, 20, 256)

context, weights = attention(query, encoder_outputs)
# context.shape = (4, 256)
# weights.shape = (4, 20)
\`\`\``,
			hint1: 'Diqqat ballari = v(tanh(W_q * query + W_k * keys))',
			hint2: "Kontekst = diqqat vaznlari bo'yicha values (keys) ning og'irlikli yig'indisi",
			whyItMatters: `Diqqat ketma-ketlik modellashtirishda inqilob qildi:

- **Dinamik fokus**: Kirish ning tegishli qismlariga e'tibor
- **Uzoq muddatli bog'liqliklar**: Ma'lumot tiqinchisi yo'q
- **Interpretatsiya**: Vaznlar model nimaga qarayotganini ko'rsatadi
- **Transformerlar asosi**: Self-attention - bu hamma joyda diqqat`,
		},
	},
};

export default task;
