import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-multi-head-attention',
	title: 'Multi-Head Attention',
	difficulty: 'hard',
	tags: ['pytorch', 'attention', 'transformer'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Multi-Head Attention

Implement multi-head attention that allows the model to attend to different representation subspaces.

## Task

Implement a \`MultiHeadAttention\` class with:
- Separate projections for Q, K, V
- Split into multiple heads
- Concatenate and project output

## Example

\`\`\`python
mha = MultiHeadAttention(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)  # batch, seq_len, d_model
output, weights = mha(x, x, x)  # Self-attention
# output.shape = (2, 10, 512)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Your code here - define projections
        pass

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        # TODO: Project Q/K/V, split into heads, apply attention, concat, return (output, weights)
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final projection
        output = self.W_o(context)

        return output, attention_weights
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        output, weights = mha(x, x, x)
        self.assertEqual(output.shape, (2, 10, 512))
        self.assertEqual(weights.shape, (2, 8, 10, 10))

    def test_different_heads(self):
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        x = torch.randn(4, 8, 256)
        output, weights = mha(x, x, x)
        self.assertEqual(output.shape, (4, 8, 256))

    def test_cross_attention(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        query = torch.randn(2, 5, 512)
        key = torch.randn(2, 10, 512)
        value = torch.randn(2, 10, 512)
        output, weights = mha(query, key, value)
        self.assertEqual(output.shape, (2, 5, 512))
        self.assertEqual(weights.shape, (2, 8, 5, 10))

    def test_is_nn_module(self):
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        self.assertIsInstance(mha, nn.Module)

    def test_has_projections(self):
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        self.assertTrue(hasattr(mha, 'W_q'))
        self.assertTrue(hasattr(mha, 'W_k'))
        self.assertTrue(hasattr(mha, 'W_v'))
        self.assertTrue(hasattr(mha, 'W_o'))

    def test_output_not_nan(self):
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        x = torch.randn(2, 8, 256)
        output, _ = mha(x, x, x)
        self.assertFalse(torch.isnan(output).any())

    def test_single_sample(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        x = torch.randn(1, 6, 512)
        output, weights = mha(x, x, x)
        self.assertEqual(output.shape, (1, 6, 512))

    def test_has_dropout(self):
        mha = MultiHeadAttention(d_model=256, num_heads=4, dropout=0.5)
        self.assertTrue(hasattr(mha, 'dropout'))

    def test_d_k_calculation(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        self.assertEqual(mha.d_k, 64)

    def test_weights_shape_with_heads(self):
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        x = torch.randn(2, 10, 256)
        _, weights = mha(x, x, x)
        self.assertEqual(weights.shape, (2, 4, 10, 10))
`,

	hint1: 'Reshape with view and transpose to split into heads',
	hint2: 'Use contiguous() before final view to avoid memory errors',

	whyItMatters: `Multi-head attention is key to transformer power:

- **Multiple perspectives**: Each head learns different relationships
- **Rich representations**: Capture various types of dependencies
- **Efficient**: Parallel heads with reduced dimension
- **Standard component**: Used in every transformer model

Understanding MHA is essential for working with modern NLP.`,

	translations: {
		ru: {
			title: 'Multi-Head Attention',
			description: `# Multi-Head Attention

Реализуйте multi-head attention, позволяющий модели обращать внимание на разные подпространства представлений.

## Задача

Реализуйте класс \`MultiHeadAttention\` с:
- Отдельными проекциями для Q, K, V
- Разделением на несколько голов
- Конкатенацией и проекцией выхода

## Пример

\`\`\`python
mha = MultiHeadAttention(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)  # batch, seq_len, d_model
output, weights = mha(x, x, x)  # Self-attention
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: 'Используйте view и transpose для разделения на головы',
			hint2: 'Используйте contiguous() перед финальным view для избежания ошибок памяти',
			whyItMatters: `Multi-head attention ключ к силе трансформеров:

- **Множество перспектив**: Каждая голова учит разные связи
- **Богатые представления**: Захват различных типов зависимостей
- **Эффективность**: Параллельные головы с уменьшенной размерностью
- **Стандартный компонент**: Используется в каждой модели трансформера`,
		},
		uz: {
			title: 'Multi-Head Attention',
			description: `# Multi-Head Attention

Modelga turli tasvirlash kichik fazolariga e'tibor berish imkonini beradigan multi-head attention ni amalga oshiring.

## Topshiriq

\`MultiHeadAttention\` sinfini amalga oshiring:
- Q, K, V uchun alohida proyeksiyalar
- Bir nechta boshlarga bo'linish
- Birlashtirish va chiqishni proyeksiya qilish

## Misol

\`\`\`python
mha = MultiHeadAttention(d_model=512, num_heads=8)

x = torch.randn(2, 10, 512)  # batch, seq_len, d_model
output, weights = mha(x, x, x)  # Self-attention
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: "Boshlarga bo'lish uchun view va transpose dan foydalaning",
			hint2: "Xotira xatolaridan qochish uchun oxirgi view dan oldin contiguous() dan foydalaning",
			whyItMatters: `Multi-head attention transformer kuchining kaliti:

- **Ko'p nuqtai nazarlar**: Har bir bosh turli munosabatlarni o'rganadi
- **Boy tasvirlar**: Turli xil bog'liqliklarni qamrab oladi
- **Samarali**: Kamaytirilgan o'lchamli parallel boshlar
- **Standart komponent**: Har bir transformer modelida ishlatiladi`,
		},
	},
};

export default task;
