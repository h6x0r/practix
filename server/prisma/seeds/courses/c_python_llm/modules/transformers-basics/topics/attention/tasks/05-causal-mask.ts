import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-causal-attention-mask',
	title: 'Causal Attention Mask',
	difficulty: 'easy',
	tags: ['pytorch', 'transformer', 'mask'],
	estimatedTime: '10m',
	isPremium: false,
	order: 5,
	description: `# Causal Attention Mask

Create masks for causal (autoregressive) attention used in decoder models like GPT.

## Task

Implement two functions:
1. \`create_causal_mask\` - Create a triangular mask for autoregressive attention
2. \`create_padding_mask\` - Create a mask for padding tokens

## Example

\`\`\`python
# Causal mask: prevents attending to future positions
mask = create_causal_mask(seq_len=4)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# Padding mask: masks out padding tokens
padding_mask = create_padding_mask(lengths=[3, 2], max_len=4)
\`\`\``,

	initialCode: `import torch

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask."""
    # Your code here
    pass

def create_padding_mask(lengths: list, max_len: int) -> torch.Tensor:
    """Create a padding mask from sequence lengths."""
    # Your code here
    pass

def combine_masks(causal_mask: torch.Tensor,
                  padding_mask: torch.Tensor) -> torch.Tensor:
    """Combine causal and padding masks."""
    # Your code here
    pass
`,

	solutionCode: `import torch

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

def create_padding_mask(lengths: list, max_len: int) -> torch.Tensor:
    """Create a padding mask from sequence lengths."""
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask

def combine_masks(causal_mask: torch.Tensor,
                  padding_mask: torch.Tensor) -> torch.Tensor:
    """Combine causal and padding masks."""
    # Expand causal mask for batch: (seq, seq) -> (batch, 1, seq, seq)
    batch_size = padding_mask.size(0)
    seq_len = causal_mask.size(0)

    causal = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
    causal = causal.expand(batch_size, 1, seq_len, seq_len)

    # Expand padding mask: (batch, seq) -> (batch, 1, 1, seq)
    padding = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)

    # Combine: both must be 1 to allow attention
    combined = causal * padding

    return combined
`,

	testCode: `import torch
import unittest

class TestMasks(unittest.TestCase):
    def test_causal_mask_shape(self):
        mask = create_causal_mask(4)
        self.assertEqual(mask.shape, (4, 4))

    def test_causal_mask_values(self):
        mask = create_causal_mask(3)
        expected = torch.tensor([
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_padding_mask(self):
        mask = create_padding_mask([3, 2], max_len=4)
        self.assertEqual(mask.shape, (2, 4))
        self.assertEqual(mask[0].sum().item(), 3)
        self.assertEqual(mask[1].sum().item(), 2)

    def test_combine_masks(self):
        causal = create_causal_mask(4)
        padding = create_padding_mask([4, 3], max_len=4)
        combined = combine_masks(causal, padding)
        self.assertEqual(combined.shape, (2, 1, 4, 4))

    def test_causal_mask_diagonal(self):
        mask = create_causal_mask(5)
        for i in range(5):
            self.assertEqual(mask[i, i].item(), 1.0)

    def test_causal_mask_is_tensor(self):
        mask = create_causal_mask(4)
        self.assertIsInstance(mask, torch.Tensor)

    def test_padding_mask_single_sequence(self):
        mask = create_padding_mask([5], max_len=10)
        self.assertEqual(mask.shape, (1, 10))
        self.assertEqual(mask[0, :5].sum().item(), 5)
        self.assertEqual(mask[0, 5:].sum().item(), 0)

    def test_padding_mask_full_length(self):
        mask = create_padding_mask([4, 4], max_len=4)
        self.assertTrue(torch.all(mask == 1))

    def test_combine_masks_zeros_padding(self):
        causal = create_causal_mask(3)
        padding = create_padding_mask([2, 3], max_len=3)
        combined = combine_masks(causal, padding)
        self.assertEqual(combined[0, 0, 0, 2].item(), 0)

    def test_causal_mask_upper_triangle_zero(self):
        mask = create_causal_mask(4)
        upper = torch.triu(mask, diagonal=1)
        self.assertEqual(upper.sum().item(), 0)
`,

	hint1: 'Use torch.tril() to create lower triangular matrix',
	hint2: 'Multiply masks element-wise to combine them',

	whyItMatters: `Attention masks control information flow:

- **Causal mask**: Prevents looking at future tokens (GPT-style)
- **Padding mask**: Ignores padding tokens
- **Combined masks**: Both constraints together
- **Efficient training**: Process sequences in parallel

Proper masking is essential for autoregressive generation.`,

	translations: {
		ru: {
			title: 'Каузальная маска внимания',
			description: `# Каузальная маска внимания

Создайте маски для каузального (авторегрессивного) внимания, используемого в decoder моделях типа GPT.

## Задача

Реализуйте две функции:
1. \`create_causal_mask\` - Создание треугольной маски для авторегрессивного внимания
2. \`create_padding_mask\` - Создание маски для padding токенов

## Пример

\`\`\`python
# Causal mask: prevents attending to future positions
mask = create_causal_mask(seq_len=4)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# Padding mask: masks out padding tokens
padding_mask = create_padding_mask(lengths=[3, 2], max_len=4)
\`\`\``,
			hint1: 'Используйте torch.tril() для создания нижнетреугольной матрицы',
			hint2: 'Перемножайте маски поэлементно для их комбинирования',
			whyItMatters: `Маски внимания контролируют поток информации:

- **Каузальная маска**: Предотвращает просмотр будущих токенов (GPT-style)
- **Padding маска**: Игнорирует padding токены
- **Комбинированные маски**: Оба ограничения вместе
- **Эффективное обучение**: Обработка последовательностей параллельно`,
		},
		uz: {
			title: 'Kauzal diqqat niqobi',
			description: `# Kauzal diqqat niqobi

GPT kabi decoder modellarda ishlatiladigan kauzal (avtoregessiv) diqqat uchun niqoblar yarating.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`create_causal_mask\` - Avtoregessiv diqqat uchun uchburchak niqob yaratish
2. \`create_padding_mask\` - Padding tokenlar uchun niqob yaratish

## Misol

\`\`\`python
# Causal mask: prevents attending to future positions
mask = create_causal_mask(seq_len=4)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# Padding mask: masks out padding tokens
padding_mask = create_padding_mask(lengths=[3, 2], max_len=4)
\`\`\``,
			hint1: "Pastki uchburchak matrisa yaratish uchun torch.tril() dan foydalaning",
			hint2: "Niqoblarni birlashtirish uchun ularni element bo'yicha ko'paytiring",
			whyItMatters: `Diqqat niqoblari ma'lumot oqimini boshqaradi:

- **Kauzal niqob**: Kelajak tokenlarni ko'rishni oldini oladi (GPT uslubi)
- **Padding niqobi**: Padding tokenlarni e'tiborsiz qoldiradi
- **Birlashtirilgan niqoblar**: Ikkala cheklov birga
- **Samarali o'qitish**: Ketma-ketliklarni parallel qayta ishlash`,
		},
	},
};

export default task;
