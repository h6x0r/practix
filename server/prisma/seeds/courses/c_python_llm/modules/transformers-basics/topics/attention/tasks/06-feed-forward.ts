import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-feed-forward-network',
	title: 'Feed-Forward Network',
	difficulty: 'easy',
	tags: ['pytorch', 'transformer', 'mlp'],
	estimatedTime: '10m',
	isPremium: false,
	order: 6,
	description: `# Feed-Forward Network

Implement the position-wise feed-forward network used in transformers.

## Task

Implement a \`FeedForward\` class with:
- Two linear layers with expansion factor
- GELU activation
- Dropout

## Example

\`\`\`python
ff = FeedForward(d_model=512, d_ff=2048)

x = torch.randn(2, 10, 512)
output = ff(x)
# output.shape = (2, 10, 512)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestFeedForward(unittest.TestCase):
    def test_output_shape(self):
        ff = FeedForward(d_model=512)
        x = torch.randn(2, 10, 512)
        output = ff(x)
        self.assertEqual(output.shape, (2, 10, 512))

    def test_custom_d_ff(self):
        ff = FeedForward(d_model=256, d_ff=1024)
        self.assertEqual(ff.linear1.out_features, 1024)
        self.assertEqual(ff.linear2.in_features, 1024)

    def test_default_expansion(self):
        ff = FeedForward(d_model=256)
        self.assertEqual(ff.linear1.out_features, 1024)  # 256 * 4

    def test_is_nn_module(self):
        ff = FeedForward(d_model=512)
        self.assertIsInstance(ff, nn.Module)

    def test_has_linear_layers(self):
        ff = FeedForward(d_model=256)
        self.assertTrue(hasattr(ff, 'linear1'))
        self.assertTrue(hasattr(ff, 'linear2'))

    def test_has_dropout(self):
        ff = FeedForward(d_model=256, dropout=0.2)
        self.assertTrue(hasattr(ff, 'dropout'))

    def test_output_not_nan(self):
        ff = FeedForward(d_model=128)
        x = torch.randn(2, 8, 128)
        output = ff(x)
        self.assertFalse(torch.isnan(output).any())

    def test_single_sample(self):
        ff = FeedForward(d_model=256)
        x = torch.randn(1, 5, 256)
        output = ff(x)
        self.assertEqual(output.shape, (1, 5, 256))

    def test_linear2_output_matches_d_model(self):
        ff = FeedForward(d_model=512, d_ff=2048)
        self.assertEqual(ff.linear2.out_features, 512)

    def test_different_seq_lengths(self):
        ff = FeedForward(d_model=128)
        x1 = torch.randn(2, 5, 128)
        x2 = torch.randn(2, 20, 128)
        out1 = ff(x1)
        out2 = ff(x2)
        self.assertEqual(out1.shape, (2, 5, 128))
        self.assertEqual(out2.shape, (2, 20, 128))
`,

	hint1: 'Default d_ff is 4 * d_model (expansion factor)',
	hint2: 'GELU is preferred over ReLU in modern transformers',

	whyItMatters: `Feed-forward networks add non-linearity to transformers:

- **Position-wise**: Same MLP applied to each position
- **Expansion**: Increase capacity with wider hidden layer
- **GELU activation**: Smoother than ReLU, used in BERT/GPT
- **Memory bottleneck**: Often the largest part of transformer

FFN is half of each transformer layer's computation.`,

	translations: {
		ru: {
			title: 'Feed-Forward сеть',
			description: `# Feed-Forward сеть

Реализуйте position-wise feed-forward сеть, используемую в трансформерах.

## Задача

Реализуйте класс \`FeedForward\` с:
- Два линейных слоя с фактором расширения
- GELU активация
- Dropout

## Пример

\`\`\`python
ff = FeedForward(d_model=512, d_ff=2048)

x = torch.randn(2, 10, 512)
output = ff(x)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: 'По умолчанию d_ff = 4 * d_model (фактор расширения)',
			hint2: 'GELU предпочтительнее ReLU в современных трансформерах',
			whyItMatters: `Feed-forward сети добавляют нелинейность в трансформеры:

- **Position-wise**: Одинаковый MLP применяется к каждой позиции
- **Расширение**: Увеличение емкости через более широкий скрытый слой
- **GELU активация**: Более гладкая чем ReLU, используется в BERT/GPT
- **Узкое место памяти**: Часто самая большая часть трансформера`,
		},
		uz: {
			title: 'Feed-Forward tarmoq',
			description: `# Feed-Forward tarmoq

Transformerlarda ishlatiladigan position-wise feed-forward tarmoqni amalga oshiring.

## Topshiriq

\`FeedForward\` sinfini amalga oshiring:
- Kengayish koeffitsienti bilan ikkita chiziqli qatlam
- GELU aktivatsiyasi
- Dropout

## Misol

\`\`\`python
ff = FeedForward(d_model=512, d_ff=2048)

x = torch.randn(2, 10, 512)
output = ff(x)
# output.shape = (2, 10, 512)
\`\`\``,
			hint1: "Standart d_ff = 4 * d_model (kengayish koeffitsienti)",
			hint2: "Zamonaviy transformerlarda GELU ReLU dan afzalroq",
			whyItMatters: `Feed-forward tarmoqlar transformerlarga nolinearlik qo'shadi:

- **Position-wise**: Bir xil MLP har bir pozitsiyaga qo'llaniladi
- **Kengayish**: Kengroq yashirin qatlam bilan sig'imni oshirish
- **GELU aktivatsiyasi**: ReLU dan tekisroq, BERT/GPT da ishlatiladi
- **Xotira tiqinchisi**: Ko'pincha transformerning eng katta qismi`,
		},
	},
};

export default task;
