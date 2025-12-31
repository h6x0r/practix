import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-gru',
	title: 'GRU Networks',
	difficulty: 'medium',
	tags: ['pytorch', 'gru', 'sequence'],
	estimatedTime: '12m',
	isPremium: false,
	order: 3,
	description: `# GRU Networks

Learn Gated Recurrent Units - a simpler alternative to LSTM.

## Task

Implement a \`GRUModel\` class that:
- Uses a GRU layer for sequence processing
- Has a fully connected output layer
- Returns predictions for classification

## Example

\`\`\`python
model = GRUModel(input_size=10, hidden_size=64, num_classes=5)
x = torch.randn(4, 20, 10)  # batch, seq_len, features
output = model(x)
# output.shape = (4, 5)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """GRU-based sequence classifier."""

    def __init__(self, input_size: int, hidden_size: int,
                 num_classes: int, num_layers: int = 1):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """GRU-based sequence classifier."""

    def __init__(self, input_size: int, hidden_size: int,
                 num_classes: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process sequence
        _, hidden = self.gru(x)
        # Use last layer's hidden state
        out = hidden[-1]
        # Classify
        out = self.fc(out)
        return out
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestGRU(unittest.TestCase):
    def test_output_shape(self):
        model = GRUModel(10, 64, 5)
        x = torch.randn(4, 20, 10)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_different_configs(self):
        model = GRUModel(5, 32, 10, num_layers=2)
        x = torch.randn(2, 15, 5)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_has_gru(self):
        model = GRUModel(10, 64, 5)
        self.assertTrue(hasattr(model, 'gru'))
        self.assertIsInstance(model.gru, nn.GRU)

    def test_has_fc(self):
        model = GRUModel(10, 64, 5)
        self.assertTrue(hasattr(model, 'fc'))
        self.assertIsInstance(model.fc, nn.Linear)

    def test_model_is_module(self):
        model = GRUModel(10, 64, 5)
        self.assertIsInstance(model, nn.Module)

    def test_single_sample(self):
        model = GRUModel(10, 64, 5)
        x = torch.randn(1, 20, 10)
        out = model(x)
        self.assertEqual(out.shape, (1, 5))

    def test_larger_batch(self):
        model = GRUModel(10, 64, 5)
        x = torch.randn(16, 20, 10)
        out = model(x)
        self.assertEqual(out.shape, (16, 5))

    def test_different_seq_len(self):
        model = GRUModel(10, 64, 5)
        for seq_len in [5, 10, 30]:
            x = torch.randn(4, seq_len, 10)
            out = model(x)
            self.assertEqual(out.shape, (4, 5))

    def test_output_not_all_zeros(self):
        model = GRUModel(10, 64, 5)
        x = torch.randn(4, 20, 10)
        out = model(x)
        self.assertGreater(out.abs().sum().item(), 0)

    def test_model_trainable(self):
        model = GRUModel(10, 64, 5)
        x = torch.randn(4, 20, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.fc.weight.grad)
`,

	hint1: 'Use the last hidden state (hidden[-1]) for classification',
	hint2: 'GRU has no cell state, only hidden state - simpler than LSTM',

	whyItMatters: `GRU is a popular LSTM alternative:

- **Simpler**: Only 2 gates instead of 3, fewer parameters
- **Faster**: Less computation per step
- **Similar performance**: Often matches LSTM accuracy
- **Reset & update gates**: Control information flow

GRU is preferred when computational efficiency matters.`,

	translations: {
		ru: {
			title: 'Сети GRU',
			description: `# Сети GRU

Изучите Gated Recurrent Units - более простую альтернативу LSTM.

## Задача

Реализуйте класс \`GRUModel\`, который:
- Использует слой GRU для обработки последовательности
- Имеет полносвязный выходной слой
- Возвращает предсказания для классификации

## Пример

\`\`\`python
model = GRUModel(input_size=10, hidden_size=64, num_classes=5)
x = torch.randn(4, 20, 10)  # batch, seq_len, features
output = model(x)
# output.shape = (4, 5)
\`\`\``,
			hint1: 'Используйте последнее скрытое состояние (hidden[-1]) для классификации',
			hint2: 'GRU не имеет cell state, только hidden state - проще чем LSTM',
			whyItMatters: `GRU - популярная альтернатива LSTM:

- **Проще**: Только 2 гейта вместо 3, меньше параметров
- **Быстрее**: Меньше вычислений на шаг
- **Похожая точность**: Часто не уступает LSTM
- **Reset & update gates**: Контролируют поток информации`,
		},
		uz: {
			title: 'GRU tarmoqlari',
			description: `# GRU tarmoqlari

Gated Recurrent Units - LSTM ga oddiyroq alternativani o'rganing.

## Topshiriq

\`GRUModel\` sinfini amalga oshiring:
- Ketma-ketlikni qayta ishlash uchun GRU qatlamidan foydalanadi
- To'liq bog'langan chiqish qatlamiga ega
- Tasniflash uchun bashoratlarni qaytaradi

## Misol

\`\`\`python
model = GRUModel(input_size=10, hidden_size=64, num_classes=5)
x = torch.randn(4, 20, 10)  # batch, seq_len, features
output = model(x)
# output.shape = (4, 5)
\`\`\``,
			hint1: "Tasniflash uchun oxirgi yashirin holatdan (hidden[-1]) foydalaning",
			hint2: "GRU da cell state yo'q, faqat hidden state - LSTM dan oddiyroq",
			whyItMatters: `GRU LSTM ga mashhur alternativa:

- **Oddiyroq**: 3 ta o'rniga faqat 2 ta gate, kamroq parametrlar
- **Tezroq**: Har bir qadamda kamroq hisoblash
- **O'xshash ishlash**: Ko'pincha LSTM aniqligiga teng
- **Reset & update gates**: Ma'lumot oqimini boshqaradi`,
		},
	},
};

export default task;
