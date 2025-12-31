import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-bidirectional-rnn',
	title: 'Bidirectional RNNs',
	difficulty: 'medium',
	tags: ['pytorch', 'rnn', 'bidirectional'],
	estimatedTime: '15m',
	isPremium: true,
	order: 4,
	description: `# Bidirectional RNNs

Learn to process sequences in both directions for better context.

## Task

Implement a \`BiLSTMClassifier\` class that:
- Uses bidirectional LSTM
- Concatenates forward and backward hidden states
- Classifies the full sequence

## Example

\`\`\`python
model = BiLSTMClassifier(input_size=10, hidden_size=64, num_classes=5)
x = torch.randn(4, 20, 10)
output = model(x)
# output.shape = (4, 5)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""

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

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""

    def __init__(self, input_size: int, hidden_size: int,
                 num_classes: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True)
        # Output is 2*hidden_size due to bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output: (batch, seq, 2*hidden)
        output, (h_n, c_n) = self.lstm(x)
        # Use last timestep output (has both directions)
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        return out
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestBiLSTM(unittest.TestCase):
    def test_output_shape(self):
        model = BiLSTMClassifier(10, 64, 5)
        x = torch.randn(4, 20, 10)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_bidirectional_lstm(self):
        model = BiLSTMClassifier(10, 64, 5)
        self.assertTrue(model.lstm.bidirectional)

    def test_fc_input_size(self):
        model = BiLSTMClassifier(10, 64, 5)
        # FC should accept 2*hidden due to bidirectional
        self.assertEqual(model.fc.in_features, 128)

    def test_is_nn_module(self):
        model = BiLSTMClassifier(10, 64, 5)
        self.assertIsInstance(model, nn.Module)

    def test_has_lstm(self):
        model = BiLSTMClassifier(10, 64, 5)
        self.assertTrue(hasattr(model, 'lstm'))
        self.assertIsInstance(model.lstm, nn.LSTM)

    def test_has_fc(self):
        model = BiLSTMClassifier(10, 64, 5)
        self.assertTrue(hasattr(model, 'fc'))
        self.assertIsInstance(model.fc, nn.Linear)

    def test_single_sample(self):
        model = BiLSTMClassifier(10, 64, 5)
        x = torch.randn(1, 20, 10)
        out = model(x)
        self.assertEqual(out.shape, (1, 5))

    def test_different_seq_lengths(self):
        model = BiLSTMClassifier(10, 64, 5)
        for seq_len in [5, 15, 30]:
            x = torch.randn(4, seq_len, 10)
            out = model(x)
            self.assertEqual(out.shape, (4, 5))

    def test_different_num_classes(self):
        for num_classes in [2, 10, 20]:
            model = BiLSTMClassifier(10, 64, num_classes)
            x = torch.randn(4, 20, 10)
            out = model(x)
            self.assertEqual(out.shape, (4, num_classes))

    def test_output_not_nan(self):
        model = BiLSTMClassifier(10, 64, 5)
        x = torch.randn(4, 20, 10)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())
`,

	hint1: 'Set bidirectional=True in LSTM constructor',
	hint2: 'Bidirectional output is 2*hidden_size (forward + backward)',

	whyItMatters: `Bidirectional RNNs capture context from both directions:

- **Full context**: Each position sees past and future
- **Better representations**: Especially useful for NLP
- **Named entity recognition**: Know "Apple" is company from context
- **Common pattern**: Used in BERT, ELMo, and many NLP models

Bidirectional is essential when full sequence is available at inference.`,

	translations: {
		ru: {
			title: 'Двунаправленные RNN',
			description: `# Двунаправленные RNN

Научитесь обрабатывать последовательности в обоих направлениях для лучшего контекста.

## Задача

Реализуйте класс \`BiLSTMClassifier\`, который:
- Использует двунаправленный LSTM
- Конкатенирует прямое и обратное скрытые состояния
- Классифицирует всю последовательность

## Пример

\`\`\`python
model = BiLSTMClassifier(input_size=10, hidden_size=64, num_classes=5)
x = torch.randn(4, 20, 10)
output = model(x)
# output.shape = (4, 5)
\`\`\``,
			hint1: 'Установите bidirectional=True в конструкторе LSTM',
			hint2: 'Двунаправленный выход имеет размер 2*hidden_size (forward + backward)',
			whyItMatters: `Двунаправленные RNN захватывают контекст с обоих направлений:

- **Полный контекст**: Каждая позиция видит прошлое и будущее
- **Лучшие представления**: Особенно полезно для NLP
- **NER**: Знает что "Apple" - компания по контексту
- **Распространенный паттерн**: Используется в BERT, ELMo`,
		},
		uz: {
			title: "Ikki yo'nalishli RNN",
			description: `# Ikki yo'nalishli RNN

Yaxshiroq kontekst uchun ketma-ketliklarni ikkala yo'nalishda qayta ishlashni o'rganing.

## Topshiriq

\`BiLSTMClassifier\` sinfini amalga oshiring:
- Ikki yo'nalishli LSTM dan foydalanadi
- Oldinga va orqaga yashirin holatlarni birlashtiradi
- Butun ketma-ketlikni tasniflaydi

## Misol

\`\`\`python
model = BiLSTMClassifier(input_size=10, hidden_size=64, num_classes=5)
x = torch.randn(4, 20, 10)
output = model(x)
# output.shape = (4, 5)
\`\`\``,
			hint1: "LSTM konstruktorida bidirectional=True qo'ying",
			hint2: "Ikki yo'nalishli chiqish 2*hidden_size ga teng (forward + backward)",
			whyItMatters: `Ikki yo'nalishli RNN lar ikkala tomondan kontekstni oladi:

- **To'liq kontekst**: Har bir pozitsiya o'tmish va kelajakni ko'radi
- **Yaxshiroq tasvirlar**: Ayniqsa NLP uchun foydali
- **NER**: Kontekstdan "Apple" kompaniya ekanini biladi
- **Keng tarqalgan naqsh**: BERT, ELMo va ko'p NLP modellarida ishlatiladi`,
		},
	},
};

export default task;
