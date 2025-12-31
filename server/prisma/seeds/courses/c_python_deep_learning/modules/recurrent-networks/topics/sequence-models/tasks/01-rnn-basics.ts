import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-rnn-basics',
	title: 'RNN Basics',
	difficulty: 'medium',
	tags: ['pytorch', 'rnn', 'sequence'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# RNN Basics

Learn the fundamentals of Recurrent Neural Networks in PyTorch.

## Task

Implement two functions:
1. \`create_rnn\` - Create an RNN layer with specified parameters
2. \`process_sequence\` - Process a sequence through RNN, return all hidden states

## Example

\`\`\`python
rnn = create_rnn(input_size=10, hidden_size=20, num_layers=2)

# Process sequence: (batch=4, seq_len=15, features=10)
x = torch.randn(4, 15, 10)
outputs, final_hidden = process_sequence(rnn, x)
# outputs.shape = (4, 15, 20)
# final_hidden.shape = (2, 4, 20)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def create_rnn(input_size: int, hidden_size: int,
               num_layers: int = 1, batch_first: bool = True) -> nn.RNN:
    """Create an RNN layer."""
    # Your code here
    pass

def process_sequence(rnn: nn.RNN, x: torch.Tensor) -> tuple:
    """Process sequence through RNN. Return (outputs, hidden_state)."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

def create_rnn(input_size: int, hidden_size: int,
               num_layers: int = 1, batch_first: bool = True) -> nn.RNN:
    """Create an RNN layer."""
    return nn.RNN(input_size, hidden_size, num_layers=num_layers,
                  batch_first=batch_first)

def process_sequence(rnn: nn.RNN, x: torch.Tensor) -> tuple:
    """Process sequence through RNN. Return (outputs, hidden_state)."""
    outputs, hidden = rnn(x)
    return outputs, hidden
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestRNNBasics(unittest.TestCase):
    def test_create_rnn(self):
        rnn = create_rnn(10, 20, num_layers=2)
        self.assertIsInstance(rnn, nn.RNN)
        self.assertEqual(rnn.input_size, 10)
        self.assertEqual(rnn.hidden_size, 20)

    def test_process_sequence_shapes(self):
        rnn = create_rnn(10, 20, num_layers=2)
        x = torch.randn(4, 15, 10)
        outputs, hidden = process_sequence(rnn, x)
        self.assertEqual(outputs.shape, (4, 15, 20))
        self.assertEqual(hidden.shape, (2, 4, 20))

    def test_single_layer(self):
        rnn = create_rnn(5, 10, num_layers=1)
        x = torch.randn(2, 8, 5)
        outputs, hidden = process_sequence(rnn, x)
        self.assertEqual(hidden.shape, (1, 2, 10))

    def test_rnn_has_hidden_size(self):
        rnn = create_rnn(10, 20)
        self.assertEqual(rnn.hidden_size, 20)

    def test_rnn_has_num_layers(self):
        rnn = create_rnn(10, 20, num_layers=3)
        self.assertEqual(rnn.num_layers, 3)

    def test_outputs_is_tensor(self):
        rnn = create_rnn(10, 20)
        x = torch.randn(2, 5, 10)
        outputs, _ = process_sequence(rnn, x)
        self.assertIsInstance(outputs, torch.Tensor)

    def test_hidden_is_tensor(self):
        rnn = create_rnn(10, 20)
        x = torch.randn(2, 5, 10)
        _, hidden = process_sequence(rnn, x)
        self.assertIsInstance(hidden, torch.Tensor)

    def test_different_seq_lengths(self):
        rnn = create_rnn(10, 20)
        for seq_len in [5, 10, 20]:
            x = torch.randn(2, seq_len, 10)
            outputs, _ = process_sequence(rnn, x)
            self.assertEqual(outputs.shape[1], seq_len)

    def test_different_batch_sizes(self):
        rnn = create_rnn(10, 20)
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 5, 10)
            outputs, _ = process_sequence(rnn, x)
            self.assertEqual(outputs.shape[0], batch_size)

    def test_outputs_not_all_zeros(self):
        rnn = create_rnn(10, 20)
        x = torch.randn(2, 5, 10)
        outputs, _ = process_sequence(rnn, x)
        self.assertGreater(outputs.abs().sum().item(), 0)
`,

	hint1: 'Use nn.RNN(input_size, hidden_size, num_layers, batch_first)',
	hint2: 'RNN returns (outputs, hidden) - outputs has all timesteps',

	whyItMatters: `RNNs process sequential data by maintaining hidden state:

- **Hidden state**: Carries information across timesteps
- **Temporal dependencies**: Model relationships in sequences
- **Variable length**: Handle sequences of any length
- **Foundation**: Basis for LSTM and GRU architectures

RNNs are essential for time series, NLP, and any sequential data.`,

	translations: {
		ru: {
			title: 'Основы RNN',
			description: `# Основы RNN

Изучите основы рекуррентных нейронных сетей в PyTorch.

## Задача

Реализуйте две функции:
1. \`create_rnn\` - Создание слоя RNN с заданными параметрами
2. \`process_sequence\` - Обработка последовательности через RNN

## Пример

\`\`\`python
rnn = create_rnn(input_size=10, hidden_size=20, num_layers=2)

# Process sequence: (batch=4, seq_len=15, features=10)
x = torch.randn(4, 15, 10)
outputs, final_hidden = process_sequence(rnn, x)
# outputs.shape = (4, 15, 20)
# final_hidden.shape = (2, 4, 20)
\`\`\``,
			hint1: 'Используйте nn.RNN(input_size, hidden_size, num_layers, batch_first)',
			hint2: 'RNN возвращает (outputs, hidden) - outputs содержит все временные шаги',
			whyItMatters: `RNN обрабатывают последовательные данные через скрытое состояние:

- **Скрытое состояние**: Переносит информацию между шагами
- **Временные зависимости**: Моделирование связей в последовательностях
- **Переменная длина**: Работа с последовательностями любой длины
- **Основа**: База для архитектур LSTM и GRU`,
		},
		uz: {
			title: 'RNN asoslari',
			description: `# RNN asoslari

PyTorch da rekurrent neyron tarmoqlarning asoslarini o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`create_rnn\` - Berilgan parametrlar bilan RNN qatlami yaratish
2. \`process_sequence\` - Ketma-ketlikni RNN orqali qayta ishlash

## Misol

\`\`\`python
rnn = create_rnn(input_size=10, hidden_size=20, num_layers=2)

# Process sequence: (batch=4, seq_len=15, features=10)
x = torch.randn(4, 15, 10)
outputs, final_hidden = process_sequence(rnn, x)
# outputs.shape = (4, 15, 20)
# final_hidden.shape = (2, 4, 20)
\`\`\``,
			hint1: 'nn.RNN(input_size, hidden_size, num_layers, batch_first) dan foydalaning',
			hint2: "RNN (outputs, hidden) qaytaradi - outputs barcha vaqt qadamlarini o'z ichiga oladi",
			whyItMatters: `RNN lar yashirin holat orqali ketma-ket ma'lumotlarni qayta ishlaydi:

- **Yashirin holat**: Vaqt qadamlari o'rtasida ma'lumot tashiydi
- **Vaqtinchalik bog'liqliklar**: Ketma-ketliklardagi munosabatlarni modellashtiradi
- **O'zgaruvchan uzunlik**: Har qanday uzunlikdagi ketma-ketliklar bilan ishlaydi
- **Asos**: LSTM va GRU arxitekturalari uchun asos`,
		},
	},
};

export default task;
