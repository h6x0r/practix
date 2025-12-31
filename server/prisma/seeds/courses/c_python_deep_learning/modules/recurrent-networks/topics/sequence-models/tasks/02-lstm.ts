import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-lstm',
	title: 'LSTM Networks',
	difficulty: 'medium',
	tags: ['pytorch', 'lstm', 'sequence'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# LSTM Networks

Learn Long Short-Term Memory networks for long-range dependencies.

## Task

Implement two functions:
1. \`create_lstm\` - Create an LSTM layer
2. \`lstm_forward\` - Process sequence, return outputs and both hidden states

## Example

\`\`\`python
lstm = create_lstm(input_size=10, hidden_size=32, num_layers=2)

x = torch.randn(4, 20, 10)  # batch, seq_len, features
outputs, (h_n, c_n) = lstm_forward(lstm, x)
# outputs.shape = (4, 20, 32)
# h_n.shape = (2, 4, 32)  # hidden state
# c_n.shape = (2, 4, 32)  # cell state
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

def create_lstm(input_size: int, hidden_size: int,
                num_layers: int = 1, batch_first: bool = True) -> nn.LSTM:
    """Create an LSTM layer."""
    # Your code here
    pass

def lstm_forward(lstm: nn.LSTM, x: torch.Tensor) -> tuple:
    """Process sequence through LSTM. Return (outputs, (h_n, c_n))."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

def create_lstm(input_size: int, hidden_size: int,
                num_layers: int = 1, batch_first: bool = True) -> nn.LSTM:
    """Create an LSTM layer."""
    return nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                   batch_first=batch_first)

def lstm_forward(lstm: nn.LSTM, x: torch.Tensor) -> tuple:
    """Process sequence through LSTM. Return (outputs, (h_n, c_n))."""
    outputs, (h_n, c_n) = lstm(x)
    return outputs, (h_n, c_n)
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestLSTM(unittest.TestCase):
    def test_create_lstm(self):
        lstm = create_lstm(10, 32, num_layers=2)
        self.assertIsInstance(lstm, nn.LSTM)
        self.assertEqual(lstm.hidden_size, 32)

    def test_lstm_forward_shapes(self):
        lstm = create_lstm(10, 32, num_layers=2)
        x = torch.randn(4, 20, 10)
        outputs, (h_n, c_n) = lstm_forward(lstm, x)
        self.assertEqual(outputs.shape, (4, 20, 32))
        self.assertEqual(h_n.shape, (2, 4, 32))
        self.assertEqual(c_n.shape, (2, 4, 32))

    def test_cell_state_different(self):
        lstm = create_lstm(5, 10)
        x = torch.randn(2, 8, 5)
        outputs, (h_n, c_n) = lstm_forward(lstm, x)
        # Cell state should be different from hidden state
        self.assertFalse(torch.equal(h_n, c_n))

    def test_lstm_input_size(self):
        lstm = create_lstm(15, 32)
        self.assertEqual(lstm.input_size, 15)

    def test_lstm_num_layers(self):
        lstm = create_lstm(10, 20, num_layers=3)
        self.assertEqual(lstm.num_layers, 3)

    def test_outputs_is_tensor(self):
        lstm = create_lstm(10, 20)
        x = torch.randn(2, 5, 10)
        outputs, _ = lstm_forward(lstm, x)
        self.assertIsInstance(outputs, torch.Tensor)

    def test_single_layer_shapes(self):
        lstm = create_lstm(10, 20, num_layers=1)
        x = torch.randn(2, 5, 10)
        outputs, (h_n, c_n) = lstm_forward(lstm, x)
        self.assertEqual(h_n.shape, (1, 2, 20))

    def test_different_seq_lengths(self):
        lstm = create_lstm(10, 20)
        for seq_len in [5, 10, 15]:
            x = torch.randn(2, seq_len, 10)
            outputs, _ = lstm_forward(lstm, x)
            self.assertEqual(outputs.shape[1], seq_len)

    def test_different_batch_sizes(self):
        lstm = create_lstm(10, 20)
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 5, 10)
            outputs, (h_n, _) = lstm_forward(lstm, x)
            self.assertEqual(h_n.shape[1], batch_size)

    def test_outputs_last_timestep_matches_hidden(self):
        lstm = create_lstm(10, 20, num_layers=1)
        x = torch.randn(2, 5, 10)
        outputs, (h_n, _) = lstm_forward(lstm, x)
        # For single layer, last output should match hidden state
        self.assertTrue(torch.allclose(outputs[:, -1, :], h_n.squeeze(0), atol=1e-5))
`,

	hint1: 'LSTM returns outputs and tuple of (hidden_state, cell_state)',
	hint2: 'Cell state is the LSTM memory that prevents vanishing gradients',

	whyItMatters: `LSTMs solve the vanishing gradient problem in RNNs:

- **Cell state**: Highway for gradients to flow unchanged
- **Forget gate**: Decides what information to discard
- **Input gate**: Controls what new information to store
- **Output gate**: Controls what to output from cell

LSTMs are the workhorse of sequence modeling for NLP and time series.`,

	translations: {
		ru: {
			title: 'Сети LSTM',
			description: `# Сети LSTM

Изучите сети Long Short-Term Memory для долгосрочных зависимостей.

## Задача

Реализуйте две функции:
1. \`create_lstm\` - Создание слоя LSTM
2. \`lstm_forward\` - Обработка последовательности, возврат outputs и обоих скрытых состояний

## Пример

\`\`\`python
lstm = create_lstm(input_size=10, hidden_size=32, num_layers=2)

x = torch.randn(4, 20, 10)  # batch, seq_len, features
outputs, (h_n, c_n) = lstm_forward(lstm, x)
# outputs.shape = (4, 20, 32)
# h_n.shape = (2, 4, 32)  # hidden state
# c_n.shape = (2, 4, 32)  # cell state
\`\`\``,
			hint1: 'LSTM возвращает outputs и кортеж (hidden_state, cell_state)',
			hint2: 'Cell state - память LSTM, предотвращающая затухание градиентов',
			whyItMatters: `LSTM решает проблему затухающих градиентов в RNN:

- **Cell state**: Магистраль для неизменного потока градиентов
- **Forget gate**: Решает какую информацию отбросить
- **Input gate**: Контролирует какую новую информацию сохранить
- **Output gate**: Контролирует что выводить из ячейки`,
		},
		uz: {
			title: 'LSTM tarmoqlari',
			description: `# LSTM tarmoqlari

Uzoq muddatli bog'liqliklar uchun Long Short-Term Memory tarmoqlarini o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`create_lstm\` - LSTM qatlami yaratish
2. \`lstm_forward\` - Ketma-ketlikni qayta ishlash, outputs va ikkala yashirin holatni qaytarish

## Misol

\`\`\`python
lstm = create_lstm(input_size=10, hidden_size=32, num_layers=2)

x = torch.randn(4, 20, 10)  # batch, seq_len, features
outputs, (h_n, c_n) = lstm_forward(lstm, x)
# outputs.shape = (4, 20, 32)
# h_n.shape = (2, 4, 32)  # hidden state
# c_n.shape = (2, 4, 32)  # cell state
\`\`\``,
			hint1: "LSTM outputs va (hidden_state, cell_state) kortejini qaytaradi",
			hint2: "Cell state yo'qolib ketayotgan gradientlarni oldini oladigan LSTM xotirasi",
			whyItMatters: `LSTM RNN lardagi yo'qolib ketayotgan gradient muammosini hal qiladi:

- **Cell state**: Gradientlar o'zgarishsiz oqishi uchun magistral
- **Forget gate**: Qaysi ma'lumotni tashlashni hal qiladi
- **Input gate**: Qaysi yangi ma'lumotni saqlashni boshqaradi
- **Output gate**: Hujayradan nimani chiqarishni boshqaradi`,
		},
	},
};

export default task;
