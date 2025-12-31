import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-packed-sequences',
	title: 'Packed Sequences',
	difficulty: 'hard',
	tags: ['pytorch', 'rnn', 'padding'],
	estimatedTime: '18m',
	isPremium: true,
	order: 5,
	description: `# Packed Sequences

Learn to efficiently handle variable-length sequences with packing.

## Task

Implement two functions:
1. \`pack_batch\` - Pack padded sequences for efficient RNN processing
2. \`process_packed\` - Process packed sequences through LSTM and unpack

## Example

\`\`\`python
# Sequences of different lengths (padded)
sequences = torch.randn(4, 20, 10)  # batch, max_len, features
lengths = torch.tensor([20, 15, 10, 5])  # actual lengths

packed = pack_batch(sequences, lengths)
outputs = process_packed(lstm, packed, lengths)
# outputs.shape = (4, 20, hidden_size)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def pack_batch(sequences: torch.Tensor, lengths: torch.Tensor):
    """Pack padded sequences for efficient RNN processing."""
    # Your code here
    pass

def process_packed(lstm: nn.LSTM, packed, lengths: torch.Tensor) -> torch.Tensor:
    """Process packed sequences and return padded outputs."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def pack_batch(sequences: torch.Tensor, lengths: torch.Tensor):
    """Pack padded sequences for efficient RNN processing."""
    # Sort by length (required for packing)
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    sorted_sequences = sequences[sorted_idx]

    packed = pack_padded_sequence(sorted_sequences, sorted_lengths.cpu(),
                                   batch_first=True)
    return packed, sorted_idx

def process_packed(lstm: nn.LSTM, packed, lengths: torch.Tensor) -> torch.Tensor:
    """Process packed sequences and return padded outputs."""
    packed_seq, sorted_idx = packed

    # Process through LSTM
    packed_output, _ = lstm(packed_seq)

    # Unpack
    output, _ = pad_packed_sequence(packed_output, batch_first=True)

    # Unsort to original order
    _, unsort_idx = sorted_idx.sort()
    output = output[unsort_idx]

    return output
`,

	testCode: `import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import unittest

class TestPackedSequences(unittest.TestCase):
    def setUp(self):
        self.lstm = nn.LSTM(10, 32, batch_first=True)
        self.sequences = torch.randn(4, 20, 10)
        self.lengths = torch.tensor([20, 15, 10, 5])

    def test_pack_batch(self):
        packed, idx = pack_batch(self.sequences, self.lengths)
        self.assertIsNotNone(packed)

    def test_process_packed_shape(self):
        packed = pack_batch(self.sequences, self.lengths)
        outputs = process_packed(self.lstm, packed, self.lengths)
        self.assertEqual(outputs.shape[0], 4)
        self.assertEqual(outputs.shape[2], 32)

    def test_preserves_order(self):
        packed = pack_batch(self.sequences, self.lengths)
        outputs = process_packed(self.lstm, packed, self.lengths)
        # First sequence should have full length
        self.assertEqual(outputs.shape[1], 20)

    def test_pack_returns_tuple(self):
        packed, idx = pack_batch(self.sequences, self.lengths)
        self.assertIsInstance(idx, torch.Tensor)

    def test_sorted_idx_length(self):
        packed, idx = pack_batch(self.sequences, self.lengths)
        self.assertEqual(len(idx), 4)

    def test_output_tensor_type(self):
        packed = pack_batch(self.sequences, self.lengths)
        outputs = process_packed(self.lstm, packed, self.lengths)
        self.assertIsInstance(outputs, torch.Tensor)

    def test_different_lengths(self):
        sequences = torch.randn(3, 15, 10)
        lengths = torch.tensor([15, 10, 5])
        packed = pack_batch(sequences, lengths)
        outputs = process_packed(self.lstm, packed, lengths)
        self.assertEqual(outputs.shape[0], 3)

    def test_output_not_nan(self):
        packed = pack_batch(self.sequences, self.lengths)
        outputs = process_packed(self.lstm, packed, self.lengths)
        self.assertFalse(torch.isnan(outputs).any())

    def test_single_sequence(self):
        sequences = torch.randn(1, 10, 10)
        lengths = torch.tensor([10])
        lstm = nn.LSTM(10, 32, batch_first=True)
        packed = pack_batch(sequences, lengths)
        outputs = process_packed(lstm, packed, lengths)
        self.assertEqual(outputs.shape, (1, 10, 32))

    def test_equal_lengths(self):
        sequences = torch.randn(4, 15, 10)
        lengths = torch.tensor([15, 15, 15, 15])
        packed = pack_batch(sequences, lengths)
        outputs = process_packed(self.lstm, packed, lengths)
        self.assertEqual(outputs.shape[0], 4)
`,

	hint1: 'pack_padded_sequence requires sorted sequences by length',
	hint2: 'Remember to unsort outputs back to original order',

	whyItMatters: `Packed sequences are critical for efficient RNN training:

- **No wasted computation**: Skip padding tokens
- **Significant speedup**: Especially for variable-length batches
- **Memory efficient**: Don't process padding
- **Production use**: Essential for real NLP systems

Mastering packed sequences is key to efficient sequence processing.`,

	translations: {
		ru: {
			title: 'Упакованные последовательности',
			description: `# Упакованные последовательности

Научитесь эффективно обрабатывать последовательности переменной длины.

## Задача

Реализуйте две функции:
1. \`pack_batch\` - Упаковка padded последовательностей для эффективной обработки
2. \`process_packed\` - Обработка упакованных последовательностей через LSTM

## Пример

\`\`\`python
# Sequences of different lengths (padded)
sequences = torch.randn(4, 20, 10)  # batch, max_len, features
lengths = torch.tensor([20, 15, 10, 5])  # actual lengths

packed = pack_batch(sequences, lengths)
outputs = process_packed(lstm, packed, lengths)
# outputs.shape = (4, 20, hidden_size)
\`\`\``,
			hint1: 'pack_padded_sequence требует отсортированные по длине последовательности',
			hint2: 'Не забудьте вернуть outputs в исходный порядок',
			whyItMatters: `Упакованные последовательности критичны для эффективного обучения RNN:

- **Нет лишних вычислений**: Пропуск padding токенов
- **Значительное ускорение**: Особенно для батчей переменной длины
- **Эффективность памяти**: Не обрабатываем padding
- **Production использование**: Необходимо для реальных NLP систем`,
		},
		uz: {
			title: "Qadoqlangan ketma-ketliklar",
			description: `# Qadoqlangan ketma-ketliklar

O'zgaruvchan uzunlikdagi ketma-ketliklarni samarali qayta ishlashni o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`pack_batch\` - Samarali RNN qayta ishlash uchun padded ketma-ketliklarni qadoqlash
2. \`process_packed\` - Qadoqlangan ketma-ketliklarni LSTM orqali qayta ishlash

## Misol

\`\`\`python
# Sequences of different lengths (padded)
sequences = torch.randn(4, 20, 10)  # batch, max_len, features
lengths = torch.tensor([20, 15, 10, 5])  # actual lengths

packed = pack_batch(sequences, lengths)
outputs = process_packed(lstm, packed, lengths)
# outputs.shape = (4, 20, hidden_size)
\`\`\``,
			hint1: "pack_padded_sequence uzunlik bo'yicha saralangan ketma-ketliklarni talab qiladi",
			hint2: "Chiqishlarni asl tartibga qaytarishni unutmang",
			whyItMatters: `Qadoqlangan ketma-ketliklar samarali RNN o'qitish uchun muhim:

- **Behuda hisoblash yo'q**: Padding tokenlarini o'tkazib yuborish
- **Sezilarli tezlashtirish**: Ayniqsa o'zgaruvchan uzunlikdagi batchlar uchun
- **Xotira samaradorligi**: Padding ni qayta ishlamaymiz
- **Production foydalanish**: Haqiqiy NLP tizimlari uchun zarur`,
		},
	},
};

export default task;
