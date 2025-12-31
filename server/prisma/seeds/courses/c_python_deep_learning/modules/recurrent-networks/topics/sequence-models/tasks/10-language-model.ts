import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-language-model',
	title: 'Language Model',
	difficulty: 'hard',
	tags: ['pytorch', 'lstm', 'nlp', 'language-model'],
	estimatedTime: '20m',
	isPremium: true,
	order: 10,
	description: `# Language Model

Build a character-level language model using LSTM.

## Task

Implement a \`CharLanguageModel\` class that:
- Embeds characters to dense vectors
- Uses LSTM to model sequences
- Predicts next character at each position

## Example

\`\`\`python
model = CharLanguageModel(vocab_size=128, embed_dim=64, hidden_size=256)

# Input: batch of character indices
x = torch.randint(0, 128, (4, 100))  # batch, seq_len
logits = model(x)
# logits.shape = (4, 100, 128) - prediction for each position
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class CharLanguageModel(nn.Module):
    """Character-level language model with LSTM."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor,
                hidden: tuple = None) -> tuple:
        # TODO: Embed input, run through LSTM, return (logits, hidden)
        pass

    def generate(self, start_char: int, length: int,
                 temperature: float = 1.0) -> list:
        """Generate text starting from start_char."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLanguageModel(nn.Module):
    """Character-level language model with LSTM."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                hidden: tuple = None) -> tuple:
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_char: int, length: int,
                 temperature: float = 1.0) -> list:
        """Generate text starting from start_char."""
        self.eval()
        generated = [start_char]
        x = torch.tensor([[start_char]])
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                logits, hidden = self.forward(x, hidden)
                logits = logits[0, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                next_char = torch.multinomial(probs, 1).item()
                generated.append(next_char)
                x = torch.tensor([[next_char]])

        return generated
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestLanguageModel(unittest.TestCase):
    def test_forward_shape(self):
        model = CharLanguageModel(128, 64, 256)
        x = torch.randint(0, 128, (4, 100))
        logits, hidden = model(x)
        self.assertEqual(logits.shape, (4, 100, 128))

    def test_hidden_state(self):
        model = CharLanguageModel(128, 64, 256, num_layers=2)
        x = torch.randint(0, 128, (4, 50))
        logits, (h, c) = model(x)
        self.assertEqual(h.shape, (2, 4, 256))
        self.assertEqual(c.shape, (2, 4, 256))

    def test_generate(self):
        model = CharLanguageModel(128, 64, 256)
        generated = model.generate(start_char=65, length=10)
        self.assertEqual(len(generated), 11)  # start + 10 new
        self.assertTrue(all(0 <= c < 128 for c in generated))

    def test_is_nn_module(self):
        model = CharLanguageModel(128, 64, 256)
        self.assertIsInstance(model, nn.Module)

    def test_has_embedding(self):
        model = CharLanguageModel(128, 64, 256)
        self.assertTrue(hasattr(model, 'embedding'))

    def test_has_lstm(self):
        model = CharLanguageModel(128, 64, 256)
        self.assertTrue(hasattr(model, 'lstm'))

    def test_has_fc(self):
        model = CharLanguageModel(128, 64, 256)
        self.assertTrue(hasattr(model, 'fc'))

    def test_single_sample(self):
        model = CharLanguageModel(128, 64, 256)
        x = torch.randint(0, 128, (1, 50))
        logits, hidden = model(x)
        self.assertEqual(logits.shape, (1, 50, 128))

    def test_logits_not_nan(self):
        model = CharLanguageModel(128, 64, 256)
        x = torch.randint(0, 128, (2, 50))
        logits, _ = model(x)
        self.assertFalse(torch.isnan(logits).any())

    def test_vocab_size_stored(self):
        model = CharLanguageModel(256, 64, 128)
        self.assertEqual(model.vocab_size, 256)
`,

	hint1: 'Predict next char at each position: logits[:, i] predicts char[:, i+1]',
	hint2: 'Use temperature to control randomness in generation',

	whyItMatters: `Language models are the foundation of modern NLP:

- **Text generation**: Complete sentences, write stories
- **Pretraining**: GPT started as a language model
- **Understanding language**: Learn grammar and semantics
- **Foundation for LLMs**: ChatGPT, Claude are language models

Building one from scratch teaches core NLP concepts.`,

	translations: {
		ru: {
			title: 'Языковая модель',
			description: `# Языковая модель

Создайте языковую модель на уровне символов с использованием LSTM.

## Задача

Реализуйте класс \`CharLanguageModel\`, который:
- Представляет символы как плотные векторы
- Использует LSTM для моделирования последовательностей
- Предсказывает следующий символ на каждой позиции

## Пример

\`\`\`python
model = CharLanguageModel(vocab_size=128, embed_dim=64, hidden_size=256)

# Input: batch of character indices
x = torch.randint(0, 128, (4, 100))  # batch, seq_len
logits = model(x)
# logits.shape = (4, 100, 128) - prediction for each position
\`\`\``,
			hint1: 'Предсказание следующего символа на каждой позиции: logits[:, i] предсказывает char[:, i+1]',
			hint2: 'Используйте temperature для контроля случайности при генерации',
			whyItMatters: `Языковые модели - основа современного NLP:

- **Генерация текста**: Завершение предложений, написание историй
- **Предобучение**: GPT начинался как языковая модель
- **Понимание языка**: Изучение грамматики и семантики
- **Основа для LLM**: ChatGPT, Claude - это языковые модели`,
		},
		uz: {
			title: 'Til modeli',
			description: `# Til modeli

LSTM yordamida belgi darajasidagi til modelini yarating.

## Topshiriq

\`CharLanguageModel\` sinfini amalga oshiring:
- Belgilarni zich vektorlarga aylantiradi
- Ketma-ketliklarni modellashtirish uchun LSTM dan foydalanadi
- Har bir pozitsiyada keyingi belgini bashorat qiladi

## Misol

\`\`\`python
model = CharLanguageModel(vocab_size=128, embed_dim=64, hidden_size=256)

# Input: batch of character indices
x = torch.randint(0, 128, (4, 100))  # batch, seq_len
logits = model(x)
# logits.shape = (4, 100, 128) - prediction for each position
\`\`\``,
			hint1: "Har bir pozitsiyada keyingi belgini bashorat qilish: logits[:, i] char[:, i+1] ni bashorat qiladi",
			hint2: "Generatsiyada tasodifiylikni boshqarish uchun temperature dan foydalaning",
			whyItMatters: `Til modellari zamonaviy NLP ning asosi:

- **Matn generatsiyasi**: Gaplarni tugatish, hikoyalar yozish
- **Oldindan o'qitish**: GPT til modeli sifatida boshlangan
- **Tilni tushunish**: Grammatika va semantikani o'rganish
- **LLM lar uchun asos**: ChatGPT, Claude til modellari`,
		},
	},
};

export default task;
