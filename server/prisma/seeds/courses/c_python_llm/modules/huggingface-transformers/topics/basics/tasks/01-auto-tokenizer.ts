import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-auto-tokenizer-basics',
	title: 'AutoTokenizer Basics',
	difficulty: 'easy',
	tags: ['huggingface', 'tokenizer', 'nlp'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# AutoTokenizer Basics

Learn to use HuggingFace's AutoTokenizer for text tokenization.

## Task

Implement functions that use AutoTokenizer to:
- Load a pretrained tokenizer
- Tokenize text into tokens and IDs
- Handle batched inputs with padding

## Example

\`\`\`python
tokenizer = load_tokenizer("bert-base-uncased")
tokens = tokenize_text(tokenizer, "Hello world!")
# tokens = {'input_ids': [...], 'attention_mask': [...]}
\`\`\``,

	initialCode: `from transformers import AutoTokenizer

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a pretrained tokenizer."""
    # Your code here
    pass

def tokenize_text(tokenizer, text: str) -> dict:
    """Tokenize a single text string."""
    # Your code here
    pass

def tokenize_batch(tokenizer, texts: list, max_length: int = 128) -> dict:
    """Tokenize a batch of texts with padding."""
    # Your code here
    pass

def decode_tokens(tokenizer, token_ids: list) -> str:
    """Decode token IDs back to text."""
    # Your code here
    pass
`,

	solutionCode: `from transformers import AutoTokenizer

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a pretrained tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_text(tokenizer, text: str) -> dict:
    """Tokenize a single text string."""
    return tokenizer(text, return_tensors="pt")

def tokenize_batch(tokenizer, texts: list, max_length: int = 128) -> dict:
    """Tokenize a batch of texts with padding."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def decode_tokens(tokenizer, token_ids: list) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)
`,

	testCode: `import unittest
from unittest.mock import MagicMock, patch

class TestAutoTokenizer(unittest.TestCase):
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_tokenizer(self, mock_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        result = load_tokenizer("bert-base-uncased")
        mock_from_pretrained.assert_called_once_with("bert-base-uncased")

    def test_tokenize_text_returns_dict(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        result = tokenize_text(mock_tokenizer, "Hello")
        self.assertIn("input_ids", result)

    def test_tokenize_batch_with_padding(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2], [3, 4]], "attention_mask": [[1, 1], [1, 1]]}
        result = tokenize_batch(mock_tokenizer, ["Hello", "World"], max_length=128)
        mock_tokenizer.assert_called_once()

    def test_decode_tokens(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Hello world"
        result = decode_tokens(mock_tokenizer, [101, 7592, 2088, 102])
        mock_tokenizer.decode.assert_called_once()

    def test_tokenize_text_calls_tokenizer(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        tokenize_text(mock_tokenizer, "Test text")
        mock_tokenizer.assert_called_once()

    def test_tokenize_batch_uses_padding(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2], [3, 4]]}
        tokenize_batch(mock_tokenizer, ["A", "B"])
        args, kwargs = mock_tokenizer.call_args
        self.assertTrue(kwargs.get('padding', False))

    def test_tokenize_batch_uses_truncation(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2], [3, 4]]}
        tokenize_batch(mock_tokenizer, ["A", "B"])
        args, kwargs = mock_tokenizer.call_args
        self.assertTrue(kwargs.get('truncation', False))

    def test_decode_returns_string(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "decoded text"
        result = decode_tokens(mock_tokenizer, [1, 2, 3])
        self.assertIsInstance(result, str)

    def test_tokenize_text_has_attention_mask(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        result = tokenize_text(mock_tokenizer, "Hello")
        self.assertIn("attention_mask", result)

    def test_tokenize_batch_respects_max_length(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2], [3, 4]]}
        tokenize_batch(mock_tokenizer, ["A", "B"], max_length=256)
        args, kwargs = mock_tokenizer.call_args
        self.assertEqual(kwargs.get('max_length'), 256)
`,

	hint1: 'Use AutoTokenizer.from_pretrained() to load tokenizers',
	hint2: 'Set padding=True and truncation=True for batch processing',

	whyItMatters: `Tokenizers are the entry point to LLMs:

- **Subword tokenization**: BPE, WordPiece, SentencePiece
- **Vocabulary**: Each model has its own vocabulary
- **Special tokens**: [CLS], [SEP], [PAD], [MASK]
- **Efficient batching**: Padding and attention masks

Proper tokenization is essential for model performance.`,

	translations: {
		ru: {
			title: 'Основы AutoTokenizer',
			description: `# Основы AutoTokenizer

Научитесь использовать AutoTokenizer от HuggingFace для токенизации текста.

## Задача

Реализуйте функции, использующие AutoTokenizer для:
- Загрузки предобученного токенизатора
- Токенизации текста в токены и ID
- Обработки батчей с padding

## Пример

\`\`\`python
tokenizer = load_tokenizer("bert-base-uncased")
tokens = tokenize_text(tokenizer, "Hello world!")
# tokens = {'input_ids': [...], 'attention_mask': [...]}
\`\`\``,
			hint1: 'Используйте AutoTokenizer.from_pretrained() для загрузки токенизаторов',
			hint2: 'Установите padding=True и truncation=True для обработки батчей',
			whyItMatters: `Токенизаторы - точка входа в LLM:

- **Subword токенизация**: BPE, WordPiece, SentencePiece
- **Словарь**: Каждая модель имеет свой словарь
- **Специальные токены**: [CLS], [SEP], [PAD], [MASK]
- **Эффективный батчинг**: Padding и attention masks`,
		},
		uz: {
			title: 'AutoTokenizer asoslari',
			description: `# AutoTokenizer asoslari

Matn tokenizatsiyasi uchun HuggingFace ning AutoTokenizer dan foydalanishni o'rganing.

## Topshiriq

AutoTokenizer dan foydalanadigan funksiyalarni amalga oshiring:
- Oldindan o'qitilgan tokenizatorni yuklash
- Matnni tokenlar va ID larga tokenizatsiya qilish
- Padding bilan batch inputlarni qayta ishlash

## Misol

\`\`\`python
tokenizer = load_tokenizer("bert-base-uncased")
tokens = tokenize_text(tokenizer, "Hello world!")
# tokens = {'input_ids': [...], 'attention_mask': [...]}
\`\`\``,
			hint1: "Tokenizatorlarni yuklash uchun AutoTokenizer.from_pretrained() dan foydalaning",
			hint2: "Batch qayta ishlash uchun padding=True va truncation=True o'rnating",
			whyItMatters: `Tokenizatorlar LLM larga kirish nuqtasi:

- **Subword tokenizatsiya**: BPE, WordPiece, SentencePiece
- **Lug'at**: Har bir model o'z lug'atiga ega
- **Maxsus tokenlar**: [CLS], [SEP], [PAD], [MASK]
- **Samarali batching**: Padding va attention masks`,
		},
	},
};

export default task;
