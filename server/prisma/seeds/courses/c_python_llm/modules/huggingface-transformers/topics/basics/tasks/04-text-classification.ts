import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-text-classification-hf',
	title: 'Text Classification',
	difficulty: 'medium',
	tags: ['huggingface', 'classification', 'bert'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Text Classification with HuggingFace

Build a complete text classification pipeline with proper preprocessing and inference.

## Task

Implement a \`TextClassifier\` class that:
- Loads a pretrained classification model
- Handles tokenization
- Returns predictions with confidence scores
- Supports batch inference

## Example

\`\`\`python
classifier = TextClassifier("distilbert-base-uncased-finetuned-sst-2-english")

result = classifier.predict("This movie was amazing!")
# {'label': 'POSITIVE', 'confidence': 0.9998}

results = classifier.predict_batch(["Great!", "Terrible..."])
\`\`\``,

	initialCode: `import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextClassifier:
    """Text classification using HuggingFace models."""

    def __init__(self, model_name: str, device: str = None):
        # Your code here
        pass

    def predict(self, text: str) -> dict:
        """Classify a single text."""
        # Your code here
        pass

    def predict_batch(self, texts: list) -> list:
        """Classify multiple texts."""
        # Your code here
        pass

    def get_probabilities(self, text: str) -> dict:
        """Get probabilities for all classes."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextClassifier:
    """Text classification using HuggingFace models."""

    def __init__(self, model_name: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> dict:
        """Classify a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_id].item()

        return {
            "label": self.id2label[pred_id],
            "confidence": confidence
        }

    def predict_batch(self, texts: list) -> list:
        """Classify multiple texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        confidences = probs.gather(1, pred_ids.unsqueeze(1)).squeeze()

        results = []
        for i, (pred_id, conf) in enumerate(zip(pred_ids, confidences)):
            results.append({
                "label": self.id2label[pred_id.item()],
                "confidence": conf.item()
            })

        return results

    def get_probabilities(self, text: str) -> dict:
        """Get probabilities for all classes."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)[0]

        return {
            self.id2label[i]: prob.item()
            for i, prob in enumerate(probs)
        }
`,

	testCode: `import torch
import unittest
from unittest.mock import MagicMock, patch

class TestTextClassifier(unittest.TestCase):
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_init(self, mock_model, mock_tokenizer):
        mock_model.return_value = MagicMock()
        mock_model.return_value.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_tokenizer.return_value = MagicMock()
        classifier = TextClassifier("test-model", device="cpu")
        self.assertIsNotNone(classifier.model)
        self.assertIsNotNone(classifier.tokenizer)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_predict_returns_dict(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9]])
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        classifier = TextClassifier("test-model", device="cpu")
        result = classifier.predict("Great!")

        self.assertIn("label", result)
        self.assertIn("confidence", result)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_has_tokenizer(self, mock_model, mock_tokenizer):
        mock_model.return_value = MagicMock()
        mock_model.return_value.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_tokenizer.return_value = MagicMock()
        classifier = TextClassifier("test-model", device="cpu")
        self.assertIsNotNone(classifier.tokenizer)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_has_device(self, mock_model, mock_tokenizer):
        mock_model.return_value = MagicMock()
        mock_model.return_value.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_tokenizer.return_value = MagicMock()
        classifier = TextClassifier("test-model", device="cpu")
        self.assertEqual(classifier.device, "cpu")

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_has_id2label(self, mock_model, mock_tokenizer):
        mock_model.return_value = MagicMock()
        mock_model.return_value.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_tokenizer.return_value = MagicMock()
        classifier = TextClassifier("test-model", device="cpu")
        self.assertIn(0, classifier.id2label)
        self.assertIn(1, classifier.id2label)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_predict_batch_returns_list(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {"input_ids": torch.tensor([[1, 2], [3, 4]]), "attention_mask": torch.tensor([[1, 1], [1, 1]])}

        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        classifier = TextClassifier("test-model", device="cpu")
        result = classifier.predict_batch(["Good", "Bad"])
        self.assertIsInstance(result, list)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_get_probabilities_returns_dict(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9]])
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        classifier = TextClassifier("test-model", device="cpu")
        result = classifier.get_probabilities("Test")
        self.assertIsInstance(result, dict)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_predict_has_confidence(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9]])
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        classifier = TextClassifier("test-model", device="cpu")
        result = classifier.predict("Great!")
        self.assertGreater(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_model_is_in_eval_mode(self, mock_model, mock_tokenizer):
        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = MagicMock()
        classifier = TextClassifier("test-model", device="cpu")
        mock_model_instance.eval.assert_called_once()

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_get_probs_has_all_labels(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9]])
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        classifier = TextClassifier("test-model", device="cpu")
        result = classifier.get_probabilities("Test")
        self.assertIn("NEGATIVE", result)
        self.assertIn("POSITIVE", result)
`,

	hint1: 'Use model.config.id2label to map prediction IDs to labels',
	hint2: 'Apply F.softmax to logits to get probabilities',

	whyItMatters: `Text classification is a fundamental NLP task:

- **Sentiment analysis**: Detect opinions in reviews, social media
- **Content moderation**: Identify harmful content
- **Intent detection**: Understand user queries in chatbots
- **Topic classification**: Categorize documents

Production-ready classification requires proper batching and device handling.`,

	translations: {
		ru: {
			title: 'Классификация текста',
			description: `# Классификация текста с HuggingFace

Создайте полный pipeline классификации текста с правильной предобработкой и инференсом.

## Задача

Реализуйте класс \`TextClassifier\`:
- Загружает предобученную модель классификации
- Обрабатывает токенизацию
- Возвращает предсказания с confidence scores
- Поддерживает batch инференс

## Пример

\`\`\`python
classifier = TextClassifier("distilbert-base-uncased-finetuned-sst-2-english")

result = classifier.predict("This movie was amazing!")
# {'label': 'POSITIVE', 'confidence': 0.9998}

results = classifier.predict_batch(["Great!", "Terrible..."])
\`\`\``,
			hint1: 'Используйте model.config.id2label для маппинга ID предсказаний в метки',
			hint2: 'Примените F.softmax к logits для получения вероятностей',
			whyItMatters: `Классификация текста - фундаментальная задача NLP:

- **Анализ тональности**: Определение мнений в отзывах, соцсетях
- **Модерация контента**: Выявление вредного контента
- **Определение интента**: Понимание запросов пользователей в чатботах
- **Классификация тем**: Категоризация документов`,
		},
		uz: {
			title: 'Matn klassifikatsiyasi',
			description: `# HuggingFace bilan matn klassifikatsiyasi

To'g'ri oldindan qayta ishlash va inference bilan to'liq matn klassifikatsiyasi pipeline yarating.

## Topshiriq

\`TextClassifier\` sinfini amalga oshiring:
- Oldindan o'qitilgan klassifikatsiya modelini yuklaydi
- Tokenizatsiyani boshqaradi
- Confidence scores bilan bashoratlarni qaytaradi
- Batch inference ni qo'llab-quvvatlaydi

## Misol

\`\`\`python
classifier = TextClassifier("distilbert-base-uncased-finetuned-sst-2-english")

result = classifier.predict("This movie was amazing!")
# {'label': 'POSITIVE', 'confidence': 0.9998}

results = classifier.predict_batch(["Great!", "Terrible..."])
\`\`\``,
			hint1: "Bashorat ID larini teglarga map qilish uchun model.config.id2label dan foydalaning",
			hint2: "Ehtimolliklarni olish uchun logits ga F.softmax qo'llang",
			whyItMatters: `Matn klassifikatsiyasi asosiy NLP vazifasi:

- **Sentiment tahlili**: Sharhlarda, ijtimoiy tarmoqlarda fikrlarni aniqlash
- **Kontent moderatsiyasi**: Zararli kontentni aniqlash
- **Intent aniqlash**: Chatbotlarda foydalanuvchi so'rovlarini tushunish
- **Mavzu klassifikatsiyasi**: Hujjatlarni kategoriyalash`,
		},
	},
};

export default task;
