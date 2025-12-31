import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-huggingface-pipelines',
	title: 'HuggingFace Pipelines',
	difficulty: 'easy',
	tags: ['huggingface', 'pipeline', 'inference'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,
	description: `# HuggingFace Pipelines

Use pipelines for quick inference without manual model/tokenizer handling.

## Task

Implement functions using different pipeline types:
- Text classification
- Named Entity Recognition (NER)
- Question answering
- Text generation

## Example

\`\`\`python
classifier = create_classifier("distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.99}]
\`\`\``,

	initialCode: `from transformers import pipeline

def create_classifier(model_name: str = None):
    """Create a text classification pipeline."""
    # Your code here
    pass

def create_ner_pipeline(model_name: str = None):
    """Create a Named Entity Recognition pipeline."""
    # Your code here
    pass

def create_qa_pipeline(model_name: str = None):
    """Create a question answering pipeline."""
    # Your code here
    pass

def create_generator(model_name: str = None):
    """Create a text generation pipeline."""
    # Your code here
    pass

def batch_classify(classifier, texts: list) -> list:
    """Classify multiple texts at once."""
    # Your code here
    pass
`,

	solutionCode: `from transformers import pipeline

def create_classifier(model_name: str = None):
    """Create a text classification pipeline."""
    if model_name:
        return pipeline("text-classification", model=model_name)
    return pipeline("text-classification")

def create_ner_pipeline(model_name: str = None):
    """Create a Named Entity Recognition pipeline."""
    if model_name:
        return pipeline("ner", model=model_name, aggregation_strategy="simple")
    return pipeline("ner", aggregation_strategy="simple")

def create_qa_pipeline(model_name: str = None):
    """Create a question answering pipeline."""
    if model_name:
        return pipeline("question-answering", model=model_name)
    return pipeline("question-answering")

def create_generator(model_name: str = None):
    """Create a text generation pipeline."""
    if model_name:
        return pipeline("text-generation", model=model_name)
    return pipeline("text-generation")

def batch_classify(classifier, texts: list) -> list:
    """Classify multiple texts at once."""
    return classifier(texts)
`,

	testCode: `import unittest
from unittest.mock import MagicMock, patch

class TestPipelines(unittest.TestCase):
    @patch('transformers.pipeline')
    def test_create_classifier(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_classifier()
        mock_pipeline.assert_called_once_with("text-classification")

    @patch('transformers.pipeline')
    def test_create_classifier_with_model(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_classifier("my-model")
        mock_pipeline.assert_called_once_with("text-classification", model="my-model")

    @patch('transformers.pipeline')
    def test_create_ner_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_ner_pipeline()
        mock_pipeline.assert_called_once()

    def test_batch_classify(self):
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "POSITIVE"}, {"label": "NEGATIVE"}]
        result = batch_classify(mock_classifier, ["good", "bad"])
        self.assertEqual(len(result), 2)

    @patch('transformers.pipeline')
    def test_create_qa_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_qa_pipeline()
        mock_pipeline.assert_called_once()

    @patch('transformers.pipeline')
    def test_create_generator(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_generator()
        mock_pipeline.assert_called_once()

    @patch('transformers.pipeline')
    def test_create_generator_with_model(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_generator("gpt2")
        mock_pipeline.assert_called_once_with("text-generation", model="gpt2")

    @patch('transformers.pipeline')
    def test_create_qa_with_model(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        result = create_qa_pipeline("my-qa-model")
        mock_pipeline.assert_called_once_with("question-answering", model="my-qa-model")

    def test_batch_classify_calls_classifier(self):
        mock_classifier = MagicMock()
        mock_classifier.return_value = []
        batch_classify(mock_classifier, ["text1", "text2"])
        mock_classifier.assert_called_once_with(["text1", "text2"])

    def test_batch_classify_returns_list(self):
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "A"}, {"label": "B"}]
        result = batch_classify(mock_classifier, ["a", "b"])
        self.assertIsInstance(result, list)
`,

	hint1: 'Use pipeline("task_name") to create pipelines with default models',
	hint2: 'Pass a list of texts to pipeline for batch inference',

	whyItMatters: `Pipelines make NLP accessible:

- **Zero-config**: Works with sensible defaults
- **Best practices**: Handles batching, device placement
- **Many tasks**: Classification, NER, QA, generation, translation
- **Quick prototyping**: Test ideas without boilerplate

Great for rapid experimentation and production inference.`,

	translations: {
		ru: {
			title: 'HuggingFace Pipelines',
			description: `# HuggingFace Pipelines

Используйте pipelines для быстрого инференса без ручной обработки модели/токенизатора.

## Задача

Реализуйте функции с разными типами pipeline:
- Классификация текста
- Named Entity Recognition (NER)
- Question answering
- Генерация текста

## Пример

\`\`\`python
classifier = create_classifier("distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.99}]
\`\`\``,
			hint1: 'Используйте pipeline("task_name") для создания pipelines с моделями по умолчанию',
			hint2: 'Передайте список текстов в pipeline для batch инференса',
			whyItMatters: `Pipelines делают NLP доступным:

- **Zero-config**: Работает с разумными настройками по умолчанию
- **Best practices**: Обрабатывает батчинг, размещение на устройстве
- **Много задач**: Классификация, NER, QA, генерация, перевод
- **Быстрое прототипирование**: Тестируйте идеи без boilerplate`,
		},
		uz: {
			title: 'HuggingFace Pipelines',
			description: `# HuggingFace Pipelines

Model/tokenizatorni qo'lda boshqarmasdan tez inference uchun pipelines dan foydalaning.

## Topshiriq

Turli pipeline turlari bilan funksiyalarni amalga oshiring:
- Matn klassifikatsiyasi
- Named Entity Recognition (NER)
- Question answering
- Matn generatsiyasi

## Misol

\`\`\`python
classifier = create_classifier("distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.99}]
\`\`\``,
			hint1: "Standart modellar bilan pipelines yaratish uchun pipeline(\"task_name\") dan foydalaning",
			hint2: "Batch inference uchun pipeline ga matnlar ro'yxatini bering",
			whyItMatters: `Pipelines NLP ni oson qiladi:

- **Zero-config**: Oqilona standart sozlamalar bilan ishlaydi
- **Best practices**: Batching, qurilmaga joylashtirish
- **Ko'p vazifalar**: Klassifikatsiya, NER, QA, generatsiya, tarjima
- **Tez prototiplash**: Boilerplate siz g'oyalarni sinab ko'ring`,
		},
	},
};

export default task;
