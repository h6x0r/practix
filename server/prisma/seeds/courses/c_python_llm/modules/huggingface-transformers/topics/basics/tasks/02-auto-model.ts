import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-auto-model-loading',
	title: 'AutoModel Loading',
	difficulty: 'easy',
	tags: ['huggingface', 'model', 'bert'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,
	description: `# AutoModel Loading

Learn to load pretrained models with HuggingFace AutoModel classes.

## Task

Implement functions to:
- Load different types of models (base, sequence classification, token classification)
- Get model embeddings
- Check model configuration

## AutoModel Variants

- \`AutoModel\`: Base model, outputs hidden states
- \`AutoModelForSequenceClassification\`: For text classification
- \`AutoModelForTokenClassification\`: For NER, POS tagging
- \`AutoModelForCausalLM\`: For text generation (GPT-style)

## Example

\`\`\`python
model = load_base_model("bert-base-uncased")
embeddings = get_embeddings(model, tokenizer, "Hello world")
\`\`\``,

	initialCode: `import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

def load_base_model(model_name: str):
    """Load a base transformer model."""
    # Your code here
    pass

def load_classifier(model_name: str, num_labels: int):
    """Load a model for sequence classification."""
    # Your code here
    pass

def get_embeddings(model, tokenizer, text: str) -> torch.Tensor:
    """Get the [CLS] token embedding for a text."""
    # Your code here
    pass

def get_model_config(model) -> dict:
    """Extract key configuration from model."""
    # Your code here
    pass
`,

	solutionCode: `import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

def load_base_model(model_name: str):
    """Load a base transformer model."""
    return AutoModel.from_pretrained(model_name)

def load_classifier(model_name: str, num_labels: int):
    """Load a model for sequence classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

def get_embeddings(model, tokenizer, text: str) -> torch.Tensor:
    """Get the [CLS] token embedding for a text."""
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Get [CLS] token embedding (first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

def get_model_config(model) -> dict:
    """Extract key configuration from model."""
    config = model.config
    return {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "model_type": config.model_type
    }
`,

	testCode: `import torch
import unittest
from unittest.mock import MagicMock, patch

class TestAutoModel(unittest.TestCase):
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_base_model(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        result = load_base_model("bert-base-uncased")
        mock_from_pretrained.assert_called_once_with("bert-base-uncased")

    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_load_classifier(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        result = load_classifier("bert-base-uncased", num_labels=2)
        mock_from_pretrained.assert_called_once_with("bert-base-uncased", num_labels=2)

    def test_get_embeddings_shape(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 768)
        mock_model.return_value = mock_outputs
        result = get_embeddings(mock_model, mock_tokenizer, "Hello")
        self.assertEqual(result.shape, (1, 768))

    def test_get_model_config(self):
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.vocab_size = 30522
        mock_model.config.model_type = "bert"
        result = get_model_config(mock_model)
        self.assertEqual(result["hidden_size"], 768)
        self.assertEqual(result["num_layers"], 12)

    def test_get_model_config_has_vocab_size(self):
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.vocab_size = 30522
        mock_model.config.model_type = "bert"
        result = get_model_config(mock_model)
        self.assertEqual(result["vocab_size"], 30522)

    def test_get_model_config_has_model_type(self):
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.vocab_size = 30522
        mock_model.config.model_type = "roberta"
        result = get_model_config(mock_model)
        self.assertEqual(result["model_type"], "roberta")

    def test_get_model_config_has_num_heads(self):
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 8
        mock_model.config.vocab_size = 30522
        mock_model.config.model_type = "bert"
        result = get_model_config(mock_model)
        self.assertEqual(result["num_heads"], 8)

    def test_get_model_config_returns_dict(self):
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.vocab_size = 30522
        mock_model.config.model_type = "bert"
        result = get_model_config(mock_model)
        self.assertIsInstance(result, dict)

    def test_get_embeddings_returns_tensor(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 768)
        mock_model.return_value = mock_outputs
        result = get_embeddings(mock_model, mock_tokenizer, "Hello")
        self.assertIsInstance(result, torch.Tensor)
`,

	hint1: 'Use AutoModel.from_pretrained() for base models',
	hint2: 'Access [CLS] embedding with outputs.last_hidden_state[:, 0, :]',

	whyItMatters: `AutoModel simplifies working with different architectures:

- **Unified API**: Same interface for BERT, RoBERTa, GPT, etc.
- **Task-specific heads**: Classification, NER, generation
- **Config access**: Model hyperparameters and architecture
- **Transfer learning**: Start from pretrained weights

This is the standard way to use transformers in production.`,

	translations: {
		ru: {
			title: 'Загрузка AutoModel',
			description: `# Загрузка AutoModel

Научитесь загружать предобученные модели с помощью классов AutoModel от HuggingFace.

## Задача

Реализуйте функции для:
- Загрузки разных типов моделей (base, sequence classification, token classification)
- Получения embeddings модели
- Проверки конфигурации модели

## Пример

\`\`\`python
model = load_base_model("bert-base-uncased")
embeddings = get_embeddings(model, tokenizer, "Hello world")
\`\`\``,
			hint1: 'Используйте AutoModel.from_pretrained() для базовых моделей',
			hint2: 'Получите [CLS] embedding через outputs.last_hidden_state[:, 0, :]',
			whyItMatters: `AutoModel упрощает работу с разными архитектурами:

- **Единый API**: Одинаковый интерфейс для BERT, RoBERTa, GPT и т.д.
- **Task-specific heads**: Классификация, NER, генерация
- **Доступ к конфигу**: Гиперпараметры и архитектура модели
- **Transfer learning**: Старт с предобученных весов`,
		},
		uz: {
			title: 'AutoModel yuklash',
			description: `# AutoModel yuklash

HuggingFace ning AutoModel sinflari bilan oldindan o'qitilgan modellarni yuklashni o'rganing.

## Topshiriq

Funksiyalarni amalga oshiring:
- Turli xil modellarni yuklash (base, sequence classification, token classification)
- Model embeddinglarini olish
- Model konfiguratsiyasini tekshirish

## Misol

\`\`\`python
model = load_base_model("bert-base-uncased")
embeddings = get_embeddings(model, tokenizer, "Hello world")
\`\`\``,
			hint1: "Bazaviy modellar uchun AutoModel.from_pretrained() dan foydalaning",
			hint2: "[CLS] embedding ni outputs.last_hidden_state[:, 0, :] orqali oling",
			whyItMatters: `AutoModel turli arxitekturalar bilan ishlashni soddalashtiradi:

- **Yagona API**: BERT, RoBERTa, GPT va boshqalar uchun bir xil interfeys
- **Vazifaga xos boshlar**: Klassifikatsiya, NER, generatsiya
- **Konfigga kirish**: Model giperparametrlari va arxitekturasi
- **Transfer learning**: Oldindan o'qitilgan og'irliklardan boshlash`,
		},
	},
};

export default task;
