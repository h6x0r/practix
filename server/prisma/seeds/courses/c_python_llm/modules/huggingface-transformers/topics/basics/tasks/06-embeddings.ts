import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-sentence-embeddings',
	title: 'Sentence Embeddings',
	difficulty: 'medium',
	tags: ['huggingface', 'embeddings', 'similarity'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,
	description: `# Sentence Embeddings

Create sentence embeddings for semantic similarity and retrieval.

## Task

Implement an \`EmbeddingModel\` class that:
- Creates sentence embeddings using mean pooling
- Computes cosine similarity between texts
- Finds most similar texts in a corpus

## Pooling Strategies

- **CLS pooling**: Use [CLS] token embedding
- **Mean pooling**: Average all token embeddings
- **Max pooling**: Max over token embeddings

## Example

\`\`\`python
model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

emb1 = model.encode("The cat sat on the mat")
emb2 = model.encode("A feline rested on the rug")

similarity = model.similarity(emb1, emb2)  # ~0.85
\`\`\``,

	initialCode: `import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel:
    """Create sentence embeddings for semantic search."""

    def __init__(self, model_name: str, device: str = None):
        # Your code here
        pass

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text to embedding."""
        # Your code here
        pass

    def encode_batch(self, texts: list) -> torch.Tensor:
        """Encode multiple texts."""
        # Your code here
        pass

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between embeddings."""
        # Your code here
        pass

    def find_similar(self, query: str, corpus: list, top_k: int = 5) -> list:
        """Find most similar texts in corpus."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel:
    """Create sentence embeddings for semantic search."""

    def __init__(self, model_name: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text to embedding."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        return F.normalize(embedding, p=2, dim=1).squeeze()

    def encode_batch(self, texts: list) -> torch.Tensor:
        """Encode multiple texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between embeddings."""
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        return F.cosine_similarity(emb1, emb2).item()

    def find_similar(self, query: str, corpus: list, top_k: int = 5) -> list:
        """Find most similar texts in corpus."""
        query_emb = self.encode(query).unsqueeze(0)
        corpus_embs = self.encode_batch(corpus)

        similarities = F.cosine_similarity(query_emb, corpus_embs)

        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": corpus[idx],
                "score": similarities[idx].item()
            })

        return results
`,

	testCode: `import torch
import unittest
from unittest.mock import MagicMock, patch

class TestEmbeddingModel(unittest.TestCase):
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_encode_returns_tensor(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 384)
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        model = EmbeddingModel("test-model", device="cpu")
        emb = model.encode("Hello")
        self.assertIsInstance(emb, torch.Tensor)

    def test_similarity_calculation(self):
        emb1 = torch.tensor([1.0, 0.0, 0.0])
        emb2 = torch.tensor([1.0, 0.0, 0.0])
        # Manual calculation
        sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_similarity_orthogonal(self):
        emb1 = torch.tensor([1.0, 0.0, 0.0])
        emb2 = torch.tensor([0.0, 1.0, 0.0])
        sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        self.assertAlmostEqual(sim, 0.0, places=5)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_has_tokenizer(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        model = EmbeddingModel("test-model", device="cpu")
        self.assertTrue(hasattr(model, 'tokenizer'))

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_has_model(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        model = EmbeddingModel("test-model", device="cpu")
        self.assertTrue(hasattr(model, 'model'))

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_has_device(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        model = EmbeddingModel("test-model", device="cpu")
        self.assertEqual(model.device, "cpu")

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_model_in_eval_mode(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        model = EmbeddingModel("test-model", device="cpu")
        mock_model_instance.eval.assert_called_once()

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_encode_batch_returns_tensor(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]])
        }

        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(2, 2, 384)
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        model = EmbeddingModel("test-model", device="cpu")
        embs = model.encode_batch(["Hello", "World"])
        self.assertIsInstance(embs, torch.Tensor)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_similarity_returns_float(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        model = EmbeddingModel("test-model", device="cpu")
        emb1 = torch.tensor([1.0, 0.0, 0.0])
        emb2 = torch.tensor([0.5, 0.5, 0.0])
        sim = model.similarity(emb1, emb2)
        self.assertIsInstance(sim, float)
`,

	hint1: 'Mean pooling: average token embeddings weighted by attention mask',
	hint2: 'Normalize embeddings before computing cosine similarity',

	whyItMatters: `Embeddings enable semantic understanding:

- **Semantic search**: Find documents by meaning, not keywords
- **Clustering**: Group similar texts automatically
- **Retrieval-Augmented Generation (RAG)**: Find relevant context for LLMs
- **Deduplication**: Identify similar content

Embeddings are the foundation of modern search and retrieval.`,

	translations: {
		ru: {
			title: 'Sentence Embeddings',
			description: `# Sentence Embeddings

Создайте sentence embeddings для семантического сходства и поиска.

## Задача

Реализуйте класс \`EmbeddingModel\`:
- Создает sentence embeddings с помощью mean pooling
- Вычисляет косинусное сходство между текстами
- Находит наиболее похожие тексты в корпусе

## Пример

\`\`\`python
model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

emb1 = model.encode("The cat sat on the mat")
emb2 = model.encode("A feline rested on the rug")

similarity = model.similarity(emb1, emb2)  # ~0.85
\`\`\``,
			hint1: 'Mean pooling: усреднение token embeddings с весами attention mask',
			hint2: 'Нормализуйте embeddings перед вычислением косинусного сходства',
			whyItMatters: `Embeddings обеспечивают семантическое понимание:

- **Семантический поиск**: Поиск документов по смыслу, а не ключевым словам
- **Кластеризация**: Автоматическая группировка похожих текстов
- **RAG**: Поиск релевантного контекста для LLM
- **Дедупликация**: Выявление похожего контента`,
		},
		uz: {
			title: 'Sentence Embeddings',
			description: `# Sentence Embeddings

Semantik o'xshashlik va qidirish uchun sentence embeddings yarating.

## Topshiriq

\`EmbeddingModel\` sinfini amalga oshiring:
- Mean pooling yordamida sentence embeddings yaratadi
- Matnlar o'rtasida kosinus o'xshashligini hisoblaydi
- Korpusda eng o'xshash matnlarni topadi

## Misol

\`\`\`python
model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

emb1 = model.encode("The cat sat on the mat")
emb2 = model.encode("A feline rested on the rug")

similarity = model.similarity(emb1, emb2)  # ~0.85
\`\`\``,
			hint1: "Mean pooling: attention mask og'irliklari bilan token embeddings ni o'rtachalashtirish",
			hint2: "Kosinus o'xshashligini hisoblashdan oldin embeddings ni normallang",
			whyItMatters: `Embeddings semantik tushunishni ta'minlaydi:

- **Semantik qidirish**: Kalit so'zlar emas, ma'no bo'yicha hujjatlarni topish
- **Klasterlash**: O'xshash matnlarni avtomatik guruhlash
- **RAG**: LLM lar uchun tegishli kontekstni topish
- **Deduplikatsiya**: O'xshash kontentni aniqlash`,
		},
	},
};

export default task;
