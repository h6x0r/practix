import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-vllm-serving',
	title: 'vLLM High-Performance Serving',
	difficulty: 'hard',
	tags: ['vllm', 'serving', 'production'],
	estimatedTime: '25m',
	isPremium: true,
	order: 2,
	description: `# vLLM High-Performance Serving

Use vLLM for high-throughput LLM inference.

## What is vLLM?

vLLM provides:
- **PagedAttention**: Efficient memory management
- **Continuous batching**: Dynamic batch scheduling
- **Tensor parallelism**: Multi-GPU support
- **OpenAI-compatible API**: Drop-in replacement

## Performance

- 24x higher throughput than HuggingFace
- Lower latency with batching
- Optimal GPU memory utilization

## Example

\`\`\`python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
\`\`\``,

	initialCode: `from vllm import LLM, SamplingParams
from typing import List, Dict

class VLLMServer:
    """High-performance LLM server using vLLM."""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1,
                 max_model_len: int = 4096):
        # Your code here
        pass

    def generate(self, prompts: List[str], max_tokens: int = 100,
                 temperature: float = 0.7, top_p: float = 0.9) -> List[str]:
        """Generate completions for multiple prompts."""
        # Your code here
        pass

    def generate_with_params(self, prompts: List[str],
                             params: SamplingParams) -> List[Dict]:
        """Generate with custom sampling parameters."""
        # Your code here
        pass

    def batch_generate(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """Generate in batches for very long prompt lists."""
        # Your code here
        pass
`,

	solutionCode: `from vllm import LLM, SamplingParams
from typing import List, Dict

class VLLMServer:
    """High-performance LLM server using vLLM."""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1,
                 max_model_len: int = 4096):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True
        )

    def generate(self, prompts: List[str], max_tokens: int = 100,
                 temperature: float = 0.7, top_p: float = 0.9) -> List[str]:
        """Generate completions for multiple prompts."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        outputs = self.llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def generate_with_params(self, prompts: List[str],
                             params: SamplingParams) -> List[Dict]:
        """Generate with custom sampling parameters."""
        outputs = self.llm.generate(prompts, params)

        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason
            })

        return results

    def batch_generate(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """Generate in batches for very long prompt lists."""
        all_results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            results = self.generate(batch)
            all_results.extend(results)

        return all_results

def create_vllm_sampling_params(
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = -1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop: List[str] = None
) -> SamplingParams:
    """Create vLLM sampling parameters."""
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=stop or []
    )
`,

	testCode: `import unittest
from unittest.mock import MagicMock, patch

class TestVLLMServer(unittest.TestCase):
    @patch('vllm.LLM')
    def test_init(self, mock_llm_class):
        mock_llm_class.return_value = MagicMock()
        server = VLLMServer("test-model", tensor_parallel_size=2)
        mock_llm_class.assert_called_once()

    @patch('vllm.LLM')
    def test_generate(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text")]
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm

        server = VLLMServer("test-model")
        results = server.generate(["Hello"])
        self.assertEqual(results, ["Generated text"])

    def test_create_sampling_params(self):
        params = create_vllm_sampling_params(max_tokens=50, temperature=0.5)
        self.assertEqual(params.max_tokens, 50)
        self.assertEqual(params.temperature, 0.5)

    def test_sampling_params_returns_sampling_params(self):
        params = create_vllm_sampling_params()
        self.assertIsInstance(params, SamplingParams)

    def test_sampling_params_default_top_p(self):
        params = create_vllm_sampling_params()
        self.assertEqual(params.top_p, 0.9)

    @patch('vllm.LLM')
    def test_server_has_llm(self, mock_llm_class):
        mock_llm_class.return_value = MagicMock()
        server = VLLMServer("test-model")
        self.assertIsNotNone(server.llm)

    @patch('vllm.LLM')
    def test_batch_generate(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Gen")]
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm
        server = VLLMServer("test")
        results = server.batch_generate(["p1", "p2"], batch_size=1)
        self.assertEqual(len(results), 2)

    @patch('vllm.LLM')
    def test_generate_returns_list(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = []
        mock_llm_class.return_value = mock_llm
        server = VLLMServer("test")
        results = server.generate([])
        self.assertIsInstance(results, list)

    def test_sampling_params_has_top_k(self):
        params = create_vllm_sampling_params(top_k=50)
        self.assertEqual(params.top_k, 50)
`,

	hint1: 'Use tensor_parallel_size > 1 for multi-GPU inference',
	hint2: 'vLLM automatically handles batching and scheduling',

	whyItMatters: `vLLM is the industry standard for LLM serving:

- **Used by**: Anyscale, Databricks, many startups
- **Performance**: 24x throughput improvement
- **Memory efficient**: PagedAttention reduces memory waste
- **Easy migration**: OpenAI-compatible API

Essential for production LLM deployments.`,

	translations: {
		ru: {
			title: 'vLLM высокопроизводительный сервинг',
			description: `# vLLM высокопроизводительный сервинг

Используйте vLLM для высокопроизводительного инференса LLM.

## Что такое vLLM?

vLLM предоставляет:
- **PagedAttention**: Эффективное управление памятью
- **Continuous batching**: Динамическое планирование батчей
- **Tensor parallelism**: Поддержка нескольких GPU
- **OpenAI-совместимый API**: Прямая замена

## Пример

\`\`\`python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
\`\`\``,
			hint1: 'Используйте tensor_parallel_size > 1 для инференса на нескольких GPU',
			hint2: 'vLLM автоматически обрабатывает батчинг и планирование',
			whyItMatters: `vLLM - индустриальный стандарт сервинга LLM:

- **Используется**: Anyscale, Databricks, многие стартапы
- **Производительность**: 24x улучшение пропускной способности
- **Эффективность памяти**: PagedAttention уменьшает потери памяти
- **Легкая миграция**: OpenAI-совместимый API`,
		},
		uz: {
			title: 'vLLM yuqori samarali serving',
			description: `# vLLM yuqori samarali serving

Yuqori o'tkazuvchanlikli LLM inference uchun vLLM dan foydalaning.

## vLLM nima?

vLLM taqdim etadi:
- **PagedAttention**: Samarali xotira boshqaruvi
- **Continuous batching**: Dinamik batch rejalashtirish
- **Tensor parallelism**: Ko'p GPU qo'llab-quvvatlash
- **OpenAI-mos API**: To'g'ridan-to'g'ri almashtirish

## Misol

\`\`\`python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
\`\`\``,
			hint1: "Ko'p GPU inference uchun tensor_parallel_size > 1 dan foydalaning",
			hint2: "vLLM avtomatik ravishda batching va rejalashtirishni boshqaradi",
			whyItMatters: `vLLM LLM serving uchun sanoat standarti:

- **Foydalanadi**: Anyscale, Databricks, ko'p startaplar
- **Samaradorlik**: 24x o'tkazuvchanlik yaxshilanishi
- **Xotira samaradorligi**: PagedAttention xotira isrofini kamaytiradi
- **Oson migratsiya**: OpenAI-mos API`,
		},
	},
};

export default task;
