import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-model-quantization',
	title: 'Model Quantization',
	difficulty: 'hard',
	tags: ['quantization', 'gptq', 'awq'],
	estimatedTime: '25m',
	isPremium: true,
	order: 3,
	description: `# Model Quantization

Reduce model size and speed up inference with quantization.

## Quantization Types

- **GPTQ**: Post-training quantization, 4-bit
- **AWQ**: Activation-aware quantization
- **GGUF/GGML**: CPU-friendly formats (llama.cpp)
- **bitsandbytes**: Dynamic quantization

## Trade-offs

| Method | Size Reduction | Quality Loss | Speed |
|--------|---------------|--------------|-------|
| FP16   | 2x            | None         | Fast  |
| INT8   | 4x            | Minimal      | Fast  |
| INT4   | 8x            | Small        | Medium|
| GPTQ   | 8x            | Very Small   | Fast  |

## Example

\`\`\`python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)
\`\`\``,

	initialCode: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_8bit_model(model_name: str):
    """Load model with 8-bit quantization."""
    # Your code here
    pass

def load_4bit_model(model_name: str, compute_dtype=torch.float16):
    """Load model with 4-bit quantization."""
    # Your code here
    pass

def load_gptq_model(model_name: str):
    """Load a GPTQ quantized model."""
    # Your code here
    pass

def compare_model_sizes(fp16_model, quantized_model) -> dict:
    """Compare memory usage between models."""
    # Your code here
    pass

def benchmark_inference(model, tokenizer, prompts: list) -> dict:
    """Benchmark inference speed."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig

def load_8bit_model(model_name: str):
    """Load model with 8-bit quantization."""
    config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto"
    )

    return model

def load_4bit_model(model_name: str, compute_dtype=torch.float16):
    """Load model with 4-bit quantization."""
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto"
    )

    return model

def load_gptq_model(model_name: str):
    """Load a GPTQ quantized model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )

    return model

def compare_model_sizes(fp16_model, quantized_model) -> dict:
    """Compare memory usage between models."""
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return (param_size + buffer_size) / 1024**3  # GB

    fp16_size = get_model_size(fp16_model)
    quantized_size = get_model_size(quantized_model)

    return {
        "fp16_size_gb": fp16_size,
        "quantized_size_gb": quantized_size,
        "reduction_factor": fp16_size / quantized_size if quantized_size > 0 else 0,
        "savings_gb": fp16_size - quantized_size
    }

def benchmark_inference(model, tokenizer, prompts: list) -> dict:
    """Benchmark inference speed."""
    model.eval()

    # Warmup
    with torch.no_grad():
        inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    total_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50)
            total_tokens += outputs.shape[1]

    end_time = time.time()
    elapsed = end_time - start_time

    return {
        "total_time": elapsed,
        "prompts_processed": len(prompts),
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed,
        "latency_per_prompt": elapsed / len(prompts)
    }
`,

	testCode: `import torch
import unittest
from unittest.mock import MagicMock, patch

class TestQuantization(unittest.TestCase):
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_8bit_model(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        model = load_8bit_model("test-model")
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertTrue(call_kwargs["quantization_config"].load_in_8bit)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_4bit_model(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        model = load_4bit_model("test-model")
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertTrue(call_kwargs["quantization_config"].load_in_4bit)

    def test_compare_model_sizes(self):
        # Create mock models with known sizes
        mock_fp16 = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 1000000
        mock_param.element_size.return_value = 2  # FP16
        mock_fp16.parameters.return_value = [mock_param]
        mock_fp16.buffers.return_value = []

        mock_quant = MagicMock()
        mock_param_q = MagicMock()
        mock_param_q.numel.return_value = 1000000
        mock_param_q.element_size.return_value = 1  # INT8
        mock_quant.parameters.return_value = [mock_param_q]
        mock_quant.buffers.return_value = []

        result = compare_model_sizes(mock_fp16, mock_quant)
        self.assertIn("reduction_factor", result)
        self.assertGreater(result["reduction_factor"], 1)

    def test_compare_sizes_returns_dict(self):
        mock1 = MagicMock()
        mock1.parameters.return_value = []
        mock1.buffers.return_value = []
        mock2 = MagicMock()
        mock2.parameters.return_value = []
        mock2.buffers.return_value = []
        result = compare_model_sizes(mock1, mock2)
        self.assertIsInstance(result, dict)

    def test_compare_sizes_has_fp16_size(self):
        mock1 = MagicMock()
        mock1.parameters.return_value = []
        mock1.buffers.return_value = []
        mock2 = MagicMock()
        mock2.parameters.return_value = []
        mock2.buffers.return_value = []
        result = compare_model_sizes(mock1, mock2)
        self.assertIn("fp16_size_gb", result)

    def test_compare_sizes_has_savings(self):
        mock1 = MagicMock()
        mock1.parameters.return_value = []
        mock1.buffers.return_value = []
        mock2 = MagicMock()
        mock2.parameters.return_value = []
        mock2.buffers.return_value = []
        result = compare_model_sizes(mock1, mock2)
        self.assertIn("savings_gb", result)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_4bit_uses_nf4(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        load_4bit_model("test")
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["quantization_config"].bnb_4bit_quant_type, "nf4")

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_gptq_model(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        model = load_gptq_model("test-gptq")
        mock_from_pretrained.assert_called_once()

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_8bit_uses_auto_device(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        load_8bit_model("test")
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["device_map"], "auto")

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_4bit_double_quant(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        load_4bit_model("test")
        call_kwargs = mock_from_pretrained.call_args[1]
        self.assertTrue(call_kwargs["quantization_config"].bnb_4bit_use_double_quant)
`,

	hint1: 'Use nf4 quantization type for best quality with 4-bit',
	hint2: 'GPTQ models are pre-quantized and ready to use',

	whyItMatters: `Quantization makes large models accessible:

- **7B on consumer GPUs**: Run LLaMA-7B on RTX 3080
- **Cost reduction**: Smaller models = cheaper inference
- **Faster inference**: Less memory movement
- **Mobile deployment**: Run on edge devices

Essential for practical LLM deployment.`,

	translations: {
		ru: {
			title: 'Квантизация моделей',
			description: `# Квантизация моделей

Уменьшите размер модели и ускорьте инференс с помощью квантизации.

## Типы квантизации

- **GPTQ**: Post-training квантизация, 4-bit
- **AWQ**: Activation-aware квантизация
- **GGUF/GGML**: CPU-friendly форматы (llama.cpp)
- **bitsandbytes**: Динамическая квантизация

## Пример

\`\`\`python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)
\`\`\``,
			hint1: 'Используйте nf4 тип квантизации для лучшего качества с 4-bit',
			hint2: 'GPTQ модели предварительно квантизированы и готовы к использованию',
			whyItMatters: `Квантизация делает большие модели доступными:

- **7B на потребительских GPU**: LLaMA-7B на RTX 3080
- **Снижение стоимости**: Меньшие модели = дешевле инференс
- **Быстрее инференс**: Меньше перемещения памяти
- **Mobile деплой**: Работа на edge устройствах`,
		},
		uz: {
			title: 'Model kvantizatsiyasi',
			description: `# Model kvantizatsiyasi

Kvantizatsiya bilan model hajmini kamaytiring va inference ni tezlashtiring.

## Kvantizatsiya turlari

- **GPTQ**: Post-training kvantizatsiya, 4-bit
- **AWQ**: Activation-aware kvantizatsiya
- **GGUF/GGML**: CPU-do'st formatlar (llama.cpp)
- **bitsandbytes**: Dinamik kvantizatsiya

## Misol

\`\`\`python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)
\`\`\``,
			hint1: "4-bit bilan eng yaxshi sifat uchun nf4 kvantizatsiya turidan foydalaning",
			hint2: "GPTQ modellari oldindan kvantizatsiya qilingan va foydalanishga tayyor",
			whyItMatters: `Kvantizatsiya katta modellarni qulay qiladi:

- **Iste'molchi GPU larida 7B**: RTX 3080 da LLaMA-7B
- **Xarajat kamayishi**: Kichikroq modellar = arzonroq inference
- **Tezroq inference**: Kamroq xotira harakati
- **Mobil deploy**: Edge qurilmalarida ishlash`,
		},
	},
};

export default task;
