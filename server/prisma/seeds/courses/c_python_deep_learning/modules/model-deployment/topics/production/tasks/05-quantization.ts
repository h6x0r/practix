import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-model-quantization',
	title: 'Model Quantization',
	difficulty: 'hard',
	tags: ['pytorch', 'quantization', 'optimization'],
	estimatedTime: '18m',
	isPremium: true,
	order: 5,
	description: `# Model Quantization

Reduce model size and improve inference speed with quantization.

## Task

Implement three functions:
1. \`dynamic_quantize\` - Apply dynamic quantization
2. \`static_quantize\` - Apply static quantization with calibration
3. \`compare_models\` - Compare original vs quantized model size and speed

## Example

\`\`\`python
model = SimpleModel()
calibration_data = torch.randn(100, 10)

quantized = dynamic_quantize(model)
static_quantized = static_quantize(model, calibration_data)

stats = compare_models(model, quantized, torch.randn(10, 10))
# {'size_reduction': 0.75, 'speedup': 2.1}
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.quantization as quant
import time
import os
import tempfile
from typing import Dict

def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to model."""
    # Your code here
    pass

def get_model_size(model: nn.Module) -> int:
    """Get model size in bytes."""
    # Your code here
    pass

def compare_models(original: nn.Module, quantized: nn.Module,
                   test_input: torch.Tensor, num_runs: int = 100) -> Dict:
    """Compare original and quantized models."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.quantization as quant
import time
import os
import tempfile
from typing import Dict

def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to model."""
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8
    )
    return quantized_model

def get_model_size(model: nn.Module) -> int:
    """Get model size in bytes."""
    with tempfile.NamedTemporaryFile() as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name)

def compare_models(original: nn.Module, quantized: nn.Module,
                   test_input: torch.Tensor, num_runs: int = 100) -> Dict:
    """Compare original and quantized models."""
    original.eval()
    quantized.eval()

    # Size comparison
    original_size = get_model_size(original)
    quantized_size = get_model_size(quantized)
    size_reduction = 1 - (quantized_size / original_size)

    # Speed comparison
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            original(test_input)
            quantized(test_input)

        # Time original
        start = time.time()
        for _ in range(num_runs):
            original(test_input)
        original_time = time.time() - start

        # Time quantized
        start = time.time()
        for _ in range(num_runs):
            quantized(test_input)
        quantized_time = time.time() - start

    speedup = original_time / quantized_time if quantized_time > 0 else 1.0

    return {
        'original_size_bytes': original_size,
        'quantized_size_bytes': quantized_size,
        'size_reduction': size_reduction,
        'original_time': original_time,
        'quantized_time': quantized_time,
        'speedup': speedup
    }
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.test_input = torch.randn(4, 100)

    def test_dynamic_quantize(self):
        quantized = dynamic_quantize(self.model)
        self.assertIsNotNone(quantized)
        # Should still work
        output = quantized(self.test_input)
        self.assertEqual(output.shape, (4, 10))

    def test_compare_models(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=10)
        self.assertIn('size_reduction', stats)
        self.assertIn('speedup', stats)
        # Quantized should be smaller
        self.assertGreater(stats['size_reduction'], 0)

    def test_quantized_model_type(self):
        quantized = dynamic_quantize(self.model)
        self.assertIsInstance(quantized, nn.Module)

    def test_compare_returns_dict(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=5)
        self.assertIsInstance(stats, dict)

    def test_compare_has_original_size(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=5)
        self.assertIn('original_size_bytes', stats)

    def test_compare_has_quantized_size(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=5)
        self.assertIn('quantized_size_bytes', stats)

    def test_compare_has_times(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=5)
        self.assertIn('original_time', stats)
        self.assertIn('quantized_time', stats)

    def test_quantized_smaller(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=5)
        self.assertLess(stats['quantized_size_bytes'], stats['original_size_bytes'])

    def test_speedup_positive(self):
        quantized = dynamic_quantize(self.model)
        stats = compare_models(self.model, quantized, self.test_input, num_runs=5)
        self.assertGreater(stats['speedup'], 0)

    def test_get_model_size(self):
        size = get_model_size(self.model)
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)
`,

	hint1: 'quantize_dynamic targets Linear, LSTM, GRU layers',
	hint2: 'Save model to temp file to measure size',

	whyItMatters: `Quantization reduces deployment costs:

- **4x smaller**: INT8 vs FP32 weights
- **2-4x faster**: Integer operations are faster
- **Edge deployment**: Enables running on mobile/IoT
- **Lower costs**: Less memory, compute, and bandwidth

Quantization is essential for efficient production deployment.`,

	translations: {
		ru: {
			title: 'Квантизация модели',
			description: `# Квантизация модели

Уменьшите размер модели и улучшите скорость инференса с помощью квантизации.

## Задача

Реализуйте три функции:
1. \`dynamic_quantize\` - Применение динамической квантизации
2. \`static_quantize\` - Применение статической квантизации с калибровкой
3. \`compare_models\` - Сравнение размера и скорости оригинала и квантизированной модели

## Пример

\`\`\`python
model = SimpleModel()
calibration_data = torch.randn(100, 10)

quantized = dynamic_quantize(model)
static_quantized = static_quantize(model, calibration_data)

stats = compare_models(model, quantized, torch.randn(10, 10))
# {'size_reduction': 0.75, 'speedup': 2.1}
\`\`\``,
			hint1: 'quantize_dynamic нацелен на слои Linear, LSTM, GRU',
			hint2: 'Сохраните модель во временный файл для измерения размера',
			whyItMatters: `Квантизация снижает затраты на развертывание:

- **В 4 раза меньше**: INT8 вместо FP32 весов
- **В 2-4 раза быстрее**: Целочисленные операции быстрее
- **Edge развертывание**: Запуск на мобильных/IoT устройствах
- **Снижение затрат**: Меньше памяти, вычислений и трафика`,
		},
		uz: {
			title: 'Model kvantizatsiyasi',
			description: `# Model kvantizatsiyasi

Kvantizatsiya yordamida model hajmini kamaytiring va inference tezligini yaxshilang.

## Topshiriq

Uchta funksiya amalga oshiring:
1. \`dynamic_quantize\` - Dinamik kvantizatsiyani qo'llash
2. \`static_quantize\` - Kalibrlash bilan statik kvantizatsiyani qo'llash
3. \`compare_models\` - Original va kvantizatsiya qilingan model hajmi va tezligini solishtirish

## Misol

\`\`\`python
model = SimpleModel()
calibration_data = torch.randn(100, 10)

quantized = dynamic_quantize(model)
static_quantized = static_quantize(model, calibration_data)

stats = compare_models(model, quantized, torch.randn(10, 10))
# {'size_reduction': 0.75, 'speedup': 2.1}
\`\`\``,
			hint1: 'quantize_dynamic Linear, LSTM, GRU qatlamlarini nishonga oladi',
			hint2: "O'lchamni o'lchash uchun modelni vaqtinchalik faylga saqlang",
			whyItMatters: `Kvantizatsiya joylashtirish xarajatlarini kamaytiradi:

- **4 marta kichikroq**: FP32 vaznlar o'rniga INT8
- **2-4 marta tezroq**: Butun sonli operatsiyalar tezroq
- **Edge joylashtirish**: Mobil/IoT da ishlash imkoniyati
- **Kam xarajat**: Kamroq xotira, hisoblash va tarmoqli kengligi`,
		},
	},
};

export default task;
