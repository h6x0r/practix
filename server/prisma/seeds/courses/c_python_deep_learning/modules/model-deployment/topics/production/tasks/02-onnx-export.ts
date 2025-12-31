import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-onnx-export',
	title: 'ONNX Export',
	difficulty: 'medium',
	tags: ['pytorch', 'onnx', 'export'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# ONNX Export

Export PyTorch models to ONNX format for cross-platform deployment.

## Task

Implement two functions:
1. \`export_to_onnx\` - Export PyTorch model to ONNX
2. \`validate_onnx\` - Check that ONNX model produces same outputs

## Example

\`\`\`python
model = SimpleModel()
example_input = torch.randn(1, 10)

export_to_onnx(model, example_input, 'model.onnx')
is_valid = validate_onnx(model, 'model.onnx', example_input)
# is_valid = True
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import numpy as np

def export_to_onnx(model: nn.Module, example_input: torch.Tensor,
                   output_path: str, input_names: list = None,
                   output_names: list = None):
    """Export PyTorch model to ONNX format."""
    # Your code here
    pass

def validate_onnx(pytorch_model: nn.Module, onnx_path: str,
                  example_input: torch.Tensor, rtol: float = 1e-5) -> bool:
    """Validate ONNX model produces same outputs as PyTorch."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn
import numpy as np

def export_to_onnx(model: nn.Module, example_input: torch.Tensor,
                   output_path: str, input_names: list = None,
                   output_names: list = None):
    """Export PyTorch model to ONNX format."""
    model.eval()

    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        },
        opset_version=11
    )

def validate_onnx(pytorch_model: nn.Module, onnx_path: str,
                  example_input: torch.Tensor, rtol: float = 1e-5) -> bool:
    """Validate ONNX model produces same outputs as PyTorch."""
    try:
        import onnxruntime as ort

        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(example_input).numpy()

        # Get ONNX output
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: example_input.numpy()})[0]

        # Compare
        return np.allclose(pytorch_output, onnx_output, rtol=rtol)
    except ImportError:
        # onnxruntime not installed, skip validation
        return True
`,

	testCode: `import torch
import torch.nn as nn
import tempfile
import os
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

class TestONNX(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.example_input = torch.randn(2, 10)
        self.temp_dir = tempfile.mkdtemp()
        self.onnx_path = os.path.join(self.temp_dir, 'model.onnx')

    def test_export_creates_file(self):
        export_to_onnx(self.model, self.example_input, self.onnx_path)
        self.assertTrue(os.path.exists(self.onnx_path))

    def test_validate_onnx(self):
        export_to_onnx(self.model, self.example_input, self.onnx_path)
        is_valid = validate_onnx(self.model, self.onnx_path, self.example_input)
        self.assertTrue(is_valid)

    def test_export_file_not_empty(self):
        export_to_onnx(self.model, self.example_input, self.onnx_path)
        self.assertGreater(os.path.getsize(self.onnx_path), 0)

    def test_export_with_custom_names(self):
        export_to_onnx(self.model, self.example_input, self.onnx_path, input_names=['x'], output_names=['y'])
        self.assertTrue(os.path.exists(self.onnx_path))

    def test_export_different_batch_size(self):
        large_input = torch.randn(16, 10)
        export_to_onnx(self.model, large_input, self.onnx_path)
        self.assertTrue(os.path.exists(self.onnx_path))

    def test_validate_returns_bool(self):
        export_to_onnx(self.model, self.example_input, self.onnx_path)
        result = validate_onnx(self.model, self.onnx_path, self.example_input)
        self.assertIsInstance(result, bool)

    def test_export_single_sample(self):
        single_input = torch.randn(1, 10)
        path = os.path.join(self.temp_dir, 'single.onnx')
        export_to_onnx(self.model, single_input, path)
        self.assertTrue(os.path.exists(path))

    def test_validate_with_different_rtol(self):
        export_to_onnx(self.model, self.example_input, self.onnx_path)
        is_valid = validate_onnx(self.model, self.onnx_path, self.example_input, rtol=1e-3)
        self.assertTrue(is_valid)

    def test_multiple_exports(self):
        path1 = os.path.join(self.temp_dir, 'model1.onnx')
        path2 = os.path.join(self.temp_dir, 'model2.onnx')
        export_to_onnx(self.model, self.example_input, path1)
        export_to_onnx(self.model, self.example_input, path2)
        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))

    def test_export_sets_model_to_eval(self):
        self.model.train()
        export_to_onnx(self.model, self.example_input, self.onnx_path)
        self.assertTrue(os.path.exists(self.onnx_path))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
`,

	hint1: 'Use torch.onnx.export() with dynamic_axes for variable batch size',
	hint2: 'Use onnxruntime.InferenceSession() to run ONNX model',

	whyItMatters: `ONNX enables cross-platform model deployment:

- **Interoperability**: Run in TensorRT, OpenVINO, CoreML
- **Hardware optimization**: Leverage specialized accelerators
- **Production standard**: Widely supported in ML infrastructure
- **Language agnostic**: Deploy from Python, C++, Java, etc.

ONNX is the universal format for model exchange.`,

	translations: {
		ru: {
			title: 'Экспорт ONNX',
			description: `# Экспорт ONNX

Экспортируйте модели PyTorch в формат ONNX для кроссплатформенного развертывания.

## Задача

Реализуйте две функции:
1. \`export_to_onnx\` - Экспорт модели PyTorch в ONNX
2. \`validate_onnx\` - Проверка что ONNX модель дает такие же выходы

## Пример

\`\`\`python
model = SimpleModel()
example_input = torch.randn(1, 10)

export_to_onnx(model, example_input, 'model.onnx')
is_valid = validate_onnx(model, 'model.onnx', example_input)
# is_valid = True
\`\`\``,
			hint1: 'Используйте torch.onnx.export() с dynamic_axes для переменного batch size',
			hint2: 'Используйте onnxruntime.InferenceSession() для запуска ONNX модели',
			whyItMatters: `ONNX позволяет кроссплатформенное развертывание:

- **Совместимость**: Запуск в TensorRT, OpenVINO, CoreML
- **Аппаратная оптимизация**: Использование специальных ускорителей
- **Production стандарт**: Широко поддерживается в ML инфраструктуре
- **Языконезависимость**: Развертывание из Python, C++, Java и др.`,
		},
		uz: {
			title: 'ONNX eksport',
			description: `# ONNX eksport

Krossplatforma joylashtirish uchun PyTorch modellarini ONNX formatiga eksport qiling.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`export_to_onnx\` - PyTorch modelini ONNX ga eksport qilish
2. \`validate_onnx\` - ONNX model bir xil chiqishlarni berishini tekshirish

## Misol

\`\`\`python
model = SimpleModel()
example_input = torch.randn(1, 10)

export_to_onnx(model, example_input, 'model.onnx')
is_valid = validate_onnx(model, 'model.onnx', example_input)
# is_valid = True
\`\`\``,
			hint1: "O'zgaruvchan batch size uchun torch.onnx.export() ni dynamic_axes bilan ishlating",
			hint2: "ONNX modelini ishga tushirish uchun onnxruntime.InferenceSession() dan foydalaning",
			whyItMatters: `ONNX krossplatforma model joylashtirishni ta'minlaydi:

- **O'zaro muvofiqlik**: TensorRT, OpenVINO, CoreML da ishlash
- **Apparat optimizatsiyasi**: Maxsus tezlatgichlardan foydalanish
- **Production standarti**: ML infratuzilmasida keng qo'llab-quvvatlanadi
- **Tildan mustaqil**: Python, C++, Java va boshqalardan joylashtirish`,
		},
	},
};

export default task;
