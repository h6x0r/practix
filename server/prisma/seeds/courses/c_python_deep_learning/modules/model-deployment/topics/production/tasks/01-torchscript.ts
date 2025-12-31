import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-torchscript-export',
	title: 'TorchScript Export',
	difficulty: 'medium',
	tags: ['pytorch', 'torchscript', 'export'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# TorchScript Export

Learn to export PyTorch models to TorchScript for production deployment.

## Task

Implement two functions:
1. \`export_traced\` - Export model using tracing
2. \`export_scripted\` - Export model using scripting

## Example

\`\`\`python
model = SimpleModel()

traced = export_traced(model, torch.randn(1, 10))
scripted = export_scripted(model)

# Both can be saved and loaded without Python
traced.save('model_traced.pt')
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def export_traced(model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
    """Export model using tracing."""
    # Your code here
    pass

def export_scripted(model: nn.Module) -> torch.jit.ScriptModule:
    """Export model using scripting."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def export_traced(model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
    """Export model using tracing."""
    model.eval()
    traced = torch.jit.trace(model, example_input)
    return traced

def export_scripted(model: nn.Module) -> torch.jit.ScriptModule:
    """Export model using scripting."""
    model.eval()
    scripted = torch.jit.script(model)
    return scripted
`,

	testCode: `import torch
import torch.nn as nn
import tempfile
import os
import unittest

class TestTorchScript(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.example_input = torch.randn(2, 10)

    def test_trace_export(self):
        traced = export_traced(self.model, self.example_input)
        self.assertIsInstance(traced, torch.jit.ScriptModule)

    def test_script_export(self):
        scripted = export_scripted(self.model)
        self.assertIsInstance(scripted, torch.jit.ScriptModule)

    def test_traced_output(self):
        traced = export_traced(self.model, self.example_input)
        with torch.no_grad():
            original_out = self.model(self.example_input)
            traced_out = traced(self.example_input)
        self.assertTrue(torch.allclose(original_out, traced_out))

    def test_save_load(self):
        traced = export_traced(self.model, self.example_input)
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            traced.save(f.name)
            loaded = torch.jit.load(f.name)
            self.assertIsInstance(loaded, torch.jit.ScriptModule)

    def test_scripted_output(self):
        scripted = export_scripted(self.model)
        with torch.no_grad():
            original_out = self.model(self.example_input)
            scripted_out = scripted(self.example_input)
        self.assertTrue(torch.allclose(original_out, scripted_out))

    def test_traced_different_batch(self):
        traced = export_traced(self.model, self.example_input)
        new_input = torch.randn(4, 10)
        out = traced(new_input)
        self.assertEqual(out.shape, (4, 5))

    def test_scripted_different_batch(self):
        scripted = export_scripted(self.model)
        new_input = torch.randn(8, 10)
        out = scripted(new_input)
        self.assertEqual(out.shape, (8, 5))

    def test_traced_single_sample(self):
        single_input = torch.randn(1, 10)
        traced = export_traced(self.model, single_input)
        self.assertIsInstance(traced, torch.jit.ScriptModule)

    def test_scripted_callable(self):
        scripted = export_scripted(self.model)
        self.assertTrue(callable(scripted))

    def test_traced_output_not_nan(self):
        traced = export_traced(self.model, self.example_input)
        out = traced(self.example_input)
        self.assertFalse(torch.isnan(out).any())
`,

	hint1: 'Use torch.jit.trace(model, example_input) for tracing',
	hint2: 'Use torch.jit.script(model) for scripting',

	whyItMatters: `TorchScript is essential for production deployment:

- **No Python needed**: Run without Python interpreter
- **Optimization**: JIT compilation improves speed
- **Serialization**: Save entire model as single file
- **C++ deployment**: Load in C++ for embedded/mobile

TorchScript is the standard way to deploy PyTorch models.`,

	translations: {
		ru: {
			title: 'Экспорт TorchScript',
			description: `# Экспорт TorchScript

Научитесь экспортировать модели PyTorch в TorchScript для production.

## Задача

Реализуйте две функции:
1. \`export_traced\` - Экспорт модели через tracing
2. \`export_scripted\` - Экспорт модели через scripting

## Пример

\`\`\`python
model = SimpleModel()

traced = export_traced(model, torch.randn(1, 10))
scripted = export_scripted(model)

# Both can be saved and loaded without Python
traced.save('model_traced.pt')
\`\`\``,
			hint1: 'Используйте torch.jit.trace(model, example_input) для tracing',
			hint2: 'Используйте torch.jit.script(model) для scripting',
			whyItMatters: `TorchScript необходим для production развертывания:

- **Без Python**: Запуск без интерпретатора Python
- **Оптимизация**: JIT компиляция улучшает скорость
- **Сериализация**: Сохранение модели как единый файл
- **C++ развертывание**: Загрузка в C++ для embedded/mobile`,
		},
		uz: {
			title: 'TorchScript eksport',
			description: `# TorchScript eksport

Production uchun PyTorch modellarini TorchScript ga eksport qilishni o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`export_traced\` - Tracing orqali modelni eksport qilish
2. \`export_scripted\` - Scripting orqali modelni eksport qilish

## Misol

\`\`\`python
model = SimpleModel()

traced = export_traced(model, torch.randn(1, 10))
scripted = export_scripted(model)

# Both can be saved and loaded without Python
traced.save('model_traced.pt')
\`\`\``,
			hint1: 'Tracing uchun torch.jit.trace(model, example_input) dan foydalaning',
			hint2: 'Scripting uchun torch.jit.script(model) dan foydalaning',
			whyItMatters: `TorchScript production joylashtirishida muhim:

- **Python kerak emas**: Python interpretatorisiz ishlash
- **Optimallashtirish**: JIT kompilyatsiya tezlikni yaxshilaydi
- **Serializatsiya**: Butun modelni bitta fayl sifatida saqlash
- **C++ joylashtirish**: Embedded/mobile uchun C++ da yuklash`,
		},
	},
};

export default task;
