import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-adapter-merging',
	title: 'Adapter Merging',
	difficulty: 'medium',
	tags: ['peft', 'lora', 'inference'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Adapter Merging

Learn to merge LoRA adapters with base model for efficient inference.

## Why Merge?

During training:
- Keep adapter separate for flexibility
- Easy to swap or combine adapters

For inference:
- Merge for zero additional latency
- Single model file for deployment

## Task

Implement functions to:
- Merge LoRA weights into base model
- Unmerge (revert to adapter mode)
- Combine multiple adapters

## Example

\`\`\`python
# Merge for inference
model = merge_adapter(model)
model.save_pretrained("merged_model")

# Or keep separate
model.merge_and_unload()  # Built-in PEFT method
\`\`\``,

	initialCode: `import torch
from peft import PeftModel

def merge_adapter(model: PeftModel) -> torch.nn.Module:
    """Merge LoRA adapter into base model."""
    # Your code here
    pass

def unmerge_adapter(model: PeftModel):
    """Unmerge adapter from base model."""
    # Your code here
    pass

def merge_and_save(model: PeftModel, save_path: str):
    """Merge adapter and save as regular model."""
    # Your code here
    pass

def load_multiple_adapters(base_model, adapter_paths: list) -> PeftModel:
    """Load multiple adapters onto same base model."""
    # Your code here
    pass

def combine_adapters(model: PeftModel, adapter_names: list,
                     weights: list) -> PeftModel:
    """Combine multiple adapters with specified weights."""
    # Your code here
    pass
`,

	solutionCode: `import torch
from peft import PeftModel

def merge_adapter(model: PeftModel) -> torch.nn.Module:
    """Merge LoRA adapter into base model."""
    # Merge LoRA weights into base model
    model = model.merge_and_unload()
    return model

def unmerge_adapter(model: PeftModel):
    """Unmerge adapter from base model."""
    # This only works if merge hasn't been called yet
    # or if the model was created with allow_merge=True
    model.unmerge_adapter()
    return model

def merge_and_save(model: PeftModel, save_path: str):
    """Merge adapter and save as regular model."""
    # Merge weights
    merged_model = model.merge_and_unload()

    # Save as regular HuggingFace model
    merged_model.save_pretrained(save_path)

    return merged_model

def load_multiple_adapters(base_model, adapter_paths: list) -> PeftModel:
    """Load multiple adapters onto same base model."""
    # Load first adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name="adapter_0"
    )

    # Load additional adapters
    for i, path in enumerate(adapter_paths[1:], 1):
        model.load_adapter(path, adapter_name=f"adapter_{i}")

    return model

def combine_adapters(model: PeftModel, adapter_names: list,
                     weights: list) -> PeftModel:
    """Combine multiple adapters with specified weights."""
    # Create weighted combination
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="combined",
        combination_type="linear"
    )

    # Set combined adapter as active
    model.set_adapter("combined")

    return model
`,

	testCode: `import torch
import unittest
from unittest.mock import MagicMock, patch

class TestAdapterMerging(unittest.TestCase):
    def test_merge_adapter(self):
        mock_model = MagicMock()
        mock_model.merge_and_unload.return_value = MagicMock()
        result = merge_adapter(mock_model)
        mock_model.merge_and_unload.assert_called_once()

    def test_unmerge_adapter(self):
        mock_model = MagicMock()
        result = unmerge_adapter(mock_model)
        mock_model.unmerge_adapter.assert_called_once()

    def test_merge_and_save(self):
        mock_model = MagicMock()
        merged = MagicMock()
        mock_model.merge_and_unload.return_value = merged
        result = merge_and_save(mock_model, "./output")
        merged.save_pretrained.assert_called_once_with("./output")

    @patch('peft.PeftModel.from_pretrained')
    def test_load_multiple_adapters(self, mock_from_pretrained):
        mock_peft = MagicMock()
        mock_from_pretrained.return_value = mock_peft
        base_model = MagicMock()
        result = load_multiple_adapters(base_model, ["path1", "path2"])
        mock_peft.load_adapter.assert_called_once()

    def test_combine_adapters(self):
        mock_model = MagicMock()
        result = combine_adapters(mock_model, ["a1", "a2"], [0.5, 0.5])
        mock_model.add_weighted_adapter.assert_called_once()
        mock_model.set_adapter.assert_called_with("combined")

    def test_merge_adapter_returns_model(self):
        mock_model = MagicMock()
        merged = MagicMock()
        mock_model.merge_and_unload.return_value = merged
        result = merge_adapter(mock_model)
        self.assertEqual(result, merged)

    def test_unmerge_returns_model(self):
        mock_model = MagicMock()
        result = unmerge_adapter(mock_model)
        self.assertIsNotNone(result)

    def test_merge_and_save_returns_merged(self):
        mock_model = MagicMock()
        merged = MagicMock()
        mock_model.merge_and_unload.return_value = merged
        result = merge_and_save(mock_model, "./path")
        self.assertEqual(result, merged)

    @patch('peft.PeftModel.from_pretrained')
    def test_load_multiple_returns_peft_model(self, mock_from_pretrained):
        mock_peft = MagicMock()
        mock_from_pretrained.return_value = mock_peft
        result = load_multiple_adapters(MagicMock(), ["path1"])
        self.assertEqual(result, mock_peft)

    def test_combine_returns_model(self):
        mock_model = MagicMock()
        result = combine_adapters(mock_model, ["a"], [1.0])
        self.assertIsNotNone(result)
`,

	hint1: 'merge_and_unload() returns a regular nn.Module without PEFT wrapper',
	hint2: 'Use add_weighted_adapter to combine adapters with custom weights',

	whyItMatters: `Adapter management enables flexible deployment:

- **Merged inference**: Zero latency overhead
- **Multi-task**: Switch adapters at runtime
- **Adapter arithmetic**: Combine skills from different adapters
- **A/B testing**: Compare adapter versions easily

Key technique for production LLM systems.`,

	translations: {
		ru: {
			title: 'Слияние адаптеров',
			description: `# Слияние адаптеров

Научитесь объединять LoRA адаптеры с базовой моделью для эффективного инференса.

## Зачем объединять?

При обучении:
- Держите адаптер отдельно для гибкости
- Легко менять или комбинировать адаптеры

Для инференса:
- Объедините для нулевой дополнительной задержки
- Один файл модели для деплоя

## Пример

\`\`\`python
# Merge for inference
model = merge_adapter(model)
model.save_pretrained("merged_model")

# Or keep separate
model.merge_and_unload()  # Built-in PEFT method
\`\`\``,
			hint1: 'merge_and_unload() возвращает обычный nn.Module без PEFT обертки',
			hint2: 'Используйте add_weighted_adapter для комбинирования адаптеров с весами',
			whyItMatters: `Управление адаптерами обеспечивает гибкий деплой:

- **Merged inference**: Нулевые дополнительные задержки
- **Multi-task**: Переключение адаптеров во время работы
- **Арифметика адаптеров**: Комбинирование навыков из разных адаптеров
- **A/B тестирование**: Легкое сравнение версий адаптеров`,
		},
		uz: {
			title: 'Adapterlarni birlashtirish',
			description: `# Adapterlarni birlashtirish

Samarali inference uchun LoRA adapterlarni bazaviy model bilan birlashtirishni o'rganing.

## Nega birlashtirish?

O'qitish vaqtida:
- Moslashuvchanlik uchun adapterni alohida saqlang
- Adapterlarni oson almashtirish yoki birlashtirish

Inference uchun:
- Nol qo'shimcha kechikish uchun birlashtiring
- Deploy uchun bitta model fayli

## Misol

\`\`\`python
# Merge for inference
model = merge_adapter(model)
model.save_pretrained("merged_model")

# Or keep separate
model.merge_and_unload()  # Built-in PEFT method
\`\`\``,
			hint1: "merge_and_unload() PEFT wrapper siz oddiy nn.Module qaytaradi",
			hint2: "Adapterlarni maxsus og'irliklar bilan birlashtirish uchun add_weighted_adapter dan foydalaning",
			whyItMatters: `Adapter boshqaruvi moslashuvchan deployni ta'minlaydi:

- **Birlashtirilgan inference**: Nol kechikish qo'shimchasi
- **Multi-task**: Ish vaqtida adapterlarni almashtirish
- **Adapter arifmetikasi**: Turli adapterlardan ko'nikmalarni birlashtirish
- **A/B test**: Adapter versiyalarini oson solishtirish`,
		},
	},
};

export default task;
