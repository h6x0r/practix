import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-few-shot-prompting',
	title: 'Few-Shot Prompting',
	difficulty: 'medium',
	tags: ['prompting', 'few-shot', 'examples'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Few-Shot Prompting

Use examples to guide LLM behavior for better outputs.

## What is Few-Shot?

- **Zero-shot**: No examples, just instructions
- **One-shot**: Single example
- **Few-shot**: Multiple examples (2-5)

## When to Use

- Complex formatting requirements
- Domain-specific terminology
- Consistent output structure
- Classification tasks

## Example

\`\`\`python
prompt = """
Classify the sentiment of the text.

Text: "I love this product!"
Sentiment: positive

Text: "Terrible experience, never again."
Sentiment: negative

Text: "It's okay, nothing special."
Sentiment: neutral

Text: "Best purchase I've ever made!"
Sentiment:
"""
\`\`\``,

	initialCode: `def create_few_shot_prompt(task: str, examples: list,
                            query: str) -> str:
    """Create a few-shot prompt with examples."""
    # Your code here
    pass

def create_classification_prompt(labels: list, examples: list,
                                  text: str) -> str:
    """Create a few-shot classification prompt."""
    # Your code here
    pass

def create_qa_few_shot(examples: list, question: str) -> str:
    """Create a few-shot Q&A prompt."""
    # Your code here
    pass

def create_formatting_prompt(format_spec: str, examples: list,
                              input_text: str) -> str:
    """Create a prompt to format text in a specific way."""
    # Your code here
    pass
`,

	solutionCode: `def create_few_shot_prompt(task: str, examples: list,
                            query: str) -> str:
    """Create a few-shot prompt with examples."""
    prompt = f"{task}\\n\\n"

    for example in examples:
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        prompt += f"Input: {input_text}\\nOutput: {output_text}\\n\\n"

    prompt += f"Input: {query}\\nOutput:"

    return prompt

def create_classification_prompt(labels: list, examples: list,
                                  text: str) -> str:
    """Create a few-shot classification prompt."""
    labels_str = ", ".join(labels)
    prompt = f"Classify the text into one of these categories: {labels_str}\\n\\n"

    for example in examples:
        prompt += f"Text: {example['text']}\\n"
        prompt += f"Category: {example['label']}\\n\\n"

    prompt += f"Text: {text}\\nCategory:"

    return prompt

def create_qa_few_shot(examples: list, question: str) -> str:
    """Create a few-shot Q&A prompt."""
    prompt = "Answer the question based on your knowledge.\\n\\n"

    for example in examples:
        prompt += f"Question: {example['question']}\\n"
        prompt += f"Answer: {example['answer']}\\n\\n"

    prompt += f"Question: {question}\\nAnswer:"

    return prompt

def create_formatting_prompt(format_spec: str, examples: list,
                              input_text: str) -> str:
    """Create a prompt to format text in a specific way."""
    prompt = f"Format the input according to this specification: {format_spec}\\n\\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\\n"
        prompt += f"Input: {example['input']}\\n"
        prompt += f"Output: {example['output']}\\n\\n"

    prompt += f"Now format this:\\nInput: {input_text}\\nOutput:"

    return prompt
`,

	testCode: `import unittest

class TestFewShotPrompting(unittest.TestCase):
    def test_create_few_shot_prompt(self):
        examples = [
            {"input": "Hello", "output": "Привет"},
            {"input": "Goodbye", "output": "До свидания"}
        ]
        result = create_few_shot_prompt("Translate to Russian", examples, "Thank you")
        self.assertIn("Hello", result)
        self.assertIn("Привет", result)
        self.assertIn("Thank you", result)

    def test_create_classification_prompt(self):
        labels = ["positive", "negative"]
        examples = [{"text": "Great!", "label": "positive"}]
        result = create_classification_prompt(labels, examples, "Bad!")
        self.assertIn("positive, negative", result)
        self.assertIn("Great!", result)
        self.assertIn("Bad!", result)

    def test_create_qa_few_shot(self):
        examples = [{"question": "What is 2+2?", "answer": "4"}]
        result = create_qa_few_shot(examples, "What is 3+3?")
        self.assertIn("What is 2+2?", result)
        self.assertIn("What is 3+3?", result)

    def test_create_formatting_prompt(self):
        examples = [{"input": "john doe", "output": "John Doe"}]
        result = create_formatting_prompt("Title case", examples, "jane smith")
        self.assertIn("john doe", result)
        self.assertIn("Jane Doe", result)
        self.assertIn("jane smith", result)

    def test_few_shot_returns_string(self):
        examples = [{"input": "a", "output": "b"}]
        result = create_few_shot_prompt("Task", examples, "c")
        self.assertIsInstance(result, str)

    def test_classification_includes_labels(self):
        labels = ["cat", "dog", "bird"]
        examples = [{"text": "Meow", "label": "cat"}]
        result = create_classification_prompt(labels, examples, "Bark")
        self.assertIn("cat, dog, bird", result)

    def test_qa_includes_answer_prompt(self):
        examples = [{"question": "Q1?", "answer": "A1"}]
        result = create_qa_few_shot(examples, "Q2?")
        self.assertIn("Answer:", result)

    def test_formatting_includes_example_numbers(self):
        examples = [{"input": "a", "output": "A"}, {"input": "b", "output": "B"}]
        result = create_formatting_prompt("Uppercase", examples, "c")
        self.assertIn("Example 1:", result)
        self.assertIn("Example 2:", result)

    def test_few_shot_ends_with_output(self):
        examples = [{"input": "x", "output": "y"}]
        result = create_few_shot_prompt("Task", examples, "z")
        self.assertTrue(result.strip().endswith("Output:"))

    def test_classification_ends_with_category(self):
        labels = ["a", "b"]
        examples = [{"text": "test", "label": "a"}]
        result = create_classification_prompt(labels, examples, "new")
        self.assertTrue(result.strip().endswith("Category:"))
`,

	hint1: 'Use consistent formatting between examples',
	hint2: 'Include diverse examples that cover edge cases',

	whyItMatters: `Few-shot learning is powerful for LLM applications:

- **No training needed**: Works with any LLM instantly
- **Flexible**: Change behavior with new examples
- **Precise**: Show exactly what you want
- **Classification**: Achieves high accuracy with few examples

Essential technique for production LLM systems.`,

	translations: {
		ru: {
			title: 'Few-Shot промптинг',
			description: `# Few-Shot промптинг

Используйте примеры для направления поведения LLM.

## Что такое Few-Shot?

- **Zero-shot**: Без примеров, только инструкции
- **One-shot**: Один пример
- **Few-shot**: Несколько примеров (2-5)

## Когда использовать

- Сложные требования к форматированию
- Доменная терминология
- Консистентная структура вывода
- Задачи классификации

## Пример

\`\`\`python
prompt = """
Classify the sentiment of the text.

Text: "I love this product!"
Sentiment: positive

Text: "Terrible experience, never again."
Sentiment: negative

Text: "It's okay, nothing special."
Sentiment: neutral

Text: "Best purchase I've ever made!"
Sentiment:
"""
\`\`\``,
			hint1: 'Используйте консистентное форматирование между примерами',
			hint2: 'Включайте разнообразные примеры, покрывающие крайние случаи',
			whyItMatters: `Few-shot learning мощен для LLM приложений:

- **Без обучения**: Работает с любой LLM мгновенно
- **Гибкость**: Меняйте поведение новыми примерами
- **Точность**: Показывайте именно то, что нужно
- **Классификация**: Высокая точность с малым числом примеров`,
		},
		uz: {
			title: 'Few-Shot prompting',
			description: `# Few-Shot prompting

LLM xatti-harakatini yo'naltirish uchun misollardan foydalaning.

## Few-Shot nima?

- **Zero-shot**: Misolsiz, faqat ko'rsatmalar
- **One-shot**: Bitta misol
- **Few-shot**: Bir nechta misol (2-5)

## Qachon foydalanish

- Murakkab formatlash talablari
- Sohaga xos terminologiya
- Izchil chiqish tuzilmasi
- Klassifikatsiya vazifalari

## Misol

\`\`\`python
prompt = """
Classify the sentiment of the text.

Text: "I love this product!"
Sentiment: positive

Text: "Terrible experience, never again."
Sentiment: negative

Text: "It's okay, nothing special."
Sentiment: neutral

Text: "Best purchase I've ever made!"
Sentiment:
"""
\`\`\``,
			hint1: "Misollar orasida izchil formatlashdan foydalaning",
			hint2: "Chekka holatlarni qamrab oladigan turli xil misollarni kiriting",
			whyItMatters: `Few-shot learning LLM ilovalari uchun kuchli:

- **O'qitish kerak emas**: Har qanday LLM bilan darhol ishlaydi
- **Moslashuvchan**: Yangi misollar bilan xatti-harakatni o'zgartiring
- **Aniq**: Nima xohlayotganingizni ko'rsating
- **Klassifikatsiya**: Oz misollar bilan yuqori aniqlik`,
		},
	},
};

export default task;
