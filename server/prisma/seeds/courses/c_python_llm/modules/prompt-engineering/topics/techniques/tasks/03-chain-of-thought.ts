import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-chain-of-thought',
	title: 'Chain of Thought',
	difficulty: 'medium',
	tags: ['prompting', 'reasoning', 'cot'],
	estimatedTime: '15m',
	isPremium: true,
	order: 3,
	description: `# Chain of Thought Prompting

Enable step-by-step reasoning for complex problem solving.

## What is CoT?

Chain of Thought prompting encourages LLMs to:
- Break down complex problems
- Show intermediate reasoning steps
- Arrive at answers systematically

## Techniques

1. **Manual CoT**: Add "Let's think step by step"
2. **Few-shot CoT**: Show reasoning examples
3. **Zero-shot CoT**: Just add the trigger phrase

## Example

\`\`\`python
prompt = """
Q: A store had 23 apples. They sold 15 and received 8 more.
   How many apples do they have now?

A: Let's think step by step.
1. Started with 23 apples
2. Sold 15: 23 - 15 = 8 apples
3. Received 8 more: 8 + 8 = 16 apples
Answer: 16 apples
"""
\`\`\``,

	initialCode: `def add_cot_trigger(prompt: str) -> str:
    """Add chain-of-thought trigger to prompt."""
    # Your code here
    pass

def create_cot_example(question: str, steps: list, answer: str) -> str:
    """Create a CoT example with reasoning steps."""
    # Your code here
    pass

def create_cot_prompt(examples: list, question: str) -> str:
    """Create a few-shot CoT prompt."""
    # Your code here
    pass

def create_math_cot_prompt(problem: str) -> str:
    """Create a CoT prompt for math problems."""
    # Your code here
    pass

def create_code_reasoning_prompt(task: str, requirements: list) -> str:
    """Create a prompt for code with reasoning."""
    # Your code here
    pass
`,

	solutionCode: `def add_cot_trigger(prompt: str) -> str:
    """Add chain-of-thought trigger to prompt."""
    return f"{prompt}\\n\\nLet's think step by step."

def create_cot_example(question: str, steps: list, answer: str) -> str:
    """Create a CoT example with reasoning steps."""
    example = f"Q: {question}\\n\\n"
    example += "A: Let's think step by step.\\n"

    for i, step in enumerate(steps, 1):
        example += f"{i}. {step}\\n"

    example += f"\\nAnswer: {answer}"
    return example

def create_cot_prompt(examples: list, question: str) -> str:
    """Create a few-shot CoT prompt."""
    prompt = "Solve the following problems by thinking step by step.\\n\\n"

    for example in examples:
        prompt += create_cot_example(
            example["question"],
            example["steps"],
            example["answer"]
        )
        prompt += "\\n\\n---\\n\\n"

    prompt += f"Q: {question}\\n\\n"
    prompt += "A: Let's think step by step."

    return prompt

def create_math_cot_prompt(problem: str) -> str:
    """Create a CoT prompt for math problems."""
    return f"""Solve this math problem step by step.

Problem: {problem}

Instructions:
1. Identify what we need to find
2. List the given information
3. Apply relevant formulas or operations
4. Show each calculation step
5. State the final answer clearly

Solution:
Let's think step by step."""

def create_code_reasoning_prompt(task: str, requirements: list) -> str:
    """Create a prompt for code with reasoning."""
    reqs = "\\n".join(f"- {req}" for req in requirements)

    return f"""Write code for the following task, explaining your reasoning.

Task: {task}

Requirements:
{reqs}

Approach:
1. First, let's understand what we need to do
2. Then, identify the key components
3. Plan the implementation
4. Write the code with explanations

Solution:"""
`,

	testCode: `import unittest

class TestChainOfThought(unittest.TestCase):
    def test_add_cot_trigger(self):
        result = add_cot_trigger("Solve this problem")
        self.assertIn("Let's think step by step", result)
        self.assertIn("Solve this problem", result)

    def test_create_cot_example(self):
        result = create_cot_example(
            "What is 2+3?",
            ["Add 2 and 3", "2 + 3 = 5"],
            "5"
        )
        self.assertIn("Q: What is 2+3?", result)
        self.assertIn("step by step", result)
        self.assertIn("Answer: 5", result)

    def test_create_cot_prompt(self):
        examples = [{
            "question": "What is 1+1?",
            "steps": ["Add 1 and 1"],
            "answer": "2"
        }]
        result = create_cot_prompt(examples, "What is 2+2?")
        self.assertIn("What is 1+1?", result)
        self.assertIn("What is 2+2?", result)

    def test_create_math_cot_prompt(self):
        result = create_math_cot_prompt("Calculate 15% of 200")
        self.assertIn("15% of 200", result)
        self.assertIn("step by step", result)

    def test_create_code_reasoning_prompt(self):
        result = create_code_reasoning_prompt("Sort a list", ["Use O(n log n)"])
        self.assertIn("Sort a list", result)
        self.assertIn("O(n log n)", result)

    def test_add_cot_trigger_returns_string(self):
        result = add_cot_trigger("Test")
        self.assertIsInstance(result, str)

    def test_cot_example_includes_numbered_steps(self):
        result = create_cot_example("Q?", ["Step A", "Step B"], "X")
        self.assertIn("1. Step A", result)
        self.assertIn("2. Step B", result)

    def test_cot_prompt_includes_separator(self):
        examples = [{"question": "Q1", "steps": ["S1"], "answer": "A1"}]
        result = create_cot_prompt(examples, "Q2")
        self.assertIn("---", result)

    def test_math_cot_includes_instructions(self):
        result = create_math_cot_prompt("2+2")
        self.assertIn("Instructions:", result)

    def test_code_reasoning_includes_requirements(self):
        result = create_code_reasoning_prompt("Task", ["Req1", "Req2"])
        self.assertIn("Requirements:", result)
        self.assertIn("- Req1", result)
        self.assertIn("- Req2", result)
`,

	hint1: '"Let\'s think step by step" significantly improves reasoning',
	hint2: 'Number your steps for clearer structure',

	whyItMatters: `Chain of Thought dramatically improves LLM reasoning:

- **Math problems**: 50%+ accuracy improvement
- **Logic puzzles**: Better step-by-step deduction
- **Code generation**: More thoughtful implementations
- **Debugging**: Systematic error analysis

Essential for any task requiring multi-step reasoning.`,

	translations: {
		ru: {
			title: 'Chain of Thought',
			description: `# Chain of Thought промптинг

Включите пошаговое рассуждение для решения сложных задач.

## Что такое CoT?

Chain of Thought промптинг побуждает LLM:
- Разбивать сложные задачи
- Показывать промежуточные шаги рассуждения
- Приходить к ответам систематически

## Техники

1. **Manual CoT**: Добавьте "Let's think step by step"
2. **Few-shot CoT**: Покажите примеры рассуждений
3. **Zero-shot CoT**: Просто добавьте триггер-фразу

## Пример

\`\`\`python
prompt = """
Q: A store had 23 apples. They sold 15 and received 8 more.
   How many apples do they have now?

A: Let's think step by step.
1. Started with 23 apples
2. Sold 15: 23 - 15 = 8 apples
3. Received 8 more: 8 + 8 = 16 apples
Answer: 16 apples
"""
\`\`\``,
			hint1: '"Let\'s think step by step" значительно улучшает рассуждение',
			hint2: 'Нумеруйте шаги для более четкой структуры',
			whyItMatters: `Chain of Thought драматически улучшает рассуждение LLM:

- **Математика**: 50%+ улучшение точности
- **Логические задачи**: Лучшая пошаговая дедукция
- **Генерация кода**: Более продуманные реализации
- **Отладка**: Систематический анализ ошибок`,
		},
		uz: {
			title: 'Chain of Thought',
			description: `# Chain of Thought prompting

Murakkab muammolarni yechish uchun bosqichma-bosqich fikrlashni yoqing.

## CoT nima?

Chain of Thought prompting LLM larni quyidagilarga undaydi:
- Murakkab muammolarni bo'laklarga ajratish
- Oraliq fikrlash bosqichlarini ko'rsatish
- Javoblarga tizimli yondashish

## Texnikalar

1. **Manual CoT**: "Let's think step by step" qo'shing
2. **Few-shot CoT**: Fikrlash misollarini ko'rsating
3. **Zero-shot CoT**: Faqat trigger iborani qo'shing

## Misol

\`\`\`python
prompt = """
Q: A store had 23 apples. They sold 15 and received 8 more.
   How many apples do they have now?

A: Let's think step by step.
1. Started with 23 apples
2. Sold 15: 23 - 15 = 8 apples
3. Received 8 more: 8 + 8 = 16 apples
Answer: 16 apples
"""
\`\`\``,
			hint1: '"Let\'s think step by step" fikrlashni sezilarli darajada yaxshilaydi',
			hint2: "Aniqroq tuzilma uchun qadamlarni raqamlang",
			whyItMatters: `Chain of Thought LLM fikrlashini keskin yaxshilaydi:

- **Matematika muammolari**: 50%+ aniqlik yaxshilanishi
- **Mantiqiy jumboqlar**: Yaxshiroq bosqichma-bosqich deduksiya
- **Kod generatsiyasi**: Ko'proq o'ylangan implementatsiyalar
- **Debugging**: Tizimli xato tahlili`,
		},
	},
};

export default task;
