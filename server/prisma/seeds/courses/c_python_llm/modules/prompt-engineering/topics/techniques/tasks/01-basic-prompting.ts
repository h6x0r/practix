import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-basic-prompting',
	title: 'Basic Prompting',
	difficulty: 'easy',
	tags: ['prompting', 'llm', 'basics'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Basic Prompting

Learn fundamental prompting techniques for effective LLM interaction.

## Key Principles

1. **Be specific**: Clear instructions get better results
2. **Provide context**: Give relevant background information
3. **Use examples**: Show the format you want
4. **Specify output format**: JSON, markdown, etc.

## Prompt Structure

\`\`\`
[Role/Persona]
[Context]
[Task]
[Output Format]
[Examples]
[Constraints]
\`\`\`

## Example

\`\`\`python
prompt = """
You are a helpful coding assistant.

Task: Explain the following Python code.

Code:
\`\`\`python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\`

Provide your explanation in bullet points.
"""
\`\`\``,

	initialCode: `def create_prompt(task: str, context: str = "",
                   output_format: str = "") -> str:
    """Create a structured prompt."""
    # Your code here
    pass

def create_role_prompt(role: str, task: str, constraints: list = None) -> str:
    """Create a prompt with a specific role."""
    # Your code here
    pass

def create_code_review_prompt(code: str, language: str = "python") -> str:
    """Create a prompt for code review."""
    # Your code here
    pass

def create_summarization_prompt(text: str, max_words: int = 100) -> str:
    """Create a prompt for text summarization."""
    # Your code here
    pass
`,

	solutionCode: `def create_prompt(task: str, context: str = "",
                   output_format: str = "") -> str:
    """Create a structured prompt."""
    parts = []

    if context:
        parts.append(f"Context: {context}")

    parts.append(f"Task: {task}")

    if output_format:
        parts.append(f"Output Format: {output_format}")

    return "\\n\\n".join(parts)

def create_role_prompt(role: str, task: str, constraints: list = None) -> str:
    """Create a prompt with a specific role."""
    prompt = f"You are {role}.\\n\\n"
    prompt += f"Task: {task}"

    if constraints:
        prompt += "\\n\\nConstraints:\\n"
        for constraint in constraints:
            prompt += f"- {constraint}\\n"

    return prompt

def create_code_review_prompt(code: str, language: str = "python") -> str:
    """Create a prompt for code review."""
    return f"""You are an experienced software engineer conducting a code review.

Review the following {language} code for:
1. Bugs and potential issues
2. Code style and best practices
3. Performance improvements
4. Security vulnerabilities

Code:
\`\`\`{language}
{code}
\`\`\`

Provide your review in the following format:
- **Issues**: List any bugs or problems
- **Suggestions**: Improvements to consider
- **Positive aspects**: What's done well
"""

def create_summarization_prompt(text: str, max_words: int = 100) -> str:
    """Create a prompt for text summarization."""
    return f"""Summarize the following text in {max_words} words or less.

Text:
{text}

Requirements:
- Capture the main points
- Use clear, concise language
- Maintain the original meaning

Summary:"""
`,

	testCode: `import unittest

class TestBasicPrompting(unittest.TestCase):
    def test_create_prompt_basic(self):
        result = create_prompt("Write a poem")
        self.assertIn("Task: Write a poem", result)

    def test_create_prompt_with_context(self):
        result = create_prompt("Analyze this", context="User data analysis")
        self.assertIn("Context:", result)
        self.assertIn("User data analysis", result)

    def test_create_role_prompt(self):
        result = create_role_prompt("a Python expert", "Explain decorators")
        self.assertIn("You are a Python expert", result)
        self.assertIn("Explain decorators", result)

    def test_create_role_prompt_with_constraints(self):
        result = create_role_prompt("an expert", "Help", constraints=["Be brief"])
        self.assertIn("Constraints:", result)
        self.assertIn("Be brief", result)

    def test_create_code_review_prompt(self):
        result = create_code_review_prompt("print('hello')", "python")
        self.assertIn("code review", result.lower())
        self.assertIn("print('hello')", result)

    def test_create_summarization_prompt(self):
        result = create_summarization_prompt("Long text here", max_words=50)
        self.assertIn("50 words", result)

    def test_create_prompt_with_output_format(self):
        result = create_prompt("Analyze data", output_format="JSON")
        self.assertIn("Output Format:", result)
        self.assertIn("JSON", result)

    def test_create_prompt_returns_string(self):
        result = create_prompt("Test task")
        self.assertIsInstance(result, str)

    def test_create_code_review_includes_language(self):
        result = create_code_review_prompt("code", "javascript")
        self.assertIn("javascript", result)

    def test_summarization_includes_requirements(self):
        result = create_summarization_prompt("Some text")
        self.assertIn("Requirements:", result)
`,

	hint1: 'Structure prompts with clear sections separated by newlines',
	hint2: 'Always specify the expected output format for consistency',

	whyItMatters: `Good prompting is the foundation of LLM applications:

- **Quality**: Better prompts = better outputs
- **Consistency**: Structured prompts get consistent results
- **Efficiency**: Clear prompts reduce iterations
- **Cost**: Fewer tokens needed with precise prompts

This is the most important skill for working with LLMs.`,

	translations: {
		ru: {
			title: 'Основы промптинга',
			description: `# Основы промптинга

Изучите фундаментальные техники промптинга для эффективного взаимодействия с LLM.

## Ключевые принципы

1. **Будьте конкретны**: Четкие инструкции дают лучшие результаты
2. **Предоставляйте контекст**: Давайте релевантную информацию
3. **Используйте примеры**: Показывайте желаемый формат
4. **Указывайте формат вывода**: JSON, markdown и т.д.

## Пример

\`\`\`python
prompt = """
You are a helpful coding assistant.

Task: Explain the following Python code.

Code:
\`\`\`python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\`

Provide your explanation in bullet points.
"""
\`\`\``,
			hint1: 'Структурируйте промпты с четкими секциями, разделенными переносами строк',
			hint2: 'Всегда указывайте ожидаемый формат вывода для консистентности',
			whyItMatters: `Хороший промптинг - основа LLM приложений:

- **Качество**: Лучшие промпты = лучшие результаты
- **Консистентность**: Структурированные промпты дают стабильные результаты
- **Эффективность**: Четкие промпты уменьшают итерации
- **Стоимость**: Меньше токенов с точными промптами`,
		},
		uz: {
			title: 'Asosiy prompting',
			description: `# Asosiy prompting

LLM bilan samarali muloqot uchun asosiy prompting texnikalarini o'rganing.

## Asosiy tamoyillar

1. **Aniq bo'ling**: Aniq ko'rsatmalar yaxshiroq natijalar beradi
2. **Kontekst bering**: Tegishli ma'lumotlarni taqdim eting
3. **Misollar foydalaning**: Kerakli formatni ko'rsating
4. **Chiqish formatini belgilang**: JSON, markdown va h.k.

## Misol

\`\`\`python
prompt = """
You are a helpful coding assistant.

Task: Explain the following Python code.

Code:
\`\`\`python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\`

Provide your explanation in bullet points.
"""
\`\`\``,
			hint1: "Promptlarni yangi qatorlar bilan ajratilgan aniq bo'limlar bilan tuzilmalang",
			hint2: "Izchillik uchun har doim kutilgan chiqish formatini belgilang",
			whyItMatters: `Yaxshi prompting LLM ilovalarining asosi:

- **Sifat**: Yaxshiroq promptlar = yaxshiroq natijalar
- **Izchillik**: Tuzilgan promptlar barqaror natijalar beradi
- **Samaradorlik**: Aniq promptlar iteratsiyalarni kamaytiradi
- **Narx**: Aniq promptlar bilan kamroq tokenlar kerak`,
		},
	},
};

export default task;
