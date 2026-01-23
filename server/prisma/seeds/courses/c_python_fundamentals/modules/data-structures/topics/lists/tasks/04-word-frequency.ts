import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-word-frequency',
	title: 'Word Frequency',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'dictionaries', 'counting'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,

	description: `# Word Frequency

Learn to use dictionaries for counting occurrences.

## Task

Implement the function \`word_frequency(words)\` that counts how many times each word appears.

## Requirements

- Return a dictionary with words as keys and counts as values
- Words are case-sensitive
- Handle empty lists

## Examples

\`\`\`python
>>> word_frequency(["apple", "banana", "apple"])
{"apple": 2, "banana": 1}

>>> word_frequency(["a", "b", "a", "a", "c"])
{"a": 3, "b": 1, "c": 1}

>>> word_frequency([])
{}
\`\`\``,

	initialCode: `def word_frequency(words: list[str]) -> dict[str, int]:
    """Count the frequency of each word in a list.

    Args:
        words: List of words

    Returns:
        Dictionary mapping words to their counts
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def word_frequency(words: list[str]) -> dict[str, int]:
    """Count the frequency of each word in a list.

    Args:
        words: List of words

    Returns:
        Dictionary mapping words to their counts
    """
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic counting"""
        result = word_frequency(["apple", "banana", "apple"])
        self.assertEqual(result, {"apple": 2, "banana": 1})

    def test_2(self):
        """Multiple occurrences"""
        result = word_frequency(["a", "b", "a", "a", "c"])
        self.assertEqual(result, {"a": 3, "b": 1, "c": 1})

    def test_3(self):
        """Empty list"""
        self.assertEqual(word_frequency([]), {})

    def test_4(self):
        """Single word"""
        self.assertEqual(word_frequency(["hello"]), {"hello": 1})

    def test_5(self):
        """All same"""
        self.assertEqual(word_frequency(["x", "x", "x"]), {"x": 3})

    def test_6(self):
        """Case sensitive"""
        result = word_frequency(["Hello", "hello"])
        self.assertEqual(result, {"Hello": 1, "hello": 1})

    def test_7(self):
        """Numbers as strings"""
        result = word_frequency(["1", "2", "1"])
        self.assertEqual(result, {"1": 2, "2": 1})

    def test_8(self):
        """Long words"""
        result = word_frequency(["python", "python", "java"])
        self.assertEqual(result["python"], 2)

    def test_9(self):
        """Special characters"""
        result = word_frequency(["@", "#", "@"])
        self.assertEqual(result, {"@": 2, "#": 1})

    def test_10(self):
        """Many unique words"""
        result = word_frequency(["a", "b", "c", "d", "e"])
        self.assertEqual(len(result), 5)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use a dictionary to store counts. Check if word exists before incrementing.',
	hint2: 'Alternatively, use dict.get(word, 0) + 1 to handle missing keys.',

	whyItMatters: `Counting occurrences is fundamental to data analysis and text processing.

**Production Pattern:**

\`\`\`python
from collections import Counter

def analyze_log_levels(logs: list[str]) -> dict[str, int]:
    """Count log levels in application logs."""
    levels = [log.split("]")[0].strip("[") for log in logs]
    return dict(Counter(levels))

def most_common_errors(errors: list[str], top_n: int = 5) -> list[tuple]:
    """Get most common error messages."""
    return Counter(errors).most_common(top_n)

# Pro tip: Counter is optimized for counting
\`\`\`

**Practical Benefits:**
- Dictionary counting is O(n) time complexity
- Counter from collections is even more convenient
- Essential for analytics and monitoring`,

	translations: {
		ru: {
			title: 'Частота слов',
			description: `# Частота слов

Научитесь использовать словари для подсчёта вхождений.

## Задача

Реализуйте функцию \`word_frequency(words)\`, которая подсчитывает, сколько раз встречается каждое слово.

## Требования

- Верните словарь со словами как ключами и счётчиками как значениями
- Слова чувствительны к регистру
- Обработайте пустые списки

## Примеры

\`\`\`python
>>> word_frequency(["apple", "banana", "apple"])
{"apple": 2, "banana": 1}

>>> word_frequency(["a", "b", "a", "a", "c"])
{"a": 3, "b": 1, "c": 1}

>>> word_frequency([])
{}
\`\`\``,
			hint1: 'Используйте словарь для хранения счётчиков. Проверяйте существование ключа.',
			hint2: 'Альтернатива: используйте dict.get(word, 0) + 1',
			whyItMatters: `Подсчёт вхождений — основа анализа данных и обработки текста.

**Продакшен паттерн:**

\`\`\`python
from collections import Counter

def analyze_log_levels(logs: list[str]) -> dict[str, int]:
    """Подсчёт уровней логов."""
    levels = [log.split("]")[0].strip("[") for log in logs]
    return dict(Counter(levels))
\`\`\`

**Практические преимущества:**
- Подсчёт в словаре имеет сложность O(n)
- Counter из collections ещё удобнее
- Необходим для аналитики и мониторинга`,
		},
		uz: {
			title: "So'zlar chastotasi",
			description: `# So'zlar chastotasi

Uchrashuvlarni hisoblash uchun lug'atlardan foydalanishni o'rganing.

## Vazifa

Har bir so'z necha marta uchraganini hisoblovchi \`word_frequency(words)\` funksiyasini amalga oshiring.

## Talablar

- So'zlar kalit va hisoblar qiymat sifatida bo'lgan lug'at qaytaring
- So'zlar registrga sezgir
- Bo'sh ro'yxatlarni ham ishlov bering

## Misollar

\`\`\`python
>>> word_frequency(["apple", "banana", "apple"])
{"apple": 2, "banana": 1}

>>> word_frequency(["a", "b", "a", "a", "c"])
{"a": 3, "b": 1, "c": 1}

>>> word_frequency([])
{}
\`\`\``,
			hint1: "Hisoblarni saqlash uchun lug'atdan foydalaning. Kalit mavjudligini tekshiring.",
			hint2: "Alternativa: dict.get(word, 0) + 1 dan foydalaning",
			whyItMatters: `Uchrashuvlarni hisoblash ma'lumotlarni tahlil qilish va matnni qayta ishlashning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
from collections import Counter

def analyze_log_levels(logs: list[str]) -> dict[str, int]:
    """Log darajalarini hisoblash."""
    levels = [log.split("]")[0].strip("[") for log in logs]
    return dict(Counter(levels))
\`\`\`

**Amaliy foydalari:**
- Lug'atda hisoblash O(n) vaqt murakkabligiga ega
- collections dan Counter yanada qulayroq
- Analitika va monitoring uchun zarur`,
		},
	},
};

export default task;
