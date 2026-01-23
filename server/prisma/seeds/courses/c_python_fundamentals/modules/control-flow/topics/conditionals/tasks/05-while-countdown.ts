import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-while-countdown',
	title: 'Countdown with While',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'loops', 'while'],
	estimatedTime: '10m',
	isPremium: false,
	order: 5,

	description: `# Countdown with While

Learn to use while loops for iteration with a condition.

## Task

Implement the function \`countdown(n)\` that returns a list counting down from n to 1.

## Requirements

- Start from n and go down to 1 (inclusive)
- If n <= 0, return an empty list
- Use a while loop (not for loop)

## Examples

\`\`\`python
>>> countdown(5)
[5, 4, 3, 2, 1]

>>> countdown(3)
[3, 2, 1]

>>> countdown(0)
[]
\`\`\``,

	initialCode: `def countdown(n: int) -> list[int]:
    """Create a countdown list from n to 1.

    Args:
        n: Starting number

    Returns:
        List of integers from n down to 1
    """
    # TODO: Implement using a while loop
    pass`,

	solutionCode: `def countdown(n: int) -> list[int]:
    """Create a countdown list from n to 1.

    Args:
        n: Starting number

    Returns:
        List of integers from n down to 1
    """
    result = []
    while n > 0:
        result.append(n)
        n -= 1
    return result`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic countdown"""
        self.assertEqual(countdown(5), [5, 4, 3, 2, 1])

    def test_2(self):
        """Short countdown"""
        self.assertEqual(countdown(3), [3, 2, 1])

    def test_3(self):
        """Zero input"""
        self.assertEqual(countdown(0), [])

    def test_4(self):
        """Negative input"""
        self.assertEqual(countdown(-5), [])

    def test_5(self):
        """Countdown from 1"""
        self.assertEqual(countdown(1), [1])

    def test_6(self):
        """Countdown from 10"""
        self.assertEqual(countdown(10), [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    def test_7(self):
        """Length check"""
        self.assertEqual(len(countdown(7)), 7)

    def test_8(self):
        """First element"""
        self.assertEqual(countdown(100)[0], 100)

    def test_9(self):
        """Last element"""
        self.assertEqual(countdown(50)[-1], 1)

    def test_10(self):
        """Countdown from 2"""
        self.assertEqual(countdown(2), [2, 1])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Initialize an empty list, then use while n > 0 to loop.',
	hint2: 'Inside the loop: append n to result, then decrement n by 1.',

	whyItMatters: `While loops are essential when you don't know the number of iterations in advance.

**Production Pattern:**

\`\`\`python
def retry_request(url: str, max_attempts: int = 3) -> dict | None:
    """Retry failed requests with exponential backoff."""
    attempt = 0
    wait_time = 1

    while attempt < max_attempts:
        try:
            response = make_request(url)
            return response
        except RequestError:
            attempt += 1
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff

    return None

def read_until_eof(stream) -> list[str]:
    """Read lines until end of stream."""
    lines = []
    while True:
        line = stream.readline()
        if not line:
            break
        lines.append(line.strip())
    return lines
\`\`\`

**Practical Benefits:**
- While loops handle unknown iteration counts
- Retry logic requires condition-based loops
- Stream processing often uses while True with break`,

	translations: {
		ru: {
			title: 'Обратный отсчёт с While',
			description: `# Обратный отсчёт с While

Научитесь использовать циклы while для итерации с условием.

## Задача

Реализуйте функцию \`countdown(n)\`, которая возвращает список с обратным отсчётом от n до 1.

## Требования

- Начните с n и идите до 1 (включительно)
- Если n <= 0, верните пустой список
- Используйте цикл while (не for)

## Примеры

\`\`\`python
>>> countdown(5)
[5, 4, 3, 2, 1]

>>> countdown(3)
[3, 2, 1]

>>> countdown(0)
[]
\`\`\``,
			hint1: 'Инициализируйте пустой список, затем используйте while n > 0.',
			hint2: 'Внутри цикла: добавьте n в result, затем уменьшите n на 1.',
			whyItMatters: `Циклы while необходимы, когда количество итераций заранее неизвестно.

**Продакшен паттерн:**

\`\`\`python
def retry_request(url: str, max_attempts: int = 3) -> dict | None:
    """Повторные запросы с экспоненциальной задержкой."""
    attempt = 0
    wait_time = 1

    while attempt < max_attempts:
        try:
            response = make_request(url)
            return response
        except RequestError:
            attempt += 1
            time.sleep(wait_time)
            wait_time *= 2

    return None
\`\`\`

**Практические преимущества:**
- While обрабатывает неизвестное количество итераций
- Логика повторов требует условных циклов`,
		},
		uz: {
			title: 'While bilan teskari hisoblash',
			description: `# While bilan teskari hisoblash

Shart bilan iteratsiya uchun while sikllaridan foydalanishni o'rganing.

## Vazifa

n dan 1 gacha teskari hisoblash ro'yxatini qaytaruvchi \`countdown(n)\` funksiyasini amalga oshiring.

## Talablar

- n dan boshlab 1 gacha (shu jumladan) boring
- Agar n <= 0 bo'lsa, bo'sh ro'yxat qaytaring
- while sikldan foydalaning (for emas)

## Misollar

\`\`\`python
>>> countdown(5)
[5, 4, 3, 2, 1]

>>> countdown(3)
[3, 2, 1]

>>> countdown(0)
[]
\`\`\``,
			hint1: "Bo'sh ro'yxat yarating, keyin while n > 0 dan foydalaning.",
			hint2: "Sikl ichida: n ni result ga qo'shing, keyin n ni 1 ga kamaytiring.",
			whyItMatters: `While sikllari iteratsiyalar soni oldindan noma'lum bo'lganda zarur.

**Ishlab chiqarish patterni:**

\`\`\`python
def retry_request(url: str, max_attempts: int = 3) -> dict | None:
    """Eksponentsial kechikish bilan qayta urinishlar."""
    attempt = 0
    wait_time = 1

    while attempt < max_attempts:
        try:
            response = make_request(url)
            return response
        except RequestError:
            attempt += 1
            time.sleep(wait_time)
            wait_time *= 2

    return None
\`\`\`

**Amaliy foydalari:**
- While noma'lum iteratsiyalar sonini boshqaradi
- Qayta urinish mantig'i shartli sikllarni talab qiladi`,
		},
	},
};

export default task;
