import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-iterator',
	title: 'Custom Iterator',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'oop', 'iterator'],
	estimatedTime: '15m',
	isPremium: false,
	order: 8,

	description: `# Custom Iterator

Make your class iterable by implementing \`__iter__\` and \`__next__\`.

## Task

Create a class \`Range\` that works like Python's built-in range() but as a class.

## Requirements

- \`__init__(self, start, stop)\`: Initialize range boundaries
- \`__iter__(self)\`: Return self (the iterator object)
- \`__next__(self)\`: Return next value or raise StopIteration
- Should be usable in for loops

## Examples

\`\`\`python
>>> r = Range(1, 4)
>>> list(r)
[1, 2, 3]

>>> for num in Range(5, 8):
...     print(num)
5
6
7
\`\`\``,

	initialCode: `class Range:
    """Custom iterable range class."""

    def __init__(self, start: int, stop: int):
        # TODO: Initialize
        pass

    def __iter__(self):
        # TODO: Return iterator
        pass

    def __next__(self) -> int:
        # TODO: Return next or raise StopIteration
        pass`,

	solutionCode: `class Range:
    """Custom iterable range class."""

    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop
        self.current = start

    def __iter__(self):
        # Reset current and return self
        self.current = self.start
        return self

    def __next__(self) -> int:
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += 1
        return value`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        r = Range(1, 4)
        self.assertEqual(list(r), [1, 2, 3])

    def test_2(self):
        r = Range(5, 8)
        self.assertEqual(list(r), [5, 6, 7])

    def test_3(self):
        r = Range(0, 0)
        self.assertEqual(list(r), [])

    def test_4(self):
        r = Range(0, 1)
        self.assertEqual(list(r), [0])

    def test_5(self):
        r = Range(10, 15)
        self.assertEqual(len(list(r)), 5)

    def test_6(self):
        r = Range(1, 4)
        result = []
        for x in r:
            result.append(x)
        self.assertEqual(result, [1, 2, 3])

    def test_7(self):
        r = Range(-2, 2)
        self.assertEqual(list(r), [-2, -1, 0, 1])

    def test_8(self):
        r = Range(100, 103)
        self.assertEqual(list(r), [100, 101, 102])

    def test_9(self):
        r = Range(0, 3)
        self.assertEqual(list(r), list(r))

    def test_10(self):
        r = Range(1, 2)
        it = iter(r)
        self.assertEqual(next(it), 1)

if __name__ == '__main__':
    unittest.main()`,

	hint1: '__iter__ should reset current to start and return self. This allows re-iteration.',
	hint2: 'In __next__, check if current >= stop. If so, raise StopIteration. Otherwise, return current and increment.',

	whyItMatters: `Custom iterators enable memory-efficient data processing and integration with Python's iteration protocol.`,

	translations: {
		ru: {
			title: 'Пользовательский итератор',
			description: `# Пользовательский итератор

Сделайте класс итерируемым через \`__iter__\` и \`__next__\`.

## Задача

Создайте класс \`Range\`, работающий как встроенный range().`,
			hint1: '__iter__ должен сбросить current в start и вернуть self. Это позволяет повторную итерацию.',
			hint2: 'В __next__ проверьте current >= stop. Если да — raise StopIteration. Иначе верните current и увеличьте.',
			whyItMatters: `Пользовательские итераторы обеспечивают эффективную по памяти обработку данных.`,
		},
		uz: {
			title: "Maxsus iterator",
			description: `# Maxsus iterator

\`__iter__\` va \`__next__\` ni amalga oshirib klassni iteratsiyalanadigan qiling.

## Vazifa

O'rnatilgan range() kabi ishlaydigan \`Range\` klassini yarating.`,
			hint1: "__iter__ current ni start ga qaytarishi va self qaytarishi kerak. Bu qayta iteratsiyaga imkon beradi.",
			hint2: "__next__ da current >= stop ni tekshiring. Bo'lsa — StopIteration ko'taring. Aks holda current ni qaytaring va oshiring.",
			whyItMatters: `Maxsus iteratorlar xotirani tejaydigan ma'lumotlarni qayta ishlashni ta'minlaydi.`,
		},
	},
};

export default task;
