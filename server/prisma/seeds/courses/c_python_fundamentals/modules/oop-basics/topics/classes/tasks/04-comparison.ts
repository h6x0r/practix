import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-comparison',
	title: 'Comparison Methods',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'oop', 'dunder'],
	estimatedTime: '10m',
	isPremium: false,
	order: 4,

	description: `# Comparison Methods

Implement \`__eq__\` and \`__lt__\` to enable object comparison.

## Task

Create a class \`Product\` that can be compared by price.

## Requirements

- \`__init__(self, name, price)\`: Initialize with name and price
- \`__eq__(self, other)\`: Return True if prices are equal
- \`__lt__(self, other)\`: Return True if self.price < other.price
- Products should be sortable by price

## Examples

\`\`\`python
>>> p1 = Product("Apple", 1.5)
>>> p2 = Product("Banana", 1.5)
>>> p3 = Product("Cherry", 3.0)
>>> p1 == p2
True
>>> p1 < p3
True
>>> sorted([p3, p1, p2], key=lambda x: x.price)
[Apple, Banana, Cherry]  # sorted by price
\`\`\``,

	initialCode: `class Product:
    """A product with name and price."""

    def __init__(self, name: str, price: float):
        # TODO: Initialize
        pass

    def __eq__(self, other) -> bool:
        # TODO: Compare prices
        pass

    def __lt__(self, other) -> bool:
        # TODO: Less than comparison
        pass`,

	solutionCode: `class Product:
    """A product with name and price."""

    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def __eq__(self, other) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self.price == other.price

    def __lt__(self, other) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self.price < other.price`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        p1 = Product("A", 1.5)
        p2 = Product("B", 1.5)
        self.assertTrue(p1 == p2)

    def test_2(self):
        p1 = Product("A", 1.0)
        p2 = Product("B", 2.0)
        self.assertTrue(p1 < p2)

    def test_3(self):
        p1 = Product("A", 2.0)
        p2 = Product("B", 1.0)
        self.assertFalse(p1 < p2)

    def test_4(self):
        p1 = Product("A", 1.0)
        p2 = Product("B", 2.0)
        self.assertFalse(p1 == p2)

    def test_5(self):
        p1 = Product("Test", 10)
        self.assertEqual(p1.name, "Test")

    def test_6(self):
        p1 = Product("Test", 10)
        self.assertEqual(p1.price, 10)

    def test_7(self):
        p1 = Product("A", 5)
        p2 = Product("B", 5)
        self.assertFalse(p1 < p2)

    def test_8(self):
        p1 = Product("A", 0)
        p2 = Product("B", 1)
        self.assertTrue(p1 < p2)

    def test_9(self):
        products = [Product("C", 3), Product("A", 1), Product("B", 2)]
        sorted_prices = [p.price for p in sorted(products)]
        self.assertEqual(sorted_prices, [1, 2, 3])

    def test_10(self):
        p1 = Product("Same", 100)
        p2 = Product("Same", 100)
        self.assertEqual(p1, p2)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'In __eq__, compare self.price with other.price. Check isinstance(other, Product) first.',
	hint2: 'With __eq__ and __lt__ defined, Python can derive other comparisons automatically.',

	whyItMatters: `Comparison methods enable sorting, searching, and intuitive object comparisons.`,

	translations: {
		ru: {
			title: 'Методы сравнения',
			description: `# Методы сравнения

Реализуйте \`__eq__\` и \`__lt__\` для сравнения объектов.

## Задача

Создайте класс \`Product\`, который можно сравнивать по цене.`,
			hint1: 'В __eq__ сравните self.price с other.price. Сначала проверьте isinstance.',
			hint2: 'С __eq__ и __lt__ Python автоматически выводит другие сравнения.',
			whyItMatters: `Методы сравнения позволяют сортировать и искать объекты.`,
		},
		uz: {
			title: 'Taqqoslash metodlari',
			description: `# Taqqoslash metodlari

Ob'ektlarni taqqoslash uchun \`__eq__\` va \`__lt__\` ni amalga oshiring.

## Vazifa

Narx bo'yicha taqqoslanadigan \`Product\` klassini yarating.`,
			hint1: "__eq__ da self.price ni other.price bilan taqqoslang. Avval isinstance tekshiring.",
			hint2: "__eq__ va __lt__ bilan Python boshqa taqqoslashlarni avtomatik chiqaradi.",
			whyItMatters: `Taqqoslash metodlari ob'ektlarni saralash va qidirishga imkon beradi.`,
		},
	},
};

export default task;
