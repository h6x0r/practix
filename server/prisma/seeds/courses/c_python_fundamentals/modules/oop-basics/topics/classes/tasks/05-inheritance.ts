import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-inheritance',
	title: 'Basic Inheritance',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'oop', 'inheritance'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,

	description: `# Basic Inheritance

Inheritance allows a class to inherit attributes and methods from a parent class.

## Task

Create a class hierarchy: \`Animal\` (base) → \`Dog\` (child).

## Requirements

**Animal class:**
- \`__init__(self, name)\`: Initialize with name
- \`speak(self)\`: Return "Some sound"

**Dog class (inherits from Animal):**
- \`__init__(self, name, breed)\`: Initialize with name and breed
- \`speak(self)\`: Override to return "Woof!"
- \`fetch(self)\`: Return "{name} is fetching!"

## Examples

\`\`\`python
>>> animal = Animal("Generic")
>>> animal.speak()
"Some sound"

>>> dog = Dog("Buddy", "Golden Retriever")
>>> dog.speak()
"Woof!"
>>> dog.fetch()
"Buddy is fetching!"
>>> dog.name
"Buddy"
\`\`\``,

	initialCode: `class Animal:
    """Base class for animals."""

    def __init__(self, name: str):
        # TODO: Initialize name
        pass

    def speak(self) -> str:
        # TODO: Return generic sound
        pass


class Dog(Animal):
    """Dog class inheriting from Animal."""

    def __init__(self, name: str, breed: str):
        # TODO: Call parent __init__ and set breed
        pass

    def speak(self) -> str:
        # TODO: Override with dog sound
        pass

    def fetch(self) -> str:
        # TODO: Dog-specific method
        pass`,

	solutionCode: `class Animal:
    """Base class for animals."""

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "Some sound"


class Dog(Animal):
    """Dog class inheriting from Animal."""

    def __init__(self, name: str, breed: str):
        # Call parent class __init__ using super()
        super().__init__(name)
        self.breed = breed

    def speak(self) -> str:
        # Override parent method
        return "Woof!"

    def fetch(self) -> str:
        # New method specific to Dog
        return f"{self.name} is fetching!"`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        animal = Animal("Generic")
        self.assertEqual(animal.speak(), "Some sound")

    def test_2(self):
        dog = Dog("Buddy", "Golden")
        self.assertEqual(dog.speak(), "Woof!")

    def test_3(self):
        dog = Dog("Buddy", "Golden")
        self.assertEqual(dog.fetch(), "Buddy is fetching!")

    def test_4(self):
        dog = Dog("Max", "Labrador")
        self.assertEqual(dog.name, "Max")

    def test_5(self):
        dog = Dog("Max", "Labrador")
        self.assertEqual(dog.breed, "Labrador")

    def test_6(self):
        animal = Animal("Test")
        self.assertEqual(animal.name, "Test")

    def test_7(self):
        dog = Dog("Rex", "German Shepherd")
        self.assertIsInstance(dog, Animal)

    def test_8(self):
        dog = Dog("Rex", "German Shepherd")
        self.assertIsInstance(dog, Dog)

    def test_9(self):
        dog = Dog("Spot", "Dalmatian")
        self.assertEqual(dog.fetch(), "Spot is fetching!")

    def test_10(self):
        animal = Animal("A")
        dog = Dog("D", "B")
        self.assertNotEqual(animal.speak(), dog.speak())

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use class Dog(Animal): to inherit from Animal. Call super().__init__(name) in Dog\'s __init__.',
	hint2: 'Override speak() in Dog by simply defining a new speak() method that returns "Woof!".',

	whyItMatters: `Inheritance enables code reuse and establishes "is-a" relationships between classes.

**Production Pattern:**

\`\`\`python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2
\`\`\``,

	translations: {
		ru: {
			title: 'Основы наследования',
			description: `# Основы наследования

Наследование позволяет классу получать атрибуты и методы от родительского класса.

## Задача

Создайте иерархию классов: \`Animal\` (базовый) → \`Dog\` (дочерний).`,
			hint1: 'Используйте class Dog(Animal): для наследования. Вызовите super().__init__(name) в __init__ Dog.',
			hint2: 'Переопределите speak() в Dog, определив новый метод speak() с "Woof!".',
			whyItMatters: `Наследование обеспечивает повторное использование кода и отношения "является".`,
		},
		uz: {
			title: 'Meros asoslari',
			description: `# Meros asoslari

Meros klassga ota klassdan atributlar va metodlarni olish imkonini beradi.

## Vazifa

Klasslar ierarxiyasini yarating: \`Animal\` (asos) → \`Dog\` (bola).`,
			hint1: "Merosxo'rlik uchun class Dog(Animal): ishlating. Dog ning __init__ da super().__init__(name) chaqiring.",
			hint2: 'Dog da speak() ni qayta aniqlang, "Woof!" qaytaruvchi yangi speak() metodi bilan.',
			whyItMatters: `Meros kodni qayta ishlatish va "...dir" munosabatlarini o'rnatishga imkon beradi.`,
		},
	},
};

export default task;
