import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-basic-class',
	title: 'Basic Class',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'oop', 'classes'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,

	description: `# Basic Class

Classes are blueprints for creating objects with attributes and methods.

## Task

Create a class \`Rectangle\` that represents a rectangle with width and height.

## Requirements

- \`__init__(self, width, height)\`: Initialize with width and height
- \`area(self)\`: Return the area (width × height)
- \`perimeter(self)\`: Return the perimeter (2 × (width + height))
- \`is_square(self)\`: Return True if width equals height

## Examples

\`\`\`python
>>> rect = Rectangle(4, 5)
>>> rect.area()
20
>>> rect.perimeter()
18
>>> rect.is_square()
False

>>> square = Rectangle(5, 5)
>>> square.is_square()
True
\`\`\``,

	initialCode: `class Rectangle:
    """A class representing a rectangle.

    Attributes:
        width: The width of the rectangle
        height: The height of the rectangle
    """

    def __init__(self, width: float, height: float):
        """Initialize rectangle with width and height."""
        # TODO: Store width and height
        pass

    def area(self) -> float:
        """Calculate and return the area."""
        # TODO: Implement
        pass

    def perimeter(self) -> float:
        """Calculate and return the perimeter."""
        # TODO: Implement
        pass

    def is_square(self) -> bool:
        """Check if the rectangle is a square."""
        # TODO: Implement
        pass`,

	solutionCode: `class Rectangle:
    """A class representing a rectangle.

    Attributes:
        width: The width of the rectangle
        height: The height of the rectangle
    """

    def __init__(self, width: float, height: float):
        """Initialize rectangle with width and height."""
        # Store dimensions as instance attributes
        self.width = width
        self.height = height

    def area(self) -> float:
        """Calculate and return the area."""
        # Area = width × height
        return self.width * self.height

    def perimeter(self) -> float:
        """Calculate and return the perimeter."""
        # Perimeter = 2 × (width + height)
        return 2 * (self.width + self.height)

    def is_square(self) -> bool:
        """Check if the rectangle is a square."""
        # Square has equal width and height
        return self.width == self.height`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic area calculation"""
        rect = Rectangle(4, 5)
        self.assertEqual(rect.area(), 20)

    def test_2(self):
        """Basic perimeter calculation"""
        rect = Rectangle(4, 5)
        self.assertEqual(rect.perimeter(), 18)

    def test_3(self):
        """Rectangle is not a square"""
        rect = Rectangle(4, 5)
        self.assertFalse(rect.is_square())

    def test_4(self):
        """Square detection"""
        square = Rectangle(5, 5)
        self.assertTrue(square.is_square())

    def test_5(self):
        """Width attribute accessible"""
        rect = Rectangle(3, 7)
        self.assertEqual(rect.width, 3)

    def test_6(self):
        """Height attribute accessible"""
        rect = Rectangle(3, 7)
        self.assertEqual(rect.height, 7)

    def test_7(self):
        """Float dimensions"""
        rect = Rectangle(2.5, 4.0)
        self.assertEqual(rect.area(), 10.0)

    def test_8(self):
        """Unit square"""
        rect = Rectangle(1, 1)
        self.assertEqual(rect.area(), 1)
        self.assertEqual(rect.perimeter(), 4)

    def test_9(self):
        """Large dimensions"""
        rect = Rectangle(100, 200)
        self.assertEqual(rect.area(), 20000)

    def test_10(self):
        """Zero dimension"""
        rect = Rectangle(0, 5)
        self.assertEqual(rect.area(), 0)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'In __init__, store width and height using self.width = width and self.height = height.',
	hint2: 'Access attributes in methods using self.width and self.height.',

	whyItMatters: `Classes are the foundation of object-oriented programming, enabling code organization and reuse.

**Production Pattern:**

\`\`\`python
from dataclasses import dataclass
from typing import Self

@dataclass
class Rectangle:
    """Modern Python dataclass approach."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def scale(self, factor: float) -> Self:
        """Return a new scaled rectangle."""
        return Rectangle(self.width * factor, self.height * factor)

    def __str__(self) -> str:
        return f"Rectangle({self.width}x{self.height})"

class Shape(ABC):
    """Abstract base class for shapes."""
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass
\`\`\`

**Practical Benefits:**
- Encapsulate related data and behavior
- Create reusable, modular code
- Model real-world concepts clearly`,

	translations: {
		ru: {
			title: 'Базовый класс',
			description: `# Базовый класс

Классы — это шаблоны для создания объектов с атрибутами и методами.

## Задача

Создайте класс \`Rectangle\`, представляющий прямоугольник с шириной и высотой.

## Требования

- \`__init__(self, width, height)\`: Инициализация с шириной и высотой
- \`area(self)\`: Возврат площади (ширина × высота)
- \`perimeter(self)\`: Возврат периметра (2 × (ширина + высота))
- \`is_square(self)\`: Возврат True, если ширина равна высоте

## Примеры

\`\`\`python
>>> rect = Rectangle(4, 5)
>>> rect.area()
20
>>> rect.perimeter()
18
>>> rect.is_square()
False

>>> square = Rectangle(5, 5)
>>> square.is_square()
True
\`\`\``,
			hint1: 'В __init__ сохраните width и height через self.width = width и self.height = height.',
			hint2: 'Обращайтесь к атрибутам в методах через self.width и self.height.',
			whyItMatters: `Классы — основа ООП, обеспечивающая организацию и повторное использование кода.

**Продакшен паттерн:**

\`\`\`python
from dataclasses import dataclass

@dataclass
class Rectangle:
    """Современный подход с dataclass."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def scale(self, factor: float):
        return Rectangle(self.width * factor, self.height * factor)
\`\`\`

**Практические преимущества:**
- Инкапсуляция данных и поведения
- Создание переиспользуемого модульного кода`,
		},
		uz: {
			title: 'Asosiy klass',
			description: `# Asosiy klass

Klasslar atributlar va metodlarga ega ob'ektlar yaratish uchun shablonlardir.

## Vazifa

Kenglik va balandlikka ega to'rtburchakni ifodalovchi \`Rectangle\` klassini yarating.

## Talablar

- \`__init__(self, width, height)\`: Kenglik va balandlik bilan ishga tushirish
- \`area(self)\`: Maydonni qaytarish (kenglik × balandlik)
- \`perimeter(self)\`: Perimetrni qaytarish (2 × (kenglik + balandlik))
- \`is_square(self)\`: Kenglik balandlikka teng bo'lsa True qaytarish

## Misollar

\`\`\`python
>>> rect = Rectangle(4, 5)
>>> rect.area()
20
>>> rect.perimeter()
18
>>> rect.is_square()
False

>>> square = Rectangle(5, 5)
>>> square.is_square()
True
\`\`\``,
			hint1: "__init__ da width va height ni self.width = width va self.height = height orqali saqlang.",
			hint2: "Metodlarda atributlarga self.width va self.height orqali kiring.",
			whyItMatters: `Klasslar OOP ning asosidir va kodni tashkil etish hamda qayta foydalanishni ta'minlaydi.

**Ishlab chiqarish patterni:**

\`\`\`python
from dataclasses import dataclass

@dataclass
class Rectangle:
    """dataclass bilan zamonaviy yondashuv."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def scale(self, factor: float):
        return Rectangle(self.width * factor, self.height * factor)
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlar va xatti-harakatlarni inkapsulatsiya qilish
- Qayta foydalaniladigan modulli kod yaratish`,
		},
	},
};

export default task;
