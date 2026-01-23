import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-property',
	title: 'Properties',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'oop', 'property'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,

	description: `# Properties

Properties allow controlled access to attributes with getters and setters.

## Task

Create a class \`Temperature\` with a property that converts between Celsius and Fahrenheit.

## Requirements

- \`__init__(self, celsius)\`: Initialize with temperature in Celsius
- \`celsius\` property: Get/set temperature in Celsius
- \`fahrenheit\` property: Get/set temperature in Fahrenheit (read-write)
- Formula: F = C × 9/5 + 32, C = (F - 32) × 5/9

## Examples

\`\`\`python
>>> t = Temperature(0)
>>> t.celsius
0
>>> t.fahrenheit
32.0
>>> t.fahrenheit = 212
>>> t.celsius
100.0
\`\`\``,

	initialCode: `class Temperature:
    """Temperature class with Celsius/Fahrenheit conversion."""

    def __init__(self, celsius: float):
        # TODO: Store temperature
        pass

    @property
    def celsius(self) -> float:
        # TODO: Return celsius
        pass

    @celsius.setter
    def celsius(self, value: float):
        # TODO: Set celsius
        pass

    @property
    def fahrenheit(self) -> float:
        # TODO: Convert and return fahrenheit
        pass

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        # TODO: Convert from fahrenheit and store as celsius
        pass`,

	solutionCode: `class Temperature:
    """Temperature class with Celsius/Fahrenheit conversion."""

    def __init__(self, celsius: float):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self._celsius = (value - 32) * 5/9`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        t = Temperature(0)
        self.assertEqual(t.celsius, 0)

    def test_2(self):
        t = Temperature(0)
        self.assertEqual(t.fahrenheit, 32.0)

    def test_3(self):
        t = Temperature(100)
        self.assertEqual(t.fahrenheit, 212.0)

    def test_4(self):
        t = Temperature(0)
        t.fahrenheit = 212
        self.assertAlmostEqual(t.celsius, 100.0, places=1)

    def test_5(self):
        t = Temperature(25)
        self.assertAlmostEqual(t.fahrenheit, 77.0, places=1)

    def test_6(self):
        t = Temperature(0)
        t.celsius = 50
        self.assertEqual(t.celsius, 50)

    def test_7(self):
        t = Temperature(-40)
        self.assertAlmostEqual(t.fahrenheit, -40.0, places=1)

    def test_8(self):
        t = Temperature(0)
        t.fahrenheit = 32
        self.assertAlmostEqual(t.celsius, 0, places=1)

    def test_9(self):
        t = Temperature(37)
        self.assertAlmostEqual(t.fahrenheit, 98.6, places=1)

    def test_10(self):
        t = Temperature(20)
        t.fahrenheit = 68
        self.assertAlmostEqual(t.celsius, 20, places=1)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Store temperature internally as _celsius. The celsius property just gets/sets this value.',
	hint2: 'The fahrenheit property converts using formulas: get returns C*9/5+32, set stores (F-32)*5/9.',

	whyItMatters: `Properties enable computed attributes and data validation while keeping a clean API.`,

	translations: {
		ru: {
			title: 'Свойства',
			description: `# Свойства

Свойства позволяют контролировать доступ к атрибутам через геттеры и сеттеры.

## Задача

Создайте класс \`Temperature\` со свойством для конвертации между Цельсием и Фаренгейтом.`,
			hint1: 'Храните температуру как _celsius. Свойство celsius просто получает/устанавливает это значение.',
			hint2: 'Свойство fahrenheit конвертирует: get возвращает C*9/5+32, set сохраняет (F-32)*5/9.',
			whyItMatters: `Свойства позволяют создавать вычисляемые атрибуты и валидацию данных.`,
		},
		uz: {
			title: 'Xususiyatlar',
			description: `# Xususiyatlar

Xususiyatlar getterlar va setterlar orqali atributlarga boshqariladigan kirishni ta'minlaydi.

## Vazifa

Selsiy va Farengeyt orasida konvertatsiya qiluvchi xususiyatga ega \`Temperature\` klassini yarating.`,
			hint1: "Temperaturani ichki _celsius sifatida saqlang. celsius xususiyati shunchaki bu qiymatni oladi/o'rnatadi.",
			hint2: "fahrenheit xususiyati konvertatsiya qiladi: get C*9/5+32 qaytaradi, set (F-32)*5/9 saqlaydi.",
			whyItMatters: `Xususiyatlar hisoblangan atributlar va ma'lumotlarni tekshirishni ta'minlaydi.`,
		},
	},
};

export default task;
