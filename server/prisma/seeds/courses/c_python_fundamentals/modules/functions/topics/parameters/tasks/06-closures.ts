import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-closures',
	title: 'Closures',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'functions', 'closures'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,

	description: `# Closures

A closure is a function that remembers values from its enclosing scope.

## Task

Implement the function \`make_counter(start)\` that creates a counter function.

## Requirements

- \`make_counter(start)\` returns a function
- Each call to the returned function increments and returns the counter
- Start from the given start value
- Multiple counters should be independent

## Examples

\`\`\`python
>>> counter1 = make_counter(0)
>>> counter1()
1
>>> counter1()
2
>>> counter1()
3

>>> counter2 = make_counter(10)
>>> counter2()
11
>>> counter1()  # counter1 is independent
4
\`\`\``,

	initialCode: `def make_counter(start: int):
    """Create a counter function that increments and returns its value.

    Args:
        start: Initial value for the counter

    Returns:
        A function that increments and returns the counter each time called
    """
    # TODO: Implement closure
    pass`,

	solutionCode: `def make_counter(start: int):
    """Create a counter function that increments and returns its value.

    Args:
        start: Initial value for the counter

    Returns:
        A function that increments and returns the counter each time called
    """
    # This variable is "enclosed" by the inner function
    count = start

    def counter():
        # nonlocal allows us to modify the enclosed variable
        nonlocal count
        count += 1
        return count

    return counter`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Counter starts from 0"""
        counter = make_counter(0)
        self.assertEqual(counter(), 1)
        self.assertEqual(counter(), 2)
        self.assertEqual(counter(), 3)

    def test_2(self):
        """Counter starts from 10"""
        counter = make_counter(10)
        self.assertEqual(counter(), 11)
        self.assertEqual(counter(), 12)

    def test_3(self):
        """Independent counters"""
        c1 = make_counter(0)
        c2 = make_counter(100)
        self.assertEqual(c1(), 1)
        self.assertEqual(c2(), 101)
        self.assertEqual(c1(), 2)
        self.assertEqual(c2(), 102)

    def test_4(self):
        """Negative start"""
        counter = make_counter(-5)
        self.assertEqual(counter(), -4)
        self.assertEqual(counter(), -3)

    def test_5(self):
        """Many calls"""
        counter = make_counter(0)
        for i in range(1, 11):
            self.assertEqual(counter(), i)

    def test_6(self):
        """Start from 1"""
        counter = make_counter(1)
        self.assertEqual(counter(), 2)

    def test_7(self):
        """Three independent counters"""
        c1 = make_counter(0)
        c2 = make_counter(0)
        c3 = make_counter(0)
        c1()
        c1()
        c2()
        self.assertEqual(c1(), 3)
        self.assertEqual(c2(), 2)
        self.assertEqual(c3(), 1)

    def test_8(self):
        """Large start value"""
        counter = make_counter(1000000)
        self.assertEqual(counter(), 1000001)

    def test_9(self):
        """Zero increments correctly"""
        counter = make_counter(0)
        self.assertEqual(counter(), 1)

    def test_10(self):
        """Returns callable"""
        counter = make_counter(5)
        self.assertTrue(callable(counter))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Define an inner function that accesses a variable from the outer scope. Use "nonlocal" to modify it.',
	hint2: 'The pattern is: outer function sets up state, inner function uses/modifies that state. Return the inner function.',

	whyItMatters: `Closures are fundamental to functional programming and are used extensively in callbacks, event handlers, and factory functions.

**Production Pattern:**

\`\`\`python
def create_logger(prefix: str):
    """Factory for creating prefixed loggers."""
    def log(message: str):
        print(f"[{prefix}] {message}")
    return log

info = create_logger("INFO")
error = create_logger("ERROR")
info("Server started")  # [INFO] Server started
error("Connection failed")  # [ERROR] Connection failed

def create_rate_limiter(max_calls: int, period_seconds: float):
    """Create a rate limiter using closure."""
    import time
    calls = []

    def is_allowed() -> bool:
        nonlocal calls
        now = time.time()
        # Remove old calls outside the period
        calls = [t for t in calls if now - t < period_seconds]
        if len(calls) < max_calls:
            calls.append(now)
            return True
        return False

    return is_allowed

def create_cache():
    """Simple memoization cache using closure."""
    cache = {}

    def cached_func(func):
        def wrapper(*args):
            if args not in cache:
                cache[args] = func(*args)
            return cache[args]
        return wrapper

    return cached_func
\`\`\`

**Practical Benefits:**
- Encapsulation of state without classes
- Factory functions for customized behavior
- Event handlers and callbacks in async code`,

	translations: {
		ru: {
			title: 'Замыкания',
			description: `# Замыкания

Замыкание — это функция, которая запоминает значения из своей внешней области видимости.

## Задача

Реализуйте функцию \`make_counter(start)\`, которая создаёт функцию-счётчик.

## Требования

- \`make_counter(start)\` возвращает функцию
- Каждый вызов возвращённой функции увеличивает и возвращает счётчик
- Начните с заданного значения start
- Несколько счётчиков должны быть независимы

## Примеры

\`\`\`python
>>> counter1 = make_counter(0)
>>> counter1()
1
>>> counter1()
2
>>> counter1()
3

>>> counter2 = make_counter(10)
>>> counter2()
11
>>> counter1()  # counter1 независим
4
\`\`\``,
			hint1: 'Определите внутреннюю функцию с доступом к переменной внешней области. Используйте "nonlocal" для изменения.',
			hint2: 'Паттерн: внешняя функция задаёт состояние, внутренняя использует/изменяет его. Верните внутреннюю функцию.',
			whyItMatters: `Замыкания фундаментальны для функционального программирования и используются в колбэках и фабриках.

**Продакшен паттерн:**

\`\`\`python
def create_logger(prefix: str):
    """Фабрика для создания логгеров с префиксом."""
    def log(message: str):
        print(f"[{prefix}] {message}")
    return log

info = create_logger("INFO")
error = create_logger("ERROR")
info("Сервер запущен")  # [INFO] Сервер запущен

def create_rate_limiter(max_calls: int, period_seconds: float):
    """Создание ограничителя частоты через замыкание."""
    import time
    calls = []

    def is_allowed() -> bool:
        nonlocal calls
        now = time.time()
        calls = [t for t in calls if now - t < period_seconds]
        if len(calls) < max_calls:
            calls.append(now)
            return True
        return False

    return is_allowed
\`\`\`

**Практические преимущества:**
- Инкапсуляция состояния без классов
- Фабричные функции для кастомизации`,
		},
		uz: {
			title: 'Yopilmalar',
			description: `# Yopilmalar

Yopilma — o'z tashqi sohasidagi qiymatlarni eslab qoladigan funksiya.

## Vazifa

Hisoblagich funksiyasini yaratuvchi \`make_counter(start)\` funksiyasini amalga oshiring.

## Talablar

- \`make_counter(start)\` funksiya qaytaradi
- Qaytarilgan funksiyaning har bir chaqiruvi hisoblagichni oshiradi va qaytaradi
- Berilgan start qiymatidan boshlang
- Bir nechta hisoblagichlar mustaqil bo'lishi kerak

## Misollar

\`\`\`python
>>> counter1 = make_counter(0)
>>> counter1()
1
>>> counter1()
2
>>> counter1()
3

>>> counter2 = make_counter(10)
>>> counter2()
11
>>> counter1()  # counter1 mustaqil
4
\`\`\``,
			hint1: "Tashqi sohadagi o'zgaruvchiga kiradigan ichki funksiya belgilang. Uni o'zgartirish uchun \"nonlocal\" ishlating.",
			hint2: "Pattern: tashqi funksiya holatni o'rnatadi, ichki foydalanadi/o'zgartiradi. Ichki funksiyani qaytaring.",
			whyItMatters: `Yopilmalar funksional dasturlash uchun asosiy va callback va fabrika funksiyalarida keng ishlatiladi.

**Ishlab chiqarish patterni:**

\`\`\`python
def create_logger(prefix: str):
    """Prefiksli loggerlar yaratish fabrikasi."""
    def log(message: str):
        print(f"[{prefix}] {message}")
    return log

info = create_logger("INFO")
error = create_logger("ERROR")
info("Server ishga tushdi")  # [INFO] Server ishga tushdi

def create_rate_limiter(max_calls: int, period_seconds: float):
    """Yopilma orqali tezlik cheklovchisi yaratish."""
    import time
    calls = []

    def is_allowed() -> bool:
        nonlocal calls
        now = time.time()
        calls = [t for t in calls if now - t < period_seconds]
        if len(calls) < max_calls:
            calls.append(now)
            return True
        return False

    return is_allowed
\`\`\`

**Amaliy foydalari:**
- Klasslarsiz holatni inkapsulatsiya qilish
- Moslashtirish uchun fabrika funksiyalari`,
		},
	},
};

export default task;
