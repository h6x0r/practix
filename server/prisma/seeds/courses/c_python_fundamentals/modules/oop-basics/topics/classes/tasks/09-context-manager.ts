import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-context-manager',
	title: 'Context Manager',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'oop', 'context-manager'],
	estimatedTime: '15m',
	isPremium: false,
	order: 9,

	description: `# Context Manager

Context managers enable setup/cleanup with the \`with\` statement using \`__enter__\` and \`__exit__\`.

## Task

Create a class \`Timer\` that measures execution time of code blocks.

## Requirements

- \`__init__(self)\`: Initialize timer (elapsed_time = 0)
- \`__enter__(self)\`: Record start time, return self
- \`__exit__(self, *args)\`: Calculate and store elapsed time
- \`elapsed_time\`: Attribute with elapsed seconds (float)

## Examples

\`\`\`python
>>> import time
>>> timer = Timer()
>>> with timer:
...     time.sleep(0.1)
>>> 0.09 < timer.elapsed_time < 0.15
True
\`\`\``,

	initialCode: `import time

class Timer:
    """Context manager that measures execution time."""

    def __init__(self):
        # TODO: Initialize elapsed_time
        pass

    def __enter__(self):
        # TODO: Record start time and return self
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Calculate elapsed time
        pass`,

	solutionCode: `import time

class Timer:
    """Context manager that measures execution time."""

    def __init__(self):
        self.elapsed_time = 0.0
        self._start_time = None

    def __enter__(self):
        # Record start time using high-precision timer
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate elapsed time
        self.elapsed_time = time.perf_counter() - self._start_time
        # Return False to propagate any exceptions
        return False`,

	testCode: `import unittest
import time

class Test(unittest.TestCase):
    def test_1(self):
        timer = Timer()
        with timer:
            pass
        self.assertGreaterEqual(timer.elapsed_time, 0)

    def test_2(self):
        timer = Timer()
        with timer:
            time.sleep(0.05)
        self.assertGreater(timer.elapsed_time, 0.04)

    def test_3(self):
        timer = Timer()
        self.assertEqual(timer.elapsed_time, 0.0)

    def test_4(self):
        timer = Timer()
        with timer:
            x = sum(range(1000))
        self.assertIsInstance(timer.elapsed_time, float)

    def test_5(self):
        timer = Timer()
        with timer:
            pass
        self.assertLess(timer.elapsed_time, 1.0)

    def test_6(self):
        timer = Timer()
        with timer:
            time.sleep(0.01)
        t1 = timer.elapsed_time
        with timer:
            time.sleep(0.02)
        self.assertGreater(timer.elapsed_time, t1)

    def test_7(self):
        timer = Timer()
        with timer as t:
            pass
        self.assertIs(t, timer)

    def test_8(self):
        timer = Timer()
        with timer:
            _ = [i*2 for i in range(100)]
        self.assertGreater(timer.elapsed_time, 0)

    def test_9(self):
        timer1 = Timer()
        timer2 = Timer()
        with timer1:
            with timer2:
                time.sleep(0.01)
        self.assertGreater(timer1.elapsed_time, timer2.elapsed_time)

    def test_10(self):
        timer = Timer()
        with timer:
            time.sleep(0.1)
        self.assertGreater(timer.elapsed_time, 0.09)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use time.perf_counter() for high-precision timing. Store start time in __enter__.',
	hint2: '__enter__ must return self so you can use "with Timer() as timer:". __exit__ calculates elapsed.',

	whyItMatters: `Context managers ensure proper resource cleanup and enable clean setup/teardown patterns.

**Production Pattern:**

\`\`\`python
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.4f}s")
\`\`\``,

	translations: {
		ru: {
			title: 'Контекстный менеджер',
			description: `# Контекстный менеджер

Контекстные менеджеры обеспечивают настройку/очистку с оператором \`with\` через \`__enter__\` и \`__exit__\`.

## Задача

Создайте класс \`Timer\`, измеряющий время выполнения блоков кода.`,
			hint1: 'Используйте time.perf_counter() для точного измерения. Сохраните время в __enter__.',
			hint2: '__enter__ должен вернуть self для "with Timer() as timer:". __exit__ вычисляет elapsed.',
			whyItMatters: `Контекстные менеджеры обеспечивают правильную очистку ресурсов.`,
		},
		uz: {
			title: 'Kontekst menejeri',
			description: `# Kontekst menejeri

Kontekst menejerlari \`__enter__\` va \`__exit__\` orqali \`with\` operatori bilan sozlash/tozalashni ta'minlaydi.

## Vazifa

Kod bloklari bajarilish vaqtini o'lchovchi \`Timer\` klassini yarating.`,
			hint1: "Aniq vaqt o'lchash uchun time.perf_counter() ishlating. __enter__ da boshlang'ich vaqtni saqlang.",
			hint2: '"with Timer() as timer:" uchun __enter__ self qaytarishi kerak. __exit__ elapsed ni hisoblaydi.',
			whyItMatters: `Kontekst menejerlari resurslarni to'g'ri tozalashni ta'minlaydi.`,
		},
	},
};

export default task;
