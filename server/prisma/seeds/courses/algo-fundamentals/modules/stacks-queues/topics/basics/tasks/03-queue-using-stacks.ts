import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-queue-using-stacks',
	title: 'Queue Using Stacks',
	difficulty: 'easy',
	tags: ['python', 'stack', 'queue', 'design'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a queue using two stacks.

**Problem:**

Implement a first-in-first-out (FIFO) queue using only two stacks. The queue should support:
- \`Push(x)\` - push element to back of queue
- \`Pop()\` - remove and return front element
- \`Peek()\` - return front element without removing
- \`Empty()\` - return whether queue is empty

**Examples:**

\`\`\`
queue.Push(1);
queue.Push(2);
queue.Peek();  // Returns 1
queue.Pop();   // Returns 1
queue.Empty(); // Returns false
\`\`\`

**Approach:**

Use two stacks: input and output.
- Push: always push to input stack
- Pop/Peek: if output empty, transfer all from input to output (reverses order)

**Why it works:**

Transferring from input to output reverses the LIFO order, giving us FIFO.

**Time Complexity:** O(1) amortized for all operations
**Space Complexity:** O(n)`,
	initialCode: `class MyQueue:
    # TODO: Implement a queue using two stacks

    def __init__(self):
        pass

    def push(self, x: int) -> None:
        pass

    def pop(self) -> int:
        return 0

    def peek(self) -> int:
        return 0

    def empty(self) -> bool:
        return True`,
	solutionCode: `class MyQueue:
    """
    Queue implementation using two stacks.
    """

    def __init__(self):
        """Initialize the queue with two stacks."""
        self.input = []   # For pushing
        self.output = []  # For popping/peeking

    def push(self, x: int) -> None:
        """Push element to back of queue."""
        self.input.append(x)

    def pop(self) -> int:
        """Remove and return front element."""
        self._move()
        if not self.output:
            return 0
        return self.output.pop()

    def peek(self) -> int:
        """Return front element without removing."""
        self._move()
        if not self.output:
            return 0
        return self.output[-1]

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return not self.input and not self.output

    def _move(self) -> None:
        """Transfer elements from input to output if output is empty."""
        if not self.output:
            while self.input:
                # Pop from input, push to output
                self.output.append(self.input.pop())`,
	testCode: `import pytest
from solution import MyQueue


class TestMyQueue:
    def test_basic_operations(self):
        """Test basic push, pop, peek, empty"""
        queue = MyQueue()

        queue.push(1)
        queue.push(2)

        assert queue.peek() == 1
        assert queue.pop() == 1
        assert queue.empty() == False

        queue.push(3)

        assert queue.pop() == 2
        assert queue.pop() == 3
        assert queue.empty() == True

    def test_fifo_order(self):
        """Test FIFO ordering"""
        queue = MyQueue()

        for i in range(5):
            queue.push(i)

        for i in range(5):
            assert queue.pop() == i

    def test_interleaved(self):
        """Test interleaved push and pop"""
        queue = MyQueue()

        queue.push(1)
        assert queue.pop() == 1

        queue.push(2)
        queue.push(3)
        assert queue.pop() == 2

        queue.push(4)
        assert queue.pop() == 3
        assert queue.pop() == 4

    def test_empty_queue(self):
        """Test empty queue"""
        queue = MyQueue()
        assert queue.empty() == True

    def test_single_element(self):
        """Test single element push and pop"""
        queue = MyQueue()
        queue.push(1)
        assert queue.peek() == 1
        assert queue.pop() == 1
        assert queue.empty() == True

    def test_peek_without_pop(self):
        """Test multiple peeks without popping"""
        queue = MyQueue()
        queue.push(1)
        queue.push(2)
        assert queue.peek() == 1
        assert queue.peek() == 1
        assert queue.pop() == 1

    def test_large_sequence(self):
        """Test with larger sequence"""
        queue = MyQueue()
        for i in range(10):
            queue.push(i)
        for i in range(10):
            assert queue.pop() == i
        assert queue.empty() == True

    def test_repeated_empty_checks(self):
        """Test empty checks after operations"""
        queue = MyQueue()
        assert queue.empty() == True
        queue.push(1)
        assert queue.empty() == False
        queue.pop()
        assert queue.empty() == True

    def test_multiple_interleaved(self):
        """Test multiple interleaved operations"""
        queue = MyQueue()
        queue.push(1)
        queue.push(2)
        assert queue.pop() == 1
        queue.push(3)
        queue.push(4)
        assert queue.pop() == 2
        assert queue.pop() == 3
        assert queue.pop() == 4
        assert queue.empty() == True

    def test_peek_after_operations(self):
        """Test peek after various operations"""
        queue = MyQueue()
        queue.push(5)
        queue.push(10)
        queue.push(15)
        queue.pop()
        assert queue.peek() == 10
        queue.pop()
        assert queue.peek() == 15`,
	hint1: `Use two stacks: 'input' for push operations and 'output' for pop/peek. Only transfer from input to output when output is empty.`,
	hint2: `The transfer operation pops all elements from input and pushes to output. This reverses the order, converting LIFO to FIFO.`,
	whyItMatters: `This problem demonstrates how to build one data structure from another.

**Why This Matters:**

**1. Amortized Analysis**

\`\`\`python
# Each element is pushed/popped from each stack at most once
# Total: 2n operations for n elements
# Amortized O(1) per operation

# Worst case single pop: O(n) if all in input
# But this is rare and "pays" for future pops
\`\`\`

**2. Building Data Structures**

Understanding how to compose primitives:
- Queue from stacks (this problem)
- Stack from queues (related problem)
- Deque from arrays

**3. Lazy Evaluation**

We don't transfer immediately; we wait until needed:
\`\`\`python
# Eager: transfer after every push (slower)
# Lazy: transfer only when output empty (faster)
\`\`\`

**4. Real-World Application**

This pattern appears in:
- Message queues (batch processing)
- Event systems (buffering)
- Undo/redo functionality`,
	order: 3,
	translations: {
		ru: {
			title: 'Очередь на стеках',
			description: `Реализуйте очередь используя два стека.

**Задача:**

Реализуйте очередь FIFO (first-in-first-out) используя только два стека. Очередь должна поддерживать:
- \`Push(x)\` - добавить элемент в конец очереди
- \`Pop()\` - удалить и вернуть первый элемент
- \`Peek()\` - вернуть первый элемент без удаления
- \`Empty()\` - проверить пустоту очереди

**Примеры:**

\`\`\`
queue.Push(1);
queue.Push(2);
queue.Peek();  // Возвращает 1
queue.Pop();   // Возвращает 1
queue.Empty(); // Возвращает false
\`\`\`

**Подход:**

Используйте два стека: input и output.
- Push: всегда добавлять в input стек
- Pop/Peek: если output пуст, перенести всё из input в output

**Временная сложность:** O(1) амортизированно
**Пространственная сложность:** O(n)`,
			hint1: `Используйте два стека: 'input' для push операций и 'output' для pop/peek. Переносите из input в output только когда output пуст.`,
			hint2: `Операция переноса извлекает все элементы из input и добавляет в output. Это переворачивает порядок, конвертируя LIFO в FIFO.`,
			whyItMatters: `Эта задача демонстрирует как построить одну структуру данных из другой.

**Почему это важно:**

**1. Амортизированный анализ**

Каждый элемент push/pop из каждого стека максимум один раз.

**2. Построение структур данных**

Понимание как компоновать примитивы.`,
			solutionCode: `class MyQueue:
    """
    Реализация очереди используя два стека.
    """

    def __init__(self):
        """Инициализирует очередь двумя стеками."""
        self.input = []   # Для добавления
        self.output = []  # Для извлечения

    def push(self, x: int) -> None:
        """Добавляет элемент в конец очереди."""
        self.input.append(x)

    def pop(self) -> int:
        """Удаляет и возвращает первый элемент."""
        self._move()
        if not self.output:
            return 0
        return self.output.pop()

    def peek(self) -> int:
        """Возвращает первый элемент без удаления."""
        self._move()
        if not self.output:
            return 0
        return self.output[-1]

    def empty(self) -> bool:
        """Возвращает True если очередь пуста."""
        return not self.input and not self.output

    def _move(self) -> None:
        """Переносит элементы из input в output если output пуст."""
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())`
		},
		uz: {
			title: 'Steklar yordamida navbat',
			description: `Ikki stek yordamida navbat amalga oshiring.

**Masala:**

Faqat ikki stek yordamida FIFO (birinchi kirgan - birinchi chiqadi) navbatni amalga oshiring. Navbat quyidagilarni qo'llab-quvvatlashi kerak:
- \`Push(x)\` - elementni navbat oxiriga qo'shadi
- \`Pop()\` - old elementni olib tashlaydi va qaytaradi
- \`Peek()\` - old elementni olib tashlamay qaytaradi
- \`Empty()\` - navbat bo'sh ekanligini qaytaradi

**Misollar:**

\`\`\`
queue.Push(1);
queue.Push(2);
queue.Peek();  // 1 qaytaradi
queue.Pop();   // 1 qaytaradi
queue.Empty(); // false qaytaradi
\`\`\`

**Yondashuv:**

Ikki stekdan foydalaning: input va output.
- Push: doimo input stekga qo'shing
- Pop/Peek: agar output bo'sh bo'lsa, hammasini input dan output ga o'tkazing

**Vaqt murakkabligi:** Amortizatsiyalangan O(1)
**Xotira murakkabligi:** O(n)`,
			hint1: `Ikki stekdan foydalaning: push operatsiyalari uchun 'input' va pop/peek uchun 'output'. Faqat output bo'sh bo'lganda input dan output ga o'tkazing.`,
			hint2: `O'tkazish operatsiyasi barcha elementlarni input dan oladi va output ga qo'shadi. Bu tartibni teskari aylantiradi, LIFO ni FIFO ga aylantiradi.`,
			whyItMatters: `Bu masala boshqa ma'lumotlar tuzilmasidan bittasini qanday qurish mumkinligini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Amortizatsiyalangan tahlil**

Har bir element har bir stekdan ko'pi bilan bir marta push/pop qilinadi.

**2. Ma'lumotlar tuzilmalarini qurish**

Primitivlarni qanday birlashtirish tushunish.`,
			solutionCode: `class MyQueue:
    """
    Ikki stek yordamida navbat amalga oshirish.
    """

    def __init__(self):
        """Navbatni ikki stek bilan ishga tushiradi."""
        self.input = []   # Qo'shish uchun
        self.output = []  # Olish uchun

    def push(self, x: int) -> None:
        """Elementni navbat oxiriga qo'shadi."""
        self.input.append(x)

    def pop(self) -> int:
        """Old elementni olib tashlaydi va qaytaradi."""
        self._move()
        if not self.output:
            return 0
        return self.output.pop()

    def peek(self) -> int:
        """Old elementni olib tashlamay qaytaradi."""
        self._move()
        if not self.output:
            return 0
        return self.output[-1]

    def empty(self) -> bool:
        """Navbat bo'sh bo'lsa True qaytaradi."""
        return not self.input and not self.output

    def _move(self) -> None:
        """Output bo'sh bo'lsa elementlarni input dan output ga o'tkazadi."""
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())`
		}
	}
};

export default task;
