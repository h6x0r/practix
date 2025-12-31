import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-deque-operations',
    title: 'Deque Double-Ended Operations',
    difficulty: 'easy',
    tags: ['java', 'collections', 'deque', 'stack'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn Deque (Double-Ended Queue) operations in Java.

**Requirements:**
1. Create a Deque using LinkedList
2. Add elements at both ends: addFirst() and addLast()
3. Remove elements from both ends: removeFirst() and removeLast()
4. Peek at both ends: peekFirst() and peekLast()
5. Demonstrate using Deque as a Stack (LIFO)
6. Demonstrate using Deque as a Queue (FIFO)
7. Use offerFirst/offerLast and pollFirst/pollLast
8. Show size and iteration

Deque extends Queue and provides operations for both ends, making it versatile for Stack and Queue implementations.`,
    initialCode: `import java.util.Deque;
import java.util.LinkedList;

public class DequeOperations {
    public static void main(String[] args) {
        // Create a Deque using LinkedList

        // Add elements at both ends

        // Peek at both ends

        // Remove from both ends

        // Use Deque as Stack (LIFO)

        // Use Deque as Queue (FIFO)

        // Use offer/poll methods
    }
}`,
    solutionCode: `import java.util.Deque;
import java.util.LinkedList;

public class DequeOperations {
    public static void main(String[] args) {
        // Create a Deque using LinkedList
        Deque<String> deque = new LinkedList<>();

        // Add elements at both ends
        deque.addFirst("First");
        deque.addLast("Last");
        deque.addFirst("NewFirst");
        deque.addLast("NewLast");
        System.out.println("Deque after adding: " + deque);

        // Peek at both ends without removal
        System.out.println("\\nPeeking at ends:");
        System.out.println("peekFirst(): " + deque.peekFirst());
        System.out.println("peekLast(): " + deque.peekLast());
        System.out.println("Deque unchanged: " + deque);

        // Remove from both ends
        System.out.println("\\nRemoving from ends:");
        System.out.println("removeFirst(): " + deque.removeFirst());
        System.out.println("removeLast(): " + deque.removeLast());
        System.out.println("Deque after removal: " + deque);

        // Use Deque as Stack (LIFO - Last In First Out)
        System.out.println("\\nDeque as Stack (LIFO):");
        Deque<Integer> stack = new LinkedList<>();
        stack.push(1);  // addFirst()
        stack.push(2);
        stack.push(3);
        System.out.println("Stack: " + stack);
        System.out.println("pop(): " + stack.pop());  // removeFirst()
        System.out.println("pop(): " + stack.pop());
        System.out.println("Stack after pops: " + stack);

        // Use Deque as Queue (FIFO - First In First Out)
        System.out.println("\\nDeque as Queue (FIFO):");
        Deque<String> queue = new LinkedList<>();
        queue.offer("A");     // addLast()
        queue.offer("B");
        queue.offer("C");
        System.out.println("Queue: " + queue);
        System.out.println("poll(): " + queue.poll());  // removeFirst()
        System.out.println("poll(): " + queue.poll());
        System.out.println("Queue after polls: " + queue);

        // Use offer/poll methods (safe versions)
        System.out.println("\\nUsing offer/poll methods:");
        Deque<String> safeDeque = new LinkedList<>();
        safeDeque.offerFirst("Front");
        safeDeque.offerLast("Back");
        System.out.println("SafeDeque: " + safeDeque);
        System.out.println("pollFirst(): " + safeDeque.pollFirst());
        System.out.println("pollLast(): " + safeDeque.pollLast());
        System.out.println("pollFirst() on empty: " + safeDeque.pollFirst()); // Returns null

        // Show size and iteration
        Deque<String> items = new LinkedList<>();
        items.add("Item1");
        items.add("Item2");
        items.add("Item3");
        System.out.println("\\nIteration:");
        System.out.println("Size: " + items.size());
        for (String item : items) {
            System.out.println("  - " + item);
        }
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.Deque;
import java.util.LinkedList;

// Test1: Verify Deque creation and addFirst/addLast
class Test1 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        deque.addFirst("First");
        deque.addLast("Last");
        assertEquals(2, deque.size());
        assertEquals("First", deque.peekFirst());
        assertEquals("Last", deque.peekLast());
    }
}

// Test2: Verify peekFirst and peekLast don't remove elements
class Test2 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        deque.addFirst("A");
        deque.addLast("B");
        assertEquals("A", deque.peekFirst());
        assertEquals("B", deque.peekLast());
        assertEquals(2, deque.size());
    }
}

// Test3: Verify removeFirst and removeLast
class Test3 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        deque.addFirst("A");
        deque.addLast("B");
        deque.addLast("C");
        assertEquals("A", deque.removeFirst());
        assertEquals("C", deque.removeLast());
        assertEquals(1, deque.size());
    }
}

// Test4: Verify Deque as Stack (LIFO) using push/pop
class Test4 {
    @Test
    public void test() {
        Deque<Integer> stack = new LinkedList<>();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        assertEquals(Integer.valueOf(3), stack.pop());
        assertEquals(Integer.valueOf(2), stack.pop());
        assertEquals(Integer.valueOf(1), stack.pop());
    }
}

// Test5: Verify Deque as Queue (FIFO) using offer/poll
class Test5 {
    @Test
    public void test() {
        Deque<String> queue = new LinkedList<>();
        queue.offer("A");
        queue.offer("B");
        queue.offer("C");
        assertEquals("A", queue.poll());
        assertEquals("B", queue.poll());
        assertEquals("C", queue.poll());
    }
}

// Test6: Verify offerFirst and offerLast
class Test6 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        assertTrue(deque.offerFirst("Front"));
        assertTrue(deque.offerLast("Back"));
        assertEquals("Front", deque.peekFirst());
        assertEquals("Back", deque.peekLast());
    }
}

// Test7: Verify pollFirst and pollLast
class Test7 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        deque.offer("A");
        deque.offer("B");
        deque.offer("C");
        assertEquals("A", deque.pollFirst());
        assertEquals("C", deque.pollLast());
        assertEquals(1, deque.size());
    }
}

// Test8: Verify pollFirst returns null on empty deque
class Test8 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        assertNull(deque.pollFirst());
        assertNull(deque.pollLast());
    }
}

// Test9: Verify size and isEmpty methods
class Test9 {
    @Test
    public void test() {
        Deque<String> deque = new LinkedList<>();
        assertTrue(deque.isEmpty());
        deque.add("Test");
        assertFalse(deque.isEmpty());
        assertEquals(1, deque.size());
    }
}

// Test10: Verify iteration over Deque
class Test10 {
    @Test
    public void test() {
        Deque<Integer> deque = new LinkedList<>();
        deque.add(1);
        deque.add(2);
        deque.add(3);
        int sum = 0;
        for (Integer num : deque) {
            sum += num;
        }
        assertEquals(6, sum);
    }
}
`,
    hint1: `Deque has methods for both ends: addFirst/addLast, removeFirst/removeLast, peekFirst/peekLast. It also has push/pop methods for Stack operations.`,
    hint2: `Use push/pop for Stack (LIFO) behavior. Use offer/poll or addLast/removeFirst for Queue (FIFO) behavior. The offer/poll methods are safer as they don't throw exceptions.`,
    whyItMatters: `Deque is one of the most versatile collections in Java. Understanding its double-ended operations allows you to implement both Stacks and Queues efficiently, and handle complex data flow patterns.

**Production Pattern:**
\`\`\`java
// Action history with undo (undo/redo)
Deque<Action> undoStack = new LinkedList<>();
undoStack.push(userAction);  // Add action
Action lastAction = undoStack.pop();  // Undo last
lastAction.undo();

// Sliding window for metrics
Deque<Metric> recentMetrics = new LinkedList<>();
recentMetrics.addLast(new Metric(value, timestamp));
if (recentMetrics.size() > WINDOW_SIZE) {
    recentMetrics.removeFirst();  // Remove old ones
}
double average = calculateAverage(recentMetrics);
\`\`\`

**Practical Benefits:**
- Efficient undo/redo functionality implementation
- Sliding window for temporal data analysis
- O(1) operations from both ends`,
    order: 1,
    translations: {
        ru: {
            title: 'Двусторонние операции Deque',
            solutionCode: `import java.util.Deque;
import java.util.LinkedList;

public class DequeOperations {
    public static void main(String[] args) {
        // Создаем Deque используя LinkedList
        Deque<String> deque = new LinkedList<>();

        // Добавляем элементы с обоих концов
        deque.addFirst("First");
        deque.addLast("Last");
        deque.addFirst("NewFirst");
        deque.addLast("NewLast");
        System.out.println("Deque после добавления: " + deque);

        // Смотрим на оба конца без удаления
        System.out.println("\\nПросмотр концов:");
        System.out.println("peekFirst(): " + deque.peekFirst());
        System.out.println("peekLast(): " + deque.peekLast());
        System.out.println("Deque не изменен: " + deque);

        // Удаляем с обоих концов
        System.out.println("\\nУдаление с концов:");
        System.out.println("removeFirst(): " + deque.removeFirst());
        System.out.println("removeLast(): " + deque.removeLast());
        System.out.println("Deque после удаления: " + deque);

        // Используем Deque как Stack (LIFO - Last In First Out)
        System.out.println("\\nDeque как Stack (LIFO):");
        Deque<Integer> stack = new LinkedList<>();
        stack.push(1);  // addFirst()
        stack.push(2);
        stack.push(3);
        System.out.println("Stack: " + stack);
        System.out.println("pop(): " + stack.pop());  // removeFirst()
        System.out.println("pop(): " + stack.pop());
        System.out.println("Stack после pop: " + stack);

        // Используем Deque как Queue (FIFO - First In First Out)
        System.out.println("\\nDeque как Queue (FIFO):");
        Deque<String> queue = new LinkedList<>();
        queue.offer("A");     // addLast()
        queue.offer("B");
        queue.offer("C");
        System.out.println("Queue: " + queue);
        System.out.println("poll(): " + queue.poll());  // removeFirst()
        System.out.println("poll(): " + queue.poll());
        System.out.println("Queue после poll: " + queue);

        // Используем методы offer/poll (безопасные версии)
        System.out.println("\\nИспользование методов offer/poll:");
        Deque<String> safeDeque = new LinkedList<>();
        safeDeque.offerFirst("Front");
        safeDeque.offerLast("Back");
        System.out.println("SafeDeque: " + safeDeque);
        System.out.println("pollFirst(): " + safeDeque.pollFirst());
        System.out.println("pollLast(): " + safeDeque.pollLast());
        System.out.println("pollFirst() на пустом: " + safeDeque.pollFirst()); // Возвращает null

        // Показываем размер и итерацию
        Deque<String> items = new LinkedList<>();
        items.add("Item1");
        items.add("Item2");
        items.add("Item3");
        System.out.println("\\nИтерация:");
        System.out.println("Размер: " + items.size());
        for (String item : items) {
            System.out.println("  - " + item);
        }
    }
}`,
            description: `Изучите операции Deque (двусторонней очереди) в Java.

**Требования:**
1. Создайте Deque используя LinkedList
2. Добавьте элементы с обоих концов: addFirst() и addLast()
3. Удалите элементы с обоих концов: removeFirst() и removeLast()
4. Просмотрите оба конца: peekFirst() и peekLast()
5. Продемонстрируйте использование Deque как Stack (LIFO)
6. Продемонстрируйте использование Deque как Queue (FIFO)
7. Используйте offerFirst/offerLast и pollFirst/pollLast
8. Покажите размер и итерацию

Deque расширяет Queue и предоставляет операции для обоих концов, что делает его универсальным для реализации Stack и Queue.`,
            hint1: `Deque имеет методы для обоих концов: addFirst/addLast, removeFirst/removeLast, peekFirst/peekLast. Также есть методы push/pop для операций Stack.`,
            hint2: `Используйте push/pop для поведения Stack (LIFO). Используйте offer/poll или addLast/removeFirst для поведения Queue (FIFO). Методы offer/poll безопаснее, так как не выбрасывают исключения.`,
            whyItMatters: `Deque - одна из самых универсальных коллекций в Java. Понимание его двусторонних операций позволяет эффективно реализовывать как Stack, так и Queue, и обрабатывать сложные паттерны потоков данных.

**Продакшен паттерн:**
\`\`\`java
// История действий с отменой (undo/redo)
Deque<Action> undoStack = new LinkedList<>();
undoStack.push(userAction);  // Добавляем действие
Action lastAction = undoStack.pop();  // Отменяем последнее
lastAction.undo();

// Скользящее окно для метрик
Deque<Metric> recentMetrics = new LinkedList<>();
recentMetrics.addLast(new Metric(value, timestamp));
if (recentMetrics.size() > WINDOW_SIZE) {
    recentMetrics.removeFirst();  // Удаляем старые
}
double average = calculateAverage(recentMetrics);
\`\`\`

**Практические преимущества:**
- Эффективная реализация undo/redo функциональности
- Скользящее окно для анализа временных данных
- O(1) операции с обоих концов`
        },
        uz: {
            title: 'Deque Ikki Tomonlama Operatsiyalari',
            solutionCode: `import java.util.Deque;
import java.util.LinkedList;

public class DequeOperations {
    public static void main(String[] args) {
        // LinkedList yordamida Deque yaratamiz
        Deque<String> deque = new LinkedList<>();

        // Ikki tomondan elementlar qo'shamiz
        deque.addFirst("First");
        deque.addLast("Last");
        deque.addFirst("NewFirst");
        deque.addLast("NewLast");
        System.out.println("Qo'shgandan keyin Deque: " + deque);

        // Ikki tomonni o'chirmasdan ko'ramiz
        System.out.println("\\nTomonlarni ko'rish:");
        System.out.println("peekFirst(): " + deque.peekFirst());
        System.out.println("peekLast(): " + deque.peekLast());
        System.out.println("Deque o'zgarmagan: " + deque);

        // Ikki tomondan o'chiramiz
        System.out.println("\\nTomonlardan o'chirish:");
        System.out.println("removeFirst(): " + deque.removeFirst());
        System.out.println("removeLast(): " + deque.removeLast());
        System.out.println("O'chirishdan keyin Deque: " + deque);

        // Deque ni Stack sifatida ishlatamiz (LIFO - Last In First Out)
        System.out.println("\\nDeque Stack sifatida (LIFO):");
        Deque<Integer> stack = new LinkedList<>();
        stack.push(1);  // addFirst()
        stack.push(2);
        stack.push(3);
        System.out.println("Stack: " + stack);
        System.out.println("pop(): " + stack.pop());  // removeFirst()
        System.out.println("pop(): " + stack.pop());
        System.out.println("pop dan keyin Stack: " + stack);

        // Deque ni Queue sifatida ishlatamiz (FIFO - First In First Out)
        System.out.println("\\nDeque Queue sifatida (FIFO):");
        Deque<String> queue = new LinkedList<>();
        queue.offer("A");     // addLast()
        queue.offer("B");
        queue.offer("C");
        System.out.println("Queue: " + queue);
        System.out.println("poll(): " + queue.poll());  // removeFirst()
        System.out.println("poll(): " + queue.poll());
        System.out.println("poll dan keyin Queue: " + queue);

        // offer/poll metodlarini ishlatamiz (xavfsiz versiyalar)
        System.out.println("\\noffer/poll metodlarini ishlatish:");
        Deque<String> safeDeque = new LinkedList<>();
        safeDeque.offerFirst("Front");
        safeDeque.offerLast("Back");
        System.out.println("SafeDeque: " + safeDeque);
        System.out.println("pollFirst(): " + safeDeque.pollFirst());
        System.out.println("pollLast(): " + safeDeque.pollLast());
        System.out.println("Bo'shda pollFirst(): " + safeDeque.pollFirst()); // null qaytaradi

        // O'lcham va iteratsiyani ko'rsatamiz
        Deque<String> items = new LinkedList<>();
        items.add("Item1");
        items.add("Item2");
        items.add("Item3");
        System.out.println("\\nIteratsiya:");
        System.out.println("O'lchami: " + items.size());
        for (String item : items) {
            System.out.println("  - " + item);
        }
    }
}`,
            description: `Java da Deque (ikki tomonlama navbat) operatsiyalarini o'rganing.

**Talablar:**
1. LinkedList yordamida Deque yarating
2. Ikki tomondan elementlar qo'shing: addFirst() va addLast()
3. Ikki tomondan elementlarni o'chiring: removeFirst() va removeLast()
4. Ikki tomonni ko'ring: peekFirst() va peekLast()
5. Deque ni Stack sifatida (LIFO) ishlatishni ko'rsating
6. Deque ni Queue sifatida (FIFO) ishlatishni ko'rsating
7. offerFirst/offerLast va pollFirst/pollLast dan foydalaning
8. O'lcham va iteratsiyani ko'rsating

Deque Queue ni kengaytiradi va ikki tomon uchun operatsiyalarni taqdim etadi, bu uni Stack va Queue implementatsiyalari uchun universal qiladi.`,
            hint1: `Deque ikki tomon uchun metodlarga ega: addFirst/addLast, removeFirst/removeLast, peekFirst/peekLast. Stack operatsiyalari uchun push/pop metodlari ham bor.`,
            hint2: `Stack (LIFO) xatti-harakati uchun push/pop dan foydalaning. Queue (FIFO) xatti-harakati uchun offer/poll yoki addLast/removeFirst dan foydalaning. offer/poll metodlari xavfsizroq, chunki istisno chiqarmaydi.`,
            whyItMatters: `Deque Java dagi eng universal kolleksiyalardan biri. Uning ikki tomonlama operatsiyalarini tushunish Stack va Queue ni samarali amalga oshirish va murakkab ma'lumot oqimi naqshlarini boshqarish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Bekor qilish (undo/redo) bilan harakatlar tarixi
Deque<Action> undoStack = new LinkedList<>();
undoStack.push(userAction);  // Harakatni qo'shamiz
Action lastAction = undoStack.pop();  // Oxirgisini bekor qilamiz
lastAction.undo();

// Metrikalar uchun sirg'aluvchan oyna
Deque<Metric> recentMetrics = new LinkedList<>();
recentMetrics.addLast(new Metric(value, timestamp));
if (recentMetrics.size() > WINDOW_SIZE) {
    recentMetrics.removeFirst();  // Eskilarni o'chiramiz
}
double average = calculateAverage(recentMetrics);
\`\`\`

**Amaliy foydalari:**
- undo/redo funksionalligini samarali amalga oshirish
- Vaqt ma'lumotlarini tahlil qilish uchun sirg'aluvchan oyna
- Ikki uchdan O(1) operatsiyalar`
        }
    }
};

export default task;
