import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-arraydeque',
    title: 'ArrayDeque Performance',
    difficulty: 'easy',
    tags: ['java', 'collections', 'arraydeque', 'performance'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn ArrayDeque performance characteristics and usage patterns.

**Requirements:**
1. Create an ArrayDeque and demonstrate basic operations
2. Use ArrayDeque as a Stack (push, pop, peek)
3. Use ArrayDeque as a Queue (offer, poll)
4. Compare ArrayDeque vs LinkedList performance characteristics
5. Demonstrate null elements are not allowed
6. Show capacity growth (mention circular array concept)
7. Iterate through ArrayDeque in both directions
8. Use ArrayDeque for both LIFO and FIFO operations

ArrayDeque is faster than LinkedList for both Stack and Queue operations due to better cache locality and no node overhead.`,
    initialCode: `import java.util.ArrayDeque;
import java.util.Deque;

public class ArrayDequeDemo {
    public static void main(String[] args) {
        // Create ArrayDeque

        // Use as Stack

        // Use as Queue

        // Show null not allowed

        // Iterate in both directions
    }
}`,
    solutionCode: `import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

public class ArrayDequeDemo {
    public static void main(String[] args) {
        // Create ArrayDeque with initial capacity
        Deque<String> deque = new ArrayDeque<>(10);

        // Basic operations
        System.out.println("Basic ArrayDeque operations:");
        deque.add("A");
        deque.add("B");
        deque.add("C");
        System.out.println("Deque: " + deque);

        // Use as Stack (LIFO - Last In First Out)
        System.out.println("\\nArrayDeque as Stack:");
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(1);  // addFirst
        stack.push(2);
        stack.push(3);
        System.out.println("Stack after pushes: " + stack);
        System.out.println("peek(): " + stack.peek());  // peekFirst
        System.out.println("pop(): " + stack.pop());    // removeFirst
        System.out.println("pop(): " + stack.pop());
        System.out.println("Stack after pops: " + stack);

        // Use as Queue (FIFO - First In First Out)
        System.out.println("\\nArrayDeque as Queue:");
        Deque<String> queue = new ArrayDeque<>();
        queue.offer("First");   // addLast
        queue.offer("Second");
        queue.offer("Third");
        System.out.println("Queue after offers: " + queue);
        System.out.println("poll(): " + queue.poll());  // removeFirst
        System.out.println("poll(): " + queue.poll());
        System.out.println("Queue after polls: " + queue);

        // Performance characteristics
        System.out.println("\\nPerformance comparison (ArrayDeque vs LinkedList):");
        System.out.println("ArrayDeque advantages:");
        System.out.println("  - Faster for both Stack and Queue operations");
        System.out.println("  - Better cache locality (contiguous memory)");
        System.out.println("  - No per-element overhead (no Node objects)");
        System.out.println("  - Backed by circular resizable array");
        System.out.println("LinkedList advantages:");
        System.out.println("  - Better for frequent insertions in middle");
        System.out.println("  - Implements List interface (positional access)");

        // Show null not allowed
        System.out.println("\\nNull handling:");
        Deque<String> nullTest = new ArrayDeque<>();
        try {
            nullTest.add(null);
        } catch (NullPointerException e) {
            System.out.println("ArrayDeque does NOT allow null elements");
            System.out.println("Exception: " + e.getClass().getSimpleName());
        }

        // Circular array concept
        System.out.println("\\nCircular array growth:");
        Deque<Integer> growingDeque = new ArrayDeque<>(4); // Small initial capacity
        System.out.println("Initial capacity: 4");
        for (int i = 1; i <= 10; i++) {
            growingDeque.add(i);
        }
        System.out.println("Added 10 elements: " + growingDeque);
        System.out.println("Array automatically resized (doubled) when needed");

        // Iterate in both directions
        System.out.println("\\nBidirectional iteration:");
        Deque<String> biDeque = new ArrayDeque<>();
        biDeque.add("First");
        biDeque.add("Second");
        biDeque.add("Third");

        System.out.println("Forward iteration:");
        for (String item : biDeque) {
            System.out.println("  " + item);
        }

        System.out.println("Reverse iteration:");
        Iterator<String> descending = biDeque.descendingIterator();
        while (descending.hasNext()) {
            System.out.println("  " + descending.next());
        }

        // Combined LIFO and FIFO
        System.out.println("\\nCombined operations:");
        Deque<String> combined = new ArrayDeque<>();
        combined.addFirst("Front1");
        combined.addLast("Back1");
        combined.addFirst("Front2");
        combined.addLast("Back2");
        System.out.println("After mixed adds: " + combined);
        System.out.println("removeFirst(): " + combined.removeFirst());
        System.out.println("removeLast(): " + combined.removeLast());
        System.out.println("Final: " + combined);
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

// Test1: Verify ArrayDeque creation and basic operations
class Test1 {
    @Test
    public void test() {
        Deque<String> deque = new ArrayDeque<>();
        deque.add("A");
        deque.add("B");
        deque.add("C");
        assertEquals(3, deque.size());
    }
}

// Test2: Verify ArrayDeque as Stack (push/pop)
class Test2 {
    @Test
    public void test() {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        assertEquals(Integer.valueOf(3), stack.pop());
        assertEquals(Integer.valueOf(2), stack.pop());
    }
}

// Test3: Verify ArrayDeque as Queue (offer/poll)
class Test3 {
    @Test
    public void test() {
        Deque<String> queue = new ArrayDeque<>();
        queue.offer("First");
        queue.offer("Second");
        queue.offer("Third");
        assertEquals("First", queue.poll());
        assertEquals("Second", queue.poll());
    }
}

// Test4: Verify peek doesn't remove elements
class Test4 {
    @Test
    public void test() {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(10);
        stack.push(20);
        assertEquals(Integer.valueOf(20), stack.peek());
        assertEquals(2, stack.size());
    }
}

// Test5: Verify null elements are not allowed
class Test5 {
    @Test(expected = NullPointerException.class)
    public void test() {
        Deque<String> deque = new ArrayDeque<>();
        deque.add(null);
    }
}

// Test6: Verify capacity growth
class Test6 {
    @Test
    public void test() {
        Deque<Integer> deque = new ArrayDeque<>(4);
        for (int i = 1; i <= 10; i++) {
            deque.add(i);
        }
        assertEquals(10, deque.size());
    }
}

// Test7: Verify forward iteration
class Test7 {
    @Test
    public void test() {
        Deque<String> deque = new ArrayDeque<>();
        deque.add("A");
        deque.add("B");
        deque.add("C");
        StringBuilder sb = new StringBuilder();
        for (String item : deque) {
            sb.append(item);
        }
        assertEquals("ABC", sb.toString());
    }
}

// Test8: Verify reverse iteration with descendingIterator
class Test8 {
    @Test
    public void test() {
        Deque<String> deque = new ArrayDeque<>();
        deque.add("A");
        deque.add("B");
        deque.add("C");
        Iterator<String> desc = deque.descendingIterator();
        assertEquals("C", desc.next());
        assertEquals("B", desc.next());
        assertEquals("A", desc.next());
    }
}

// Test9: Verify combined LIFO and FIFO operations
class Test9 {
    @Test
    public void test() {
        Deque<String> deque = new ArrayDeque<>();
        deque.addFirst("Front");
        deque.addLast("Back");
        assertEquals("Front", deque.removeFirst());
        assertEquals("Back", deque.removeLast());
        assertTrue(deque.isEmpty());
    }
}

// Test10: Verify isEmpty and size methods
class Test10 {
    @Test
    public void test() {
        Deque<String> deque = new ArrayDeque<>();
        assertTrue(deque.isEmpty());
        deque.add("Test");
        assertFalse(deque.isEmpty());
        assertEquals(1, deque.size());
    }
}
`,
    hint1: `ArrayDeque is preferred over Stack and LinkedList for both stack and queue operations. It's backed by a resizable circular array, providing O(1) operations at both ends.`,
    hint2: `Use push/pop for stack operations, offer/poll for queue operations. ArrayDeque doesn't allow null elements. Use descendingIterator() for reverse iteration.`,
    whyItMatters: `ArrayDeque is the recommended implementation for both Stack and Queue in modern Java. It's faster than Stack (legacy) and LinkedList, with better memory efficiency and cache performance. Understanding its circular array backing helps optimize collection choices.

**Production Pattern:**
\`\`\`java
// High-performance command processing (stack)
Deque<Command> commandStack = new ArrayDeque<>();
commandStack.push(command);  // Faster than Stack
Command last = commandStack.pop();

// High-throughput message buffer
Deque<Message> messageBuffer = new ArrayDeque<>(INITIAL_SIZE);
messageBuffer.offerLast(incomingMessage);  // O(1) to end
Message next = messageBuffer.pollFirst();  // O(1) from start

// BFS with ArrayDeque for better performance
Deque<Node> queue = new ArrayDeque<>();
queue.offer(root);
while (!queue.isEmpty()) {
    Node current = queue.poll();
    process(current);
    queue.addAll(current.getChildren());
}
\`\`\`

**Practical Benefits:**
- Better performance than LinkedList for stacks/queues
- Minimal memory overhead without nodes
- Predictable performance due to array backing`,
    order: 3,
    translations: {
        ru: {
            title: 'Производительность ArrayDeque',
            solutionCode: `import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

public class ArrayDequeDemo {
    public static void main(String[] args) {
        // Создаем ArrayDeque с начальной емкостью
        Deque<String> deque = new ArrayDeque<>(10);

        // Базовые операции
        System.out.println("Базовые операции ArrayDeque:");
        deque.add("A");
        deque.add("B");
        deque.add("C");
        System.out.println("Deque: " + deque);

        // Используем как Stack (LIFO - Last In First Out)
        System.out.println("\\nArrayDeque как Stack:");
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(1);  // addFirst
        stack.push(2);
        stack.push(3);
        System.out.println("Stack после push: " + stack);
        System.out.println("peek(): " + stack.peek());  // peekFirst
        System.out.println("pop(): " + stack.pop());    // removeFirst
        System.out.println("pop(): " + stack.pop());
        System.out.println("Stack после pop: " + stack);

        // Используем как Queue (FIFO - First In First Out)
        System.out.println("\\nArrayDeque как Queue:");
        Deque<String> queue = new ArrayDeque<>();
        queue.offer("First");   // addLast
        queue.offer("Second");
        queue.offer("Third");
        System.out.println("Queue после offer: " + queue);
        System.out.println("poll(): " + queue.poll());  // removeFirst
        System.out.println("poll(): " + queue.poll());
        System.out.println("Queue после poll: " + queue);

        // Характеристики производительности
        System.out.println("\\nСравнение производительности (ArrayDeque vs LinkedList):");
        System.out.println("Преимущества ArrayDeque:");
        System.out.println("  - Быстрее для операций Stack и Queue");
        System.out.println("  - Лучшая локальность кэша (непрерывная память)");
        System.out.println("  - Нет накладных расходов на элемент (нет объектов Node)");
        System.out.println("  - Основан на циклическом изменяемом массиве");
        System.out.println("Преимущества LinkedList:");
        System.out.println("  - Лучше для частых вставок в середину");
        System.out.println("  - Реализует интерфейс List (позиционный доступ)");

        // Показываем, что null не разрешен
        System.out.println("\\nОбработка null:");
        Deque<String> nullTest = new ArrayDeque<>();
        try {
            nullTest.add(null);
        } catch (NullPointerException e) {
            System.out.println("ArrayDeque НЕ позволяет null элементы");
            System.out.println("Исключение: " + e.getClass().getSimpleName());
        }

        // Концепция циклического массива
        System.out.println("\\nРост циклического массива:");
        Deque<Integer> growingDeque = new ArrayDeque<>(4); // Малая начальная емкость
        System.out.println("Начальная емкость: 4");
        for (int i = 1; i <= 10; i++) {
            growingDeque.add(i);
        }
        System.out.println("Добавлено 10 элементов: " + growingDeque);
        System.out.println("Массив автоматически изменен (удвоен) при необходимости");

        // Итерация в обоих направлениях
        System.out.println("\\nДвунаправленная итерация:");
        Deque<String> biDeque = new ArrayDeque<>();
        biDeque.add("First");
        biDeque.add("Second");
        biDeque.add("Third");

        System.out.println("Прямая итерация:");
        for (String item : biDeque) {
            System.out.println("  " + item);
        }

        System.out.println("Обратная итерация:");
        Iterator<String> descending = biDeque.descendingIterator();
        while (descending.hasNext()) {
            System.out.println("  " + descending.next());
        }

        // Комбинированные LIFO и FIFO
        System.out.println("\\nКомбинированные операции:");
        Deque<String> combined = new ArrayDeque<>();
        combined.addFirst("Front1");
        combined.addLast("Back1");
        combined.addFirst("Front2");
        combined.addLast("Back2");
        System.out.println("После смешанных добавлений: " + combined);
        System.out.println("removeFirst(): " + combined.removeFirst());
        System.out.println("removeLast(): " + combined.removeLast());
        System.out.println("Финальный: " + combined);
    }
}`,
            description: `Изучите характеристики производительности и паттерны использования ArrayDeque.

**Требования:**
1. Создайте ArrayDeque и продемонстрируйте базовые операции
2. Используйте ArrayDeque как Stack (push, pop, peek)
3. Используйте ArrayDeque как Queue (offer, poll)
4. Сравните характеристики производительности ArrayDeque и LinkedList
5. Продемонстрируйте, что null элементы не разрешены
6. Покажите рост емкости (упомяните концепцию циклического массива)
7. Выполните итерацию по ArrayDeque в обоих направлениях
8. Используйте ArrayDeque для операций LIFO и FIFO

ArrayDeque быстрее чем LinkedList для операций Stack и Queue благодаря лучшей локальности кэша и отсутствию накладных расходов на узлы.`,
            hint1: `ArrayDeque предпочтителен вместо Stack и LinkedList для операций stack и queue. Он основан на изменяемом циклическом массиве, предоставляя O(1) операции на обоих концах.`,
            hint2: `Используйте push/pop для операций stack, offer/poll для операций queue. ArrayDeque не позволяет null элементы. Используйте descendingIterator() для обратной итерации.`,
            whyItMatters: `ArrayDeque - рекомендуемая реализация как для Stack, так и для Queue в современной Java. Он быстрее чем Stack (устаревший) и LinkedList, с лучшей эффективностью памяти и производительностью кэша. Понимание его циклического массива помогает оптимизировать выбор коллекций.

**Продакшен паттерн:**
\`\`\`java
// Высокопроизводительная обработка команд (стек)
Deque<Command> commandStack = new ArrayDeque<>();
commandStack.push(command);  // Быстрее чем Stack
Command last = commandStack.pop();

// Буфер сообщений с высокой пропускной способностью
Deque<Message> messageBuffer = new ArrayDeque<>(INITIAL_SIZE);
messageBuffer.offerLast(incomingMessage);  // O(1) в конец
Message next = messageBuffer.pollFirst();  // O(1) из начала

// BFS с ArrayDeque для лучшей производительности
Deque<Node> queue = new ArrayDeque<>();
queue.offer(root);
while (!queue.isEmpty()) {
    Node current = queue.poll();
    process(current);
    queue.addAll(current.getChildren());
}
\`\`\`

**Практические преимущества:**
- Лучшая производительность чем LinkedList для стеков/очередей
- Минимальные накладные расходы памяти без узлов
- Предсказуемая производительность благодаря массиву`
        },
        uz: {
            title: 'ArrayDeque Samaradorligi',
            solutionCode: `import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

public class ArrayDequeDemo {
    public static void main(String[] args) {
        // Boshlang'ich sig'im bilan ArrayDeque yaratamiz
        Deque<String> deque = new ArrayDeque<>(10);

        // Asosiy operatsiyalar
        System.out.println("ArrayDeque asosiy operatsiyalari:");
        deque.add("A");
        deque.add("B");
        deque.add("C");
        System.out.println("Deque: " + deque);

        // Stack sifatida ishlatamiz (LIFO - Last In First Out)
        System.out.println("\\nArrayDeque Stack sifatida:");
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(1);  // addFirst
        stack.push(2);
        stack.push(3);
        System.out.println("push dan keyin Stack: " + stack);
        System.out.println("peek(): " + stack.peek());  // peekFirst
        System.out.println("pop(): " + stack.pop());    // removeFirst
        System.out.println("pop(): " + stack.pop());
        System.out.println("pop dan keyin Stack: " + stack);

        // Queue sifatida ishlatamiz (FIFO - First In First Out)
        System.out.println("\\nArrayDeque Queue sifatida:");
        Deque<String> queue = new ArrayDeque<>();
        queue.offer("First");   // addLast
        queue.offer("Second");
        queue.offer("Third");
        System.out.println("offer dan keyin Queue: " + queue);
        System.out.println("poll(): " + queue.poll());  // removeFirst
        System.out.println("poll(): " + queue.poll());
        System.out.println("poll dan keyin Queue: " + queue);

        // Samaradorlik xususiyatlari
        System.out.println("\\nSamaradorlik taqqoslash (ArrayDeque vs LinkedList):");
        System.out.println("ArrayDeque afzalliklari:");
        System.out.println("  - Stack va Queue operatsiyalari uchun tezroq");
        System.out.println("  - Yaxshiroq kesh lokalligi (uzluksiz xotira)");
        System.out.println("  - Element uchun qo'shimcha xarajat yo'q (Node obyektlari yo'q)");
        System.out.println("  - Aylanma o'zgaruvchan massivga asoslangan");
        System.out.println("LinkedList afzalliklari:");
        System.out.println("  - O'rtaga tez-tez qo'shish uchun yaxshiroq");
        System.out.println("  - List interfeysini amalga oshiradi (pozitsion kirish)");

        // null ruxsat berilmaganligini ko'rsatamiz
        System.out.println("\\nnull bilan ishlash:");
        Deque<String> nullTest = new ArrayDeque<>();
        try {
            nullTest.add(null);
        } catch (NullPointerException e) {
            System.out.println("ArrayDeque null elementlarga ruxsat bermaydi");
            System.out.println("Istisno: " + e.getClass().getSimpleName());
        }

        // Aylanma massiv tushunchasi
        System.out.println("\\nAylanma massiv o'sishi:");
        Deque<Integer> growingDeque = new ArrayDeque<>(4); // Kichik boshlang'ich sig'im
        System.out.println("Boshlang'ich sig'im: 4");
        for (int i = 1; i <= 10; i++) {
            growingDeque.add(i);
        }
        System.out.println("10 element qo'shildi: " + growingDeque);
        System.out.println("Kerak bo'lganda massiv avtomatik o'zgartiriladi (ikki barobar)");

        // Ikki yo'nalishda iteratsiya
        System.out.println("\\nIkki tomonlama iteratsiya:");
        Deque<String> biDeque = new ArrayDeque<>();
        biDeque.add("First");
        biDeque.add("Second");
        biDeque.add("Third");

        System.out.println("To'g'ri yo'nalishda iteratsiya:");
        for (String item : biDeque) {
            System.out.println("  " + item);
        }

        System.out.println("Teskari yo'nalishda iteratsiya:");
        Iterator<String> descending = biDeque.descendingIterator();
        while (descending.hasNext()) {
            System.out.println("  " + descending.next());
        }

        // Birlashtirilgan LIFO va FIFO
        System.out.println("\\nBirlashtirilgan operatsiyalar:");
        Deque<String> combined = new ArrayDeque<>();
        combined.addFirst("Front1");
        combined.addLast("Back1");
        combined.addFirst("Front2");
        combined.addLast("Back2");
        System.out.println("Aralash qo'shishlardan keyin: " + combined);
        System.out.println("removeFirst(): " + combined.removeFirst());
        System.out.println("removeLast(): " + combined.removeLast());
        System.out.println("Yakuniy: " + combined);
    }
}`,
            description: `ArrayDeque samaradorlik xususiyatlari va foydalanish naqshlarini o'rganing.

**Talablar:**
1. ArrayDeque yarating va asosiy operatsiyalarni ko'rsating
2. ArrayDeque ni Stack sifatida ishlating (push, pop, peek)
3. ArrayDeque ni Queue sifatida ishlating (offer, poll)
4. ArrayDeque va LinkedList samaradorlik xususiyatlarini solishtiring
5. null elementlarga ruxsat berilmaganligini ko'rsating
6. Sig'im o'sishini ko'rsating (aylanma massiv tushunchasini eslatib o'ting)
7. ArrayDeque bo'ylab ikki yo'nalishda iteratsiya qiling
8. LIFO va FIFO operatsiyalari uchun ArrayDeque dan foydalaning

ArrayDeque yaxshiroq kesh lokalligi va tugun uchun qo'shimcha xarajat yo'qligi tufayli Stack va Queue operatsiyalari uchun LinkedList dan tezroq.`,
            hint1: `ArrayDeque stack va queue operatsiyalari uchun Stack va LinkedList dan afzalroq. U o'zgaruvchan aylanma massivga asoslangan va ikki uchida O(1) operatsiyalarni taqdim etadi.`,
            hint2: `Stack operatsiyalari uchun push/pop, queue operatsiyalari uchun offer/poll dan foydalaning. ArrayDeque null elementlarga ruxsat bermaydi. Teskari iteratsiya uchun descendingIterator() dan foydalaning.`,
            whyItMatters: `ArrayDeque zamonaviy Java da Stack va Queue uchun tavsiya etilgan implementatsiya. U Stack (eski) va LinkedList dan tezroq, yaxshiroq xotira samaradorligi va kesh ishlashiga ega. Uning aylanma massiv asosini tushunish kolleksiya tanlovlarini optimallashtirish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Yuqori samaradorlikdagi buyruqlarni qayta ishlash (stek)
Deque<Command> commandStack = new ArrayDeque<>();
commandStack.push(command);  // Stack dan tezroq
Command last = commandStack.pop();

// Yuqori o'tkazuvchanlikdagi xabarlar buferi
Deque<Message> messageBuffer = new ArrayDeque<>(INITIAL_SIZE);
messageBuffer.offerLast(incomingMessage);  // Oxiriga O(1)
Message next = messageBuffer.pollFirst();  // Boshidan O(1)

// Yaxshi samaradorlik uchun ArrayDeque bilan BFS
Deque<Node> queue = new ArrayDeque<>();
queue.offer(root);
while (!queue.isEmpty()) {
    Node current = queue.poll();
    process(current);
    queue.addAll(current.getChildren());
}
\`\`\`

**Amaliy foydalari:**
- Stek/navbatlar uchun LinkedList dan yaxshiroq samaradorlik
- Tugunsiz minimal xotira qo'shimcha xarajatlari
- Massiv tufayli bashorat qilinadigan samaradorlik`
        }
    }
};

export default task;
