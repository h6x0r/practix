import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-queue-basics',
    title: 'Queue Interface Fundamentals',
    difficulty: 'easy',
    tags: ['java', 'collections', 'queue', 'fifo'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn Queue interface fundamentals in Java.

**Requirements:**
1. Create a Queue using LinkedList implementation
2. Add elements using offer(): "First", "Second", "Third"
3. Use peek() to view the head without removal
4. Use poll() to retrieve and remove elements
5. Compare add() vs offer() behavior
6. Compare remove() vs poll() when queue is empty
7. Compare element() vs peek() when queue is empty
8. Demonstrate FIFO (First-In-First-Out) order

Queue provides two sets of methods: throwing exceptions (add, remove, element) and returning special values (offer, poll, peek).`,
    initialCode: `import java.util.LinkedList;
import java.util.Queue;

public class QueueBasics {
    public static void main(String[] args) {
        // Create a Queue using LinkedList

        // Add elements using offer()

        // View head with peek()

        // Poll elements to demonstrate FIFO

        // Show difference: add vs offer

        // Show difference: remove vs poll (empty queue)

        // Show difference: element vs peek (empty queue)
    }
}`,
    solutionCode: `import java.util.LinkedList;
import java.util.Queue;

public class QueueBasics {
    public static void main(String[] args) {
        // Create a Queue using LinkedList
        Queue<String> queue = new LinkedList<>();

        // Add elements using offer() - returns false if fails
        queue.offer("First");
        queue.offer("Second");
        queue.offer("Third");
        System.out.println("Queue after offer(): " + queue);

        // View head with peek() - returns null if empty
        String head = queue.peek();
        System.out.println("Head (peek): " + head);
        System.out.println("Queue after peek: " + queue);

        // Poll elements to demonstrate FIFO
        System.out.println("\\nFIFO demonstration:");
        while (!queue.isEmpty()) {
            String element = queue.poll(); // Returns null if empty
            System.out.println("Polled: " + element);
        }
        System.out.println("Queue after polling all: " + queue);

        // Show difference: add vs offer
        System.out.println("\\nadd() vs offer():");
        queue.add("Item1");    // Throws exception if fails
        queue.offer("Item2");  // Returns false if fails
        System.out.println("Both succeeded: " + queue);

        // Show difference: remove vs poll (empty queue)
        queue.poll(); // Remove Item1
        queue.poll(); // Remove Item2
        System.out.println("\\nEmpty queue behavior:");
        String pollResult = queue.poll(); // Returns null
        System.out.println("poll() on empty queue: " + pollResult);

        try {
            queue.remove(); // Throws NoSuchElementException
        } catch (Exception e) {
            System.out.println("remove() on empty queue: " + e.getClass().getSimpleName());
        }

        // Show difference: element vs peek (empty queue)
        String peekResult = queue.peek(); // Returns null
        System.out.println("peek() on empty queue: " + peekResult);

        try {
            queue.element(); // Throws NoSuchElementException
        } catch (Exception e) {
            System.out.println("element() on empty queue: " + e.getClass().getSimpleName());
        }
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.LinkedList;
import java.util.Queue;
import java.util.NoSuchElementException;

// Test1: Verify Queue creation and offer() method
class Test1 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        assertTrue(queue.offer("First"));
        assertTrue(queue.offer("Second"));
        assertEquals(2, queue.size());
    }
}

// Test2: Verify peek() doesn't remove elements
class Test2 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        queue.offer("First");
        queue.offer("Second");
        assertEquals("First", queue.peek());
        assertEquals(2, queue.size());
    }
}

// Test3: Verify poll() removes and returns elements
class Test3 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        queue.offer("First");
        queue.offer("Second");
        assertEquals("First", queue.poll());
        assertEquals(1, queue.size());
    }
}

// Test4: Verify FIFO order
class Test4 {
    @Test
    public void test() {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(1);
        queue.offer(2);
        queue.offer(3);
        assertEquals(Integer.valueOf(1), queue.poll());
        assertEquals(Integer.valueOf(2), queue.poll());
        assertEquals(Integer.valueOf(3), queue.poll());
    }
}

// Test5: Verify add() vs offer() behavior
class Test5 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        assertTrue(queue.add("Item1"));
        assertTrue(queue.offer("Item2"));
        assertEquals(2, queue.size());
    }
}

// Test6: Verify poll() returns null on empty queue
class Test6 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        assertNull(queue.poll());
    }
}

// Test7: Verify remove() throws exception on empty queue
class Test7 {
    @Test(expected = NoSuchElementException.class)
    public void test() {
        Queue<String> queue = new LinkedList<>();
        queue.remove();
    }
}

// Test8: Verify peek() returns null on empty queue
class Test8 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        assertNull(queue.peek());
    }
}

// Test9: Verify element() throws exception on empty queue
class Test9 {
    @Test(expected = NoSuchElementException.class)
    public void test() {
        Queue<String> queue = new LinkedList<>();
        queue.element();
    }
}

// Test10: Verify isEmpty() and size() methods
class Test10 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        assertTrue(queue.isEmpty());
        queue.offer("Test");
        assertFalse(queue.isEmpty());
        assertEquals(1, queue.size());
    }
}
`,
    hint1: `Use LinkedList<>() to create a Queue. The offer() method adds elements and returns true/false, while add() throws an exception on failure.`,
    hint2: `peek() and poll() return null for empty queues, while element() and remove() throw NoSuchElementException. This makes peek/poll safer for unknown queue states.`,
    whyItMatters: `Understanding Queue's two method sets (throwing vs returning null) is crucial for robust code. Use offer/poll/peek for safe operations and add/remove/element when you want exceptions for unexpected states.

**Production Pattern:**
\`\`\`java
// Processing requests in arrival order
Queue<Request> requestQueue = new LinkedList<>();
requestQueue.offer(newRequest);  // Safe addition

// Request processor
while (!requestQueue.isEmpty()) {
    Request req = requestQueue.poll();  // Safe retrieval
    if (req != null) {
        processRequest(req);
    }
}

// Check next element without removal
Request next = requestQueue.peek();
if (next != null && next.isPriority()) {
    // Priority processing
}
\`\`\`

**Practical Benefits:**
- FIFO guarantees fair processing in arrival order
- Safe poll/peek methods don't throw exceptions on empty queue
- Perfect for asynchronous task processing`,
    order: 0,
    translations: {
        ru: {
            title: 'Основы интерфейса Queue',
            solutionCode: `import java.util.LinkedList;
import java.util.Queue;

public class QueueBasics {
    public static void main(String[] args) {
        // Создаем Queue используя LinkedList
        Queue<String> queue = new LinkedList<>();

        // Добавляем элементы используя offer() - возвращает false при неудаче
        queue.offer("First");
        queue.offer("Second");
        queue.offer("Third");
        System.out.println("Очередь после offer(): " + queue);

        // Смотрим голову с peek() - возвращает null если пусто
        String head = queue.peek();
        System.out.println("Голова (peek): " + head);
        System.out.println("Очередь после peek: " + queue);

        // Извлекаем элементы для демонстрации FIFO
        System.out.println("\\nДемонстрация FIFO:");
        while (!queue.isEmpty()) {
            String element = queue.poll(); // Возвращает null если пусто
            System.out.println("Извлечено: " + element);
        }
        System.out.println("Очередь после извлечения всех: " + queue);

        // Показываем разницу: add vs offer
        System.out.println("\\nadd() vs offer():");
        queue.add("Item1");    // Выбрасывает исключение при неудаче
        queue.offer("Item2");  // Возвращает false при неудаче
        System.out.println("Оба успешны: " + queue);

        // Показываем разницу: remove vs poll (пустая очередь)
        queue.poll(); // Удаляем Item1
        queue.poll(); // Удаляем Item2
        System.out.println("\\nПоведение пустой очереди:");
        String pollResult = queue.poll(); // Возвращает null
        System.out.println("poll() на пустой очереди: " + pollResult);

        try {
            queue.remove(); // Выбрасывает NoSuchElementException
        } catch (Exception e) {
            System.out.println("remove() на пустой очереди: " + e.getClass().getSimpleName());
        }

        // Показываем разницу: element vs peek (пустая очередь)
        String peekResult = queue.peek(); // Возвращает null
        System.out.println("peek() на пустой очереди: " + peekResult);

        try {
            queue.element(); // Выбрасывает NoSuchElementException
        } catch (Exception e) {
            System.out.println("element() на пустой очереди: " + e.getClass().getSimpleName());
        }
    }
}`,
            description: `Изучите основы интерфейса Queue в Java.

**Требования:**
1. Создайте Queue используя реализацию LinkedList
2. Добавьте элементы используя offer(): "First", "Second", "Third"
3. Используйте peek() для просмотра головы без удаления
4. Используйте poll() для извлечения и удаления элементов
5. Сравните поведение add() и offer()
6. Сравните remove() и poll() когда очередь пуста
7. Сравните element() и peek() когда очередь пуста
8. Продемонстрируйте порядок FIFO (First-In-First-Out)

Queue предоставляет два набора методов: выбрасывающие исключения (add, remove, element) и возвращающие специальные значения (offer, poll, peek).`,
            hint1: `Используйте LinkedList<>() для создания Queue. Метод offer() добавляет элементы и возвращает true/false, в то время как add() выбрасывает исключение при неудаче.`,
            hint2: `peek() и poll() возвращают null для пустых очередей, в то время как element() и remove() выбрасывают NoSuchElementException. Это делает peek/poll безопаснее для неизвестных состояний очереди.`,
            whyItMatters: `Понимание двух наборов методов Queue (выбрасывающих исключения vs возвращающих null) критично для надежного кода. Используйте offer/poll/peek для безопасных операций и add/remove/element когда хотите исключения для неожиданных состояний.

**Продакшен паттерн:**
\`\`\`java
// Обработка запросов в порядке поступления
Queue<Request> requestQueue = new LinkedList<>();
requestQueue.offer(newRequest);  // Безопасное добавление

// Обработчик запросов
while (!requestQueue.isEmpty()) {
    Request req = requestQueue.poll();  // Безопасное извлечение
    if (req != null) {
        processRequest(req);
    }
}

// Проверка следующего элемента без удаления
Request next = requestQueue.peek();
if (next != null && next.isPriority()) {
    // Приоритетная обработка
}
\`\`\`

**Практические преимущества:**
- FIFO гарантирует справедливую обработку в порядке поступления
- Безопасные методы poll/peek не выбрасывают исключения при пустой очереди
- Идеально для асинхронной обработки задач`
        },
        uz: {
            title: 'Queue Interfeysi Asoslari',
            solutionCode: `import java.util.LinkedList;
import java.util.Queue;

public class QueueBasics {
    public static void main(String[] args) {
        // LinkedList yordamida Queue yaratamiz
        Queue<String> queue = new LinkedList<>();

        // offer() yordamida elementlar qo'shamiz - muvaffaqiyatsiz bo'lsa false qaytaradi
        queue.offer("First");
        queue.offer("Second");
        queue.offer("Third");
        System.out.println("offer() dan keyin navbat: " + queue);

        // peek() bilan boshni ko'ramiz - bo'sh bo'lsa null qaytaradi
        String head = queue.peek();
        System.out.println("Bosh (peek): " + head);
        System.out.println("peek dan keyin navbat: " + queue);

        // FIFO ni ko'rsatish uchun elementlarni olamiz
        System.out.println("\\nFIFO namoyishi:");
        while (!queue.isEmpty()) {
            String element = queue.poll(); // Bo'sh bo'lsa null qaytaradi
            System.out.println("Olingan: " + element);
        }
        System.out.println("Hammasini olgandan keyin navbat: " + queue);

        // Farqini ko'rsatamiz: add vs offer
        System.out.println("\\nadd() vs offer():");
        queue.add("Item1");    // Muvaffaqiyatsiz bo'lsa istisno chiqaradi
        queue.offer("Item2");  // Muvaffaqiyatsiz bo'lsa false qaytaradi
        System.out.println("Ikkalasi ham muvaffaqiyatli: " + queue);

        // Farqini ko'rsatamiz: remove vs poll (bo'sh navbat)
        queue.poll(); // Item1 ni o'chiramiz
        queue.poll(); // Item2 ni o'chiramiz
        System.out.println("\\nBo'sh navbat xatti-harakati:");
        String pollResult = queue.poll(); // null qaytaradi
        System.out.println("Bo'sh navbatda poll(): " + pollResult);

        try {
            queue.remove(); // NoSuchElementException chiqaradi
        } catch (Exception e) {
            System.out.println("Bo'sh navbatda remove(): " + e.getClass().getSimpleName());
        }

        // Farqini ko'rsatamiz: element vs peek (bo'sh navbat)
        String peekResult = queue.peek(); // null qaytaradi
        System.out.println("Bo'sh navbatda peek(): " + peekResult);

        try {
            queue.element(); // NoSuchElementException chiqaradi
        } catch (Exception e) {
            System.out.println("Bo'sh navbatda element(): " + e.getClass().getSimpleName());
        }
    }
}`,
            description: `Java da Queue interfeysi asoslarini o'rganing.

**Talablar:**
1. LinkedList implementatsiyasidan foydalanib Queue yarating
2. offer() yordamida elementlar qo'shing: "First", "Second", "Third"
3. peek() dan foydalanib boshni o'chirmasdan ko'ring
4. poll() dan foydalanib elementlarni oling va o'chiring
5. add() va offer() xatti-harakatini solishtiring
6. Navbat bo'sh bo'lganda remove() va poll() ni solishtiring
7. Navbat bo'sh bo'lganda element() va peek() ni solishtiring
8. FIFO (First-In-First-Out) tartibini ko'rsating

Queue ikki to'plam metodlarni taqdim etadi: istisno chiqaruvchilar (add, remove, element) va maxsus qiymat qaytaruvchilar (offer, poll, peek).`,
            hint1: `Queue yaratish uchun LinkedList<>() dan foydalaning. offer() metodi elementlarni qo'shadi va true/false qaytaradi, add() esa muvaffaqiyatsizlikda istisno chiqaradi.`,
            hint2: `peek() va poll() bo'sh navbatlar uchun null qaytaradi, element() va remove() esa NoSuchElementException chiqaradi. Bu peek/poll ni noma'lum navbat holatlari uchun xavfsizroq qiladi.`,
            whyItMatters: `Queue ning ikki metod to'plamini (istisno chiqaruvchi vs null qaytaruvchi) tushunish ishonchli kod uchun muhim. Xavfsiz operatsiyalar uchun offer/poll/peek dan, kutilmagan holatlar uchun istisnolar kerak bo'lsa add/remove/element dan foydalaning.

**Ishlab chiqarish patterni:**
\`\`\`java
// So'rovlarni kelish tartibida qayta ishlash
Queue<Request> requestQueue = new LinkedList<>();
requestQueue.offer(newRequest);  // Xavfsiz qo'shish

// So'rovlar qayta ishlovchisi
while (!requestQueue.isEmpty()) {
    Request req = requestQueue.poll();  // Xavfsiz olish
    if (req != null) {
        processRequest(req);
    }
}

// Keyingi elementni o'chirmasdan tekshirish
Request next = requestQueue.peek();
if (next != null && next.isPriority()) {
    // Ustuvor qayta ishlash
}
\`\`\`

**Amaliy foydalari:**
- FIFO kelish tartibida adolatli qayta ishlashni kafolatlaydi
- Xavfsiz poll/peek metodlari bo'sh navbatda istisno chiqarmaydi
- Asinxron vazifalarni qayta ishlash uchun ideal`
        }
    }
};

export default task;
