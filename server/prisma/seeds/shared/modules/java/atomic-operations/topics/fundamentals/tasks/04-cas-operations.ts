import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-cas-operations',
    title: 'Compare-And-Swap Operations',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'atomic', 'cas', 'lock-free'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master compare-and-swap (CAS) pattern for lock-free algorithms.

**Requirements:**
1. Create a Counter class using AtomicInteger
2. Implement increment() method using CAS in a loop
3. Implement decrement() method with CAS and validation (don't go below 0)
4. Create multiplyBy2() method using CAS to double the value
5. Test with multiple threads to verify thread-safety
6. Use weakCompareAndSet() for a performance comparison method
7. Print success rates and final counter value

Compare-and-swap is the foundation of lock-free programming, allowing threads to update shared data without locks by retrying on conflicts.`,
    initialCode: `import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

class Counter {
    private AtomicInteger value = new AtomicInteger(0);

    // Implement increment using CAS
    public void increment() {
        // Use compareAndSet in a loop
    }

    // Implement decrement using CAS (don't go below 0)
    public void decrement() {
        // Use compareAndSet in a loop with validation
    }

    // Implement multiplyBy2 using CAS
    public void multiplyBy2() {
        // Use compareAndSet in a loop
    }

    public int getValue() {
        return value.get();
    }
}

public class CasOperationsDemo {
    public static void main(String[] args) throws InterruptedException {
        // Test counter with multiple threads
    }
}`,
    solutionCode: `import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

class Counter {
    private AtomicInteger value = new AtomicInteger(0);
    private AtomicLong casAttempts = new AtomicLong(0);
    private AtomicLong casSuccesses = new AtomicLong(0);

    // Implement increment using CAS
    public void increment() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            next = current + 1;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Implement decrement using CAS (don't go below 0)
    public void decrement() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            if (current == 0) {
                return; // Don't go below 0
            }
            next = current - 1;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Implement multiplyBy2 using CAS
    public void multiplyBy2() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            next = current * 2;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Using weakCompareAndSet for performance comparison
    public void incrementWeak() {
        int current;
        int next;
        do {
            current = value.get();
            next = current + 1;
        } while (!value.weakCompareAndSet(current, next));
    }

    public int getValue() {
        return value.get();
    }

    public void printStats() {
        System.out.println("CAS attempts: " + casAttempts.get());
        System.out.println("CAS successes: " + casSuccesses.get());
        double successRate = (double) casSuccesses.get() / casAttempts.get() * 100;
        System.out.println("Success rate: " + String.format("%.2f", successRate) + "%");
    }
}

public class CasOperationsDemo {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        int numThreads = 4;
        int operationsPerThread = 1000;

        // Test increment with multiple threads
        System.out.println("Testing CAS operations with " + numThreads + " threads...");
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        // Phase 1: Increments
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                for (int j = 0; j < operationsPerThread; j++) {
                    counter.increment();
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("\\nAfter " + (numThreads * operationsPerThread) + " increments:");
        System.out.println("Counter value: " + counter.getValue());
        counter.printStats();

        // Phase 2: Test multiplyBy2
        System.out.println("\\nMultiplying by 2...");
        counter.multiplyBy2();
        System.out.println("Counter value: " + counter.getValue());

        // Phase 3: Test decrement
        System.out.println("\\nTesting decrements...");
        ExecutorService executor2 = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor2.submit(() -> {
                for (int j = 0; j < operationsPerThread; j++) {
                    counter.decrement();
                }
            });
        }

        executor2.shutdown();
        executor2.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("\\nFinal counter value: " + counter.getValue());
        counter.printStats();
    }
}`,
    hint1: `The CAS pattern is: do { current = get(); next = compute(current); } while (!compareAndSet(current, next));`,
    hint2: `For decrement with validation, check if current is 0 before attempting the CAS. Return early if you can't decrement.`,
    whyItMatters: `Compare-and-swap is the foundation of lock-free algorithms, enabling high-performance concurrent programming without the overhead and risks of traditional locks. Understanding CAS is essential for building scalable concurrent systems.

**Production Pattern:**
\`\`\`java
// Lock-free stack with CAS
public class LockFreeStack<T> {
    private final AtomicReference<Node<T>> top = new AtomicReference<>();

    private static class Node<T> {
        final T value;
        final Node<T> next;
        Node(T value, Node<T> next) {
            this.value = value;
            this.next = next;
        }
    }

    public void push(T value) {
        Node<T> newNode = new Node<>(value, null);
        Node<T> currentTop;
        do {
            currentTop = top.get();
            newNode.next = currentTop;
        } while (!top.compareAndSet(currentTop, newNode));
    }

    public T pop() {
        Node<T> currentTop;
        Node<T> newTop;
        do {
            currentTop = top.get();
            if (currentTop == null) return null;
            newTop = currentTop.next;
        } while (!top.compareAndSet(currentTop, newTop));
        return currentTop.value;
    }
}
\`\`\`

**Practical Benefits:**
- No locks means no deadlocks
- High performance under thread contention
- Guaranteed progress of at least one thread`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.atomic.AtomicInteger;

// Test1: Test basic compareAndSet success
class Test1 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        assertTrue(value.compareAndSet(10, 20));
        assertEquals(20, value.get());
    }
}

// Test2: Test basic compareAndSet failure
class Test2 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        assertFalse(value.compareAndSet(5, 20));
        assertEquals(10, value.get());
    }
}

// Test3: Test CAS loop for increment
class Test3 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        int current;
        do {
            current = value.get();
        } while (!value.compareAndSet(current, current + 1));
        assertEquals(11, value.get());
    }
}

// Test4: Test CAS with retry
class Test4 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(5);
        int expected = 5;
        int newValue = 10;
        while (!value.compareAndSet(expected, newValue)) {
            expected = value.get();
        }
        assertEquals(10, value.get());
    }
}

// Test5: Test weakCompareAndSet
class Test5 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        boolean result = value.weakCompareAndSet(10, 20);
        if (result) {
            assertEquals(20, value.get());
        }
    }
}

// Test6: Test multiple CAS operations
class Test6 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(0);
        assertTrue(value.compareAndSet(0, 1));
        assertTrue(value.compareAndSet(1, 2));
        assertTrue(value.compareAndSet(2, 3));
        assertEquals(3, value.get());
    }
}

// Test7: Test CAS for multiplication
class Test7 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(5);
        int current;
        do {
            current = value.get();
        } while (!value.compareAndSet(current, current * 2));
        assertEquals(10, value.get());
    }
}

// Test8: Test getAndUpdate with CAS
class Test8 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        int old = value.getAndUpdate(x -> x * 2);
        assertEquals(10, old);
        assertEquals(20, value.get());
    }
}

// Test9: Test updateAndGet with CAS
class Test9 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        int result = value.updateAndGet(x -> x + 5);
        assertEquals(15, result);
        assertEquals(15, value.get());
    }
}

// Test10: Test accumulateAndGet with CAS
class Test10 {
    @Test
    public void test() {
        AtomicInteger value = new AtomicInteger(10);
        int result = value.accumulateAndGet(3, (x, y) -> x * y);
        assertEquals(30, result);
        assertEquals(30, value.get());
    }
}
`,
    translations: {
        ru: {
            title: 'Операции Compare-And-Swap',
            solutionCode: `import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

class Counter {
    private AtomicInteger value = new AtomicInteger(0);
    private AtomicLong casAttempts = new AtomicLong(0);
    private AtomicLong casSuccesses = new AtomicLong(0);

    // Реализуем increment используя CAS
    public void increment() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            next = current + 1;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Реализуем decrement используя CAS (не опускаться ниже 0)
    public void decrement() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            if (current == 0) {
                return; // Не опускаемся ниже 0
            }
            next = current - 1;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Реализуем multiplyBy2 используя CAS
    public void multiplyBy2() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            next = current * 2;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Используем weakCompareAndSet для сравнения производительности
    public void incrementWeak() {
        int current;
        int next;
        do {
            current = value.get();
            next = current + 1;
        } while (!value.weakCompareAndSet(current, next));
    }

    public int getValue() {
        return value.get();
    }

    public void printStats() {
        System.out.println("Попытки CAS: " + casAttempts.get());
        System.out.println("Успешные CAS: " + casSuccesses.get());
        double successRate = (double) casSuccesses.get() / casAttempts.get() * 100;
        System.out.println("Процент успеха: " + String.format("%.2f", successRate) + "%");
    }
}

public class CasOperationsDemo {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        int numThreads = 4;
        int operationsPerThread = 1000;

        // Тестируем increment с несколькими потоками
        System.out.println("Тестирование CAS операций с " + numThreads + " потоками...");
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        // Фаза 1: Инкременты
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                for (int j = 0; j < operationsPerThread; j++) {
                    counter.increment();
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("\\nПосле " + (numThreads * operationsPerThread) + " инкрементов:");
        System.out.println("Значение счетчика: " + counter.getValue());
        counter.printStats();

        // Фаза 2: Тест multiplyBy2
        System.out.println("\\nУмножение на 2...");
        counter.multiplyBy2();
        System.out.println("Значение счетчика: " + counter.getValue());

        // Фаза 3: Тест decrement
        System.out.println("\\nТестирование декрементов...");
        ExecutorService executor2 = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor2.submit(() -> {
                for (int j = 0; j < operationsPerThread; j++) {
                    counter.decrement();
                }
            });
        }

        executor2.shutdown();
        executor2.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("\\nФинальное значение счетчика: " + counter.getValue());
        counter.printStats();
    }
}`,
            description: `Освойте паттерн compare-and-swap (CAS) для неблокирующих алгоритмов.

**Требования:**
1. Создайте класс Counter, использующий AtomicInteger
2. Реализуйте метод increment() используя CAS в цикле
3. Реализуйте метод decrement() с CAS и валидацией (не опускаться ниже 0)
4. Создайте метод multiplyBy2() используя CAS для удвоения значения
5. Протестируйте с несколькими потоками для проверки потокобезопасности
6. Используйте weakCompareAndSet() для метода сравнения производительности
7. Выведите процент успеха и финальное значение счетчика

Compare-and-swap - основа неблокирующего программирования, позволяющая потокам обновлять разделяемые данные без блокировок, повторяя попытки при конфликтах.`,
            hint1: `Паттерн CAS: do { current = get(); next = compute(current); } while (!compareAndSet(current, next));`,
            hint2: `Для decrement с валидацией проверьте, равен ли current 0, перед попыткой CAS. Возвращайтесь досрочно, если не можете уменьшить.`,
            whyItMatters: `Compare-and-swap - основа неблокирующих алгоритмов, обеспечивающая высокопроизводительное многопоточное программирование без накладных расходов и рисков традиционных блокировок. Понимание CAS необходимо для построения масштабируемых многопоточных систем.

**Продакшен паттерн:**
\`\`\`java
// Неблокирующий стек с CAS
public class LockFreeStack<T> {
    private final AtomicReference<Node<T>> top = new AtomicReference<>();

    private static class Node<T> {
        final T value;
        final Node<T> next;
        Node(T value, Node<T> next) {
            this.value = value;
            this.next = next;
        }
    }

    public void push(T value) {
        Node<T> newNode = new Node<>(value, null);
        Node<T> currentTop;
        do {
            currentTop = top.get();
            newNode.next = currentTop;
        } while (!top.compareAndSet(currentTop, newNode));
    }

    public T pop() {
        Node<T> currentTop;
        Node<T> newTop;
        do {
            currentTop = top.get();
            if (currentTop == null) return null;
            newTop = currentTop.next;
        } while (!top.compareAndSet(currentTop, newTop));
        return currentTop.value;
    }
}
\`\`\`

**Практические преимущества:**
- Отсутствие блокировок означает отсутствие взаимоблокировок
- Высокая производительность при конкуренции потоков
- Гарантированный прогресс хотя бы одного потока`
        },
        uz: {
            title: 'Compare-And-Swap Operatsiyalari',
            solutionCode: `import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

class Counter {
    private AtomicInteger value = new AtomicInteger(0);
    private AtomicLong casAttempts = new AtomicLong(0);
    private AtomicLong casSuccesses = new AtomicLong(0);

    // CAS dan foydalanib increment ni amalga oshiramiz
    public void increment() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            next = current + 1;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // CAS dan foydalanib decrement ni amalga oshiramiz (0 dan past tushmaslik)
    public void decrement() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            if (current == 0) {
                return; // 0 dan past tushmaydi
            }
            next = current - 1;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // CAS dan foydalanib multiplyBy2 ni amalga oshiramiz
    public void multiplyBy2() {
        int current;
        int next;
        do {
            casAttempts.incrementAndGet();
            current = value.get();
            next = current * 2;
        } while (!value.compareAndSet(current, next));
        casSuccesses.incrementAndGet();
    }

    // Ishlashni solishtirish uchun weakCompareAndSet dan foydalanamiz
    public void incrementWeak() {
        int current;
        int next;
        do {
            current = value.get();
            next = current + 1;
        } while (!value.weakCompareAndSet(current, next));
    }

    public int getValue() {
        return value.get();
    }

    public void printStats() {
        System.out.println("CAS urinishlari: " + casAttempts.get());
        System.out.println("CAS muvaffaqiyatlari: " + casSuccesses.get());
        double successRate = (double) casSuccesses.get() / casAttempts.get() * 100;
        System.out.println("Muvaffaqiyat darajasi: " + String.format("%.2f", successRate) + "%");
    }
}

public class CasOperationsDemo {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        int numThreads = 4;
        int operationsPerThread = 1000;

        // Ko'p oqimlar bilan increment ni sinaymiz
        System.out.println(numThreads + " oqim bilan CAS operatsiyalarini sinaymiz...");
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        // 1-bosqich: Inkrementlar
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                for (int j = 0; j < operationsPerThread; j++) {
                    counter.increment();
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("\\n" + (numThreads * operationsPerThread) + " inkrement dan keyin:");
        System.out.println("Hisoblagich qiymati: " + counter.getValue());
        counter.printStats();

        // 2-bosqich: multiplyBy2 ni sinash
        System.out.println("\\n2 ga ko'paytiryapmiz...");
        counter.multiplyBy2();
        System.out.println("Hisoblagich qiymati: " + counter.getValue());

        // 3-bosqich: decrement ni sinash
        System.out.println("\\nDekrementlarni sinaymiz...");
        ExecutorService executor2 = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor2.submit(() -> {
                for (int j = 0; j < operationsPerThread; j++) {
                    counter.decrement();
                }
            });
        }

        executor2.shutdown();
        executor2.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("\\nYakuniy hisoblagich qiymati: " + counter.getValue());
        counter.printStats();
    }
}`,
            description: `Lock-free algoritmlar uchun compare-and-swap (CAS) patternini o'zlashtirinng.

**Talablar:**
1. AtomicInteger dan foydalanadigan Counter klassi yarating
2. Tsiklda CAS dan foydalanib increment() metodini amalga oshiring
3. CAS va validatsiya bilan decrement() metodini amalga oshiring (0 dan past tushmaslik)
4. Qiymatni ikkilantirish uchun CAS dan foydalanib multiplyBy2() metodini yarating
5. Thread-xavfsizlikni tekshirish uchun ko'p oqimlar bilan sinab ko'ring
6. Ishlashni solishtirish metodi uchun weakCompareAndSet() dan foydalaning
7. Muvaffaqiyat darajasini va yakuniy hisoblagich qiymatini chiqaring

Compare-and-swap lock-free dasturlashning asosi bo'lib, oqimlarga konfliktlarda qayta urinish orqali qulflarsiz umumiy ma'lumotlarni yangilashga imkon beradi.`,
            hint1: `CAS patterni: do { current = get(); next = compute(current); } while (!compareAndSet(current, next));`,
            hint2: `Validatsiya bilan decrement uchun CAS urinishidan oldin current 0 ga tengligini tekshiring. Kamaytira olmasangiz, erta qaytaring.`,
            whyItMatters: `Compare-and-swap lock-free algoritmlarning asosi bo'lib, an'anaviy qulflarning xarajatlari va xavflari mavjud bo'lmagan holda yuqori samarali concurrent dasturlashni ta'minlaydi. CAS ni tushunish kengaytiriladigan concurrent tizimlarni qurish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// CAS bilan qulfsiz stek
public class LockFreeStack<T> {
    private final AtomicReference<Node<T>> top = new AtomicReference<>();

    private static class Node<T> {
        final T value;
        final Node<T> next;
        Node(T value, Node<T> next) {
            this.value = value;
            this.next = next;
        }
    }

    public void push(T value) {
        Node<T> newNode = new Node<>(value, null);
        Node<T> currentTop;
        do {
            currentTop = top.get();
            newNode.next = currentTop;
        } while (!top.compareAndSet(currentTop, newNode));
    }

    public T pop() {
        Node<T> currentTop;
        Node<T> newTop;
        do {
            currentTop = top.get();
            if (currentTop == null) return null;
            newTop = currentTop.next;
        } while (!top.compareAndSet(currentTop, newTop));
        return currentTop.value;
    }
}
\`\`\`

**Amaliy foydalari:**
- Qulflar yo'qligi deadlock yo'qligini anglatadi
- Oqimlar raqobatida yuqori ishlash
- Kamida bitta oqimning kafolatlangan rivojlanishi`
        }
    }
};

export default task;
