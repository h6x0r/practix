import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-atomic-integer',
    title: 'AtomicInteger Fundamentals',
    difficulty: 'easy',
    tags: ['java', 'concurrency', 'atomic', 'atomicinteger'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn AtomicInteger for thread-safe counter operations.

**Requirements:**
1. Create an AtomicInteger counter initialized to 0
2. Use incrementAndGet() to increment and print the value
3. Use getAndIncrement() to get current value, then increment
4. Use addAndGet(5) to add 5 and get the new value
5. Use compareAndSet() to change 7 to 10 (should succeed)
6. Try compareAndSet() to change 7 to 15 (should fail)
7. Print the final value using get()

AtomicInteger provides lock-free thread-safe operations on integers, making it perfect for counters and statistics in concurrent applications.`,
    initialCode: `import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerDemo {
    public static void main(String[] args) {
        // Create an AtomicInteger initialized to 0

        // Use incrementAndGet()

        // Use getAndIncrement()

        // Use addAndGet(5)

        // Use compareAndSet(7, 10)

        // Try compareAndSet(7, 15)

        // Print final value
    }
}`,
    solutionCode: `import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerDemo {
    public static void main(String[] args) {
        // Create an AtomicInteger initialized to 0
        AtomicInteger counter = new AtomicInteger(0);
        System.out.println("Initial value: " + counter.get());

        // Use incrementAndGet() - increments then returns new value
        int value1 = counter.incrementAndGet();
        System.out.println("After incrementAndGet: " + value1);

        // Use getAndIncrement() - returns current value then increments
        int value2 = counter.getAndIncrement();
        System.out.println("getAndIncrement returned: " + value2);
        System.out.println("Current value: " + counter.get());

        // Use addAndGet(5) - adds 5 then returns new value
        int value3 = counter.addAndGet(5);
        System.out.println("After addAndGet(5): " + value3);

        // Use compareAndSet(7, 10) - should succeed
        boolean success1 = counter.compareAndSet(7, 10);
        System.out.println("compareAndSet(7, 10) success: " + success1);
        System.out.println("Value after CAS: " + counter.get());

        // Try compareAndSet(7, 15) - should fail (value is 10, not 7)
        boolean success2 = counter.compareAndSet(7, 15);
        System.out.println("compareAndSet(7, 15) success: " + success2);

        // Print final value
        System.out.println("Final value: " + counter.get());
    }
}`,
    hint1: `AtomicInteger provides methods like incrementAndGet() (prefix increment) and getAndIncrement() (postfix increment).`,
    hint2: `compareAndSet(expected, new) only updates if the current value equals expected. It returns true if successful.`,
    whyItMatters: `AtomicInteger eliminates the need for locks when implementing thread-safe counters, providing better performance and avoiding deadlocks in concurrent applications.

**Production Pattern:**
\`\`\`java
// Thread-safe request counter
public class RequestCounter {
    private final AtomicInteger totalRequests = new AtomicInteger(0);
    private final AtomicInteger failedRequests = new AtomicInteger(0);

    public void recordRequest() {
        totalRequests.incrementAndGet();
    }

    public void recordFailure() {
        failedRequests.incrementAndGet();
    }

    public double getSuccessRate() {
        int total = totalRequests.get();
        if (total == 0) return 100.0;
        int failed = failedRequests.get();
        return ((total - failed) * 100.0) / total;
    }
}
\`\`\`

**Practical Benefits:**
- High performance without using synchronized
- No risk of deadlocks
- Ideal for metrics and statistics in high-load systems`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.atomic.AtomicInteger;

// Test1: Test initial value
class Test1 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertEquals(10, counter.get());
    }
}

// Test2: Test incrementAndGet
class Test2 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(5);
        assertEquals(6, counter.incrementAndGet());
        assertEquals(6, counter.get());
    }
}

// Test3: Test getAndIncrement
class Test3 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(5);
        assertEquals(5, counter.getAndIncrement());
        assertEquals(6, counter.get());
    }
}

// Test4: Test decrementAndGet
class Test4 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertEquals(9, counter.decrementAndGet());
        assertEquals(9, counter.get());
    }
}

// Test5: Test addAndGet
class Test5 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertEquals(15, counter.addAndGet(5));
        assertEquals(15, counter.get());
    }
}

// Test6: Test compareAndSet success
class Test6 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertTrue(counter.compareAndSet(10, 20));
        assertEquals(20, counter.get());
    }
}

// Test7: Test compareAndSet failure
class Test7 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertFalse(counter.compareAndSet(5, 20));
        assertEquals(10, counter.get());
    }
}

// Test8: Test getAndSet
class Test8 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertEquals(10, counter.getAndSet(20));
        assertEquals(20, counter.get());
    }
}

// Test9: Test updateAndGet
class Test9 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertEquals(20, counter.updateAndGet(x -> x * 2));
        assertEquals(20, counter.get());
    }
}

// Test10: Test accumulateAndGet
class Test10 {
    @Test
    public void test() {
        AtomicInteger counter = new AtomicInteger(10);
        assertEquals(15, counter.accumulateAndGet(5, (x, y) -> x + y));
        assertEquals(15, counter.get());
    }
}
`,
    translations: {
        ru: {
            title: 'Основы AtomicInteger',
            solutionCode: `import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerDemo {
    public static void main(String[] args) {
        // Создаем AtomicInteger, инициализированный нулем
        AtomicInteger counter = new AtomicInteger(0);
        System.out.println("Начальное значение: " + counter.get());

        // Используем incrementAndGet() - увеличивает, затем возвращает новое значение
        int value1 = counter.incrementAndGet();
        System.out.println("После incrementAndGet: " + value1);

        // Используем getAndIncrement() - возвращает текущее значение, затем увеличивает
        int value2 = counter.getAndIncrement();
        System.out.println("getAndIncrement вернул: " + value2);
        System.out.println("Текущее значение: " + counter.get());

        // Используем addAndGet(5) - добавляет 5, затем возвращает новое значение
        int value3 = counter.addAndGet(5);
        System.out.println("После addAndGet(5): " + value3);

        // Используем compareAndSet(7, 10) - должно успешно выполниться
        boolean success1 = counter.compareAndSet(7, 10);
        System.out.println("compareAndSet(7, 10) успех: " + success1);
        System.out.println("Значение после CAS: " + counter.get());

        // Пробуем compareAndSet(7, 15) - должно не выполниться (значение 10, не 7)
        boolean success2 = counter.compareAndSet(7, 15);
        System.out.println("compareAndSet(7, 15) успех: " + success2);

        // Выводим финальное значение
        System.out.println("Финальное значение: " + counter.get());
    }
}`,
            description: `Изучите AtomicInteger для потокобезопасных операций со счетчиком.

**Требования:**
1. Создайте AtomicInteger счетчик, инициализированный нулем
2. Используйте incrementAndGet() для инкремента и вывода значения
3. Используйте getAndIncrement() для получения текущего значения, затем инкремента
4. Используйте addAndGet(5) для добавления 5 и получения нового значения
5. Используйте compareAndSet() для изменения 7 на 10 (должно успешно выполниться)
6. Попробуйте compareAndSet() для изменения 7 на 15 (должно не выполниться)
7. Выведите финальное значение используя get()

AtomicInteger предоставляет неблокирующие потокобезопасные операции над целыми числами, что делает его идеальным для счетчиков и статистики в многопоточных приложениях.`,
            hint1: `AtomicInteger предоставляет методы типа incrementAndGet() (префиксный инкремент) и getAndIncrement() (постфиксный инкремент).`,
            hint2: `compareAndSet(expected, new) обновляет значение только если текущее значение равно expected. Возвращает true при успехе.`,
            whyItMatters: `AtomicInteger устраняет необходимость в блокировках при реализации потокобезопасных счетчиков, обеспечивая лучшую производительность и избегая взаимоблокировок в многопоточных приложениях.

**Продакшен паттерн:**
\`\`\`java
// Потокобезопасный счетчик запросов
public class RequestCounter {
    private final AtomicInteger totalRequests = new AtomicInteger(0);
    private final AtomicInteger failedRequests = new AtomicInteger(0);

    public void recordRequest() {
        totalRequests.incrementAndGet();
    }

    public void recordFailure() {
        failedRequests.incrementAndGet();
    }

    public double getSuccessRate() {
        int total = totalRequests.get();
        if (total == 0) return 100.0;
        int failed = failedRequests.get();
        return ((total - failed) * 100.0) / total;
    }
}
\`\`\`

**Практические преимущества:**
- Высокая производительность без использования synchronized
- Отсутствие риска взаимоблокировок (deadlocks)
- Идеально для метрик и статистики в высоконагруженных системах`
        },
        uz: {
            title: 'AtomicInteger Asoslari',
            solutionCode: `import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerDemo {
    public static void main(String[] args) {
        // Nolga ishga tushirilgan AtomicInteger yaratamiz
        AtomicInteger counter = new AtomicInteger(0);
        System.out.println("Boshlang'ich qiymat: " + counter.get());

        // incrementAndGet() dan foydalanamiz - oshiradi, keyin yangi qiymatni qaytaradi
        int value1 = counter.incrementAndGet();
        System.out.println("incrementAndGet dan keyin: " + value1);

        // getAndIncrement() dan foydalanamiz - hozirgi qiymatni qaytaradi, keyin oshiradi
        int value2 = counter.getAndIncrement();
        System.out.println("getAndIncrement qaytardi: " + value2);
        System.out.println("Hozirgi qiymat: " + counter.get());

        // addAndGet(5) dan foydalanamiz - 5 qo'shadi, keyin yangi qiymatni qaytaradi
        int value3 = counter.addAndGet(5);
        System.out.println("addAndGet(5) dan keyin: " + value3);

        // compareAndSet(7, 10) dan foydalanamiz - muvaffaqiyatli bo'lishi kerak
        boolean success1 = counter.compareAndSet(7, 10);
        System.out.println("compareAndSet(7, 10) muvaffaqiyat: " + success1);
        System.out.println("CAS dan keyingi qiymat: " + counter.get());

        // compareAndSet(7, 15) ni sinab ko'ramiz - muvaffaqiyatsiz bo'lishi kerak (qiymat 10, 7 emas)
        boolean success2 = counter.compareAndSet(7, 15);
        System.out.println("compareAndSet(7, 15) muvaffaqiyat: " + success2);

        // Yakuniy qiymatni chiqaramiz
        System.out.println("Yakuniy qiymat: " + counter.get());
    }
}`,
            description: `Thread-xavfsiz hisoblagich operatsiyalari uchun AtomicInteger ni o'rganing.

**Talablar:**
1. Nolga ishga tushirilgan AtomicInteger hisoblagich yarating
2. incrementAndGet() dan foydalanib, qiymatni oshiring va chiqaring
3. getAndIncrement() dan foydalanib, hozirgi qiymatni oling, keyin oshiring
4. addAndGet(5) dan foydalanib, 5 qo'shing va yangi qiymatni oling
5. compareAndSet() dan foydalanib, 7 ni 10 ga o'zgartiring (muvaffaqiyatli bo'lishi kerak)
6. compareAndSet() dan foydalanib, 7 ni 15 ga o'zgartirishga harakat qiling (muvaffaqiyatsiz bo'lishi kerak)
7. get() dan foydalanib, yakuniy qiymatni chiqaring

AtomicInteger butun sonlar ustida lock-free thread-xavfsiz operatsiyalarni taqdim etadi, bu uni concurrent ilovalarda hisoblagichlar va statistika uchun ideal qiladi.`,
            hint1: `AtomicInteger incrementAndGet() (prefiks inkrement) va getAndIncrement() (postfiks inkrement) kabi metodlarni taqdim etadi.`,
            hint2: `compareAndSet(expected, new) faqat hozirgi qiymat expected ga teng bo'lsa yangilaydi. Muvaffaqiyatli bo'lsa true qaytaradi.`,
            whyItMatters: `AtomicInteger thread-xavfsiz hisoblagichlarni amalga oshirishda qulflar zaruriyatini yo'q qiladi, yaxshi ishlashni ta'minlaydi va concurrent ilovalarda deadlock dan qochadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Thread-xavfsiz so'rovlar hisoblagichi
public class RequestCounter {
    private final AtomicInteger totalRequests = new AtomicInteger(0);
    private final AtomicInteger failedRequests = new AtomicInteger(0);

    public void recordRequest() {
        totalRequests.incrementAndGet();
    }

    public void recordFailure() {
        failedRequests.incrementAndGet();
    }

    public double getSuccessRate() {
        int total = totalRequests.get();
        if (total == 0) return 100.0;
        int failed = failedRequests.get();
        return ((total - failed) * 100.0) / total;
    }
}
\`\`\`

**Amaliy foydalari:**
- Synchronized ishlatmasdan yuqori ishlash
- Deadlock xavfi yo'q
- Yuqori yuklamali tizimlarda metrikalar va statistika uchun ideal`
        }
    }
};

export default task;
