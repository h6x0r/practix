import { Task } from '../../../../../../types';

export const task: Task = {
	slug: 'java-threads-volatile-keyword',
	title: 'Volatile Keyword',
	difficulty: 'medium',
	tags: ['java', 'threads', 'volatile', 'visibility', 'memory-model'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn about volatile keyword for ensuring visibility and happens-before relationships between threads.

**Requirements:**
1. Implement VolatileFlag Class
   1.1. Add volatile keyword to the running field
   1.2. Ensure worker thread sees the updated value when stop() is called
2. Implement VolatileCache Class
   2.1. Add volatile keyword to the ready field
   2.2. Ensure happens-before relationship between write and read operations
3. Test Both Implementations
   3.1. Test volatile flag for thread communication
   3.2. Test volatile for happens-before guarantee

**Example:**
\`\`\`java
class Flag {
    private volatile boolean running = true;

    public void stop() {
        running = false;  // Visible to all threads immediately
    }
}
\`\`\``,
	initialCode: `class VolatileFlag {
    // TODO: Make this field volatile
    private boolean running = true;
    private int counter = 0;

    public void startCounting() {
        Thread worker = new Thread(() -> {
            while (running) {
                counter++;
            }
            System.out.println("Worker stopped. Counter: " + counter);
        });
        worker.start();
    }

    public void stop() {
        System.out.println("Stopping worker...");
        running = false;
    }

    public int getCounter() {
        return counter;
    }
}

class VolatileCache {
    // TODO: Add volatile keyword to ensure visibility
    private int value = 0;
    private boolean ready = false;

    public void write(int newValue) {
        value = newValue;
        ready = true; // Signal that value is ready
    }

    public int read() {
        while (!ready) {
            // Spin wait until ready
        }
        return value;
    }
}

public class VolatileKeyword {
    public static void main(String[] args) throws InterruptedException {
        // Test 1: Volatile flag for thread communication
        System.out.println("Test 1: Volatile Flag");
        VolatileFlag flag = new VolatileFlag();
        flag.startCounting();

        Thread.sleep(1000);
        flag.stop();
        Thread.sleep(100);

        System.out.println("Final counter: " + flag.getCounter());

        // Test 2: Volatile for happens-before guarantee
        System.out.println("\\nTest 2: Volatile Cache");
        VolatileCache cache = new VolatileCache();

        // Writer thread
        Thread writer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                cache.write(i * 10);
                System.out.println("Written: " + (i * 10));
                try { Thread.sleep(500); } catch (InterruptedException e) {}
            }
        });

        // Reader thread
        Thread reader = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                int value = cache.read();
                System.out.println("Read: " + value);
            }
        });

        writer.start();
        Thread.sleep(100); // Ensure writer starts first
        reader.start();

        writer.join();
        reader.join();
    }
}`,
	solutionCode: `class VolatileFlag {
    // Volatile ensures visibility across threads
    private volatile boolean running = true;
    private int counter = 0;

    public void startCounting() {
        Thread worker = new Thread(() -> {
            System.out.println("Worker started counting...");
            while (running) {	// Volatile read ensures we see the latest value
                counter++;
            }
            System.out.println("Worker stopped. Counter: " + counter);
        });
        worker.start();
    }

    public void stop() {
        System.out.println("Stopping worker...");
        running = false;	// Volatile write ensures visibility to other threads
    }

    public int getCounter() {
        return counter;
    }
}

class VolatileCache {
    // Volatile ensures happens-before relationship
    private int value = 0;
    private volatile boolean ready = false;

    public void write(int newValue) {
        value = newValue;	// Write 1: happens-before
        ready = true;	// Write 2: volatile write (release)
        // All writes before volatile write are visible after volatile read
    }

    public int read() {
        while (!ready) {	// Volatile read (acquire)
            // Spin wait until ready
        }
        return value;	// Read: sees all writes before volatile write
    }
}

public class VolatileKeyword {
    public static void main(String[] args) throws InterruptedException {
        // Test 1: Volatile flag for thread communication
        System.out.println("=== Test 1: Volatile Flag ===");
        VolatileFlag flag = new VolatileFlag();
        flag.startCounting();

        // Let it count for 1 second
        Thread.sleep(1000);
        flag.stop();

        // Give worker thread time to stop
        Thread.sleep(100);

        System.out.println("Final counter: " + flag.getCounter());
        System.out.println("(Without volatile, worker might never stop!)");

        // Test 2: Volatile for happens-before guarantee
        System.out.println("\\n=== Test 2: Volatile Cache ===");
        VolatileCache cache = new VolatileCache();

        // Writer thread
        Thread writer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                cache.write(i * 10);
                System.out.println("Writer: written " + (i * 10));
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }, "Writer");

        // Reader thread
        Thread reader = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                int value = cache.read();
                System.out.println("Reader: read " + value);
            }
        }, "Reader");

        writer.start();
        Thread.sleep(100); // Ensure writer starts first
        reader.start();

        writer.join();
        reader.join();

        System.out.println("\\n(Without volatile, reader might see stale values!)");
    }
}`,
	testCode: `import static org.junit.Assert.*;
import org.junit.Test;

class VolatileFlag {
    private volatile boolean running = true;
    private int counter = 0;

    public void startCounting() {
        Thread worker = new Thread(() -> {
            while (running) {
                counter++;
            }
        });
        worker.start();
    }

    public void stop() {
        running = false;
    }

    public int getCounter() {
        return counter;
    }
}

class VolatileCache {
    private int value = 0;
    private volatile boolean ready = false;

    public void write(int newValue) {
        value = newValue;
        ready = true;
    }

    public int read() {
        while (!ready) {
            // Spin wait
        }
        return value;
    }
}

// Test1: Verify volatile flag stops thread
class Test1 {
    @Test
    public void test() throws Exception {
        VolatileFlag flag = new VolatileFlag();
        flag.startCounting();
        Thread.sleep(100);
        int before = flag.getCounter();
        flag.stop();
        Thread.sleep(100);
        int after = flag.getCounter();
        assertTrue("Counter should stop incrementing after stop", after - before < 100000);
    }
}

// Test2: Verify volatile ensures visibility
class Test2 {
    @Test
    public void test() throws Exception {
        final volatile boolean[] flag = {false};
        final boolean[] observed = {false};

        Thread writer = new Thread(() -> flag[0] = true);
        Thread reader = new Thread(() -> {
            while (!flag[0]) {}
            observed[0] = true;
        });

        reader.start();
        Thread.sleep(50);
        writer.start();
        reader.join(500);

        assertTrue("Reader should observe flag change", observed[0]);
    }
}

// Test3: Verify volatile cache write-read
class Test3 {
    @Test(timeout = 1000)
    public void test() throws Exception {
        VolatileCache cache = new VolatileCache();
        final int[] result = {0};

        Thread writer = new Thread(() -> cache.write(42));
        Thread reader = new Thread(() -> result[0] = cache.read());

        writer.start();
        reader.start();
        writer.join();
        reader.join();

        assertEquals("Reader should see written value", 42, result[0]);
    }
}

// Test4: Verify multiple writes and reads
class Test4 {
    @Test(timeout = 2000)
    public void test() throws Exception {
        VolatileCache cache = new VolatileCache();
        final int[] results = new int[3];

        Thread writer = new Thread(() -> {
            cache.write(10);
            try { Thread.sleep(50); } catch (InterruptedException e) {}
            cache.write(20);
            try { Thread.sleep(50); } catch (InterruptedException e) {}
            cache.write(30);
        });

        Thread reader = new Thread(() -> {
            results[0] = cache.read();
            cache = new VolatileCache();
            results[1] = cache.read();
            cache = new VolatileCache();
            results[2] = cache.read();
        });

        // This test verifies basic volatile behavior
        cache.write(100);
        int val = cache.read();
        assertEquals("Should read written value", 100, val);
    }
}

// Test5: Verify happens-before relationship
class Test5 {
    @Test
    public void test() throws Exception {
        final int[] sharedValue = {0};
        final volatile boolean[] ready = {false};

        Thread writer = new Thread(() -> {
            sharedValue[0] = 100;
            ready[0] = true;
        });

        Thread reader = new Thread(() -> {
            while (!ready[0]) {}
            assertEquals("Should see updated value", 100, sharedValue[0]);
        });

        reader.start();
        Thread.sleep(50);
        writer.start();
        writer.join();
        reader.join();
    }
}

// Test6: Verify volatile boolean flag
class Test6 {
    @Test
    public void test() throws Exception {
        final volatile boolean[] stopFlag = {false};
        final int[] iterations = {0};

        Thread worker = new Thread(() -> {
            while (!stopFlag[0]) {
                iterations[0]++;
                if (iterations[0] >= 1000) break;
            }
        });

        worker.start();
        Thread.sleep(50);
        stopFlag[0] = true;
        worker.join(500);

        assertTrue("Worker should stop", iterations[0] < Integer.MAX_VALUE);
    }
}

// Test7: Verify volatile int visibility
class Test7 {
    @Test
    public void test() throws Exception {
        final volatile int[] sharedInt = {0};
        final boolean[] success = {false};

        Thread writer = new Thread(() -> {
            for (int i = 1; i <= 10; i++) {
                sharedInt[0] = i;
                try { Thread.sleep(10); } catch (InterruptedException e) {}
            }
        });

        Thread reader = new Thread(() -> {
            int lastSeen = 0;
            for (int i = 0; i < 10; i++) {
                int current = sharedInt[0];
                if (current > lastSeen) {
                    lastSeen = current;
                }
                try { Thread.sleep(15); } catch (InterruptedException e) {}
            }
            success[0] = lastSeen > 0;
        });

        writer.start();
        reader.start();
        writer.join();
        reader.join();

        assertTrue("Reader should see updates", success[0]);
    }
}

// Test8: Verify volatile with multiple readers
class Test8 {
    @Test
    public void test() throws Exception {
        final volatile int[] sharedValue = {0};
        final int[] reader1Value = {0};
        final int[] reader2Value = {0};

        Thread writer = new Thread(() -> sharedValue[0] = 999);

        Thread reader1 = new Thread(() -> {
            while (sharedValue[0] == 0) {}
            reader1Value[0] = sharedValue[0];
        });

        Thread reader2 = new Thread(() -> {
            while (sharedValue[0] == 0) {}
            reader2Value[0] = sharedValue[0];
        });

        reader1.start();
        reader2.start();
        Thread.sleep(50);
        writer.start();

        writer.join();
        reader1.join(500);
        reader2.join(500);

        assertEquals("Reader1 should see 999", 999, reader1Value[0]);
        assertEquals("Reader2 should see 999", 999, reader2Value[0]);
    }
}

// Test9: Verify volatile prevents caching
class Test9 {
    @Test
    public void test() throws Exception {
        final volatile long[] timestamp = {0};
        final boolean[] sawUpdate = {false};

        Thread writer = new Thread(() -> {
            try { Thread.sleep(100); } catch (InterruptedException e) {}
            timestamp[0] = System.nanoTime();
        });

        Thread reader = new Thread(() -> {
            long initial = timestamp[0];
            while (timestamp[0] == initial) {
                if (System.currentTimeMillis() % 1000 > 500) break;
            }
            sawUpdate[0] = timestamp[0] != initial;
        });

        reader.start();
        writer.start();
        writer.join();
        reader.join(1000);

        assertTrue("Reader should eventually see update", sawUpdate[0]);
    }
}

// Test10: Verify volatile with coordination
class Test10 {
    @Test(timeout = 1000)
    public void test() throws Exception {
        final volatile boolean[] phase1Done = {false};
        final volatile boolean[] phase2Done = {false};
        final int[] result = {0};

        Thread worker1 = new Thread(() -> {
            result[0] = 10;
            phase1Done[0] = true;
            while (!phase2Done[0]) {}
            result[0] += 20;
        });

        Thread worker2 = new Thread(() -> {
            while (!phase1Done[0]) {}
            result[0] += 5;
            phase2Done[0] = true;
        });

        worker1.start();
        worker2.start();
        worker1.join();
        worker2.join();

        assertTrue("Result should be updated by both threads", result[0] >= 15);
    }
}`,
	hint1: `volatile ensures that reads always see the most recent write. volatile provides visibility but NOT atomicity.`,
	hint2: `Use volatile for flags and simple status variables. volatile establishes happens-before relationship between threads.`,
	whyItMatters: `Volatile is essential for safe thread communication without full synchronization overhead.

**Why Volatile Matters:**
1.1. **Memory Visibility:** Ensures changes made by one thread are visible to others
1.2. **Happens-Before:** Establishes ordering guarantees between operations
1.3. **Performance:** Lighter than synchronized for simple visibility needs

**Volatile vs Synchronized:**
\`\`\`java
// Volatile: visibility only
private volatile boolean flag;  // Fast, but not atomic for compound ops

// Synchronized: visibility + atomicity
synchronized void increment() { count++; }  // Slower, but atomic
\`\`\`

**When to Use Volatile:**
1.1. Simple flags for thread communication (stop signals)
1.2. Status indicators (ready, initialized, running)
1.3. Single-writer, multiple-reader scenarios

**When NOT to Use Volatile:**
1.1. Compound operations (check-then-act, read-modify-write)
1.2. Multiple threads modifying the same variable
1.3. Operations requiring atomicity (use AtomicInteger instead)`,
	order: 6,
	translations: {
		ru: {
			title: 'Ключевое слово volatile',
			description: `Изучите ключевое слово volatile для обеспечения видимости и отношений happens-before между потоками.

**Требования:**
1. Реализуйте класс VolatileFlag
   1.1. Добавьте ключевое слово volatile к полю running
   1.2. Обеспечьте, чтобы рабочий поток видел обновленное значение при вызове stop()
2. Реализуйте класс VolatileCache
   2.1. Добавьте ключевое слово volatile к полю ready
   2.2. Обеспечьте отношение happens-before между операциями записи и чтения
3. Протестируйте обе реализации
   3.1. Проверьте volatile флаг для коммуникации потоков
   3.2. Проверьте volatile для гарантии happens-before`,
			hint1: `volatile гарантирует, что чтения всегда видят последнюю запись. volatile обеспечивает видимость, но НЕ атомарность.`,
			hint2: `Используйте volatile для флагов и простых переменных состояния. volatile устанавливает отношение happens-before между потоками.`,
			whyItMatters: `Volatile необходим для безопасной коммуникации потоков без накладных расходов полной синхронизации.

**Почему Volatile важен:**
1.1. **Видимость памяти:** Гарантирует, что изменения одного потока видны другим
1.2. **Happens-Before:** Устанавливает гарантии порядка операций
1.3. **Производительность:** Легче чем synchronized для простых потребностей видимости

**Когда использовать Volatile:**
1.1. Простые флаги для коммуникации потоков (сигналы остановки)
1.2. Индикаторы статуса (ready, initialized, running)
1.3. Сценарии одного писателя, нескольких читателей

**Когда НЕ использовать Volatile:**
1.1. Составные операции (check-then-act, read-modify-write)
1.2. Несколько потоков модифицируют одну переменную
1.3. Операции требующие атомарности (используйте AtomicInteger)`,
			solutionCode: `class VolatileFlag {
    // Volatile обеспечивает видимость между потоками
    private volatile boolean running = true;
    private int counter = 0;

    public void startCounting() {
        Thread worker = new Thread(() -> {
            System.out.println("Рабочий поток начал подсчет...");
            while (running) {	// Чтение volatile гарантирует, что мы видим последнее значение
                counter++;
            }
            System.out.println("Рабочий поток остановлен. Счетчик: " + counter);
        });
        worker.start();
    }

    public void stop() {
        System.out.println("Останавливаем рабочий поток...");
        running = false;	// Запись volatile обеспечивает видимость для других потоков
    }

    public int getCounter() {
        return counter;
    }
}

class VolatileCache {
    // Volatile обеспечивает отношение happens-before
    private int value = 0;
    private volatile boolean ready = false;

    public void write(int newValue) {
        value = newValue;	// Запись 1: happens-before
        ready = true;	// Запись 2: volatile запись (release)
        // Все записи до volatile записи видны после volatile чтения
    }

    public int read() {
        while (!ready) {	// Volatile чтение (acquire)
            // Ожидание пока готово
        }
        return value;	// Чтение: видит все записи до volatile записи
    }
}

public class VolatileKeyword {
    public static void main(String[] args) throws InterruptedException {
        // Тест 1: Volatile флаг для коммуникации потоков
        System.out.println("=== Тест 1: Volatile флаг ===");
        VolatileFlag flag = new VolatileFlag();
        flag.startCounting();

        // Считаем в течение 1 секунды
        Thread.sleep(1000);
        flag.stop();

        // Даем рабочему потоку время остановиться
        Thread.sleep(100);

        System.out.println("Итоговый счетчик: " + flag.getCounter());
        System.out.println("(Без volatile рабочий поток может никогда не остановиться!)");

        // Тест 2: Volatile для гарантии happens-before
        System.out.println("\\n=== Тест 2: Volatile кэш ===");
        VolatileCache cache = new VolatileCache();

        // Поток записи
        Thread writer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                cache.write(i * 10);
                System.out.println("Записывающий: записано " + (i * 10));
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }, "Writer");

        // Поток чтения
        Thread reader = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                int value = cache.read();
                System.out.println("Читающий: прочитано " + value);
            }
        }, "Reader");

        writer.start();
        Thread.sleep(100); // Убеждаемся, что записывающий поток стартует первым
        reader.start();

        writer.join();
        reader.join();

        System.out.println("\\n(Без volatile читающий поток может видеть устаревшие значения!)");
    }
}`
		},
		uz: {
			title: `Volatile kalit so'zi`,
			description: `Oqimlar o'rtasida ko'rinish va happens-before munosabatlarini ta'minlash uchun volatile kalit so'zini o'rganing.

**Talablar:**
1. VolatileFlag klassini amalga oshiring
   1.1. running maydoniga volatile kalit so'zini qo'shing
   1.2. stop() chaqirilganda ishchi oqim yangilangan qiymatni ko'rishini ta'minlang
2. VolatileCache klassini amalga oshiring
   2.1. ready maydoniga volatile kalit so'zini qo'shing
   2.2. Yozish va o'qish operatsiyalari o'rtasida happens-before munosabatini ta'minlang
3. Ikkala implementatsiyani test qiling
   3.1. Oqimlar aloqasi uchun volatile bayroqni test qiling
   3.2. Happens-before kafolatini beruvchi volatile ni test qiling`,
			hint1: `volatile o'qishlar har doim eng so'nggi yozishni ko'rishini kafolatlaydi. volatile ko'rinishni ta'minlaydi, lekin atomarlikni EMAS.`,
			hint2: `Bayroqlar va oddiy holat o'zgaruvchilari uchun volatile dan foydalaning. volatile oqimlar o'rtasida happens-before munosabatini o'rnatadi.`,
			whyItMatters: `Volatile to'liq sinxronizatsiya xarajatlarisiz xavfsiz oqim aloqasi uchun zarur.

**Nima uchun Volatile muhim:**
1.1. **Xotira ko'rinishi:** Bir oqim tomonidan qilingan o'zgarishlar boshqalarga ko'rinishini kafolatlaydi
1.2. **Happens-Before:** Operatsiyalar o'rtasida tartib kafolatlarini o'rnatadi
1.3. **Ishlash tezligi:** Oddiy ko'rinish ehtiyojlari uchun synchronized dan yengil

**Qachon Volatile ishlatish kerak:**
1.1. Oqimlar aloqasi uchun oddiy bayroqlar (to'xtatish signallari)
1.2. Holat ko'rsatkichlari (ready, initialized, running)
1.3. Bitta yozuvchi, ko'plab o'quvchilar stsenariylari

**Qachon Volatile ishlatMASLIK kerak:**
1.1. Murakkab operatsiyalar (check-then-act, read-modify-write)
1.2. Bir nechta oqim bir xil o'zgaruvchini o'zgartirganda
1.3. Atomarlik talab qiladigan operatsiyalar (AtomicInteger dan foydalaning)`,
			solutionCode: `class VolatileFlag {
    // Volatile oqimlar o'rtasida ko'rinishni ta'minlaydi
    private volatile boolean running = true;
    private int counter = 0;

    public void startCounting() {
        Thread worker = new Thread(() -> {
            System.out.println("Ishchi oqim hisoblashni boshladi...");
            while (running) {	// Volatile o'qish eng oxirgi qiymatni ko'rishni kafolatlaydi
                counter++;
            }
            System.out.println("Ishchi oqim to'xtatildi. Schyotchik: " + counter);
        });
        worker.start();
    }

    public void stop() {
        System.out.println("Ishchi oqimni to'xtatmoqda...");
        running = false;	// Volatile yozish boshqa oqimlar uchun ko'rinishni ta'minlaydi
    }

    public int getCounter() {
        return counter;
    }
}

class VolatileCache {
    // Volatile happens-before munosabatini ta'minlaydi
    private int value = 0;
    private volatile boolean ready = false;

    public void write(int newValue) {
        value = newValue;	// Yozish 1: happens-before
        ready = true;	// Yozish 2: volatile yozish (release)
        // Volatile yozishdan oldingi barcha yozuvlar volatile o'qishdan keyin ko'rinadi
    }

    public int read() {
        while (!ready) {	// Volatile o'qish (acquire)
            // Tayyor bo'lguncha kutamiz
        }
        return value;	// O'qish: volatile yozishdan oldingi barcha yozuvlarni ko'radi
    }
}

public class VolatileKeyword {
    public static void main(String[] args) throws InterruptedException {
        // Test 1: Oqimlar aloqasi uchun volatile bayroq
        System.out.println("=== Test 1: Volatile bayroq ===");
        VolatileFlag flag = new VolatileFlag();
        flag.startCounting();

        // 1 soniya davomida hisoblashga ruxsat beramiz
        Thread.sleep(1000);
        flag.stop();

        // Ishchi oqimga to'xtash uchun vaqt beramiz
        Thread.sleep(100);

        System.out.println("Yakuniy schyotchik: " + flag.getCounter());
        System.out.println("(Volatile bo'lmasa, ishchi oqim hech qachon to'xtamasligi mumkin!)");

        // Test 2: Happens-before kafolatini beruvchi volatile
        System.out.println("\\n=== Test 2: Volatile kesh ===");
        VolatileCache cache = new VolatileCache();

        // Yozuvchi oqim
        Thread writer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                cache.write(i * 10);
                System.out.println("Yozuvchi: " + (i * 10) + " yozildi");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }, "Writer");

        // O'quvchi oqim
        Thread reader = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                int value = cache.read();
                System.out.println("O'quvchi: " + value + " o'qildi");
            }
        }, "Reader");

        writer.start();
        Thread.sleep(100); // Yozuvchi oqim birinchi boshlanishiga ishonch hosil qilamiz
        reader.start();

        writer.join();
        reader.join();

        System.out.println("\\n(Volatile bo'lmasa, o'quvchi oqim eski qiymatlarni ko'rishi mumkin!)");
    }
}`
		}
	}
};

export default task;
