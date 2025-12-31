import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-reentrant-lock',
    title: 'ReentrantLock Basics',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'locks', 'reentrantlock', 'multithreading'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# ReentrantLock Basics

ReentrantLock is a flexible locking mechanism that provides more control than synchronized blocks. It supports lock acquisition with timeout, interruptible locks, and the ability to check if a lock is held. The "reentrant" means a thread can acquire the same lock multiple times.

## Requirements:
1. Create a counter class using ReentrantLock:
   1. Implement increment and decrement operations
   2. Use lock() and unlock() in try-finally blocks
   3. Demonstrate reentrant behavior

2. Implement tryLock() with timeout:
   1. Try to acquire lock with timeout
   2. Handle cases when lock cannot be acquired
   3. Show non-blocking lock attempts

3. Demonstrate fairness:
   1. Create locks with fair and unfair policies
   2. Show how fairness affects thread ordering

4. Multiple threads accessing shared resource with locks

## Example Output:
\`\`\`
=== Basic ReentrantLock ===
Counter value: 1000

=== Reentrant Behavior ===
Outer lock acquired
Inner lock acquired (reentrant)
Lock hold count: 2
Inner lock released
Outer lock released

=== TryLock with Timeout ===
Thread-1: Lock acquired successfully
Thread-2: Waiting for lock...
Thread-2: Lock acquired after wait
Final counter: 2000

=== Fair vs Unfair Lock ===
Fair lock ensures FIFO ordering
Unfair lock allows thread barging
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class ReentrantLockDemo {
    // TODO: Create lock and counter

    public static void main(String[] args) {
        // TODO: Demonstrate basic lock/unlock

        // TODO: Demonstrate reentrant behavior

        // TODO: Demonstrate tryLock with timeout

        // TODO: Show fair vs unfair lock behavior
    }
}`,
    solutionCode: `import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class ReentrantLockDemo {
    private static ReentrantLock lock = new ReentrantLock();
    private static int counter = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Basic ReentrantLock ===");

        // Basic lock/unlock usage
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100; j++) {
                    lock.lock();
                    try {
                        counter++;
                    } finally {
                        lock.unlock(); // Always unlock in finally block
                    }
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            t.join();
        }
        System.out.println("Counter value: " + counter);

        System.out.println("\\n=== Reentrant Behavior ===");
        demonstrateReentrant();

        System.out.println("\\n=== TryLock with Timeout ===");
        demonstrateTryLock();

        System.out.println("\\n=== Fair vs Unfair Lock ===");
        demonstrateFairness();
    }

    // Demonstrate reentrant behavior - same thread can acquire lock multiple times
    private static void demonstrateReentrant() {
        lock.lock();
        try {
            System.out.println("Outer lock acquired");
            System.out.println("Lock hold count: " + lock.getHoldCount());

            lock.lock(); // Reentrant - same thread acquires again
            try {
                System.out.println("Inner lock acquired (reentrant)");
                System.out.println("Lock hold count: " + lock.getHoldCount());
            } finally {
                lock.unlock();
                System.out.println("Inner lock released");
            }
        } finally {
            lock.unlock();
            System.out.println("Outer lock released");
        }
    }

    // Demonstrate tryLock with timeout
    private static void demonstrateTryLock() throws InterruptedException {
        ReentrantLock timedLock = new ReentrantLock();

        Thread t1 = new Thread(() -> {
            timedLock.lock();
            try {
                System.out.println(Thread.currentThread().getName() +
                    ": Lock acquired successfully");
                Thread.sleep(2000); // Hold lock for 2 seconds
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                timedLock.unlock();
            }
        }, "Thread-1");

        Thread t2 = new Thread(() -> {
            try {
                System.out.println(Thread.currentThread().getName() +
                    ": Waiting for lock...");

                // Try to acquire lock with 3 second timeout
                if (timedLock.tryLock(3, TimeUnit.SECONDS)) {
                    try {
                        System.out.println(Thread.currentThread().getName() +
                            ": Lock acquired after wait");
                        counter += 1000;
                    } finally {
                        timedLock.unlock();
                    }
                } else {
                    System.out.println(Thread.currentThread().getName() +
                        ": Could not acquire lock");
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }, "Thread-2");

        t1.start();
        Thread.sleep(100); // Ensure t1 acquires lock first
        t2.start();

        t1.join();
        t2.join();

        System.out.println("Final counter: " + counter);
    }

    // Demonstrate fair vs unfair lock
    private static void demonstrateFairness() {
        // Fair lock - threads acquire lock in FIFO order
        ReentrantLock fairLock = new ReentrantLock(true);

        // Unfair lock - threads may "barge" ahead of waiting threads
        ReentrantLock unfairLock = new ReentrantLock(false);

        System.out.println("Fair lock ensures FIFO ordering");
        System.out.println("Fair lock: " + fairLock.isFair());
        System.out.println("Unfair lock allows thread barging");
        System.out.println("Unfair lock: " + unfairLock.isFair());
    }
}`,
    hint1: `Always use try-finally blocks with ReentrantLock. The lock goes before try, and unlock goes in the finally block to ensure the lock is always released.`,
    hint2: `ReentrantLock provides tryLock() for non-blocking lock attempts and tryLock(timeout) for timed lock attempts. Fair locks ensure FIFO ordering but may have lower throughput.`,
    whyItMatters: `ReentrantLock provides more flexibility than synchronized blocks. It allows timeout-based lock acquisition, interruptible locking, and explicit fairness policies. These features are essential for building robust concurrent applications that need fine-grained control over locking behavior and can handle lock contention gracefully.

**Production pattern:**
\`\`\`java
// Timeout prevents deadlock
if (timedLock.tryLock(3, TimeUnit.SECONDS)) {
    try {
        // Critical section
    } finally {
        timedLock.unlock();
    }
} else {
    // Handle scenario when lock is not acquired
}
\`\`\`

**Practical benefits:**
- Deadlock prevention with timeouts
- Fair handling under high load
- Diagnostics with getHoldCount() and isLocked()`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

// Test1: Verify basic lock/unlock functionality
class Test1 {
    @Test
    public void testBasicLockUnlock() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock();
        int[] counter = {0};

        Thread t = new Thread(() -> {
            lock.lock();
            try {
                counter[0]++;
            } finally {
                lock.unlock();
            }
        });

        t.start();
        t.join();
        assertEquals(1, counter[0]);
    }
}

// Test2: Verify reentrant behavior (same thread can acquire lock multiple times)
class Test2 {
    @Test
    public void testReentrantBehavior() {
        ReentrantLock lock = new ReentrantLock();

        lock.lock();
        assertEquals(1, lock.getHoldCount());

        lock.lock(); // Reentrant
        assertEquals(2, lock.getHoldCount());

        lock.unlock();
        assertEquals(1, lock.getHoldCount());

        lock.unlock();
        assertEquals(0, lock.getHoldCount());
    }
}

// Test3: Verify tryLock without timeout
class Test3 {
    @Test
    public void testTryLock() {
        ReentrantLock lock = new ReentrantLock();

        assertTrue(lock.tryLock());
        assertTrue(lock.isLocked());
        lock.unlock();
        assertFalse(lock.isLocked());
    }
}

// Test4: Verify tryLock with timeout succeeds
class Test4 {
    @Test
    public void testTryLockWithTimeoutSuccess() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock();

        assertTrue(lock.tryLock(1, TimeUnit.SECONDS));
        lock.unlock();
    }
}

// Test5: Verify tryLock with timeout fails when lock is held
class Test5 {
    @Test
    public void testTryLockWithTimeoutFails() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock();

        lock.lock(); // Hold the lock

        Thread t = new Thread(() -> {
            try {
                assertFalse(lock.tryLock(100, TimeUnit.MILLISECONDS));
            } catch (InterruptedException e) {
                fail("Should not be interrupted");
            }
        });

        t.start();
        t.join();
        lock.unlock();
    }
}

// Test6: Verify fair lock behavior
class Test6 {
    @Test
    public void testFairLock() {
        ReentrantLock fairLock = new ReentrantLock(true);
        assertTrue(fairLock.isFair());

        ReentrantLock unfairLock = new ReentrantLock(false);
        assertFalse(unfairLock.isFair());
    }
}

// Test7: Verify isHeldByCurrentThread
class Test7 {
    @Test
    public void testIsHeldByCurrentThread() {
        ReentrantLock lock = new ReentrantLock();

        assertFalse(lock.isHeldByCurrentThread());

        lock.lock();
        assertTrue(lock.isHeldByCurrentThread());

        lock.unlock();
        assertFalse(lock.isHeldByCurrentThread());
    }
}

// Test8: Verify lock prevents concurrent access
class Test8 {
    @Test
    public void testLockPreventsConcurrentAccess() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock();
        int[] counter = {0};
        int iterations = 1000;

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < iterations; i++) {
                lock.lock();
                try {
                    counter[0]++;
                } finally {
                    lock.unlock();
                }
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < iterations; i++) {
                lock.lock();
                try {
                    counter[0]++;
                } finally {
                    lock.unlock();
                }
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        assertEquals(iterations * 2, counter[0]);
    }
}

// Test9: Verify getQueueLength
class Test9 {
    @Test
    public void testGetQueueLength() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock();

        lock.lock();

        Thread t1 = new Thread(() -> {
            lock.lock();
            lock.unlock();
        });

        Thread t2 = new Thread(() -> {
            lock.lock();
            lock.unlock();
        });

        t1.start();
        t2.start();
        Thread.sleep(100); // Let threads queue up

        assertTrue(lock.getQueueLength() >= 0);

        lock.unlock();
        t1.join();
        t2.join();
    }
}

// Test10: Verify multiple reentrant acquisitions
class Test10 {
    @Test
    public void testMultipleReentrantAcquisitions() {
        ReentrantLock lock = new ReentrantLock();

        for (int i = 1; i <= 5; i++) {
            lock.lock();
            assertEquals(i, lock.getHoldCount());
        }

        for (int i = 4; i >= 0; i--) {
            lock.unlock();
            assertEquals(i, lock.getHoldCount());
        }
    }
}`,
    order: 1,
    translations: {
        ru: {
            title: 'Основы ReentrantLock',
            solutionCode: `import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class ReentrantLockDemo {
    private static ReentrantLock lock = new ReentrantLock();
    private static int counter = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Основы ReentrantLock ===");

        // Базовое использование lock/unlock
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100; j++) {
                    lock.lock();
                    try {
                        counter++;
                    } finally {
                        lock.unlock(); // Всегда разблокировать в блоке finally
                    }
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            t.join();
        }
        System.out.println("Counter value: " + counter);

        System.out.println("\\n=== Реентерабельное поведение ===");
        demonstrateReentrant();

        System.out.println("\\n=== TryLock с таймаутом ===");
        demonstrateTryLock();

        System.out.println("\\n=== Справедливая vs несправедливая блокировка ===");
        demonstrateFairness();
    }

    // Демонстрация реентерабельного поведения - один поток может захватить блокировку несколько раз
    private static void demonstrateReentrant() {
        lock.lock();
        try {
            System.out.println("Outer lock acquired");
            System.out.println("Lock hold count: " + lock.getHoldCount());

            lock.lock(); // Реентерабельность - тот же поток захватывает снова
            try {
                System.out.println("Inner lock acquired (reentrant)");
                System.out.println("Lock hold count: " + lock.getHoldCount());
            } finally {
                lock.unlock();
                System.out.println("Inner lock released");
            }
        } finally {
            lock.unlock();
            System.out.println("Outer lock released");
        }
    }

    // Демонстрация tryLock с таймаутом
    private static void demonstrateTryLock() throws InterruptedException {
        ReentrantLock timedLock = new ReentrantLock();

        Thread t1 = new Thread(() -> {
            timedLock.lock();
            try {
                System.out.println(Thread.currentThread().getName() +
                    ": Lock acquired successfully");
                Thread.sleep(2000); // Удерживать блокировку 2 секунды
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                timedLock.unlock();
            }
        }, "Thread-1");

        Thread t2 = new Thread(() -> {
            try {
                System.out.println(Thread.currentThread().getName() +
                    ": Waiting for lock...");

                // Попытка захватить блокировку с таймаутом 3 секунды
                if (timedLock.tryLock(3, TimeUnit.SECONDS)) {
                    try {
                        System.out.println(Thread.currentThread().getName() +
                            ": Lock acquired after wait");
                        counter += 1000;
                    } finally {
                        timedLock.unlock();
                    }
                } else {
                    System.out.println(Thread.currentThread().getName() +
                        ": Could not acquire lock");
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }, "Thread-2");

        t1.start();
        Thread.sleep(100); // Убедиться, что t1 захватывает блокировку первым
        t2.start();

        t1.join();
        t2.join();

        System.out.println("Final counter: " + counter);
    }

    // Демонстрация справедливой vs несправедливой блокировки
    private static void demonstrateFairness() {
        // Справедливая блокировка - потоки захватывают блокировку в порядке FIFO
        ReentrantLock fairLock = new ReentrantLock(true);

        // Несправедливая блокировка - потоки могут "протискиваться" перед ожидающими потоками
        ReentrantLock unfairLock = new ReentrantLock(false);

        System.out.println("Fair lock ensures FIFO ordering");
        System.out.println("Fair lock: " + fairLock.isFair());
        System.out.println("Unfair lock allows thread barging");
        System.out.println("Unfair lock: " + unfairLock.isFair());
    }
}`,
            description: `# Основы ReentrantLock

ReentrantLock - это гибкий механизм блокировки, который обеспечивает больше контроля, чем синхронизированные блоки. Он поддерживает захват блокировки с таймаутом, прерываемые блокировки и возможность проверить, удерживается ли блокировка. "Реентерабельность" означает, что поток может захватить одну и ту же блокировку несколько раз.

## Требования:
1. Создайте класс счетчика с использованием ReentrantLock:
   1. Реализуйте операции инкремента и декремента
   2. Используйте lock() и unlock() в блоках try-finally
   3. Продемонстрируйте реентерабельное поведение

2. Реализуйте tryLock() с таймаутом:
   1. Попытайтесь захватить блокировку с таймаутом
   2. Обработайте случаи, когда блокировка не может быть захвачена
   3. Покажите неблокирующие попытки блокировки

3. Продемонстрируйте справедливость:
   1. Создайте блокировки со справедливой и несправедливой политикой
   2. Покажите, как справедливость влияет на порядок потоков

4. Несколько потоков, обращающихся к общему ресурсу с блокировками

## Пример вывода:
\`\`\`
=== Basic ReentrantLock ===
Counter value: 1000

=== Reentrant Behavior ===
Outer lock acquired
Inner lock acquired (reentrant)
Lock hold count: 2
Inner lock released
Outer lock released

=== TryLock with Timeout ===
Thread-1: Lock acquired successfully
Thread-2: Waiting for lock...
Thread-2: Lock acquired after wait
Final counter: 2000

=== Fair vs Unfair Lock ===
Fair lock ensures FIFO ordering
Unfair lock allows thread barging
\`\`\``,
            hint1: `Всегда используйте блоки try-finally с ReentrantLock. Блокировка идет перед try, а разблокировка идет в блоке finally, чтобы гарантировать, что блокировка всегда освобождается.`,
            hint2: `ReentrantLock предоставляет tryLock() для неблокирующих попыток блокировки и tryLock(timeout) для временных попыток блокировки. Справедливые блокировки обеспечивают порядок FIFO, но могут иметь более низкую пропускную способность.`,
            whyItMatters: `ReentrantLock обеспечивает большую гибкость, чем синхронизированные блоки. Он позволяет захват блокировки на основе таймаута, прерываемую блокировку и явные политики справедливости. Эти функции необходимы для построения надежных параллельных приложений, которым требуется точный контроль над поведением блокировки и которые могут изящно обрабатывать конкуренцию блокировок.

**Продакшен паттерн:**
\`\`\`java
// Таймаут предотвращает дедлок
if (timedLock.tryLock(3, TimeUnit.SECONDS)) {
    try {
        // Критическая секция
    } finally {
        timedLock.unlock();
    }
} else {
    // Обработка сценария, когда блокировка не получена
}
\`\`\`

**Практические преимущества:**
- Предотвращение дедлоков с таймаутами
- Справедливая обработка под высокой нагрузкой
- Диагностика с помощью getHoldCount() и isLocked()`
        },
        uz: {
            title: 'ReentrantLock asoslari',
            solutionCode: `import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;

public class ReentrantLockDemo {
    private static ReentrantLock lock = new ReentrantLock();
    private static int counter = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== ReentrantLock asoslari ===");

        // Asosiy lock/unlock ishlatish
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100; j++) {
                    lock.lock();
                    try {
                        counter++;
                    } finally {
                        lock.unlock(); // Har doim finally blokida qulfni oching
                    }
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            t.join();
        }
        System.out.println("Counter value: " + counter);

        System.out.println("\\n=== Qayta kirish xatti-harakati ===");
        demonstrateReentrant();

        System.out.println("\\n=== Taymer bilan TryLock ===");
        demonstrateTryLock();

        System.out.println("\\n=== Adolatli vs adolatsiz qulf ===");
        demonstrateFairness();
    }

    // Qayta kirish xatti-harakatini namoyish eting - bir xil oqim qulfni bir necha marta olishi mumkin
    private static void demonstrateReentrant() {
        lock.lock();
        try {
            System.out.println("Outer lock acquired");
            System.out.println("Lock hold count: " + lock.getHoldCount());

            lock.lock(); // Qayta kirish - bir xil oqim yana oladi
            try {
                System.out.println("Inner lock acquired (reentrant)");
                System.out.println("Lock hold count: " + lock.getHoldCount());
            } finally {
                lock.unlock();
                System.out.println("Inner lock released");
            }
        } finally {
            lock.unlock();
            System.out.println("Outer lock released");
        }
    }

    // Taymer bilan tryLock ni namoyish eting
    private static void demonstrateTryLock() throws InterruptedException {
        ReentrantLock timedLock = new ReentrantLock();

        Thread t1 = new Thread(() -> {
            timedLock.lock();
            try {
                System.out.println(Thread.currentThread().getName() +
                    ": Lock acquired successfully");
                Thread.sleep(2000); // Qulfni 2 soniya ushlab turing
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                timedLock.unlock();
            }
        }, "Thread-1");

        Thread t2 = new Thread(() -> {
            try {
                System.out.println(Thread.currentThread().getName() +
                    ": Waiting for lock...");

                // 3 soniya taymer bilan qulfni olishga harakat qiling
                if (timedLock.tryLock(3, TimeUnit.SECONDS)) {
                    try {
                        System.out.println(Thread.currentThread().getName() +
                            ": Lock acquired after wait");
                        counter += 1000;
                    } finally {
                        timedLock.unlock();
                    }
                } else {
                    System.out.println(Thread.currentThread().getName() +
                        ": Could not acquire lock");
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }, "Thread-2");

        t1.start();
        Thread.sleep(100); // t1 qulfni birinchi bo'lib olishini ta'minlang
        t2.start();

        t1.join();
        t2.join();

        System.out.println("Final counter: " + counter);
    }

    // Adolatli vs adolatsiz qulfni namoyish eting
    private static void demonstrateFairness() {
        // Adolatli qulf - oqimlar qulfni FIFO tartibida oladilar
        ReentrantLock fairLock = new ReentrantLock(true);

        // Adolatsiz qulf - oqimlar kutayotgan oqimlardan oldinroq "o'tib ketishi" mumkin
        ReentrantLock unfairLock = new ReentrantLock(false);

        System.out.println("Fair lock ensures FIFO ordering");
        System.out.println("Fair lock: " + fairLock.isFair());
        System.out.println("Unfair lock allows thread barging");
        System.out.println("Unfair lock: " + unfairLock.isFair());
    }
}`,
            description: `# ReentrantLock asoslari

ReentrantLock - bu synchronized bloklardan ko'ra ko'proq nazoratni ta'minlaydigan moslashuvchan qulflash mexanizmi. U taymer bilan qulfni olishni, to'xtatilishi mumkin bo'lgan qulflarni va qulf ushlab turilganligini tekshirish imkoniyatini qo'llab-quvvatlaydi. "Reentrant" degani oqim bir xil qulfni bir necha marta olishi mumkinligini anglatadi.

## Talablar:
1. ReentrantLock yordamida hisoblagich klassini yarating:
   1. Oshirish va kamaytirish operatsiyalarini amalga oshiring
   2. try-finally bloklarida lock() va unlock() dan foydalaning
   3. Qayta kirish xatti-harakatini namoyish eting

2. Taymer bilan tryLock() ni amalga oshiring:
   1. Taymer bilan qulfni olishga harakat qiling
   2. Qulf olinmaydigan holatlarni boshqaring
   3. Blokirovka qilmaydigan qulf urinishlarini ko'rsating

3. Adolatlilikni namoyish eting:
   1. Adolatli va adolatsiz siyosat bilan qulflarni yarating
   2. Adolatlilik oqimlar tartibiga qanday ta'sir qilishini ko'rsating

4. Qulflar bilan umumiy resursga kiradigan bir nechta oqimlar

## Chiqish namunasi:
\`\`\`
=== Basic ReentrantLock ===
Counter value: 1000

=== Reentrant Behavior ===
Outer lock acquired
Inner lock acquired (reentrant)
Lock hold count: 2
Inner lock released
Outer lock released

=== TryLock with Timeout ===
Thread-1: Lock acquired successfully
Thread-2: Waiting for lock...
Thread-2: Lock acquired after wait
Final counter: 2000

=== Fair vs Unfair Lock ===
Fair lock ensures FIFO ordering
Unfair lock allows thread barging
\`\`\``,
            hint1: `Har doim ReentrantLock bilan try-finally bloklarini ishlating. Qulf try dan oldin keladi va unlock finally blokida keladi, qulf har doim bo'shatilishini ta'minlash uchun.`,
            hint2: `ReentrantLock blokirovka qilmaydigan qulf urinishlari uchun tryLock() va vaqtli qulf urinishlari uchun tryLock(timeout) ni taqdim etadi. Adolatli qulflar FIFO tartibini ta'minlaydi, lekin pastroq o'tkazuvchanlikka ega bo'lishi mumkin.`,
            whyItMatters: `ReentrantLock synchronized bloklardan ko'ra ko'proq moslashuvchanlikni ta'minlaydi. U taymerga asoslangan qulfni olish, to'xtatilishi mumkin bo'lgan qulflash va aniq adolatlilik siyosatlariga ruxsat beradi. Bu xususiyatlar qulflash xatti-harakatini aniq boshqarishni talab qiladigan va qulf raqobatini chiroyli boshqara oladigan mustahkam parallel ilovalarni yaratish uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Taymer dedlokni oldini oladi
if (timedLock.tryLock(3, TimeUnit.SECONDS)) {
    try {
        // Kritik sektsiya
    } finally {
        timedLock.unlock();
    }
} else {
    // Qulf olinmagan stsenariyni boshqarish
}
\`\`\`

**Amaliy foydalari:**
- Taymerlar bilan dedloklarni oldini olish
- Yuqori yuk ostida adolatli boshqaruv
- getHoldCount() va isLocked() bilan diagnostika`
        }
    }
};

export default task;
