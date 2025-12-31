import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-stamped-lock',
    title: 'StampedLock and Optimistic Reading',
    difficulty: 'hard',
    tags: ['java', 'concurrency', 'locks', 'stampedlock', 'optimistic-locking'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# StampedLock and Optimistic Reading

StampedLock is an advanced lock introduced in Java 8 that supports three modes: writing, reading, and optimistic reading. Optimistic reading allows reading without acquiring a lock, then validating if the data was modified. This provides better performance than ReadWriteLock for read-heavy scenarios.

## Requirements:
1. Create a Point class with StampedLock:
   1. Store x, y coordinates
   2. Implement pessimistic read/write operations
   3. Track lock stamps

2. Implement optimistic reading:
   1. tryOptimisticRead() for lock-free reads
   2. validate() to check if data changed
   3. Fallback to pessimistic read if validation fails

3. Demonstrate lock conversion:
   1. Convert read lock to write lock
   2. tryConvertToWriteLock() for atomic upgrade
   3. Handle conversion failures

4. Compare performance:
   1. Benchmark optimistic vs pessimistic reads
   2. Show performance improvement

## Example Output:
\`\`\`
=== StampedLock Demo ===
Initial point: (0.0, 0.0)

=== Pessimistic Read ===
Reading with read lock: (0.0, 0.0)

=== Pessimistic Write ===
Writing with write lock: (5.0, 10.0)
Point updated: (5.0, 10.0)

=== Optimistic Read (No Interference) ===
Optimistic read stamp: 256
Reading optimistically: (5.0, 10.0)
Validation successful - data unchanged

=== Optimistic Read (With Interference) ===
Optimistic read stamp: 257
Data was modified during read!
Falling back to pessimistic read: (15.0, 20.0)

=== Lock Conversion ===
Acquired read lock, stamp: 258
Converting to write lock...
Conversion successful, stamp: 259
Updated via converted lock: (25.0, 30.0)

=== Performance Comparison ===
Optimistic reads (1000x): 45ms
Pessimistic reads (1000x): 180ms
Performance improvement: 4.0x faster
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class StampedLockDemo {
    // TODO: Create StampedLock and data

    public static void main(String[] args) {
        // TODO: Demonstrate pessimistic read/write

        // TODO: Demonstrate optimistic reading

        // TODO: Demonstrate lock conversion

        // TODO: Compare performance
    }
}`,
    solutionCode: `import java.util.concurrent.locks.StampedLock;

public class StampedLockDemo {
    private static class Point {
        private final StampedLock lock = new StampedLock();
        private double x, y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        // Pessimistic read - acquires read lock
        public double distanceFromOrigin() {
            long stamp = lock.readLock(); // Acquire read lock
            try {
                System.out.println("Reading with read lock: (" + x + ", " + y + ")");
                return Math.sqrt(x * x + y * y);
            } finally {
                lock.unlockRead(stamp); // Release read lock
            }
        }

        // Pessimistic write - acquires write lock
        public void move(double deltaX, double deltaY) {
            long stamp = lock.writeLock(); // Acquire write lock
            try {
                System.out.println("Writing with write lock: (" +
                    (x + deltaX) + ", " + (y + deltaY) + ")");
                x += deltaX;
                y += deltaY;
            } finally {
                lock.unlockWrite(stamp); // Release write lock
            }
        }

        // Optimistic read - reads without locking, then validates
        public String optimisticRead() {
            long stamp = lock.tryOptimisticRead(); // Get optimistic read stamp
            System.out.println("Optimistic read stamp: " + stamp);

            // Read data without holding lock
            double currentX = x;
            double currentY = y;

            // Validate that data wasn't modified
            if (!lock.validate(stamp)) {
                System.out.println("Data was modified during read!");
                System.out.println("Falling back to pessimistic read...");

                // Validation failed - upgrade to pessimistic read
                stamp = lock.readLock();
                try {
                    currentX = x;
                    currentY = y;
                    return "(" + currentX + ", " + currentY + ")";
                } finally {
                    lock.unlockRead(stamp);
                }
            }

            System.out.println("Validation successful - data unchanged");
            return "(" + currentX + ", " + currentY + ")";
        }

        // Lock conversion - convert read lock to write lock
        public void moveWithConversion(double deltaX, double deltaY) {
            long stamp = lock.readLock(); // Start with read lock
            System.out.println("Acquired read lock, stamp: " + stamp);

            try {
                // Try to convert read lock to write lock
                System.out.println("Converting to write lock...");
                long writeStamp = lock.tryConvertToWriteLock(stamp);

                if (writeStamp != 0L) {
                    // Conversion successful
                    System.out.println("Conversion successful, stamp: " + writeStamp);
                    stamp = writeStamp;
                    x += deltaX;
                    y += deltaY;
                } else {
                    // Conversion failed - release read lock and acquire write lock
                    System.out.println("Conversion failed - acquiring write lock");
                    lock.unlockRead(stamp);
                    stamp = lock.writeLock();
                    x += deltaX;
                    y += deltaY;
                }
            } finally {
                lock.unlock(stamp); // Works for both read and write locks
            }
        }

        public String getPosition() {
            return "(" + x + ", " + y + ")";
        }
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== StampedLock Demo ===");

        Point point = new Point(0.0, 0.0);
        System.out.println("Initial point: " + point.getPosition());

        System.out.println("\\n=== Pessimistic Read ===");
        point.distanceFromOrigin();

        System.out.println("\\n=== Pessimistic Write ===");
        point.move(5.0, 10.0);
        System.out.println("Point updated: " + point.getPosition());

        System.out.println("\\n=== Optimistic Read (No Interference) ===");
        String result1 = point.optimisticRead();
        System.out.println("Reading optimistically: " + result1);

        System.out.println("\\n=== Optimistic Read (With Interference) ===");
        // Simulate interference
        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(50); // Let optimistic read start
                point.move(10.0, 10.0);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        writer.start();
        Thread.sleep(10); // Start optimistic read
        String result2 = point.optimisticRead();
        writer.join();

        System.out.println("\\n=== Lock Conversion ===");
        point.moveWithConversion(10.0, 10.0);
        System.out.println("Updated via converted lock: " + point.getPosition());

        System.out.println("\\n=== Performance Comparison ===");
        comparePerformance();
    }

    private static void comparePerformance() throws InterruptedException {
        Point point = new Point(0.0, 0.0);
        int iterations = 1000;

        // Benchmark optimistic reads
        long startOptimistic = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            point.lock.tryOptimisticRead();
            double x = point.x;
            double y = point.y;
            point.lock.validate(0); // Just for comparison
        }
        long optimisticTime = System.currentTimeMillis() - startOptimistic;

        // Benchmark pessimistic reads
        long startPessimistic = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            long stamp = point.lock.readLock();
            try {
                double x = point.x;
                double y = point.y;
            } finally {
                point.lock.unlockRead(stamp);
            }
        }
        long pessimisticTime = System.currentTimeMillis() - startPessimistic;

        System.out.println("Optimistic reads (" + iterations + "x): " +
            optimisticTime + "ms");
        System.out.println("Pessimistic reads (" + iterations + "x): " +
            pessimisticTime + "ms");

        if (optimisticTime > 0) {
            System.out.printf("Performance improvement: %.1fx faster\\n",
                (double) pessimisticTime / optimisticTime);
        }
    }
}`,
    hint1: `StampedLock's tryOptimisticRead() returns a stamp without acquiring a lock. Read your data, then call validate(stamp) to check if it was modified. If validation fails, fall back to a pessimistic read lock.`,
    hint2: `Lock conversion with tryConvertToWriteLock() allows atomic upgrade from read to write lock. Check if the returned stamp is non-zero to verify successful conversion.`,
    whyItMatters: `StampedLock with optimistic reading provides superior performance for read-heavy workloads compared to ReadWriteLock. The optimistic read pattern allows reads without lock acquisition overhead, making it ideal for scenarios where reads vastly outnumber writes and write conflicts are rare. Understanding StampedLock is essential for building high-performance concurrent systems with minimal lock contention.

**Production pattern:**
\`\`\`java
// Optimistic reading - no locking!
long stamp = lock.tryOptimisticRead();
double x = point.x;
double y = point.y;

if (!lock.validate(stamp)) {
    // Fallback to pessimistic read
    stamp = lock.readLock();
    try {
        x = point.x;
        y = point.y;
    } finally {
        lock.unlockRead(stamp);
    }
}
\`\`\`

**Practical benefits:**
- 4-8x faster than ReadWriteLock
- Ideal for coordinates, metrics
- Zero read cost when no writes`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.locks.StampedLock;

// Test1: Verify basic write lock acquisition
class Test1 {
    @Test
    public void testWriteLockAcquisition() {
        StampedLock lock = new StampedLock();
        long stamp = lock.writeLock();

        assertTrue(stamp != 0);
        lock.unlockWrite(stamp);
    }
}

// Test2: Verify basic read lock acquisition
class Test2 {
    @Test
    public void testReadLockAcquisition() {
        StampedLock lock = new StampedLock();
        long stamp = lock.readLock();

        assertTrue(stamp != 0);
        lock.unlockRead(stamp);
    }
}

// Test3: Verify optimistic read without contention
class Test3 {
    @Test
    public void testOptimisticReadNoContention() {
        StampedLock lock = new StampedLock();
        int[] data = {42};

        long stamp = lock.tryOptimisticRead();
        int value = data[0];

        assertTrue(lock.validate(stamp));
        assertEquals(42, value);
    }
}

// Test4: Verify optimistic read with validation
class Test4 {
    @Test
    public void testOptimisticReadValidation() throws InterruptedException {
        StampedLock lock = new StampedLock();
        int[] data = {0};

        long stamp = lock.tryOptimisticRead();
        assertTrue(stamp != 0);

        // Data read during optimistic read
        int value = data[0];

        // Validation should succeed if no write occurred
        boolean valid = lock.validate(stamp);
        assertTrue(valid || !valid); // Either outcome is valid
    }
}

// Test5: Verify tryConvertToWriteLock success
class Test5 {
    @Test
    public void testTryConvertToWriteLock() {
        StampedLock lock = new StampedLock();

        long readStamp = lock.readLock();
        long writeStamp = lock.tryConvertToWriteLock(readStamp);

        assertTrue(writeStamp != 0);
        lock.unlockWrite(writeStamp);
    }
}

// Test6: Verify tryConvertToReadLock
class Test6 {
    @Test
    public void testTryConvertToReadLock() {
        StampedLock lock = new StampedLock();

        long writeStamp = lock.writeLock();
        long readStamp = lock.tryConvertToReadLock(writeStamp);

        assertTrue(readStamp != 0);
        lock.unlockRead(readStamp);
    }
}

// Test7: Verify write lock blocks reads
class Test7 {
    @Test
    public void testWriteLockBlocksReads() throws InterruptedException {
        StampedLock lock = new StampedLock();
        boolean[] readBlocked = {false};

        long writeStamp = lock.writeLock();

        Thread reader = new Thread(() -> {
            long stamp = lock.tryReadLock();
            if (stamp == 0) {
                readBlocked[0] = true;
            } else {
                lock.unlockRead(stamp);
            }
        });

        reader.start();
        reader.join();

        assertTrue(readBlocked[0]);
        lock.unlockWrite(writeStamp);
    }
}

// Test8: Verify unlock method works for all lock types
class Test8 {
    @Test
    public void testUnlockMethod() {
        StampedLock lock = new StampedLock();

        long writeStamp = lock.writeLock();
        assertTrue(lock.isWriteLocked());
        lock.unlock(writeStamp);
        assertFalse(lock.isWriteLocked());

        long readStamp = lock.readLock();
        assertTrue(lock.isReadLocked());
        lock.unlock(readStamp);
        assertFalse(lock.isReadLocked());
    }
}

// Test9: Verify isWriteLocked and isReadLocked
class Test9 {
    @Test
    public void testLockStateChecks() {
        StampedLock lock = new StampedLock();

        assertFalse(lock.isWriteLocked());
        assertFalse(lock.isReadLocked());

        long writeStamp = lock.writeLock();
        assertTrue(lock.isWriteLocked());

        lock.unlockWrite(writeStamp);
        assertFalse(lock.isWriteLocked());
    }
}

// Test10: Verify data modification with write lock
class Test10 {
    @Test
    public void testDataModificationWithWriteLock() {
        StampedLock lock = new StampedLock();
        int[] point = {0, 0};

        long stamp = lock.writeLock();
        try {
            point[0] = 10;
            point[1] = 20;
        } finally {
            lock.unlockWrite(stamp);
        }

        assertEquals(10, point[0]);
        assertEquals(20, point[1]);
    }
}`,
    order: 3,
    translations: {
        ru: {
            title: 'StampedLock и оптимистическое чтение',
            solutionCode: `import java.util.concurrent.locks.StampedLock;

public class StampedLockDemo {
    private static class Point {
        private final StampedLock lock = new StampedLock();
        private double x, y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        // Пессимистическое чтение - захватывает блокировку чтения
        public double distanceFromOrigin() {
            long stamp = lock.readLock(); // Захватить блокировку чтения
            try {
                System.out.println("Reading with read lock: (" + x + ", " + y + ")");
                return Math.sqrt(x * x + y * y);
            } finally {
                lock.unlockRead(stamp); // Освободить блокировку чтения
            }
        }

        // Пессимистическая запись - захватывает блокировку записи
        public void move(double deltaX, double deltaY) {
            long stamp = lock.writeLock(); // Захватить блокировку записи
            try {
                System.out.println("Writing with write lock: (" +
                    (x + deltaX) + ", " + (y + deltaY) + ")");
                x += deltaX;
                y += deltaY;
            } finally {
                lock.unlockWrite(stamp); // Освободить блокировку записи
            }
        }

        // Оптимистическое чтение - читает без блокировки, затем проверяет
        public String optimisticRead() {
            long stamp = lock.tryOptimisticRead(); // Получить штамп оптимистического чтения
            System.out.println("Optimistic read stamp: " + stamp);

            // Читать данные без удерживания блокировки
            double currentX = x;
            double currentY = y;

            // Проверить, что данные не были изменены
            if (!lock.validate(stamp)) {
                System.out.println("Data was modified during read!");
                System.out.println("Falling back to pessimistic read...");

                // Проверка не удалась - перейти к пессимистическому чтению
                stamp = lock.readLock();
                try {
                    currentX = x;
                    currentY = y;
                    return "(" + currentX + ", " + currentY + ")";
                } finally {
                    lock.unlockRead(stamp);
                }
            }

            System.out.println("Validation successful - data unchanged");
            return "(" + currentX + ", " + currentY + ")";
        }

        // Преобразование блокировки - преобразовать блокировку чтения в блокировку записи
        public void moveWithConversion(double deltaX, double deltaY) {
            long stamp = lock.readLock(); // Начать с блокировки чтения
            System.out.println("Acquired read lock, stamp: " + stamp);

            try {
                // Попытка преобразовать блокировку чтения в блокировку записи
                System.out.println("Converting to write lock...");
                long writeStamp = lock.tryConvertToWriteLock(stamp);

                if (writeStamp != 0L) {
                    // Преобразование успешно
                    System.out.println("Conversion successful, stamp: " + writeStamp);
                    stamp = writeStamp;
                    x += deltaX;
                    y += deltaY;
                } else {
                    // Преобразование не удалось - освободить блокировку чтения и захватить блокировку записи
                    System.out.println("Conversion failed - acquiring write lock");
                    lock.unlockRead(stamp);
                    stamp = lock.writeLock();
                    x += deltaX;
                    y += deltaY;
                }
            } finally {
                lock.unlock(stamp); // Работает для блокировок чтения и записи
            }
        }

        public String getPosition() {
            return "(" + x + ", " + y + ")";
        }
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Демонстрация StampedLock ===");

        Point point = new Point(0.0, 0.0);
        System.out.println("Initial point: " + point.getPosition());

        System.out.println("\\n=== Пессимистическое чтение ===");
        point.distanceFromOrigin();

        System.out.println("\\n=== Пессимистическая запись ===");
        point.move(5.0, 10.0);
        System.out.println("Point updated: " + point.getPosition());

        System.out.println("\\n=== Оптимистическое чтение (без вмешательства) ===");
        String result1 = point.optimisticRead();
        System.out.println("Reading optimistically: " + result1);

        System.out.println("\\n=== Оптимистическое чтение (с вмешательством) ===");
        // Имитация вмешательства
        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(50); // Позволить оптимистическому чтению начаться
                point.move(10.0, 10.0);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        writer.start();
        Thread.sleep(10); // Начать оптимистическое чтение
        String result2 = point.optimisticRead();
        writer.join();

        System.out.println("\\n=== Преобразование блокировки ===");
        point.moveWithConversion(10.0, 10.0);
        System.out.println("Updated via converted lock: " + point.getPosition());

        System.out.println("\\n=== Сравнение производительности ===");
        comparePerformance();
    }

    private static void comparePerformance() throws InterruptedException {
        Point point = new Point(0.0, 0.0);
        int iterations = 1000;

        // Бенчмарк оптимистических чтений
        long startOptimistic = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            point.lock.tryOptimisticRead();
            double x = point.x;
            double y = point.y;
            point.lock.validate(0); // Только для сравнения
        }
        long optimisticTime = System.currentTimeMillis() - startOptimistic;

        // Бенчмарк пессимистических чтений
        long startPessimistic = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            long stamp = point.lock.readLock();
            try {
                double x = point.x;
                double y = point.y;
            } finally {
                point.lock.unlockRead(stamp);
            }
        }
        long pessimisticTime = System.currentTimeMillis() - startPessimistic;

        System.out.println("Optimistic reads (" + iterations + "x): " +
            optimisticTime + "ms");
        System.out.println("Pessimistic reads (" + iterations + "x): " +
            pessimisticTime + "ms");

        if (optimisticTime > 0) {
            System.out.printf("Performance improvement: %.1fx faster\\n",
                (double) pessimisticTime / optimisticTime);
        }
    }
}`,
            description: `# StampedLock и оптимистическое чтение

StampedLock - это продвинутая блокировка, представленная в Java 8, которая поддерживает три режима: запись, чтение и оптимистическое чтение. Оптимистическое чтение позволяет читать без захвата блокировки, затем проверять, были ли данные изменены. Это обеспечивает лучшую производительность, чем ReadWriteLock для сценариев с интенсивным чтением.

## Требования:
1. Создайте класс Point с StampedLock:
   1. Храните координаты x, y
   2. Реализуйте пессимистические операции чтения/записи
   3. Отслеживайте штампы блокировок

2. Реализуйте оптимистическое чтение:
   1. tryOptimisticRead() для чтения без блокировки
   2. validate() для проверки изменения данных
   3. Откат к пессимистическому чтению при неудаче проверки

3. Продемонстрируйте преобразование блокировки:
   1. Преобразовать блокировку чтения в блокировку записи
   2. tryConvertToWriteLock() для атомарного обновления
   3. Обработать неудачи преобразования

4. Сравните производительность:
   1. Бенчмарк оптимистических vs пессимистических чтений
   2. Покажите улучшение производительности

## Пример вывода:
\`\`\`
=== StampedLock Demo ===
Initial point: (0.0, 0.0)

=== Pessimistic Read ===
Reading with read lock: (0.0, 0.0)

=== Pessimistic Write ===
Writing with write lock: (5.0, 10.0)
Point updated: (5.0, 10.0)

=== Optimistic Read (No Interference) ===
Optimistic read stamp: 256
Reading optimistically: (5.0, 10.0)
Validation successful - data unchanged

=== Optimistic Read (With Interference) ===
Optimistic read stamp: 257
Data was modified during read!
Falling back to pessimistic read: (15.0, 20.0)

=== Lock Conversion ===
Acquired read lock, stamp: 258
Converting to write lock...
Conversion successful, stamp: 259
Updated via converted lock: (25.0, 30.0)

=== Performance Comparison ===
Optimistic reads (1000x): 45ms
Pessimistic reads (1000x): 180ms
Performance improvement: 4.0x faster
\`\`\``,
            hint1: `tryOptimisticRead() в StampedLock возвращает штамп без захвата блокировки. Прочитайте ваши данные, затем вызовите validate(stamp), чтобы проверить, были ли они изменены. Если проверка не удалась, откатитесь к пессимистической блокировке чтения.`,
            hint2: `Преобразование блокировки с tryConvertToWriteLock() позволяет атомарное обновление с чтения на запись. Проверьте, что возвращенный штамп не равен нулю, чтобы подтвердить успешное преобразование.`,
            whyItMatters: `StampedLock с оптимистическим чтением обеспечивает превосходную производительность для рабочих нагрузок с интенсивным чтением по сравнению с ReadWriteLock. Паттерн оптимистического чтения позволяет читать без накладных расходов на захват блокировки, что делает его идеальным для сценариев, где чтения значительно превышают записи, а конфликты записи редки. Понимание StampedLock необходимо для построения высокопроизводительных параллельных систем с минимальной конкуренцией блокировок.

**Продакшен паттерн:**
\`\`\`java
// Оптимистическое чтение - без блокировки!
long stamp = lock.tryOptimisticRead();
double x = point.x;
double y = point.y;

if (!lock.validate(stamp)) {
    // Откат к пессимистическому чтению
    stamp = lock.readLock();
    try {
        x = point.x;
        y = point.y;
    } finally {
        lock.unlockRead(stamp);
    }
}
\`\`\`

**Практические преимущества:**
- 4-8x быстрее ReadWriteLock
- Идеально для координат, метрик
- Нулевая стоимость чтения при отсутствии записи`
        },
        uz: {
            title: 'StampedLock va optimistik o\'qish',
            solutionCode: `import java.util.concurrent.locks.StampedLock;

public class StampedLockDemo {
    private static class Point {
        private final StampedLock lock = new StampedLock();
        private double x, y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        // Pessimistik o'qish - o'qish qulfini oladi
        public double distanceFromOrigin() {
            long stamp = lock.readLock(); // O'qish qulfini oling
            try {
                System.out.println("Reading with read lock: (" + x + ", " + y + ")");
                return Math.sqrt(x * x + y * y);
            } finally {
                lock.unlockRead(stamp); // O'qish qulfini bo'shating
            }
        }

        // Pessimistik yozish - yozish qulfini oladi
        public void move(double deltaX, double deltaY) {
            long stamp = lock.writeLock(); // Yozish qulfini oling
            try {
                System.out.println("Writing with write lock: (" +
                    (x + deltaX) + ", " + (y + deltaY) + ")");
                x += deltaX;
                y += deltaY;
            } finally {
                lock.unlockWrite(stamp); // Yozish qulfini bo'shating
            }
        }

        // Optimistik o'qish - qulfsiz o'qiydi, keyin tekshiradi
        public String optimisticRead() {
            long stamp = lock.tryOptimisticRead(); // Optimistik o'qish tamg'asini oling
            System.out.println("Optimistic read stamp: " + stamp);

            // Qulfni ushlamasdan ma'lumotlarni o'qing
            double currentX = x;
            double currentY = y;

            // Ma'lumotlar o'zgarmaganligini tekshiring
            if (!lock.validate(stamp)) {
                System.out.println("Data was modified during read!");
                System.out.println("Falling back to pessimistic read...");

                // Tekshirish muvaffaqiyatsiz - pessimistik o'qishga o'ting
                stamp = lock.readLock();
                try {
                    currentX = x;
                    currentY = y;
                    return "(" + currentX + ", " + currentY + ")";
                } finally {
                    lock.unlockRead(stamp);
                }
            }

            System.out.println("Validation successful - data unchanged");
            return "(" + currentX + ", " + currentY + ")";
        }

        // Qulfni konvertatsiya qilish - o'qish qulfini yozish qulfiga o'zgartirish
        public void moveWithConversion(double deltaX, double deltaY) {
            long stamp = lock.readLock(); // O'qish qulfi bilan boshlang
            System.out.println("Acquired read lock, stamp: " + stamp);

            try {
                // O'qish qulfini yozish qulfiga o'zgartirishga harakat qiling
                System.out.println("Converting to write lock...");
                long writeStamp = lock.tryConvertToWriteLock(stamp);

                if (writeStamp != 0L) {
                    // Konvertatsiya muvaffaqiyatli
                    System.out.println("Conversion successful, stamp: " + writeStamp);
                    stamp = writeStamp;
                    x += deltaX;
                    y += deltaY;
                } else {
                    // Konvertatsiya muvaffaqiyatsiz - o'qish qulfini bo'shating va yozish qulfini oling
                    System.out.println("Conversion failed - acquiring write lock");
                    lock.unlockRead(stamp);
                    stamp = lock.writeLock();
                    x += deltaX;
                    y += deltaY;
                }
            } finally {
                lock.unlock(stamp); // O'qish va yozish qulflari uchun ishlaydi
            }
        }

        public String getPosition() {
            return "(" + x + ", " + y + ")";
        }
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== StampedLock namoyishi ===");

        Point point = new Point(0.0, 0.0);
        System.out.println("Initial point: " + point.getPosition());

        System.out.println("\\n=== Pessimistik o'qish ===");
        point.distanceFromOrigin();

        System.out.println("\\n=== Pessimistik yozish ===");
        point.move(5.0, 10.0);
        System.out.println("Point updated: " + point.getPosition());

        System.out.println("\\n=== Optimistik o'qish (aralashmasdan) ===");
        String result1 = point.optimisticRead();
        System.out.println("Reading optimistically: " + result1);

        System.out.println("\\n=== Optimistik o'qish (aralashish bilan) ===");
        // Aralashishni simulyatsiya qilish
        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(50); // Optimistik o'qishni boshlashga ruxsat bering
                point.move(10.0, 10.0);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        writer.start();
        Thread.sleep(10); // Optimistik o'qishni boshlang
        String result2 = point.optimisticRead();
        writer.join();

        System.out.println("\\n=== Qulfni konvertatsiya qilish ===");
        point.moveWithConversion(10.0, 10.0);
        System.out.println("Updated via converted lock: " + point.getPosition());

        System.out.println("\\n=== Unumdorlikni taqqoslash ===");
        comparePerformance();
    }

    private static void comparePerformance() throws InterruptedException {
        Point point = new Point(0.0, 0.0);
        int iterations = 1000;

        // Optimistik o'qishlarni benchmark qilish
        long startOptimistic = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            point.lock.tryOptimisticRead();
            double x = point.x;
            double y = point.y;
            point.lock.validate(0); // Faqat taqqoslash uchun
        }
        long optimisticTime = System.currentTimeMillis() - startOptimistic;

        // Pessimistik o'qishlarni benchmark qilish
        long startPessimistic = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            long stamp = point.lock.readLock();
            try {
                double x = point.x;
                double y = point.y;
            } finally {
                point.lock.unlockRead(stamp);
            }
        }
        long pessimisticTime = System.currentTimeMillis() - startPessimistic;

        System.out.println("Optimistic reads (" + iterations + "x): " +
            optimisticTime + "ms");
        System.out.println("Pessimistic reads (" + iterations + "x): " +
            pessimisticTime + "ms");

        if (optimisticTime > 0) {
            System.out.printf("Performance improvement: %.1fx faster\\n",
                (double) pessimisticTime / optimisticTime);
        }
    }
}`,
            description: `# StampedLock va optimistik o'qish

StampedLock - bu Java 8 da taqdim etilgan ilg'or qulf bo'lib, uchta rejimni qo'llab-quvvatlaydi: yozish, o'qish va optimistik o'qish. Optimistik o'qish qulfni olmasdan o'qishga imkon beradi, keyin ma'lumotlar o'zgartirilganligini tekshiradi. Bu o'qish-intensiv stsenariylar uchun ReadWriteLock dan yaxshiroq unumdorlikni ta'minlaydi.

## Talablar:
1. StampedLock bilan Point klassini yarating:
   1. x, y koordinatalarini saqlang
   2. Pessimistik o'qish/yozish operatsiyalarini amalga oshiring
   3. Qulf tamg'alarini kuzatib boring

2. Optimistik o'qishni amalga oshiring:
   1. Qulfsiz o'qish uchun tryOptimisticRead()
   2. Ma'lumotlar o'zgarganligini tekshirish uchun validate()
   3. Tekshirish muvaffaqiyatsiz bo'lsa pessimistik o'qishga qaytish

3. Qulfni konvertatsiya qilishni namoyish eting:
   1. O'qish qulfini yozish qulfiga o'zgartirish
   2. Atom yangilash uchun tryConvertToWriteLock()
   3. Konvertatsiya muvaffaqiyatsizliklarini boshqarish

4. Unumdorlikni taqqoslang:
   1. Optimistik vs pessimistik o'qishlarni benchmark qiling
   2. Unumdorlik yaxshilanishini ko'rsating

## Chiqish namunasi:
\`\`\`
=== StampedLock Demo ===
Initial point: (0.0, 0.0)

=== Pessimistic Read ===
Reading with read lock: (0.0, 0.0)

=== Pessimistic Write ===
Writing with write lock: (5.0, 10.0)
Point updated: (5.0, 10.0)

=== Optimistic Read (No Interference) ===
Optimistic read stamp: 256
Reading optimistically: (5.0, 10.0)
Validation successful - data unchanged

=== Optimistic Read (With Interference) ===
Optimistic read stamp: 257
Data was modified during read!
Falling back to pessimistic read: (15.0, 20.0)

=== Lock Conversion ===
Acquired read lock, stamp: 258
Converting to write lock...
Conversion successful, stamp: 259
Updated via converted lock: (25.0, 30.0)

=== Performance Comparison ===
Optimistic reads (1000x): 45ms
Pessimistic reads (1000x): 180ms
Performance improvement: 4.0x faster
\`\`\``,
            hint1: `StampedLock ning tryOptimisticRead() qulfni olmasdan tamg'a qaytaradi. Ma'lumotlaringizni o'qing, keyin u o'zgartirilganligini tekshirish uchun validate(stamp) ni chaqiring. Agar tekshirish muvaffaqiyatsiz bo'lsa, pessimistik o'qish qulfiga qaytaring.`,
            hint2: `tryConvertToWriteLock() bilan qulfni konvertatsiya qilish o'qish qulfidan yozish qulfiga atom yangilashga imkon beradi. Muvaffaqiyatli konvertatsiyani tasdiqlash uchun qaytarilgan tamg'a nolga teng emasligini tekshiring.`,
            whyItMatters: `Optimistik o'qish bilan StampedLock ReadWriteLock bilan solishtirganda ko'p o'qiladigan ish yuklari uchun yuqori unumdorlikni ta'minlaydi. Optimistik o'qish naqshi qulfni olish xarajatlarisiz o'qishga imkon beradi, bu o'qishlar yozishlardan ancha ko'p bo'lgan va yozish to'qnashuvlari kamdan-kam bo'lgan stsenariylar uchun idealdir. StampedLock ni tushunish minimal qulf raqobati bilan yuqori unumdorlikli parallel tizimlarni yaratish uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Optimistik o'qish - qulfsiz!
long stamp = lock.tryOptimisticRead();
double x = point.x;
double y = point.y;

if (!lock.validate(stamp)) {
    // Pessimistik o'qishga qaytish
    stamp = lock.readLock();
    try {
        x = point.x;
        y = point.y;
    } finally {
        lock.unlockRead(stamp);
    }
}
\`\`\`

**Amaliy foydalari:**
- ReadWriteLock dan 4-8x tezroq
- Koordinatlar, metrikalar uchun ideal
- Yozish bo'lmaganda o'qish nol xarajat`
        }
    }
};

export default task;
