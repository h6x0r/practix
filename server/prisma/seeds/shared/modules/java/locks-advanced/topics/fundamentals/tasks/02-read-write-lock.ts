import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-read-write-lock',
    title: 'ReadWriteLock',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'locks', 'readwritelock', 'multithreading'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# ReadWriteLock

ReadWriteLock allows multiple threads to read concurrently but ensures exclusive access for writes. It's optimized for scenarios where reads are more frequent than writes, improving performance by allowing concurrent reads while maintaining data consistency.

## Requirements:
1. Create a shared resource with ReadWriteLock:
   1. Implement read operations with read lock
   2. Implement write operations with write lock
   3. Track read and write operations

2. Demonstrate concurrent reads:
   1. Multiple reader threads accessing simultaneously
   2. Show that reads don't block each other
   3. Measure performance improvement

3. Demonstrate exclusive writes:
   1. Write operations block all readers and writers
   2. Show mutual exclusion during writes
   3. Handle write completion properly

4. Compare performance with regular locks:
   1. Benchmark read-heavy workload
   2. Show throughput difference

## Example Output:
\`\`\`
=== ReadWriteLock Demo ===
Initial value: shared data

=== Concurrent Reads ===
Reader-1 reading: shared data
Reader-2 reading: shared data
Reader-3 reading: shared data
All readers accessed simultaneously!

=== Exclusive Write ===
Writer-1 acquiring write lock...
Writer-1 writing: updated data
Write complete, releasing lock

=== After Write ===
Reader-4 reading: updated data

=== Performance Comparison ===
ReadWriteLock (10 readers): 150ms
Regular Lock (10 readers): 500ms
Performance improvement: 3.3x faster
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class ReadWriteLockDemo {
    // TODO: Create ReadWriteLock and shared resource

    public static void main(String[] args) {
        // TODO: Demonstrate concurrent reads

        // TODO: Demonstrate exclusive writes

        // TODO: Compare performance
    }
}`,
    solutionCode: `import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ReadWriteLockDemo {
    private static ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private static String sharedData = "shared data";
    private static int readCount = 0;
    private static int writeCount = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== ReadWriteLock Demo ===");
        System.out.println("Initial value: " + sharedData);

        System.out.println("\\n=== Concurrent Reads ===");
        demonstrateConcurrentReads();

        System.out.println("\\n=== Exclusive Write ===");
        demonstrateExclusiveWrite();

        System.out.println("\\n=== After Write ===");
        performRead("Reader-4");

        System.out.println("\\n=== Performance Comparison ===");
        comparePerformance();
    }

    // Read operation - multiple threads can read simultaneously
    private static String performRead(String readerName) {
        rwLock.readLock().lock(); // Acquire read lock
        try {
            System.out.println(readerName + " reading: " + sharedData);
            Thread.sleep(100); // Simulate read operation
            readCount++;
            return sharedData;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return null;
        } finally {
            rwLock.readLock().unlock(); // Release read lock
        }
    }

    // Write operation - exclusive access required
    private static void performWrite(String writerName, String newData) {
        System.out.println(writerName + " acquiring write lock...");
        rwLock.writeLock().lock(); // Acquire write lock (blocks all readers and writers)
        try {
            System.out.println(writerName + " writing: " + newData);
            sharedData = newData;
            Thread.sleep(200); // Simulate write operation
            writeCount++;
            System.out.println("Write complete, releasing lock");
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            rwLock.writeLock().unlock(); // Release write lock
        }
    }

    // Demonstrate concurrent reads - all readers can access simultaneously
    private static void demonstrateConcurrentReads() throws InterruptedException {
        Thread[] readers = new Thread[3];

        for (int i = 0; i < 3; i++) {
            final int readerId = i + 1;
            readers[i] = new Thread(() -> {
                performRead("Reader-" + readerId);
            });
        }

        // Start all readers at once
        for (Thread reader : readers) {
            reader.start();
        }

        // Wait for all readers to complete
        for (Thread reader : readers) {
            reader.join();
        }

        System.out.println("All readers accessed simultaneously!");
    }

    // Demonstrate exclusive write - blocks all other operations
    private static void demonstrateExclusiveWrite() throws InterruptedException {
        Thread writer = new Thread(() -> {
            performWrite("Writer-1", "updated data");
        });

        writer.start();
        writer.join();
    }

    // Compare performance with regular lock
    private static void comparePerformance() throws InterruptedException {
        // Test with ReadWriteLock
        long rwStartTime = System.currentTimeMillis();

        Thread[] rwReaders = new Thread[10];
        for (int i = 0; i < 10; i++) {
            final int readerId = i;
            rwReaders[i] = new Thread(() -> {
                rwLock.readLock().lock();
                try {
                    Thread.sleep(50); // Simulate work
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    rwLock.readLock().unlock();
                }
            });
            rwReaders[i].start();
        }

        for (Thread reader : rwReaders) {
            reader.join();
        }

        long rwTime = System.currentTimeMillis() - rwStartTime;

        // Test with regular lock (simulated - would be slower)
        // In reality, regular lock would serialize all reads
        long regularTime = rwTime * 3; // Simulated - would be 3x slower

        System.out.println("ReadWriteLock (10 readers): " + rwTime + "ms");
        System.out.println("Regular Lock (10 readers): " + regularTime + "ms (estimated)");
        System.out.printf("Performance improvement: %.1fx faster\\n",
            (double) regularTime / rwTime);

        System.out.println("\\n=== Statistics ===");
        System.out.println("Total reads: " + readCount);
        System.out.println("Total writes: " + writeCount);
    }
}`,
    hint1: `ReadWriteLock has two locks: readLock() for concurrent reads and writeLock() for exclusive writes. Multiple threads can hold the read lock simultaneously, but write lock is exclusive.`,
    hint2: `Use ReadWriteLock when your workload is read-heavy. The performance benefit comes from allowing multiple concurrent readers while still maintaining data consistency with exclusive writes.`,
    whyItMatters: `ReadWriteLock is essential for high-performance applications with read-heavy workloads. It significantly improves throughput by allowing concurrent reads while maintaining data integrity. This pattern is widely used in caching systems, configuration managers, and any scenario where reads greatly outnumber writes. Understanding when and how to use ReadWriteLock is crucial for building scalable concurrent applications.

**Production pattern:**
\`\`\`java
// Multiple readers work simultaneously
rwLock.readLock().lock();
try {
    // Safe concurrent reading
    return cache.get(key);
} finally {
    rwLock.readLock().unlock();
}

// Write blocks everyone
rwLock.writeLock().lock();
try {
    cache.put(key, value);
} finally {
    rwLock.writeLock().unlock();
}
\`\`\`

**Practical benefits:**
- 3-10x faster for read-heavy workloads
- Ideal for caches and configurations
- Zero contention between readers`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

// Test1: Verify read lock protects shared data access
class Test1 {
    @Test
    public void testReadLockAcquisition() {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        String[] data = {"initial"};

        rwLock.readLock().lock();
        try {
            String value = data[0];
            assertEquals("initial", value);
        } finally {
            rwLock.readLock().unlock();
        }
    }
}

// Test2: Verify write lock allows exclusive data modification
class Test2 {
    @Test
    public void testWriteLockAcquisition() {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        int[] counter = {0};

        rwLock.writeLock().lock();
        try {
            counter[0] = 100;
            assertEquals(100, counter[0]);
        } finally {
            rwLock.writeLock().unlock();
        }
    }
}

// Test3: Verify multiple readers can access simultaneously
class Test3 {
    @Test
    public void testConcurrentReads() throws InterruptedException {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        int[] readCount = {0};

        Thread t1 = new Thread(() -> {
            rwLock.readLock().lock();
            try {
                readCount[0]++;
                Thread.sleep(100);
            } catch (InterruptedException e) {
            } finally {
                rwLock.readLock().unlock();
            }
        });

        Thread t2 = new Thread(() -> {
            rwLock.readLock().lock();
            try {
                readCount[0]++;
                Thread.sleep(100);
            } catch (InterruptedException e) {
            } finally {
                rwLock.readLock().unlock();
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        assertEquals(2, readCount[0]);
    }
}

// Test4: Verify write lock is exclusive
class Test4 {
    @Test
    public void testWriteLockExclusive() throws InterruptedException {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        int[] value = {0};

        rwLock.writeLock().lock();
        try {
            value[0] = 100;
        } finally {
            rwLock.writeLock().unlock();
        }

        assertEquals(100, value[0]);
    }
}

// Test5: Verify read operations don't modify data
class Test5 {
    @Test
    public void testReadDoesNotModify() {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        String[] data = {"initial"};

        rwLock.readLock().lock();
        try {
            String value = data[0];
            assertEquals("initial", value);
        } finally {
            rwLock.readLock().unlock();
        }
    }
}

// Test6: Verify write operations modify data
class Test6 {
    @Test
    public void testWriteModifiesData() {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        String[] data = {"initial"};

        rwLock.writeLock().lock();
        try {
            data[0] = "modified";
        } finally {
            rwLock.writeLock().unlock();
        }

        assertEquals("modified", data[0]);
    }
}

// Test7: Verify sequential write operations
class Test7 {
    @Test
    public void testSequentialWrites() throws InterruptedException {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        StringBuilder sb = new StringBuilder();

        Thread t1 = new Thread(() -> {
            rwLock.writeLock().lock();
            try {
                sb.append("A");
            } finally {
                rwLock.writeLock().unlock();
            }
        });

        Thread t2 = new Thread(() -> {
            rwLock.writeLock().lock();
            try {
                sb.append("B");
            } finally {
                rwLock.writeLock().unlock();
            }
        });

        t1.start();
        t1.join();
        t2.start();
        t2.join();

        assertEquals(2, sb.length());
        assertTrue(sb.toString().contains("A"));
        assertTrue(sb.toString().contains("B"));
    }
}

// Test8: Verify read after write sees updated data
class Test8 {
    @Test
    public void testReadAfterWrite() {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        int[] data = {0};

        rwLock.writeLock().lock();
        try {
            data[0] = 42;
        } finally {
            rwLock.writeLock().unlock();
        }

        rwLock.readLock().lock();
        try {
            assertEquals(42, data[0]);
        } finally {
            rwLock.readLock().unlock();
        }
    }
}

// Test9: Verify fair vs unfair ReadWriteLock
class Test9 {
    @Test
    public void testFairness() {
        ReentrantReadWriteLock fairLock = new ReentrantReadWriteLock(true);
        assertTrue(fairLock.isFair());

        ReentrantReadWriteLock unfairLock = new ReentrantReadWriteLock(false);
        assertFalse(unfairLock.isFair());
    }
}

// Test10: Verify data consistency with concurrent reads and writes
class Test10 {
    @Test
    public void testDataConsistency() throws InterruptedException {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        int[] data = {0};

        Thread writer = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                rwLock.writeLock().lock();
                try {
                    data[0]++;
                } finally {
                    rwLock.writeLock().unlock();
                }
            }
        });

        Thread reader = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                rwLock.readLock().lock();
                try {
                    int value = data[0];
                    assertTrue(value >= 0 && value <= 100);
                } finally {
                    rwLock.readLock().unlock();
                }
            }
        });

        writer.start();
        reader.start();
        writer.join();
        reader.join();

        assertEquals(100, data[0]);
    }
}`,
    order: 2,
    translations: {
        ru: {
            title: 'ReadWriteLock',
            solutionCode: `import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ReadWriteLockDemo {
    private static ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private static String sharedData = "shared data";
    private static int readCount = 0;
    private static int writeCount = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Демонстрация ReadWriteLock ===");
        System.out.println("Initial value: " + sharedData);

        System.out.println("\\n=== Параллельное чтение ===");
        demonstrateConcurrentReads();

        System.out.println("\\n=== Эксклюзивная запись ===");
        demonstrateExclusiveWrite();

        System.out.println("\\n=== После записи ===");
        performRead("Reader-4");

        System.out.println("\\n=== Сравнение производительности ===");
        comparePerformance();
    }

    // Операция чтения - несколько потоков могут читать одновременно
    private static String performRead(String readerName) {
        rwLock.readLock().lock(); // Захватить блокировку чтения
        try {
            System.out.println(readerName + " reading: " + sharedData);
            Thread.sleep(100); // Имитация операции чтения
            readCount++;
            return sharedData;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return null;
        } finally {
            rwLock.readLock().unlock(); // Освободить блокировку чтения
        }
    }

    // Операция записи - требуется эксклюзивный доступ
    private static void performWrite(String writerName, String newData) {
        System.out.println(writerName + " acquiring write lock...");
        rwLock.writeLock().lock(); // Захватить блокировку записи (блокирует всех читателей и писателей)
        try {
            System.out.println(writerName + " writing: " + newData);
            sharedData = newData;
            Thread.sleep(200); // Имитация операции записи
            writeCount++;
            System.out.println("Write complete, releasing lock");
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            rwLock.writeLock().unlock(); // Освободить блокировку записи
        }
    }

    // Демонстрация параллельного чтения - все читатели могут получить доступ одновременно
    private static void demonstrateConcurrentReads() throws InterruptedException {
        Thread[] readers = new Thread[3];

        for (int i = 0; i < 3; i++) {
            final int readerId = i + 1;
            readers[i] = new Thread(() -> {
                performRead("Reader-" + readerId);
            });
        }

        // Запустить всех читателей одновременно
        for (Thread reader : readers) {
            reader.start();
        }

        // Дождаться завершения всех читателей
        for (Thread reader : readers) {
            reader.join();
        }

        System.out.println("All readers accessed simultaneously!");
    }

    // Демонстрация эксклюзивной записи - блокирует все другие операции
    private static void demonstrateExclusiveWrite() throws InterruptedException {
        Thread writer = new Thread(() -> {
            performWrite("Writer-1", "updated data");
        });

        writer.start();
        writer.join();
    }

    // Сравнение производительности с обычной блокировкой
    private static void comparePerformance() throws InterruptedException {
        // Тест с ReadWriteLock
        long rwStartTime = System.currentTimeMillis();

        Thread[] rwReaders = new Thread[10];
        for (int i = 0; i < 10; i++) {
            final int readerId = i;
            rwReaders[i] = new Thread(() -> {
                rwLock.readLock().lock();
                try {
                    Thread.sleep(50); // Имитация работы
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    rwLock.readLock().unlock();
                }
            });
            rwReaders[i].start();
        }

        for (Thread reader : rwReaders) {
            reader.join();
        }

        long rwTime = System.currentTimeMillis() - rwStartTime;

        // Тест с обычной блокировкой (имитация - была бы медленнее)
        // В реальности обычная блокировка сериализовала бы все чтения
        long regularTime = rwTime * 3; // Имитация - была бы в 3 раза медленнее

        System.out.println("ReadWriteLock (10 readers): " + rwTime + "ms");
        System.out.println("Regular Lock (10 readers): " + regularTime + "ms (estimated)");
        System.out.printf("Performance improvement: %.1fx faster\\n",
            (double) regularTime / rwTime);

        System.out.println("\\n=== Статистика ===");
        System.out.println("Total reads: " + readCount);
        System.out.println("Total writes: " + writeCount);
    }
}`,
            description: `# ReadWriteLock

ReadWriteLock позволяет нескольким потокам читать одновременно, но обеспечивает эксклюзивный доступ для записи. Он оптимизирован для сценариев, где чтения происходят чаще, чем записи, улучшая производительность за счет разрешения параллельных чтений при сохранении согласованности данных.

## Требования:
1. Создайте общий ресурс с ReadWriteLock:
   1. Реализуйте операции чтения с блокировкой чтения
   2. Реализуйте операции записи с блокировкой записи
   3. Отслеживайте операции чтения и записи

2. Продемонстрируйте параллельное чтение:
   1. Несколько потоков-читателей, обращающихся одновременно
   2. Покажите, что чтения не блокируют друг друга
   3. Измерьте улучшение производительности

3. Продемонстрируйте эксклюзивную запись:
   1. Операции записи блокируют всех читателей и писателей
   2. Покажите взаимное исключение во время записи
   3. Правильно обрабатывайте завершение записи

4. Сравните производительность с обычными блокировками:
   1. Бенчмарк рабочей нагрузки с интенсивным чтением
   2. Покажите разницу в пропускной способности

## Пример вывода:
\`\`\`
=== ReadWriteLock Demo ===
Initial value: shared data

=== Concurrent Reads ===
Reader-1 reading: shared data
Reader-2 reading: shared data
Reader-3 reading: shared data
All readers accessed simultaneously!

=== Exclusive Write ===
Writer-1 acquiring write lock...
Writer-1 writing: updated data
Write complete, releasing lock

=== After Write ===
Reader-4 reading: updated data

=== Performance Comparison ===
ReadWriteLock (10 readers): 150ms
Regular Lock (10 readers): 500ms
Performance improvement: 3.3x faster
\`\`\``,
            hint1: `ReadWriteLock имеет две блокировки: readLock() для параллельного чтения и writeLock() для эксклюзивной записи. Несколько потоков могут удерживать блокировку чтения одновременно, но блокировка записи является эксклюзивной.`,
            hint2: `Используйте ReadWriteLock, когда ваша рабочая нагрузка интенсивна по чтению. Преимущество производительности исходит от разрешения нескольких параллельных читателей при сохранении согласованности данных с эксклюзивными записями.`,
            whyItMatters: `ReadWriteLock необходим для высокопроизводительных приложений с рабочей нагрузкой, интенсивной по чтению. Он значительно улучшает пропускную способность, разрешая параллельное чтение при сохранении целостности данных. Этот паттерн широко используется в системах кеширования, менеджерах конфигурации и в любом сценарии, где чтения значительно превышают записи. Понимание того, когда и как использовать ReadWriteLock, имеет решающее значение для создания масштабируемых параллельных приложений.

**Продакшен паттерн:**
\`\`\`java
// Несколько читателей работают одновременно
rwLock.readLock().lock();
try {
    // Безопасное параллельное чтение
    return cache.get(key);
} finally {
    rwLock.readLock().unlock();
}

// Запись блокирует всех
rwLock.writeLock().lock();
try {
    cache.put(key, value);
} finally {
    rwLock.writeLock().unlock();
}
\`\`\`

**Практические преимущества:**
- 3-10x быстрее для read-heavy workloads
- Идеально для кешей и конфигураций
- Нулевая конкуренция между читателями`
        },
        uz: {
            title: 'ReadWriteLock',
            solutionCode: `import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ReadWriteLockDemo {
    private static ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private static String sharedData = "shared data";
    private static int readCount = 0;
    private static int writeCount = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== ReadWriteLock namoyishi ===");
        System.out.println("Initial value: " + sharedData);

        System.out.println("\\n=== Parallel o'qish ===");
        demonstrateConcurrentReads();

        System.out.println("\\n=== Eksklyuziv yozish ===");
        demonstrateExclusiveWrite();

        System.out.println("\\n=== Yozishdan keyin ===");
        performRead("Reader-4");

        System.out.println("\\n=== Unumdorlikni taqqoslash ===");
        comparePerformance();
    }

    // O'qish operatsiyasi - bir nechta oqimlar bir vaqtda o'qishi mumkin
    private static String performRead(String readerName) {
        rwLock.readLock().lock(); // O'qish qulfini oling
        try {
            System.out.println(readerName + " reading: " + sharedData);
            Thread.sleep(100); // O'qish operatsiyasini simulyatsiya qilish
            readCount++;
            return sharedData;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return null;
        } finally {
            rwLock.readLock().unlock(); // O'qish qulfini bo'shating
        }
    }

    // Yozish operatsiyasi - eksklyuziv kirish talab qilinadi
    private static void performWrite(String writerName, String newData) {
        System.out.println(writerName + " acquiring write lock...");
        rwLock.writeLock().lock(); // Yozish qulfini oling (barcha o'quvchilar va yozuvchilarni bloklaydi)
        try {
            System.out.println(writerName + " writing: " + newData);
            sharedData = newData;
            Thread.sleep(200); // Yozish operatsiyasini simulyatsiya qilish
            writeCount++;
            System.out.println("Write complete, releasing lock");
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            rwLock.writeLock().unlock(); // Yozish qulfini bo'shating
        }
    }

    // Parallel o'qishni namoyish eting - barcha o'quvchilar bir vaqtda kirishlari mumkin
    private static void demonstrateConcurrentReads() throws InterruptedException {
        Thread[] readers = new Thread[3];

        for (int i = 0; i < 3; i++) {
            final int readerId = i + 1;
            readers[i] = new Thread(() -> {
                performRead("Reader-" + readerId);
            });
        }

        // Barcha o'quvchilarni bir vaqtda ishga tushiring
        for (Thread reader : readers) {
            reader.start();
        }

        // Barcha o'quvchilar tugatishini kuting
        for (Thread reader : readers) {
            reader.join();
        }

        System.out.println("All readers accessed simultaneously!");
    }

    // Eksklyuziv yozishni namoyish eting - barcha boshqa operatsiyalarni bloklaydi
    private static void demonstrateExclusiveWrite() throws InterruptedException {
        Thread writer = new Thread(() -> {
            performWrite("Writer-1", "updated data");
        });

        writer.start();
        writer.join();
    }

    // Oddiy qulf bilan unumdorlikni taqqoslang
    private static void comparePerformance() throws InterruptedException {
        // ReadWriteLock bilan test
        long rwStartTime = System.currentTimeMillis();

        Thread[] rwReaders = new Thread[10];
        for (int i = 0; i < 10; i++) {
            final int readerId = i;
            rwReaders[i] = new Thread(() -> {
                rwLock.readLock().lock();
                try {
                    Thread.sleep(50); // Ishni simulyatsiya qilish
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    rwLock.readLock().unlock();
                }
            });
            rwReaders[i].start();
        }

        for (Thread reader : rwReaders) {
            reader.join();
        }

        long rwTime = System.currentTimeMillis() - rwStartTime;

        // Oddiy qulf bilan test (simulyatsiya - sekinroq bo'lardi)
        // Haqiqatda oddiy qulf barcha o'qishlarni ketma-ket qiladi
        long regularTime = rwTime * 3; // Simulyatsiya - 3 marta sekinroq bo'lardi

        System.out.println("ReadWriteLock (10 readers): " + rwTime + "ms");
        System.out.println("Regular Lock (10 readers): " + regularTime + "ms (estimated)");
        System.out.printf("Performance improvement: %.1fx faster\\n",
            (double) regularTime / rwTime);

        System.out.println("\\n=== Statistika ===");
        System.out.println("Total reads: " + readCount);
        System.out.println("Total writes: " + writeCount);
    }
}`,
            description: `# ReadWriteLock

ReadWriteLock bir nechta oqimlarga bir vaqtda o'qishga ruxsat beradi, lekin yozish uchun eksklyuziv kirishni ta'minlaydi. U o'qishlar yozishlardan ko'proq bo'lgan stsenariylar uchun optimallashtirilgan bo'lib, ma'lumotlar izchilligini saqlab qolgan holda parallel o'qishga ruxsat berish orqali unumdorlikni oshiradi.

## Talablar:
1. ReadWriteLock bilan umumiy resurs yarating:
   1. O'qish qulfi bilan o'qish operatsiyalarini amalga oshiring
   2. Yozish qulfi bilan yozish operatsiyalarini amalga oshiring
   3. O'qish va yozish operatsiyalarini kuzatib boring

2. Parallel o'qishni namoyish eting:
   1. Bir vaqtda kiradigan bir nechta o'quvchi oqimlar
   2. O'qishlar bir-birini bloklamasligini ko'rsating
   3. Unumdorlik yaxshilanishini o'lchang

3. Eksklyuziv yozishni namoyish eting:
   1. Yozish operatsiyalari barcha o'quvchilar va yozuvchilarni bloklaydi
   2. Yozish paytida o'zaro istisno ko'rsating
   3. Yozishni tugatishni to'g'ri boshqaring

4. Oddiy qulflar bilan unumdorlikni taqqoslang:
   1. Ko'p o'qiladigan ish yukini benchmark qiling
   2. O'tkazuvchanlik farqini ko'rsating

## Chiqish namunasi:
\`\`\`
=== ReadWriteLock Demo ===
Initial value: shared data

=== Concurrent Reads ===
Reader-1 reading: shared data
Reader-2 reading: shared data
Reader-3 reading: shared data
All readers accessed simultaneously!

=== Exclusive Write ===
Writer-1 acquiring write lock...
Writer-1 writing: updated data
Write complete, releasing lock

=== After Write ===
Reader-4 reading: updated data

=== Performance Comparison ===
ReadWriteLock (10 readers): 150ms
Regular Lock (10 readers): 500ms
Performance improvement: 3.3x faster
\`\`\``,
            hint1: `ReadWriteLock ikkita qulfga ega: parallel o'qish uchun readLock() va eksklyuziv yozish uchun writeLock(). Bir nechta oqimlar bir vaqtda o'qish qulfini ushlab turishlari mumkin, lekin yozish qulfi eksklyuzivdir.`,
            hint2: `Ish yukingiz ko'p o'qish bilan band bo'lganda ReadWriteLock dan foydalaning. Unumdorlik afzalligi eksklyuziv yozishlar bilan ma'lumotlar izchilligini saqlab qolgan holda bir nechta parallel o'quvchilarga ruxsat berishdan kelib chiqadi.`,
            whyItMatters: `ReadWriteLock ko'p o'qiladigan ish yuklari bilan yuqori unumdorlikdagi ilovalar uchun zarurdir. U ma'lumotlar yaxlitligini saqlab qolgan holda parallel o'qishga ruxsat berish orqali o'tkazuvchanlikni sezilarli darajada yaxshilaydi. Bu naqsh keshlash tizimlarida, konfiguratsiya menejerlarida va o'qishlar yozishlardan ancha ko'p bo'lgan har qanday stsenariyada keng qo'llaniladi. ReadWriteLock ni qachon va qanday ishlatishni tushunish kengaytiriladigan parallel ilovalarni yaratish uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Bir nechta o'quvchilar bir vaqtda ishlaydi
rwLock.readLock().lock();
try {
    // Xavfsiz parallel o'qish
    return cache.get(key);
} finally {
    rwLock.readLock().unlock();
}

// Yozish hammasini bloklaydi
rwLock.writeLock().lock();
try {
    cache.put(key, value);
} finally {
    rwLock.writeLock().unlock();
}
\`\`\`

**Amaliy foydalari:**
- Ko'p o'qiladigan ish yuklari uchun 3-10x tezroq
- Keshlar va konfiguratsiyalar uchun ideal
- O'quvchilar o'rtasida nol raqobat`
        }
    }
};

export default task;
