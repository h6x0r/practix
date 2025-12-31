import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-virtual-basics',
    title: 'Virtual Thread Basics',
    difficulty: 'easy',
    tags: ['java', 'virtual-threads', 'concurrency', 'java21', 'project-loom'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Virtual Thread Basics

Virtual threads are lightweight threads that dramatically reduce the effort of writing, maintaining, and observing high-throughput concurrent applications. Unlike platform threads, virtual threads are cheap to create and block, enabling the use of thread-per-request style programming.

## Requirements:
1. Create virtual threads using different methods:
   1. Thread.startVirtualThread()
   2. Thread.ofVirtual().start()
   3. Thread.ofVirtual().unstarted()

2. Compare creation and execution of virtual vs platform threads:
   1. Create 1000 virtual threads
   2. Create platform threads (for comparison)
   3. Measure creation time and memory usage

3. Demonstrate virtual thread properties:
   1. Check if thread is virtual
   2. Get thread name
   3. Join threads and collect results

4. Show simple concurrent operations with virtual threads

## Example Output:
\`\`\`
=== Virtual Thread Creation ===
Started virtual thread: VirtualThread[#21]/runnable@ForkJoinPool-1-worker-1
Thread is virtual: true
Task completed in virtual thread: Result from task

=== Creating 1000 Virtual Threads ===
Created 1000 virtual threads in 15ms
All virtual threads completed: 1000 tasks done

=== Virtual vs Platform Thread ===
Platform thread: Thread[#22,Thread-0,5,main]
Platform thread is virtual: false
Virtual thread: VirtualThread[#23]/runnable@ForkJoinPool-1-worker-2
Virtual thread is virtual: true
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class VirtualBasics {
    public static void main(String[] args) throws InterruptedException {
        // TODO: Create and start a virtual thread using startVirtualThread()

        // TODO: Create a virtual thread using ofVirtual().start()

        // TODO: Create an unstarted virtual thread and start it manually

        // TODO: Create 1000 virtual threads and measure time

        // TODO: Compare with platform thread
    }
}`,
    solutionCode: `import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class VirtualBasics {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Virtual Thread Creation ===");

        // Method 1: Thread.startVirtualThread() - simplest way
        Thread vThread1 = Thread.startVirtualThread(() -> {
            System.out.println("Started virtual thread: " + Thread.currentThread());
            System.out.println("Thread is virtual: " + Thread.currentThread().isVirtual());
        });
        vThread1.join();

        // Method 2: Thread.ofVirtual().start() - with more control
        Thread vThread2 = Thread.ofVirtual()
            .name("my-virtual-thread")
            .start(() -> {
                System.out.println("Task completed in virtual thread: Result from task");
            });
        vThread2.join();

        // Method 3: Thread.ofVirtual().unstarted() - create without starting
        Thread vThread3 = Thread.ofVirtual()
            .name("unstarted-virtual")
            .unstarted(() -> {
                System.out.println("Manually started virtual thread");
            });
        // Start it manually when needed
        vThread3.start();
        vThread3.join();

        System.out.println("\\n=== Creating 1000 Virtual Threads ===");

        // Create many virtual threads - this is cheap!
        long startTime = System.currentTimeMillis();
        AtomicInteger counter = new AtomicInteger(0);

        Thread[] virtualThreads = IntStream.range(0, 1000)
            .mapToObj(i -> Thread.ofVirtual().unstarted(() -> {
                try {
                    // Simulate some work
                    Thread.sleep(Duration.ofMillis(10));
                    counter.incrementAndGet();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }))
            .toArray(Thread[]::new);

        // Start all threads
        for (Thread t : virtualThreads) {
            t.start();
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Created 1000 virtual threads in " + creationTime + "ms");

        // Wait for all to complete
        for (Thread t : virtualThreads) {
            t.join();
        }

        System.out.println("All virtual threads completed: " + counter.get() + " tasks done");

        System.out.println("\\n=== Virtual vs Platform Thread ===");

        // Platform thread (traditional)
        Thread platformThread = Thread.ofPlatform()
            .name("platform-thread")
            .start(() -> {
                System.out.println("Platform thread: " + Thread.currentThread());
                System.out.println("Platform thread is virtual: " +
                    Thread.currentThread().isVirtual());
            });
        platformThread.join();

        // Virtual thread
        Thread virtualThread = Thread.ofVirtual()
            .name("virtual-thread")
            .start(() -> {
                System.out.println("Virtual thread: " + Thread.currentThread());
                System.out.println("Virtual thread is virtual: " +
                    Thread.currentThread().isVirtual());
            });
        virtualThread.join();
    }
}`,
    hint1: `Use Thread.startVirtualThread(Runnable) for the simplest way to create and start a virtual thread. Use Thread.ofVirtual() for more control over thread properties.`,
    hint2: `Virtual threads are checked using Thread.isVirtual() method. You can create thousands of virtual threads without worrying about resource consumption like you would with platform threads.`,
    whyItMatters: `Virtual threads revolutionize concurrent programming in Java by making it practical to use thread-per-request style without the overhead of platform threads. This simplifies code, improves throughput, and makes it easier to write, maintain, and debug concurrent applications. Virtual threads are essential for modern high-performance Java applications.

**Production Pattern:**
\`\`\`java
// Optimal way to create virtual threads in production
public class ProductionVirtualThreads {
    private final ExecutorService executor =
        Executors.newVirtualThreadPerTaskExecutor();

    public CompletableFuture<Result> processRequest(Request req) {
        return CompletableFuture.supplyAsync(() -> {
            // Processing with automatic thread management
            return handleRequest(req);
        }, executor);
    }
}
\`\`\`

**Practical Benefits:**
- Simplifies scaling to millions of concurrent requests
- Eliminates need for complex thread pool configuration
- Reduces memory usage by 1000x compared to platform threads
- Improves code readability through synchronous style`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Test virtual thread creation
class Test1 {
    @Test
    public void test() throws Exception {
        Thread vThread = Thread.ofVirtual().start(() -> {
            System.out.println("Virtual thread running");
        });
        vThread.join();
        assertTrue(vThread.isVirtual());
    }
}

// Test2: Test virtual thread with task
class Test2 {
    @Test
    public void test() throws Exception {
        final int[] result = {0};
        Thread vThread = Thread.ofVirtual().start(() -> {
            result[0] = 42;
        });
        vThread.join();
        assertEquals(42, result[0]);
    }
}

// Test3: Test virtual thread factory
class Test3 {
    @Test
    public void test() throws Exception {
        Thread vThread = Thread.ofVirtual().factory().newThread(() -> {
            System.out.println("Factory thread");
        });
        vThread.start();
        vThread.join();
        assertTrue(vThread.isVirtual());
    }
}

// Test4: Test multiple virtual threads
class Test4 {
    @Test
    public void test() throws Exception {
        Thread t1 = Thread.ofVirtual().start(() -> {});
        Thread t2 = Thread.ofVirtual().start(() -> {});
        t1.join();
        t2.join();
        assertTrue(t1.isVirtual() && t2.isVirtual());
    }
}

// Test5: Test virtual thread with delay
class Test5 {
    @Test
    public void test() throws Exception {
        long start = System.currentTimeMillis();
        Thread vThread = Thread.ofVirtual().start(() -> {
            try { Thread.sleep(100); } catch (InterruptedException e) {}
        });
        vThread.join();
        assertTrue(System.currentTimeMillis() - start >= 100);
    }
}

// Test6: Test virtual thread name
class Test6 {
    @Test
    public void test() throws Exception {
        Thread vThread = Thread.ofVirtual().name("test-thread").start(() -> {});
        vThread.join();
        assertEquals("test-thread", vThread.getName());
    }
}

// Test7: Test startVirtualThread helper
class Test7 {
    @Test
    public void test() throws Exception {
        final boolean[] executed = {false};
        Thread vThread = Thread.startVirtualThread(() -> {
            executed[0] = true;
        });
        vThread.join();
        assertTrue(executed[0]);
    }
}

// Test8: Test virtual thread state
class Test8 {
    @Test
    public void test() throws Exception {
        Thread vThread = Thread.ofVirtual().start(() -> {
            try { Thread.sleep(50); } catch (InterruptedException e) {}
        });
        Thread.sleep(10);
        assertTrue(vThread.isAlive());
        vThread.join();
        assertFalse(vThread.isAlive());
    }
}

// Test9: Test unstarted virtual thread
class Test9 {
    @Test
    public void test() {
        Thread vThread = Thread.ofVirtual().unstarted(() -> {});
        assertFalse(vThread.isAlive());
        assertTrue(vThread.isVirtual());
    }
}

// Test10: Test virtual thread completion
class Test10 {
    @Test
    public void test() throws Exception {
        final String[] result = {null};
        Thread vThread = Thread.ofVirtual().start(() -> {
            result[0] = "Completed";
        });
        vThread.join();
        assertEquals("Completed", result[0]);
    }
}
`,
    translations: {
        ru: {
            title: 'Основы виртуальных потоков',
            solutionCode: `import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class VirtualBasics {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Создание виртуальных потоков ===");

        // Метод 1: Thread.startVirtualThread() - самый простой способ
        Thread vThread1 = Thread.startVirtualThread(() -> {
            System.out.println("Started virtual thread: " + Thread.currentThread());
            System.out.println("Thread is virtual: " + Thread.currentThread().isVirtual());
        });
        vThread1.join();

        // Метод 2: Thread.ofVirtual().start() - с большим контролем
        Thread vThread2 = Thread.ofVirtual()
            .name("my-virtual-thread")
            .start(() -> {
                System.out.println("Task completed in virtual thread: Result from task");
            });
        vThread2.join();

        // Метод 3: Thread.ofVirtual().unstarted() - создать без запуска
        Thread vThread3 = Thread.ofVirtual()
            .name("unstarted-virtual")
            .unstarted(() -> {
                System.out.println("Manually started virtual thread");
            });
        // Запустить вручную когда нужно
        vThread3.start();
        vThread3.join();

        System.out.println("\\n=== Создание 1000 виртуальных потоков ===");

        // Создать много виртуальных потоков - это дешево!
        long startTime = System.currentTimeMillis();
        AtomicInteger counter = new AtomicInteger(0);

        Thread[] virtualThreads = IntStream.range(0, 1000)
            .mapToObj(i -> Thread.ofVirtual().unstarted(() -> {
                try {
                    // Имитация работы
                    Thread.sleep(Duration.ofMillis(10));
                    counter.incrementAndGet();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }))
            .toArray(Thread[]::new);

        // Запустить все потоки
        for (Thread t : virtualThreads) {
            t.start();
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Created 1000 virtual threads in " + creationTime + "ms");

        // Дождаться завершения всех
        for (Thread t : virtualThreads) {
            t.join();
        }

        System.out.println("All virtual threads completed: " + counter.get() + " tasks done");

        System.out.println("\\n=== Виртуальный vs Платформенный поток ===");

        // Платформенный поток (традиционный)
        Thread platformThread = Thread.ofPlatform()
            .name("platform-thread")
            .start(() -> {
                System.out.println("Platform thread: " + Thread.currentThread());
                System.out.println("Platform thread is virtual: " +
                    Thread.currentThread().isVirtual());
            });
        platformThread.join();

        // Виртуальный поток
        Thread virtualThread = Thread.ofVirtual()
            .name("virtual-thread")
            .start(() -> {
                System.out.println("Virtual thread: " + Thread.currentThread());
                System.out.println("Virtual thread is virtual: " +
                    Thread.currentThread().isVirtual());
            });
        virtualThread.join();
    }
}`,
            description: `# Основы виртуальных потоков

Виртуальные потоки - это легковесные потоки, которые значительно упрощают написание, поддержку и наблюдение за высокопроизводительными параллельными приложениями. В отличие от платформенных потоков, виртуальные потоки дешевы в создании и блокировке, что позволяет использовать стиль программирования поток-на-запрос.

## Требования:
1. Создайте виртуальные потоки используя разные методы:
   1. Thread.startVirtualThread()
   2. Thread.ofVirtual().start()
   3. Thread.ofVirtual().unstarted()

2. Сравните создание и выполнение виртуальных и платформенных потоков:
   1. Создайте 1000 виртуальных потоков
   2. Создайте платформенные потоки (для сравнения)
   3. Измерьте время создания и использование памяти

3. Продемонстрируйте свойства виртуальных потоков:
   1. Проверьте, является ли поток виртуальным
   2. Получите имя потока
   3. Дождитесь завершения потоков и соберите результаты

4. Покажите простые параллельные операции с виртуальными потоками

## Пример вывода:
\`\`\`
=== Virtual Thread Creation ===
Started virtual thread: VirtualThread[#21]/runnable@ForkJoinPool-1-worker-1
Thread is virtual: true
Task completed in virtual thread: Result from task

=== Creating 1000 Virtual Threads ===
Created 1000 virtual threads in 15ms
All virtual threads completed: 1000 tasks done

=== Virtual vs Platform Thread ===
Platform thread: Thread[#22,Thread-0,5,main]
Platform thread is virtual: false
Virtual thread: VirtualThread[#23]/runnable@ForkJoinPool-1-worker-2
Virtual thread is virtual: true
\`\`\``,
            hint1: `Используйте Thread.startVirtualThread(Runnable) для самого простого способа создания и запуска виртуального потока. Используйте Thread.ofVirtual() для большего контроля над свойствами потока.`,
            hint2: `Виртуальные потоки проверяются методом Thread.isVirtual(). Вы можете создавать тысячи виртуальных потоков без беспокойства о потреблении ресурсов, как с платформенными потоками.`,
            whyItMatters: `Виртуальные потоки революционизируют параллельное программирование в Java, делая практичным использование стиля поток-на-запрос без накладных расходов платформенных потоков. Это упрощает код, улучшает производительность и делает проще написание, поддержку и отладку параллельных приложений. Виртуальные потоки необходимы для современных высокопроизводительных Java-приложений.

**Продакшен паттерн:**
\`\`\`java
// Оптимальный способ создания виртуальных потоков в продакшене
public class ProductionVirtualThreads {
    private final ExecutorService executor =
        Executors.newVirtualThreadPerTaskExecutor();

    public CompletableFuture<Result> processRequest(Request req) {
        return CompletableFuture.supplyAsync(() -> {
            // Обработка с автоматическим управлением потоками
            return handleRequest(req);
        }, executor);
    }
}
\`\`\`

**Практические преимущества:**
- Упрощает масштабирование до миллионов одновременных запросов
- Устраняет необходимость в сложной настройке пулов потоков
- Снижает использование памяти в 1000 раз по сравнению с платформенными потоками
- Улучшает читаемость кода за счет синхронного стиля`
        },
        uz: {
            title: `Virtual oqimlar asoslari`,
            solutionCode: `import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class VirtualBasics {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Virtual oqimlarni yaratish ===");

        // Usul 1: Thread.startVirtualThread() - eng oddiy usul
        Thread vThread1 = Thread.startVirtualThread(() -> {
            System.out.println("Started virtual thread: " + Thread.currentThread());
            System.out.println("Thread is virtual: " + Thread.currentThread().isVirtual());
        });
        vThread1.join();

        // Usul 2: Thread.ofVirtual().start() - ko'proq boshqaruv bilan
        Thread vThread2 = Thread.ofVirtual()
            .name("my-virtual-thread")
            .start(() -> {
                System.out.println("Task completed in virtual thread: Result from task");
            });
        vThread2.join();

        // Usul 3: Thread.ofVirtual().unstarted() - ishga tushirmasdan yaratish
        Thread vThread3 = Thread.ofVirtual()
            .name("unstarted-virtual")
            .unstarted(() -> {
                System.out.println("Manually started virtual thread");
            });
        // Kerak bo'lganda qo'lda ishga tushirish
        vThread3.start();
        vThread3.join();

        System.out.println("\\n=== 1000 ta virtual oqim yaratish ===");

        // Ko'p virtual oqimlar yaratish - bu arzon!
        long startTime = System.currentTimeMillis();
        AtomicInteger counter = new AtomicInteger(0);

        Thread[] virtualThreads = IntStream.range(0, 1000)
            .mapToObj(i -> Thread.ofVirtual().unstarted(() -> {
                try {
                    // Ishni taqlid qilish
                    Thread.sleep(Duration.ofMillis(10));
                    counter.incrementAndGet();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }))
            .toArray(Thread[]::new);

        // Barcha oqimlarni ishga tushirish
        for (Thread t : virtualThreads) {
            t.start();
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Created 1000 virtual threads in " + creationTime + "ms");

        // Hammasining tugashini kutish
        for (Thread t : virtualThreads) {
            t.join();
        }

        System.out.println("All virtual threads completed: " + counter.get() + " tasks done");

        System.out.println("\\n=== Virtual vs Platforma oqimi ===");

        // Platforma oqimi (an'anaviy)
        Thread platformThread = Thread.ofPlatform()
            .name("platform-thread")
            .start(() -> {
                System.out.println("Platform thread: " + Thread.currentThread());
                System.out.println("Platform thread is virtual: " +
                    Thread.currentThread().isVirtual());
            });
        platformThread.join();

        // Virtual oqim
        Thread virtualThread = Thread.ofVirtual()
            .name("virtual-thread")
            .start(() -> {
                System.out.println("Virtual thread: " + Thread.currentThread());
                System.out.println("Virtual thread is virtual: " +
                    Thread.currentThread().isVirtual());
            });
        virtualThread.join();
    }
}`,
            description: `# Virtual oqimlar asoslari

Virtual oqimlar - bu yengil vazndagi oqimlar bo'lib, ular yuqori unumdor parallel dasturlarni yozish, saqlash va kuzatishni sezilarli darajada osonlashtiradi. Platforma oqimlaridan farqli o'laroq, virtual oqimlarni yaratish va bloklash arzon, bu so'rov-uchun-oqim uslubida dasturlashni yoqadi.

## Talablar:
1. Turli usullar yordamida virtual oqimlar yarating:
   1. Thread.startVirtualThread()
   2. Thread.ofVirtual().start()
   3. Thread.ofVirtual().unstarted()

2. Virtual va platforma oqimlarini yaratish va bajarishni solishtiring:
   1. 1000 ta virtual oqim yarating
   2. Platforma oqimlarini yarating (taqqoslash uchun)
   3. Yaratish vaqti va xotira foydalanishini o'lchang

3. Virtual oqim xususiyatlarini namoyish eting:
   1. Oqim virtual ekanligini tekshiring
   2. Oqim nomini oling
   3. Oqimlarni kuting va natijalarni yig'ing

4. Virtual oqimlar bilan oddiy parallel operatsiyalarni ko'rsating

## Chiqish namunasi:
\`\`\`
=== Virtual Thread Creation ===
Started virtual thread: VirtualThread[#21]/runnable@ForkJoinPool-1-worker-1
Thread is virtual: true
Task completed in virtual thread: Result from task

=== Creating 1000 Virtual Threads ===
Created 1000 virtual threads in 15ms
All virtual threads completed: 1000 tasks done

=== Virtual vs Platform Thread ===
Platform thread: Thread[#22,Thread-0,5,main]
Platform thread is virtual: false
Virtual thread: VirtualThread[#23]/runnable@ForkJoinPool-1-worker-2
Virtual thread is virtual: true
\`\`\``,
            hint1: `Virtual oqimni yaratish va ishga tushirishning eng oddiy usuli uchun Thread.startVirtualThread(Runnable) dan foydalaning. Oqim xususiyatlari ustidan ko'proq nazorat uchun Thread.ofVirtual() dan foydalaning.`,
            hint2: `Virtual oqimlar Thread.isVirtual() metodi bilan tekshiriladi. Platforma oqimlari kabi resurs sarfidan qayg'urmasdan minglab virtual oqimlar yaratishingiz mumkin.`,
            whyItMatters: `Virtual oqimlar Java-da parallel dasturlashni inqilob qiladi, platforma oqimlarining qo'shimcha xarajatlari bo'lmagan holda so'rov-uchun-oqim uslubidan foydalanishni amaliy qiladi. Bu kodni soddalashtiradi, unumdorlikni oshiradi va parallel dasturlarni yozish, saqlash va nosozliklarni tuzatishni osonlashtiradi. Virtual oqimlar zamonaviy yuqori unumdor Java dasturlari uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ishlab chiqarishda virtual oqimlarni yaratishning optimal usuli
public class ProductionVirtualThreads {
    private final ExecutorService executor =
        Executors.newVirtualThreadPerTaskExecutor();

    public CompletableFuture<Result> processRequest(Request req) {
        return CompletableFuture.supplyAsync(() -> {
            // Avtomatik oqim boshqaruvi bilan qayta ishlash
            return handleRequest(req);
        }, executor);
    }
}
\`\`\`

**Amaliy foydalari:**
- Millionlab bir vaqtdagi so'rovlargacha miqyoslashni soddalashtiradi
- Oqim hovuzlarini murakkab sozlash zaruriyatini yo'q qiladi
- Platforma oqimlariga nisbatan xotira foydalanishini 1000 marta kamaytiradi
- Sinxron uslub orqali kod o'qilishini yaxshilaydi`
        }
    }
};

export default task;
