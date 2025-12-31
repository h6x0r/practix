import { Task } from '../../../../../../types';

export const task: Task = {
  slug: 'java-threads-synchronized-keyword',
  title: 'Synchronized Keyword',
  difficulty: 'medium',
  tags: ['threads', 'synchronized', 'concurrency', 'thread-safety'],
  estimatedTime: '30m',
  isPremium: false,
  youtubeUrl: '',
  description: `Learn how to use synchronized methods and blocks to prevent race conditions in multi-threaded applications.

**Requirements:**

1. **Implement Synchronized Method**
   1.1. Make the increment() method synchronized to ensure thread-safe increments
   1.2. Understand that synchronized methods lock on the 'this' object

2. **Create Synchronized Block**
   2.1. Implement the decrement() method using a synchronized block
   2.2. Use an explicit lock object for the synchronized block
   2.3. Ensure the lock object is final and private

3. **Create Thread Operations**
   3.1. Create two threads that increment the counter 1000 times each
   3.2. Create two threads that decrement the counter 1000 times each
   3.3. Name each thread appropriately for debugging purposes

4. **Execute and Wait for Threads**
   4.1. Start all four threads concurrently
   4.2. Use join() to wait for all threads to complete
   4.3. Display the final count (should be 0)
   4.4. Optionally, measure and display execution time`,
  initialCode: `class Counter {
    private int count = 0;

    // TODO: Make this method synchronized
    public void increment() {
        count++;
    }

    // TODO: Create a synchronized block for this method
    public void decrement() {
        count--;
    }

    public int getCount() {
        return count;
    }
}

public class SynchronizedKeyword {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();

        // TODO: Create two threads that increment counter 1000 times each


        // TODO: Create two threads that decrement counter 1000 times each


        // TODO: Start all threads and wait for completion


        System.out.println("Final count: " + counter.getCount());
    }
}`,
  solutionCode: `class Counter {
    private int count = 0;
    private final Object lock = new Object();

    // Synchronized method - locks on 'this' object
    public synchronized void increment() {
        count++;
    }

    // Synchronized block - explicit lock object
    public void decrement() {
        synchronized (lock) {
            count--;
        }
    }

    // Synchronized getter for thread-safe reading
    public synchronized int getCount() {
        return count;
    }
}

public class SynchronizedKeyword {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();

        // Create two threads that increment counter 1000 times each
        Thread incrementer1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        }, "Incrementer-1");

        Thread incrementer2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        }, "Incrementer-2");

        // Create two threads that decrement counter 1000 times each
        Thread decrementer1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.decrement();
            }
        }, "Decrementer-1");

        Thread decrementer2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.decrement();
            }
        }, "Decrementer-2");

        // Start all threads
        long startTime = System.currentTimeMillis();
        incrementer1.start();
        incrementer2.start();
        decrementer1.start();
        decrementer2.start();

        // Wait for all threads to complete
        incrementer1.join();
        incrementer2.join();
        decrementer1.join();
        decrementer2.join();

        long endTime = System.currentTimeMillis();

        System.out.println("Final count: " + counter.getCount());
        System.out.println("Expected: 0");
        System.out.println("Time taken: " + (endTime - startTime) + "ms");
    }
}`,
  testCode: `import static org.junit.Assert.*;
import org.junit.Test;

class Counter {
    private int count = 0;
    private final Object lock = new Object();

    public synchronized void increment() {
        count++;
    }

    public void decrement() {
        synchronized (lock) {
            count--;
        }
    }

    public synchronized int getCount() {
        return count;
    }
}

// Test1: Verify synchronized increment method
class Test1 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100; j++) {
                    counter.increment();
                }
            });
        }
        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();
        assertEquals("Counter should be 1000", 1000, counter.getCount());
    }
}

// Test2: Verify synchronized block decrement method
class Test2 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100; j++) {
                    counter.decrement();
                }
            });
        }
        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();
        assertEquals("Counter should be -1000", -1000, counter.getCount());
    }
}

// Test3: Verify increment and decrement balance out
class Test3 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.increment();
        });
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.decrement();
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        assertEquals("Counter should be 0", 0, counter.getCount());
    }
}

// Test4: Verify multiple incrementers work correctly
class Test4 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.increment();
        });
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.increment();
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        assertEquals("Counter should be 2000", 2000, counter.getCount());
    }
}

// Test5: Verify multiple decrementers work correctly
class Test5 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.decrement();
        });
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.decrement();
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        assertEquals("Counter should be -2000", -2000, counter.getCount());
    }
}

// Test6: Verify getCount is thread-safe
class Test6 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread writer = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                counter.increment();
                try { Thread.sleep(1); } catch (InterruptedException e) {}
            }
        });
        Thread reader = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                int count = counter.getCount();
                assertTrue("Count should be non-negative during increments", count >= 0);
                try { Thread.sleep(1); } catch (InterruptedException e) {}
            }
        });
        writer.start();
        reader.start();
        writer.join();
        reader.join();
    }
}

// Test7: Verify mixed operations produce correct result
class Test7 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread[] threads = new Thread[4];
        threads[0] = new Thread(() -> { for (int i = 0; i < 1000; i++) counter.increment(); });
        threads[1] = new Thread(() -> { for (int i = 0; i < 1000; i++) counter.increment(); });
        threads[2] = new Thread(() -> { for (int i = 0; i < 1000; i++) counter.decrement(); });
        threads[3] = new Thread(() -> { for (int i = 0; i < 1000; i++) counter.decrement(); });

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        assertEquals("Counter should be 0 (2000 - 2000)", 0, counter.getCount());
    }
}

// Test8: Verify synchronized prevents race conditions
class Test8 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        int numThreads = 50;
        Thread[] threads = new Thread[numThreads];

        for (int i = 0; i < numThreads; i++) {
            if (i % 2 == 0) {
                threads[i] = new Thread(() -> {
                    for (int j = 0; j < 20; j++) counter.increment();
                });
            } else {
                threads[i] = new Thread(() -> {
                    for (int j = 0; j < 20; j++) counter.decrement();
                });
            }
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        assertEquals("Counter should be 0", 0, counter.getCount());
    }
}

// Test9: Verify thread safety with rapid operations
class Test9 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) counter.increment();
        });
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 5000; i++) counter.decrement();
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        assertEquals("Counter should be 5000 (10000 - 5000)", 5000, counter.getCount());
    }
}

// Test10: Verify no lost updates occur
class Test10 {
    @Test
    public void test() throws Exception {
        Counter counter = new Counter();
        int numOperations = 100;
        Thread[] incrementers = new Thread[5];
        Thread[] decrementers = new Thread[5];

        for (int i = 0; i < 5; i++) {
            incrementers[i] = new Thread(() -> {
                for (int j = 0; j < numOperations; j++) counter.increment();
            });
            decrementers[i] = new Thread(() -> {
                for (int j = 0; j < numOperations; j++) counter.decrement();
            });
        }

        for (Thread t : incrementers) t.start();
        for (Thread t : decrementers) t.start();
        for (Thread t : incrementers) t.join();
        for (Thread t : decrementers) t.join();

        assertEquals("No updates should be lost", 0, counter.getCount());
    }
}`,
  hint1: 'Use synchronized keyword before method declaration for synchronized methods. For example: public synchronized void increment() {}. Use synchronized(object) {} blocks when you need more control. Synchronized methods automatically lock on the object instance (this).',
  hint2: 'Always use the same lock object for related operations to avoid race conditions. Create a private final Object lock = new Object(); and use synchronized(lock) {} for explicit locking. Remember to call join() on all threads to wait for their completion before checking the final result.',
  whyItMatters: `The synchronized keyword is fundamental to writing thread-safe Java applications. Without proper synchronization, multiple threads accessing shared data can lead to race conditions, where the final state depends on unpredictable thread scheduling. This causes bugs that are difficult to reproduce and debug.

Synchronized methods and blocks ensure mutual exclusion - only one thread can execute the synchronized code at a time, preventing data corruption. Understanding when and how to use synchronized is crucial for building reliable concurrent applications, from web servers handling multiple requests to parallel data processing systems.

While synchronized provides safety, it comes with performance trade-offs due to thread blocking. This knowledge forms the foundation for understanding more advanced concurrency tools like ReentrantLock, ReadWriteLock, and modern concurrent collections.`,
  order: 3,
  translations: {
    ru: {
      title: 'Ключевое слово synchronized',
      description: `Узнайте, как использовать синхронизированные методы и блоки для предотвращения гонки данных в многопоточных приложениях.

**Требования:**

1. **Реализация синхронизированного метода**
   1.1. Сделайте метод increment() синхронизированным для обеспечения потокобезопасного увеличения
   1.2. Поймите, что синхронизированные методы блокируют объект 'this'

2. **Создание синхронизированного блока**
   2.1. Реализуйте метод decrement() с использованием синхронизированного блока
   2.2. Используйте явный объект блокировки для синхронизированного блока
   2.3. Убедитесь, что объект блокировки является final и private

3. **Создание потоковых операций**
   3.1. Создайте два потока, которые увеличивают счетчик 1000 раз каждый
   3.2. Создайте два потока, которые уменьшают счетчик 1000 раз каждый
   3.3. Назовите каждый поток соответствующим образом для отладки

4. **Запуск и ожидание потоков**
   4.1. Запустите все четыре потока одновременно
   4.2. Используйте join() для ожидания завершения всех потоков
   4.3. Выведите итоговое значение счетчика (должно быть 0)
   4.4. Опционально измерьте и выведите время выполнения`,
      hint1: 'Используйте ключевое слово synchronized перед объявлением метода для синхронизированных методов. Например: public synchronized void increment() {}. Используйте блоки synchronized(объект) {}, когда нужен больший контроль. Синхронизированные методы автоматически блокируют экземпляр объекта (this).',
      hint2: 'Всегда используйте один и тот же объект блокировки для связанных операций, чтобы избежать гонки данных. Создайте private final Object lock = new Object(); и используйте synchronized(lock) {} для явной блокировки. Не забудьте вызвать join() для всех потоков, чтобы дождаться их завершения перед проверкой итогового результата.',
      whyItMatters: `Ключевое слово synchronized является фундаментальным для написания потокобезопасных Java-приложений. Без надлежащей синхронизации множественные потоки, обращающиеся к общим данным, могут привести к гонке данных, где конечное состояние зависит от непредсказуемого планирования потоков. Это вызывает ошибки, которые трудно воспроизвести и отладить.

Синхронизированные методы и блоки обеспечивают взаимное исключение - только один поток может выполнять синхронизированный код одновременно, предотвращая повреждение данных. Понимание того, когда и как использовать synchronized, критически важно для создания надежных параллельных приложений, от веб-серверов, обрабатывающих множественные запросы, до систем параллельной обработки данных.

Хотя synchronized обеспечивает безопасность, он имеет компромиссы производительности из-за блокировки потоков. Эти знания формируют основу для понимания более продвинутых инструментов параллелизма, таких как ReentrantLock, ReadWriteLock и современные параллельные коллекции.`,
      solutionCode: `class Counter {
    private int count = 0;
    private final Object lock = new Object();

    // Синхронизированный метод - блокировка на объекте 'this'
    public synchronized void increment() {
        count++;
    }

    // Синхронизированный блок - явный объект блокировки
    public void decrement() {
        synchronized (lock) {
            count--;
        }
    }

    // Синхронизированный геттер для безопасного чтения
    public synchronized int getCount() {
        return count;
    }
}

public class SynchronizedKeyword {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();

        // Создаем два потока, которые увеличивают счетчик 1000 раз каждый
        Thread incrementer1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        }, "Incrementer-1");

        Thread incrementer2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        }, "Incrementer-2");

        // Создаем два потока, которые уменьшают счетчик 1000 раз каждый
        Thread decrementer1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.decrement();
            }
        }, "Decrementer-1");

        Thread decrementer2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.decrement();
            }
        }, "Decrementer-2");

        // Запускаем все потоки
        long startTime = System.currentTimeMillis();
        incrementer1.start();
        incrementer2.start();
        decrementer1.start();
        decrementer2.start();

        // Ожидаем завершения всех потоков
        incrementer1.join();
        incrementer2.join();
        decrementer1.join();
        decrementer2.join();

        long endTime = System.currentTimeMillis();

        System.out.println("Итоговый счетчик: " + counter.getCount());
        System.out.println("Ожидаемое значение: 0");
        System.out.println("Затраченное время: " + (endTime - startTime) + "мс");
    }
}`
    },
    uz: {
      title: 'Synchronized kalit so\'zi',
      description: `Ko'p oqimli ilovalarda ma'lumotlar poygasini oldini olish uchun sinxronlashtirilgan metodlar va bloklardan foydalanishni o'rganing.

**Talablar:**

1. **Sinxronlashtirilgan metodini amalga oshirish**
   1.1. increment() metodini oqim-xavfsiz oshirish uchun sinxronlashtirilgan qiling
   1.2. Sinxronlashtirilgan metodlar 'this' obyektini qufllashini tushuning

2. **Sinxronlashtirilgan blok yaratish**
   2.1. decrement() metodini sinxronlashtirilgan blok yordamida amalga oshiring
   2.2. Sinxronlashtirilgan blok uchun aniq qulflash obyektidan foydalaning
   2.3. Qulflash obyekti final va private ekanligiga ishonch hosil qiling

3. **Oqim operatsiyalarini yaratish**
   3.1. Schyotchikni har biri 1000 marta oshiradigan ikkita oqim yarating
   3.2. Schyotchikni har biri 1000 marta kamaytiradigan ikkita oqim yarating
   3.3. Har bir oqimni nosozliklarni tuzatish uchun tegishli nomlang

4. **Oqimlarni ishga tushirish va kutish**
   4.1. Barcha to'rtta oqimni bir vaqtda ishga tushiring
   4.2. Barcha oqimlarning tugashini kutish uchun join() dan foydalaning
   4.3. Yakuniy schyotchik qiymatini chiqaring (0 bo'lishi kerak)
   4.4. Ixtiyoriy ravishda bajarilish vaqtini o'lchang va chiqaring`,
      hint1: 'Sinxronlashtirilgan metodlar uchun metod deklaratsiyasidan oldin synchronized kalit so\'zidan foydalaning. Masalan: public synchronized void increment() {}. Ko\'proq nazorat kerak bo\'lganda synchronized(obyekt) {} bloklaridan foydalaning. Sinxronlashtirilgan metodlar avtomatik ravishda obyekt nusxasini (this) qulflaydi.',
      hint2: 'Ma\'lumotlar poygasini oldini olish uchun bog\'liq operatsiyalar uchun har doim bir xil qulflash obyektidan foydalaning. private final Object lock = new Object(); yarating va aniq qulflash uchun synchronized(lock) {} dan foydalaning. Yakuniy natijani tekshirishdan oldin barcha oqimlarning tugashini kutish uchun join() ni chaqirishni unutmang.',
      whyItMatters: `Synchronized kalit so'zi oqim-xavfsiz Java ilovalarini yozish uchun asosiy hisoblanadi. To'g'ri sinxronizatsiyasiz, umumiy ma'lumotlarga kiruvchi ko'plab oqimlar ma'lumotlar poygasiga olib kelishi mumkin, bu erda yakuniy holat oldindan aytib bo'lmaydigan oqim rejalashtiruviga bog'liq. Bu takrorlash va nosozliklarni tuzatish qiyin bo'lgan xatolarga olib keladi.

Sinxronlashtirilgan metodlar va bloklar o'zaro istisno ta'minlaydi - faqat bitta oqim bir vaqtning o'zida sinxronlashtirilgan kodni bajarishi mumkin, bu ma'lumotlarning buzilishining oldini oladi. synchronized ni qachon va qanday ishlatishni tushunish ishonchli parallel ilovalar yaratish uchun muhim, ko'plab so'rovlarni qayta ishlovchi veb-serverlardan tortib parallel ma'lumotlarni qayta ishlash tizimlarigacha.

synchronized xavfsizlikni ta'minlasa-da, oqimlarni blokirovka qilish tufayli ishlash kompromisslari mavjud. Bu bilim ReentrantLock, ReadWriteLock va zamonaviy parallel kolleksiyalar kabi ilg'or parallelllik vositalarini tushunish uchun asos bo'ladi.`,
      solutionCode: `class Counter {
    private int count = 0;
    private final Object lock = new Object();

    // Sinxronlashtirilgan metod - 'this' obyektida qulflash
    public synchronized void increment() {
        count++;
    }

    // Sinxronlashtirilgan blok - aniq qulflash obyekti
    public void decrement() {
        synchronized (lock) {
            count--;
        }
    }

    // Xavfsiz o'qish uchun sinxronlashtirilgan getter
    public synchronized int getCount() {
        return count;
    }
}

public class SynchronizedKeyword {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();

        // Schyotchikni har biri 1000 marta oshiradigan ikkita oqim yaratamiz
        Thread incrementer1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        }, "Incrementer-1");

        Thread incrementer2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        }, "Incrementer-2");

        // Schyotchikni har biri 1000 marta kamaytiradigan ikkita oqim yaratamiz
        Thread decrementer1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.decrement();
            }
        }, "Decrementer-1");

        Thread decrementer2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.decrement();
            }
        }, "Decrementer-2");

        // Barcha oqimlarni ishga tushiramiz
        long startTime = System.currentTimeMillis();
        incrementer1.start();
        incrementer2.start();
        decrementer1.start();
        decrementer2.start();

        // Barcha oqimlarning tugashini kutamiz
        incrementer1.join();
        incrementer2.join();
        decrementer1.join();
        decrementer2.join();

        long endTime = System.currentTimeMillis();

        System.out.println("Yakuniy schyotchik: " + counter.getCount());
        System.out.println("Kutilgan qiymat: 0");
        System.out.println("Sarflangan vaqt: " + (endTime - startTime) + "ms");
    }
}`
    }
  }
};

export default task;
