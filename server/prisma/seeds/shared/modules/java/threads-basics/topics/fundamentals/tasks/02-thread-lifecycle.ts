import { Task } from '../../../../../../types';

export const task: Task = {
  slug: 'java-threads-thread-lifecycle',
  title: 'Thread Lifecycle',
  difficulty: 'easy',
  tags: ['threads', 'lifecycle', 'sleep', 'join'],
  estimatedTime: '25m',
  isPremium: false,
  youtubeUrl: '',
  description: `Understand thread states and lifecycle methods: start, sleep, join, and interrupt.

**Requirements:**

1. **Create a Worker Thread**
   1.1. Implement a thread that runs for 2 seconds total
   1.2. Print a message every 500ms (4 iterations)
   1.3. Display the thread state during execution

2. **Thread Lifecycle Management**
   2.1. Start the worker thread and observe state transitions
   2.2. Use Thread.sleep() to pause execution between iterations
   2.3. Handle InterruptedException properly

3. **Thread Synchronization**
   3.1. Use join() in the main thread to wait for worker completion
   3.2. Display thread states before start, after start, and after join
   3.3. Ensure main thread finishes only after worker thread completes`,
  initialCode: `public class ThreadLifecycle {
    public static void main(String[] args) {
        // TODO: Create a worker thread that sleeps for 2 seconds
        // and prints a message every 500ms


        // TODO: Start the worker thread


        // TODO: Use join() to wait for worker thread completion


        System.out.println("Main thread finished");
    }
}`,
  solutionCode: `public class ThreadLifecycle {
    public static void main(String[] args) {
        // Create a worker thread that sleeps and prints messages
        Thread workerThread = new Thread(() -> {
            System.out.println("Worker thread started: " + Thread.currentThread().getState());

            for (int i = 1; i <= 4; i++) {
                try {
                    // Sleep for 500 milliseconds
                    Thread.sleep(500);
                    System.out.println("Worker iteration " + i + " - State: " + Thread.currentThread().getState());
                } catch (InterruptedException e) {
                    System.out.println("Worker thread was interrupted");
                    Thread.currentThread().interrupt();
                    return;
                }
            }

            System.out.println("Worker thread finished");
        });

        // Start the worker thread (NEW -> RUNNABLE)
        System.out.println("Worker state before start: " + workerThread.getState());
        workerThread.start();
        System.out.println("Worker state after start: " + workerThread.getState());

        try {
            // Wait for worker thread to complete
            System.out.println("Main thread waiting for worker...");
            workerThread.join();
            System.out.println("Worker state after join: " + workerThread.getState());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Main thread finished");
    }
}`,
  testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify worker thread executes 4 iterations
class Test1 {
    @Test
    public void test() throws Exception {
        final int[] count = {0};
        Thread worker = new Thread(() -> {
            for (int i = 1; i <= 4; i++) {
                try {
                    Thread.sleep(50);
                    count[0]++;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        });
        worker.start();
        worker.join();
        assertEquals("Worker should complete 4 iterations", 4, count[0]);
    }
}

// Test2: Verify Thread.sleep() pauses execution
class Test2 {
    @Test
    public void test() throws Exception {
        long start = System.currentTimeMillis();
        Thread worker = new Thread(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {}
        });
        worker.start();
        worker.join();
        long duration = System.currentTimeMillis() - start;
        assertTrue("Thread should sleep for at least 200ms", duration >= 200);
    }
}

// Test3: Verify thread state is NEW before start
class Test3 {
    @Test
    public void test() {
        Thread worker = new Thread(() -> {});
        assertEquals("Thread should be in NEW state before start", Thread.State.NEW, worker.getState());
    }
}

// Test4: Verify thread state is TERMINATED after completion
class Test4 {
    @Test
    public void test() throws Exception {
        Thread worker = new Thread(() -> {});
        worker.start();
        worker.join();
        assertEquals("Thread should be in TERMINATED state after completion", Thread.State.TERMINATED, worker.getState());
    }
}

// Test5: Verify join() waits for thread to complete
class Test5 {
    @Test
    public void test() throws Exception {
        final boolean[] completed = {false};
        Thread worker = new Thread(() -> {
            try {
                Thread.sleep(100);
                completed[0] = true;
            } catch (InterruptedException e) {}
        });
        worker.start();
        assertFalse("Worker should not be completed yet", completed[0]);
        worker.join();
        assertTrue("Worker should be completed after join", completed[0]);
    }
}

// Test6: Verify InterruptedException is properly handled
class Test6 {
    @Test
    public void test() throws Exception {
        final boolean[] interrupted = {false};
        Thread worker = new Thread(() -> {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                interrupted[0] = true;
                Thread.currentThread().interrupt();
            }
        });
        worker.start();
        Thread.sleep(50);
        worker.interrupt();
        worker.join();
        assertTrue("Thread should catch InterruptedException", interrupted[0]);
    }
}

// Test7: Verify thread can print messages during iterations
class Test7 {
    @Test
    public void test() throws Exception {
        final StringBuilder messages = new StringBuilder();
        Thread worker = new Thread(() -> {
            for (int i = 1; i <= 3; i++) {
                try {
                    Thread.sleep(50);
                    messages.append(i);
                } catch (InterruptedException e) {
                    return;
                }
            }
        });
        worker.start();
        worker.join();
        assertEquals("Messages should contain all iterations", "123", messages.toString());
    }
}

// Test8: Verify main thread can wait for multiple workers
class Test8 {
    @Test
    public void test() throws Exception {
        final int[] count = {0};
        Thread w1 = new Thread(() -> {
            try {
                Thread.sleep(100);
                synchronized(count) { count[0]++; }
            } catch (InterruptedException e) {}
        });
        Thread w2 = new Thread(() -> {
            try {
                Thread.sleep(100);
                synchronized(count) { count[0]++; }
            } catch (InterruptedException e) {}
        });

        w1.start();
        w2.start();
        w1.join();
        w2.join();

        assertEquals("Both workers should complete", 2, count[0]);
    }
}

// Test9: Verify thread states during execution
class Test9 {
    @Test
    public void test() throws Exception {
        Thread worker = new Thread(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {}
        });

        assertEquals("Should start in NEW state", Thread.State.NEW, worker.getState());
        worker.start();
        Thread.sleep(50);
        assertTrue("Should be RUNNABLE or TIMED_WAITING",
            worker.getState() == Thread.State.RUNNABLE || worker.getState() == Thread.State.TIMED_WAITING);
        worker.join();
        assertEquals("Should end in TERMINATED state", Thread.State.TERMINATED, worker.getState());
    }
}

// Test10: Verify worker thread prints completion message
class Test10 {
    @Test
    public void test() throws Exception {
        final boolean[] finishMessagePrinted = {false};
        Thread worker = new Thread(() -> {
            for (int i = 1; i <= 2; i++) {
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    return;
                }
            }
            finishMessagePrinted[0] = true;
        });
        worker.start();
        worker.join();
        assertTrue("Worker should print finish message", finishMessagePrinted[0]);
    }
}`,
  hint1: 'Thread states: NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED. Use Thread.sleep(milliseconds) to pause thread execution.',
  hint2: 'Use join() method to wait for another thread to finish. Always handle InterruptedException in try-catch blocks.',
  whyItMatters: `Understanding the thread lifecycle is fundamental to writing robust multithreaded applications in Java. The lifecycle states (NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED) represent critical phases that every thread goes through, and knowing how to manage these transitions is essential for controlling program flow.

The join() method is crucial for coordinating thread execution, ensuring that dependent operations complete in the correct order. Thread.sleep() is a basic but powerful tool for implementing delays, timeouts, and rate limiting in concurrent applications.

Proper handling of InterruptedException demonstrates good exception management practices and prevents resource leaks or unexpected behavior when threads are interrupted. These concepts form the foundation for more advanced concurrency patterns like thread pools, executors, and asynchronous programming.`,
  order: 2,
  translations: {
    ru: {
      title: 'Жизненный цикл потока',
      description: `Поймите состояния потоков и методы жизненного цикла: start, sleep, join и interrupt.

**Требования:**

1. **Создайте Рабочий Поток**
   1.1. Реализуйте поток, который работает в течение 2 секунд
   1.2. Выводите сообщение каждые 500 мс (4 итерации)
   1.3. Отображайте состояние потока во время выполнения

2. **Управление Жизненным Циклом Потока**
   2.1. Запустите рабочий поток и наблюдайте за переходами состояний
   2.2. Используйте Thread.sleep() для приостановки выполнения между итерациями
   2.3. Правильно обрабатывайте InterruptedException

3. **Синхронизация Потоков**
   3.1. Используйте join() в главном потоке для ожидания завершения рабочего потока
   3.2. Отображайте состояния потока до запуска, после запуска и после join
   3.3. Убедитесь, что главный поток завершается только после завершения рабочего потока`,
      hint1: 'Состояния потоков: NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED. Используйте Thread.sleep(миллисекунды) для приостановки выполнения потока.',
      hint2: 'Используйте метод join() для ожидания завершения другого потока. Всегда обрабатывайте InterruptedException в блоках try-catch.',
      whyItMatters: `Понимание жизненного цикла потоков является основополагающим для написания надежных многопоточных приложений на Java. Состояния жизненного цикла (NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED) представляют критические фазы, через которые проходит каждый поток, и знание того, как управлять этими переходами, необходимо для контроля потока выполнения программы.

Метод join() имеет решающее значение для координации выполнения потоков, обеспечивая завершение зависимых операций в правильном порядке. Thread.sleep() - это базовый, но мощный инструмент для реализации задержек, тайм-аутов и ограничения скорости в параллельных приложениях.

Правильная обработка InterruptedException демонстрирует хорошие практики управления исключениями и предотвращает утечки ресурсов или неожиданное поведение при прерывании потоков. Эти концепции формируют основу для более продвинутых паттернов параллелизма, таких как пулы потоков, исполнители и асинхронное программирование.`,
      solutionCode: `public class ThreadLifecycle {
    public static void main(String[] args) {
        // Создаем рабочий поток, который спит и выводит сообщения
        Thread workerThread = new Thread(() -> {
            System.out.println("Рабочий поток запущен: " + Thread.currentThread().getState());

            for (int i = 1; i <= 4; i++) {
                try {
                    // Засыпаем на 500 миллисекунд
                    Thread.sleep(500);
                    System.out.println("Рабочая итерация " + i + " - Состояние: " + Thread.currentThread().getState());
                } catch (InterruptedException e) {
                    System.out.println("Рабочий поток был прерван");
                    Thread.currentThread().interrupt();
                    return;
                }
            }

            System.out.println("Рабочий поток завершен");
        });

        // Запускаем рабочий поток (NEW -> RUNNABLE)
        System.out.println("Состояние рабочего потока до запуска: " + workerThread.getState());
        workerThread.start();
        System.out.println("Состояние рабочего потока после запуска: " + workerThread.getState());

        try {
            // Ожидаем завершения рабочего потока
            System.out.println("Главный поток ожидает завершения рабочего потока...");
            workerThread.join();
            System.out.println("Состояние рабочего потока после join: " + workerThread.getState());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Главный поток завершен");
    }
}`,
    },
    uz: {
      title: 'Oqim hayotiy tsikli',
      description: `Oqim holatlari va hayotiy tsikl metodlarini tushunish: start, sleep, join va interrupt.

**Talablar:**

1. **Ishchi Oqim Yarating**
   1.1. Jami 2 soniya ishlaydigan oqimni amalga oshiring
   1.2. Har 500 msda xabar chiqaring (4 iteratsiya)
   1.3. Bajarilish davomida oqim holatini ko'rsating

2. **Oqim Hayotiy Tsiklini Boshqarish**
   2.1. Ishchi oqimni ishga tushiring va holat o'tishlarini kuzating
   2.2. Iteratsiyalar orasida bajarilishni to'xtatib turish uchun Thread.sleep() dan foydalaning
   2.3. InterruptedException ni to'g'ri qayta ishlang

3. **Oqimlarni Sinxronlashtirish**
   3.1. Ishchi oqimning tugashini kutish uchun asosiy oqimda join() dan foydalaning
   3.2. Ishga tushishdan oldin, ishga tushgandan keyin va join dan keyin oqim holatlarini ko'rsating
   3.3. Asosiy oqim faqat ishchi oqim tugagandan keyin tugashiga ishonch hosil qiling`,
      hint1: 'Oqim holatlari: NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED. Oqim bajarilishini to\'xtatib turish uchun Thread.sleep(millisekund) dan foydalaning.',
      hint2: 'Boshqa oqimning tugashini kutish uchun join() metodidan foydalaning. Har doim InterruptedException ni try-catch bloklarida qayta ishlang.',
      whyItMatters: `Oqim hayotiy tsiklini tushunish Java da mustahkam ko'p oqimli ilovalarni yozish uchun asosiy hisoblanadi. Hayotiy tsikl holatlari (NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED) har bir oqim o'tadigan muhim bosqichlarni ifodalaydi va bu o'tishlarni boshqarishni bilish dastur oqimini nazorat qilish uchun zarurdir.

join() metodi oqimlar bajarilishini muvofiqlashtirishda hal qiluvchi ahamiyatga ega bo'lib, bog'liq operatsiyalarning to'g'ri tartibda tugashini ta'minlaydi. Thread.sleep() - bu parallel ilovalarda kechikishlar, vaqt chegaralari va tezlikni cheklashni amalga oshirish uchun asosiy, lekin kuchli vosita.

InterruptedException ni to'g'ri qayta ishlash yaxshi istisno boshqaruv amaliyotlarini ko'rsatadi va oqimlar to'xtatilganda resurs oqishi yoki kutilmagan xatti-harakatlarni oldini oladi. Bu kontseptsiyalar oqim havzalari, ijrochilar va asinxron dasturlash kabi yanada ilg'or parallel naqshlar uchun asos yaratadi.`,
      solutionCode: `public class ThreadLifecycle {
    public static void main(String[] args) {
        // Uxlaydigan va xabarlarni chiqaradigan ishchi oqim yaratamiz
        Thread workerThread = new Thread(() -> {
            System.out.println("Ishchi oqim boshlandi: " + Thread.currentThread().getState());

            for (int i = 1; i <= 4; i++) {
                try {
                    // 500 millisekund uxlaymiz
                    Thread.sleep(500);
                    System.out.println("Ishchi iteratsiya " + i + " - Holat: " + Thread.currentThread().getState());
                } catch (InterruptedException e) {
                    System.out.println("Ishchi oqim to'xtatildi");
                    Thread.currentThread().interrupt();
                    return;
                }
            }

            System.out.println("Ishchi oqim tugadi");
        });

        // Ishchi oqimni ishga tushiramiz (NEW -> RUNNABLE)
        System.out.println("Ishchi oqim holati ishga tushishdan oldin: " + workerThread.getState());
        workerThread.start();
        System.out.println("Ishchi oqim holati ishga tushgandan keyin: " + workerThread.getState());

        try {
            // Ishchi oqimning tugashini kutamiz
            System.out.println("Asosiy oqim ishchi oqimni kutmoqda...");
            workerThread.join();
            System.out.println("Ishchi oqim holati join dan keyin: " + workerThread.getState());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Asosiy oqim tugadi");
    }
}`,
    },
  },
};

export default task;
