import { Task } from '../../../../../../types';

export const task: Task = {
  slug: 'java-threads-wait-notify',
  title: 'Wait and Notify',
  difficulty: 'medium',
  tags: ['threads', 'wait', 'notify', 'coordination'],
  estimatedTime: '35m',
  isPremium: false,
  youtubeUrl: '',
  description: `Learn thread coordination using wait(), notify(), and notifyAll() methods by implementing a producer-consumer pattern.

1. Implement the SharedQueue class:
   1.1. Create a synchronized produce() method that adds messages to the queue
   1.2. Use wait() to pause the producer when the queue already has a message
   1.3. Use notifyAll() to wake up waiting consumers after adding a message
   1.4. Create a synchronized consume() method that retrieves messages from the queue
   1.5. Use wait() to pause the consumer when the queue is empty
   1.6. Use notifyAll() to wake up waiting producers after consuming a message

2. Implement thread coordination in the main method:
   2.1. Create a producer thread that sends 5 messages to the queue
   2.2. Create a consumer thread that receives messages from the queue
   2.3. Start both threads and observe their coordination
   2.4. Use join() to wait for both threads to complete

3. Key requirements:
   3.1. Always call wait() and notify() within synchronized blocks or methods
   3.2. Use while loops (not if statements) to check conditions before wait()
   3.3. Handle InterruptedException properly by restoring interrupt status
   3.4. Ensure thread-safe access to shared resources`,
  initialCode: `class SharedQueue {
    private String message = null;
    private boolean hasMessage = false;

    // TODO: Implement synchronized produce method
    // Use wait() if queue already has a message
    // Use notifyAll() after adding message
    public void produce(String msg) {

    }

    // TODO: Implement synchronized consume method
    // Use wait() if queue is empty
    // Use notifyAll() after consuming message
    public String consume() {
        return null;
    }
}

public class WaitNotify {
    public static void main(String[] args) {
        SharedQueue queue = new SharedQueue();

        // TODO: Create producer thread that sends 5 messages


        // TODO: Create consumer thread that receives messages


        // TODO: Start both threads

    }
}`,
  solutionCode: `class SharedQueue {
    private String message = null;
    private boolean hasMessage = false;

    // Synchronized produce method
    public synchronized void produce(String msg) {
        // Wait while queue already has a message
        while (hasMessage) {
            try {
                System.out.println("Producer waiting... queue is full");
                wait(); // Release lock and wait
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }

        // Produce message
        this.message = msg;
        this.hasMessage = true;
        System.out.println("Produced: " + msg);

        // Notify waiting consumers
        notifyAll();
    }

    // Synchronized consume method
    public synchronized String consume() {
        // Wait while queue is empty
        while (!hasMessage) {
            try {
                System.out.println("Consumer waiting... queue is empty");
                wait(); // Release lock and wait
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }

        // Consume message
        String consumedMessage = this.message;
        this.message = null;
        this.hasMessage = false;
        System.out.println("Consumed: " + consumedMessage);

        // Notify waiting producers
        notifyAll();

        return consumedMessage;
    }
}

public class WaitNotify {
    public static void main(String[] args) {
        SharedQueue queue = new SharedQueue();

        // Create producer thread that sends 5 messages
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                queue.produce("Message-" + i);
                try {
                    Thread.sleep(1000); // Simulate production time
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            System.out.println("Producer finished");
        }, "Producer");

        // Create consumer thread that receives messages
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                String msg = queue.consume();
                try {
                    Thread.sleep(1500); // Simulate consumption time
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            System.out.println("Consumer finished");
        }, "Consumer");

        // Start both threads
        producer.start();
        consumer.start();

        try {
            producer.join();
            consumer.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Main thread finished");
    }
}`,
  testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.atomic.AtomicInteger;

class SharedQueue {
    private String message = null;
    private boolean hasMessage = false;

    public synchronized void produce(String msg) {
        while (hasMessage) {
            try {
                wait();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
        this.message = msg;
        this.hasMessage = true;
        notifyAll();
    }

    public synchronized String consume() {
        while (!hasMessage) {
            try {
                wait();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
        String consumedMessage = this.message;
        this.message = null;
        this.hasMessage = false;
        notifyAll();
        return consumedMessage;
    }
}

// Test1: Verify basic produce-consume works
class Test1 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final String[] result = {null};

        Thread producer = new Thread(() -> queue.produce("Test Message"));
        Thread consumer = new Thread(() -> result[0] = queue.consume());

        producer.start();
        Thread.sleep(50);
        consumer.start();

        producer.join();
        consumer.join();

        assertEquals("Should receive produced message", "Test Message", result[0]);
    }
}

// Test2: Verify multiple messages can be produced and consumed
class Test2 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final String[] results = new String[3];

        Thread producer = new Thread(() -> {
            queue.produce("Message1");
            queue.produce("Message2");
            queue.produce("Message3");
        });

        Thread consumer = new Thread(() -> {
            results[0] = queue.consume();
            results[1] = queue.consume();
            results[2] = queue.consume();
        });

        producer.start();
        consumer.start();
        producer.join();
        consumer.join();

        assertEquals("Message1", results[0]);
        assertEquals("Message2", results[1]);
        assertEquals("Message3", results[2]);
    }
}

// Test3: Verify producer waits when queue is full
class Test3 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        AtomicInteger produced = new AtomicInteger(0);

        Thread producer = new Thread(() -> {
            queue.produce("Msg1");
            produced.incrementAndGet();
            queue.produce("Msg2");
            produced.incrementAndGet();
        });

        producer.start();
        Thread.sleep(100);

        assertEquals("Should only produce one message when queue is full", 1, produced.get());

        queue.consume();
        Thread.sleep(100);

        assertEquals("Should produce second message after consumption", 2, produced.get());
        producer.join();
    }
}

// Test4: Verify consumer waits when queue is empty
class Test4 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final String[] result = {null};
        final boolean[] started = {false};

        Thread consumer = new Thread(() -> {
            started[0] = true;
            result[0] = queue.consume();
        });

        consumer.start();
        Thread.sleep(100);

        assertTrue("Consumer should have started", started[0]);
        assertNull("Consumer should be waiting", result[0]);

        queue.produce("Test");
        consumer.join();

        assertEquals("Should receive message after production", "Test", result[0]);
    }
}

// Test5: Verify notifyAll wakes up waiting threads
class Test5 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final AtomicInteger consumed = new AtomicInteger(0);

        Thread consumer1 = new Thread(() -> {
            queue.consume();
            consumed.incrementAndGet();
        });
        Thread consumer2 = new Thread(() -> {
            queue.consume();
            consumed.incrementAndGet();
        });

        consumer1.start();
        consumer2.start();
        Thread.sleep(100);

        queue.produce("Msg1");
        Thread.sleep(100);
        queue.produce("Msg2");

        consumer1.join();
        consumer2.join();

        assertEquals("Both consumers should consume", 2, consumed.get());
    }
}

// Test6: Verify InterruptedException handling in produce
class Test6 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        queue.produce("Block");

        Thread producer = new Thread(() -> queue.produce("ShouldNotProduce"));
        producer.start();
        Thread.sleep(50);
        producer.interrupt();
        producer.join();

        String msg = queue.consume();
        assertEquals("Should only have first message", "Block", msg);
    }
}

// Test7: Verify InterruptedException handling in consume
class Test7 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final String[] result = {null};

        Thread consumer = new Thread(() -> result[0] = queue.consume());
        consumer.start();
        Thread.sleep(50);
        consumer.interrupt();
        consumer.join();

        assertNull("Interrupted consume should return null", result[0]);
    }
}

// Test8: Verify alternating produce-consume sequence
class Test8 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final StringBuilder sequence = new StringBuilder();

        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                queue.produce("M" + i);
                sequence.append("P" + i);
            }
        });

        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                String msg = queue.consume();
                sequence.append("C" + i);
            }
        });

        producer.start();
        consumer.start();
        producer.join();
        consumer.join();

        assertEquals("Sequence length should be 10", 10, sequence.length());
    }
}

// Test9: Verify thread coordination with sleep delays
class Test9 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();
        final String[] result = {null};

        Thread producer = new Thread(() -> {
            try {
                Thread.sleep(100);
                queue.produce("Delayed Message");
            } catch (InterruptedException e) {}
        });

        Thread consumer = new Thread(() -> result[0] = queue.consume());

        consumer.start();
        producer.start();

        producer.join();
        consumer.join();

        assertEquals("Should receive delayed message", "Delayed Message", result[0]);
    }
}

// Test10: Verify queue state after operations
class Test10 {
    @Test
    public void test() throws Exception {
        SharedQueue queue = new SharedQueue();

        Thread producer = new Thread(() -> {
            queue.produce("Test1");
            queue.produce("Test2");
        });

        Thread consumer = new Thread(() -> {
            queue.consume();
            queue.consume();
        });

        producer.start();
        consumer.start();
        producer.join();
        consumer.join();

        // Queue should be empty and ready for new messages
        Thread p2 = new Thread(() -> queue.produce("Test3"));
        Thread c2 = new Thread(() -> {
            String msg = queue.consume();
            assertEquals("Should get new message", "Test3", msg);
        });

        p2.start();
        c2.start();
        p2.join();
        c2.join();
    }
}`,
  hint1: `wait() and notify() must be called within synchronized block or method. Always use while loop (not if) to check condition before wait(). This prevents spurious wakeups and ensures the condition is rechecked after the thread wakes up.`,
  hint2: `wait() releases the lock and puts thread in WAITING state, allowing other threads to acquire the lock. notifyAll() wakes up all waiting threads, while notify() wakes up only one thread. Using notifyAll() is generally safer to avoid potential deadlocks.`,
  whyItMatters: `Understanding wait() and notify() is crucial for building efficient concurrent applications. These methods form the foundation of thread coordination in Java, enabling threads to communicate and synchronize their actions without busy-waiting (consuming CPU cycles). This pattern is used extensively in real-world applications for implementing producer-consumer queues, thread pools, and event-driven systems. Mastering these primitives helps you avoid common concurrency pitfalls like deadlocks, race conditions, and resource starvation, while building scalable multi-threaded applications that efficiently utilize system resources.`,
  order: 4,
  translations: {
    ru: {
      title: 'Wait и Notify',
      description: `Изучите координацию потоков с помощью методов wait(), notify() и notifyAll(), реализуя паттерн производитель-потребитель.

1. Реализуйте класс SharedQueue:
   1.1. Создайте синхронизированный метод produce(), который добавляет сообщения в очередь
   1.2. Используйте wait() для приостановки производителя, когда очередь уже имеет сообщение
   1.3. Используйте notifyAll() для пробуждения ожидающих потребителей после добавления сообщения
   1.4. Создайте синхронизированный метод consume(), который извлекает сообщения из очереди
   1.5. Используйте wait() для приостановки потребителя, когда очередь пуста
   1.6. Используйте notifyAll() для пробуждения ожидающих производителей после потребления сообщения

2. Реализуйте координацию потоков в главном методе:
   2.1. Создайте поток-производитель, который отправляет 5 сообщений в очередь
   2.2. Создайте поток-потребитель, который получает сообщения из очереди
   2.3. Запустите оба потока и наблюдайте за их координацией
   2.4. Используйте join() для ожидания завершения обоих потоков

3. Ключевые требования:
   3.1. Всегда вызывайте wait() и notify() внутри синхронизированных блоков или методов
   3.2. Используйте циклы while (не операторы if) для проверки условий перед wait()
   3.3. Правильно обрабатывайте InterruptedException, восстанавливая статус прерывания
   3.4. Обеспечьте потокобезопасный доступ к общим ресурсам`,
      hint1: `wait() и notify() должны вызываться внутри синхронизированного блока или метода. Всегда используйте цикл while (не if) для проверки условия перед wait(). Это предотвращает ложные пробуждения и гарантирует повторную проверку условия после пробуждения потока.`,
      hint2: `wait() освобождает блокировку и переводит поток в состояние WAITING, позволяя другим потокам получить блокировку. notifyAll() пробуждает все ожидающие потоки, в то время как notify() пробуждает только один поток. Использование notifyAll() обычно безопаснее для избежания потенциальных взаимных блокировок.`,
      whyItMatters: `Понимание wait() и notify() критически важно для создания эффективных конкурентных приложений. Эти методы составляют основу координации потоков в Java, позволяя потокам общаться и синхронизировать свои действия без активного ожидания (потребления циклов процессора). Этот паттерн широко используется в реальных приложениях для реализации очередей производитель-потребитель, пулов потоков и событийно-ориентированных систем. Освоение этих примитивов помогает избежать распространенных проблем параллелизма, таких как взаимные блокировки, состояния гонки и голодание ресурсов, создавая масштабируемые многопоточные приложения, эффективно использующие системные ресурсы.`,
      solutionCode: `class SharedQueue {
    private String message = null;
    private boolean hasMessage = false;

    // Синхронизированный метод производства
    public synchronized void produce(String msg) {
        // Ожидаем, пока очередь уже имеет сообщение
        while (hasMessage) {
            try {
                System.out.println("Производитель ожидает... очередь заполнена");
                wait(); // Освобождаем блокировку и ожидаем
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }

        // Производим сообщение
        this.message = msg;
        this.hasMessage = true;
        System.out.println("Произведено: " + msg);

        // Уведомляем ожидающих потребителей
        notifyAll();
    }

    // Синхронизированный метод потребления
    public synchronized String consume() {
        // Ожидаем, пока очередь пуста
        while (!hasMessage) {
            try {
                System.out.println("Потребитель ожидает... очередь пуста");
                wait(); // Освобождаем блокировку и ожидаем
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }

        // Потребляем сообщение
        String consumedMessage = this.message;
        this.message = null;
        this.hasMessage = false;
        System.out.println("Потреблено: " + consumedMessage);

        // Уведомляем ожидающих производителей
        notifyAll();

        return consumedMessage;
    }
}

public class WaitNotify {
    public static void main(String[] args) {
        SharedQueue queue = new SharedQueue();

        // Создаем поток-производитель, который отправляет 5 сообщений
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                queue.produce("Сообщение-" + i);
                try {
                    Thread.sleep(1000); // Имитируем время производства
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            System.out.println("Производитель завершен");
        }, "Producer");

        // Создаем поток-потребитель, который получает сообщения
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                String msg = queue.consume();
                try {
                    Thread.sleep(1500); // Имитируем время потребления
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            System.out.println("Потребитель завершен");
        }, "Consumer");

        // Запускаем оба потока
        producer.start();
        consumer.start();

        try {
            producer.join();
            consumer.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Главный поток завершен");
    }
}`,
    },
    uz: {
      title: 'Wait va Notify',
      description: `wait(), notify() va notifyAll() metodlari yordamida oqimlarni muvofiqlashtirish, ishlab chiqaruvchi-iste'molchi namunasini amalga oshirish orqali o'rganing.

1. SharedQueue klassini amalga oshiring:
   1.1. Navbatga xabarlar qo'shadigan sinxronlashtirilgan produce() metodini yarating
   1.2. Navbat allaqachon xabarga ega bo'lsa, ishlab chiqaruvchini to'xtatib turish uchun wait() dan foydalaning
   1.3. Xabar qo'shgandan keyin kutayotgan iste'molchilarni uyg'otish uchun notifyAll() dan foydalaning
   1.4. Navbatdan xabarlarni oladigan sinxronlashtirilgan consume() metodini yarating
   1.5. Navbat bo'sh bo'lsa, iste'molchini to'xtatib turish uchun wait() dan foydalaning
   1.6. Xabarni iste'mol qilgandan keyin kutayotgan ishlab chiqaruvchilarni uyg'otish uchun notifyAll() dan foydalaning

2. Asosiy metodda oqimlar muvofiqligini amalga oshiring:
   2.1. Navbatga 5 ta xabar yuboradigan ishlab chiqaruvchi oqim yarating
   2.2. Navbatdan xabarlarni qabul qiladigan iste'molchi oqim yarating
   2.3. Ikkala oqimni ham ishga tushiring va ularning muvofiqligini kuzating
   2.4. Ikkala oqim tugashini kutish uchun join() dan foydalaning

3. Asosiy talablar:
   3.1. wait() va notify() ni har doim sinxronlashtirilgan bloklar yoki metodlar ichida chaqiring
   3.2. wait() dan oldin shartlarni tekshirish uchun while tsikllaridan foydalaning (if operatorlari emas)
   3.3. InterruptedException ni to'g'ri qayta ishlang, to'xtatish holatini tiklang
   3.4. Umumiy resurslarga potok-xavfsiz kirishni ta'minlang`,
      hint1: `wait() va notify() sinxronlashtirilgan blok yoki metod ichida chaqirilishi kerak. wait() dan oldin shartni tekshirish uchun har doim while tsiklidan foydalaning (if emas). Bu soxta uyg'onishlarni oldini oladi va oqim uyg'ongandan keyin shartning qayta tekshirilishini ta'minlaydi.`,
      hint2: `wait() qulfni bo'shatadi va oqimni WAITING holatiga o'tkazadi, boshqa oqimlarga qulfni olish imkonini beradi. notifyAll() barcha kutayotgan oqimlarni uyg'otadi, notify() esa faqat bitta oqimni uyg'otadi. notifyAll() dan foydalanish odatda potentsial o'zaro blokirovkalardan qochish uchun xavfsizroqdir.`,
      whyItMatters: `wait() va notify() ni tushunish samarali parallel ilovalar yaratish uchun juda muhimdir. Bu metodlar Java-da oqimlar muvofiqligining asosini tashkil etadi, oqimlarga muloqot qilish va faol kutmasdan (protsessor tsikllarini iste'mol qilmasdan) o'z harakatlarini sinxronlashtirish imkonini beradi. Bu namuna haqiqiy ilovalarda ishlab chiqaruvchi-iste'molchi navbatlarini, oqim poollarini va hodisalarga asoslangan tizimlarni amalga oshirish uchun keng qo'llaniladi. Bu primitivlarni o'zlashtirib olish deadlock, race condition va resurs ochligidan qochishga yordam beradi, tizim resurslarini samarali ishlatadigan kengaytiriladigan ko'p oqimli ilovalar yaratadi.`,
      solutionCode: `class SharedQueue {
    private String message = null;
    private boolean hasMessage = false;

    // Sinxronlashtirilgan ishlab chiqarish metodi
    public synchronized void produce(String msg) {
        // Navbat allaqachon xabarga ega bo'lsa kutamiz
        while (hasMessage) {
            try {
                System.out.println("Ishlab chiqaruvchi kutmoqda... navbat to'lgan");
                wait(); // Qulfni bo'shatamiz va kutamiz
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }

        // Xabar ishlab chiqaramiz
        this.message = msg;
        this.hasMessage = true;
        System.out.println("Ishlab chiqarildi: " + msg);

        // Kutayotgan iste'molchilarni xabardor qilamiz
        notifyAll();
    }

    // Sinxronlashtirilgan iste'mol metodi
    public synchronized String consume() {
        // Navbat bo'sh bo'lsa kutamiz
        while (!hasMessage) {
            try {
                System.out.println("Iste'molchi kutmoqda... navbat bo'sh");
                wait(); // Qulfni bo'shatamiz va kutamiz
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }

        // Xabarni iste'mol qilamiz
        String consumedMessage = this.message;
        this.message = null;
        this.hasMessage = false;
        System.out.println("Iste'mol qilindi: " + consumedMessage);

        // Kutayotgan ishlab chiqaruvchilarni xabardor qilamiz
        notifyAll();

        return consumedMessage;
    }
}

public class WaitNotify {
    public static void main(String[] args) {
        SharedQueue queue = new SharedQueue();

        // 5 ta xabar yuboradigan ishlab chiqaruvchi oqim yaratamiz
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                queue.produce("Xabar-" + i);
                try {
                    Thread.sleep(1000); // Ishlab chiqarish vaqtini taqlid qilamiz
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            System.out.println("Ishlab chiqaruvchi tugadi");
        }, "Producer");

        // Xabarlarni qabul qiladigan iste'molchi oqim yaratamiz
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                String msg = queue.consume();
                try {
                    Thread.sleep(1500); // Iste'mol vaqtini taqlid qilamiz
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            System.out.println("Iste'molchi tugadi");
        }, "Consumer");

        // Ikkala oqimni ham ishga tushiramiz
        producer.start();
        consumer.start();

        try {
            producer.join();
            consumer.join();
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
