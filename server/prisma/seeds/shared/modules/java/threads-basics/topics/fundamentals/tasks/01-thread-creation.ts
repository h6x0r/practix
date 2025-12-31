import { Task } from '../../../../../../types';

export const task: Task = {
	slug: 'java-threads-thread-creation',
	title: 'Thread Creation',
	difficulty: 'easy',
	tags: ['java', 'threads', 'concurrency', 'runnable'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn how to create threads using Thread class and Runnable interface in Java.

**Requirements:**
1. Create a thread using Thread class
   1.1. Override run() method to print "Thread using Thread class"
2. Create a thread using Runnable interface
   2.1. Implement run() method to print "Thread using Runnable interface"
3. Create a thread using lambda expression
   3.1. Print "Thread using lambda expression"
4. Start all threads using start() method

**Example:**
\`\`\`java
Thread thread1 = new Thread() {
    @Override
    public void run() {
        System.out.println("Hello from Thread class");
    }
};
thread1.start();
\`\`\``,
	initialCode: `public class ThreadCreation {
    public static void main(String[] args) {
        // TODO: Create a thread using Thread class
        // Override run() method to print "Thread using Thread class"


        // TODO: Create a thread using Runnable interface
        // Implement run() method to print "Thread using Runnable interface"


        // TODO: Create a thread using lambda expression
        // Print "Thread using lambda expression"


        // TODO: Start all threads

    }
}`,
	solutionCode: `public class ThreadCreation {
    public static void main(String[] args) {
        // Create a thread using Thread class
        Thread thread1 = new Thread() {
            @Override
            public void run() {
                System.out.println("Thread using Thread class: " + Thread.currentThread().getName());
            }
        };

        // Create a thread using Runnable interface
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("Thread using Runnable interface: " + Thread.currentThread().getName());
            }
        };
        Thread thread2 = new Thread(runnable);

        // Create a thread using lambda expression
        Thread thread3 = new Thread(() -> {
            System.out.println("Thread using lambda expression: " + Thread.currentThread().getName());
        });

        // Start all threads
        thread1.start();
        thread2.start();
        thread3.start();

        System.out.println("Main thread: " + Thread.currentThread().getName());
    }
}`,
	testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.*;

// Test1: Verify thread creation using Thread class
class Test1 {
    @Test
    public void test() throws Exception {
        final boolean[] executed = {false};
        Thread thread = new Thread() {
            @Override
            public void run() {
                executed[0] = true;
            }
        };
        thread.start();
        thread.join();
        assertTrue("Thread using Thread class should execute", executed[0]);
    }
}

// Test2: Verify thread creation using Runnable interface
class Test2 {
    @Test
    public void test() throws Exception {
        final boolean[] executed = {false};
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                executed[0] = true;
            }
        };
        Thread thread = new Thread(runnable);
        thread.start();
        thread.join();
        assertTrue("Thread using Runnable interface should execute", executed[0]);
    }
}

// Test3: Verify thread creation using lambda expression
class Test3 {
    @Test
    public void test() throws Exception {
        final boolean[] executed = {false};
        Thread thread = new Thread(() -> executed[0] = true);
        thread.start();
        thread.join();
        assertTrue("Thread using lambda expression should execute", executed[0]);
    }
}

// Test4: Verify all three threads can run concurrently
class Test4 {
    @Test
    public void test() throws Exception {
        final int[] count = {0};
        Thread t1 = new Thread() {
            @Override
            public void run() {
                synchronized(count) { count[0]++; }
            }
        };
        Runnable r = new Runnable() {
            @Override
            public void run() {
                synchronized(count) { count[0]++; }
            }
        };
        Thread t2 = new Thread(r);
        Thread t3 = new Thread(() -> { synchronized(count) { count[0]++; } });

        t1.start();
        t2.start();
        t3.start();
        t1.join();
        t2.join();
        t3.join();

        assertEquals("All three threads should execute", 3, count[0]);
    }
}

// Test5: Verify thread names can be accessed
class Test5 {
    @Test
    public void test() throws Exception {
        final String[] threadName = {null};
        Thread thread = new Thread(() -> threadName[0] = Thread.currentThread().getName());
        thread.start();
        thread.join();
        assertNotNull("Thread should have a name", threadName[0]);
        assertTrue("Thread name should start with 'Thread-'", threadName[0].startsWith("Thread-"));
    }
}

// Test6: Verify custom thread names work
class Test6 {
    @Test
    public void test() throws Exception {
        final String[] threadName = {null};
        Thread thread = new Thread(() -> threadName[0] = Thread.currentThread().getName(), "CustomThread");
        thread.start();
        thread.join();
        assertEquals("Thread should have custom name", "CustomThread", threadName[0]);
    }
}

// Test7: Verify main thread is still running
class Test7 {
    @Test
    public void test() {
        String mainThreadName = Thread.currentThread().getName();
        assertNotNull("Main thread should have a name", mainThreadName);
        assertEquals("Main thread should be named 'main'", "main", mainThreadName);
    }
}

// Test8: Verify thread state transitions
class Test8 {
    @Test
    public void test() throws Exception {
        Thread thread = new Thread(() -> {
            try { Thread.sleep(100); } catch (InterruptedException e) {}
        });
        assertEquals("New thread should be in NEW state", Thread.State.NEW, thread.getState());
        thread.start();
        Thread.sleep(10);
        assertTrue("Running thread should be RUNNABLE or TIMED_WAITING",
            thread.getState() == Thread.State.RUNNABLE || thread.getState() == Thread.State.TIMED_WAITING);
        thread.join();
        assertEquals("Finished thread should be in TERMINATED state", Thread.State.TERMINATED, thread.getState());
    }
}

// Test9: Verify multiple threads execute independently
class Test9 {
    @Test
    public void test() throws Exception {
        final long[] timestamps = new long[3];
        Thread t1 = new Thread(() -> {
            try { Thread.sleep(50); timestamps[0] = System.nanoTime(); }
            catch (InterruptedException e) {}
        });
        Thread t2 = new Thread(() -> {
            try { Thread.sleep(100); timestamps[1] = System.nanoTime(); }
            catch (InterruptedException e) {}
        });
        Thread t3 = new Thread(() -> {
            try { Thread.sleep(150); timestamps[2] = System.nanoTime(); }
            catch (InterruptedException e) {}
        });

        t1.start();
        t2.start();
        t3.start();
        t1.join();
        t2.join();
        t3.join();

        assertTrue("Threads should complete in order", timestamps[0] < timestamps[1] && timestamps[1] < timestamps[2]);
    }
}

// Test10: Verify thread execution with shared data
class Test10 {
    @Test
    public void test() throws Exception {
        final StringBuilder result = new StringBuilder();
        Thread t1 = new Thread(() -> result.append("A"));
        Thread t2 = new Thread(() -> result.append("B"));
        Thread t3 = new Thread(() -> result.append("C"));

        t1.start();
        t2.start();
        t3.start();
        t1.join();
        t2.join();
        t3.join();

        assertEquals("Result should contain all characters", 3, result.length());
        assertTrue("Result should contain A", result.toString().contains("A"));
        assertTrue("Result should contain B", result.toString().contains("B"));
        assertTrue("Result should contain C", result.toString().contains("C"));
    }
}`,
	hint1: `Use Thread class and override run() method for the first thread. Implement Runnable interface and pass it to Thread constructor.`,
	hint2: `Use lambda expression () -> {} for the third thread. Remember to call start() method, not run() directly.`,
	whyItMatters: `Understanding thread creation is fundamental to concurrent programming in Java.

**Why Thread Creation Matters:**
1.1. **Concurrent Execution:** Run multiple tasks simultaneously
1.2. **Resource Utilization:** Better use of multi-core processors
1.3. **Responsive Applications:** Keep UI responsive while performing background tasks

**Three Ways to Create Threads:**
\`\`\`java
// 1. Extend Thread class
class MyThread extends Thread {
    public void run() { /* task */ }
}

// 2. Implement Runnable interface
class MyRunnable implements Runnable {
    public void run() { /* task */ }
}

// 3. Lambda expression (Java 8+)
Thread t = new Thread(() -> { /* task */ });
\`\`\`

**Best Practice:** Prefer Runnable interface or lambda over extending Thread class, as it allows your class to extend another class.`,
	order: 1,
	translations: {
		ru: {
			title: 'Создание потоков',
			description: `Изучите, как создавать потоки с помощью класса Thread и интерфейса Runnable в Java.

**Требования:**
1. Создайте поток с помощью класса Thread
   1.1. Переопределите метод run() для вывода "Thread using Thread class"
2. Создайте поток с помощью интерфейса Runnable
   2.1. Реализуйте метод run() для вывода "Thread using Runnable interface"
3. Создайте поток с помощью лямбда-выражения
   3.1. Выведите "Thread using lambda expression"
4. Запустите все потоки с помощью метода start()`,
			hint1: `Используйте класс Thread и переопределите метод run() для первого потока. Реализуйте интерфейс Runnable и передайте его в конструктор Thread.`,
			hint2: `Используйте лямбда-выражение () -> {} для третьего потока. Не забудьте вызвать метод start(), а не run() напрямую.`,
			whyItMatters: `Понимание создания потоков является основой параллельного программирования в Java.

**Почему создание потоков важно:**
1.1. **Параллельное выполнение:** Запуск нескольких задач одновременно
1.2. **Использование ресурсов:** Лучшее использование многоядерных процессоров
1.3. **Отзывчивые приложения:** Поддержание отзывчивости UI при выполнении фоновых задач

**Три способа создания потоков:**
\`\`\`java
// 1. Расширение класса Thread
class MyThread extends Thread {
    public void run() { /* задача */ }
}

// 2. Реализация интерфейса Runnable
class MyRunnable implements Runnable {
    public void run() { /* задача */ }
}

// 3. Лямбда-выражение (Java 8+)
Thread t = new Thread(() -> { /* задача */ });
\`\`\`

**Лучшая практика:** Предпочитайте интерфейс Runnable или лямбда вместо расширения класса Thread, так как это позволяет вашему классу расширять другой класс.`,
			solutionCode: `public class ThreadCreation {
    public static void main(String[] args) {
        // Создаем поток с помощью класса Thread
        Thread thread1 = new Thread() {
            @Override
            public void run() {
                System.out.println("Поток с использованием класса Thread: " + Thread.currentThread().getName());
            }
        };

        // Создаем поток с помощью интерфейса Runnable
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("Поток с использованием интерфейса Runnable: " + Thread.currentThread().getName());
            }
        };
        Thread thread2 = new Thread(runnable);

        // Создаем поток с помощью лямбда-выражения
        Thread thread3 = new Thread(() -> {
            System.out.println("Поток с использованием лямбда-выражения: " + Thread.currentThread().getName());
        });

        // Запускаем все потоки
        thread1.start();
        thread2.start();
        thread3.start();

        System.out.println("Главный поток: " + Thread.currentThread().getName());
    }
}`
		},
		uz: {
			title: 'Oqimlarni yaratish',
			description: `Java da Thread klassi va Runnable interfeysi yordamida oqimlarni yaratishni o'rganing.

**Talablar:**
1. Thread klassi yordamida oqim yarating
   1.1. run() metodini qayta yozing va "Thread using Thread class" ni chiqaring
2. Runnable interfeysi yordamida oqim yarating
   2.1. run() metodini amalga oshiring va "Thread using Runnable interface" ni chiqaring
3. Lambda ifodasi yordamida oqim yarating
   3.1. "Thread using lambda expression" ni chiqaring
4. Barcha oqimlarni start() metodi yordamida ishga tushiring`,
			hint1: `Birinchi oqim uchun Thread klassidan foydalaning va run() metodini qayta yozing. Runnable interfeysini amalga oshiring va uni Thread konstruktoriga o'tkazing.`,
			hint2: `Uchinchi oqim uchun lambda ifodasidan foydalaning () -> {}. start() metodini chaqirishni unutmang, run() ni to'g'ridan-to'g'ri emas.`,
			whyItMatters: `Oqimlarni yaratishni tushunish Java da parallel dasturlashning asosidir.

**Nima uchun oqim yaratish muhim:**
1.1. **Parallel bajarish:** Bir nechta vazifalarni bir vaqtning o'zida bajarish
1.2. **Resurslardan foydalanish:** Ko'p yadroli protsessorlardan yaxshiroq foydalanish
1.3. **Javob beradigan ilovalar:** Fon vazifalarini bajarishda UI javob berishini saqlash

**Oqimlarni yaratishning uch usuli:**
\`\`\`java
// 1. Thread klassini kengaytirish
class MyThread extends Thread {
    public void run() { /* vazifa */ }
}

// 2. Runnable interfeysini amalga oshirish
class MyRunnable implements Runnable {
    public void run() { /* vazifa */ }
}

// 3. Lambda ifodasi (Java 8+)
Thread t = new Thread(() -> { /* vazifa */ });
\`\`\`

**Eng yaxshi amaliyot:** Thread klassini kengaytirishdan ko'ra Runnable interfeysi yoki lambdani afzal ko'ring, chunki bu sizning klassizga boshqa klassni kengaytirish imkonini beradi.`,
			solutionCode: `public class ThreadCreation {
    public static void main(String[] args) {
        // Thread klassi yordamida oqim yaratamiz
        Thread thread1 = new Thread() {
            @Override
            public void run() {
                System.out.println("Thread klassi yordamida oqim: " + Thread.currentThread().getName());
            }
        };

        // Runnable interfeysi yordamida oqim yaratamiz
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("Runnable interfeysi yordamida oqim: " + Thread.currentThread().getName());
            }
        };
        Thread thread2 = new Thread(runnable);

        // Lambda ifodasi yordamida oqim yaratamiz
        Thread thread3 = new Thread(() -> {
            System.out.println("Lambda ifodasi yordamida oqim: " + Thread.currentThread().getName());
        });

        // Barcha oqimlarni ishga tushiramiz
        thread1.start();
        thread2.start();
        thread3.start();

        System.out.println("Asosiy oqim: " + Thread.currentThread().getName());
    }
}`
		}
	}
};

export default task;
