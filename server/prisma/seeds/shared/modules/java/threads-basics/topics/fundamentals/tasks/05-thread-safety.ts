import { Task } from '../../../../../../types';

export const task: Task = {
	slug: 'java-threads-thread-safety',
	title: 'Thread Safety',
	difficulty: 'medium',
	tags: ['java', 'threads', 'thread-safety', 'race-condition', 'critical-section'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Understand race conditions, critical sections, and how to write thread-safe code.

**Requirements:**
1. Implement Thread-Safe BankAccount Class
   1.1. Make deposit() method thread-safe using synchronized block
   1.2. Make withdraw() method thread-safe using synchronized block
   1.3. Make getBalance() method thread-safe
2. Create Multiple Threads for Testing
   2.1. Create 5 threads that deposit $100 each
   2.2. Create 5 threads that withdraw $50 each
3. Execute and Verify
   3.1. Start all threads
   3.2. Wait for all threads to complete using join()
   3.3. Verify final balance is $1250.0 (1000 + 500 - 250)

**Example:**
\`\`\`java
class Counter {
    private final Object lock = new Object();
    private int count = 0;

    public void increment() {
        synchronized (lock) {  // Critical section
            count++;
        }
    }
}
\`\`\``,
	initialCode: `class BankAccount {
    private double balance;

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    // TODO: Make this method thread-safe
    public void deposit(double amount) {
        double temp = balance;
        // Simulate processing time
        try { Thread.sleep(10); } catch (InterruptedException e) {}
        balance = temp + amount;
    }

    // TODO: Make this method thread-safe
    public void withdraw(double amount) {
        double temp = balance;
        // Simulate processing time
        try { Thread.sleep(10); } catch (InterruptedException e) {}
        if (temp >= amount) {
            balance = temp - amount;
        }
    }

    // TODO: Make this method thread-safe
    public double getBalance() {
        return balance;
    }
}

public class ThreadSafety {
    public static void main(String[] args) throws InterruptedException {
        BankAccount account = new BankAccount(1000.0);

        // TODO: Create 5 threads that deposit $100 each


        // TODO: Create 5 threads that withdraw $50 each


        // TODO: Start all threads and wait for completion


        System.out.println("Final balance: $" + account.getBalance());
        System.out.println("Expected: $1250.0");
    }
}`,
	solutionCode: `class BankAccount {
    private double balance;
    private final Object lock = new Object();

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    // Thread-safe deposit method
    public void deposit(double amount) {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " depositing $" + amount);
            double temp = balance;

            // Simulate processing time - critical section
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            balance = temp + amount;
            System.out.println(Thread.currentThread().getName() + " new balance: $" + balance);
        }
    }

    // Thread-safe withdraw method
    public void withdraw(double amount) {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " withdrawing $" + amount);
            double temp = balance;

            // Simulate processing time - critical section
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if (temp >= amount) {
                balance = temp - amount;
                System.out.println(Thread.currentThread().getName() + " new balance: $" + balance);
            } else {
                System.out.println(Thread.currentThread().getName() + " insufficient funds!");
            }
        }
    }

    // Thread-safe getter
    public synchronized double getBalance() {
        return balance;
    }
}

public class ThreadSafety {
    public static void main(String[] args) throws InterruptedException {
        BankAccount account = new BankAccount(1000.0);
        System.out.println("Initial balance: $" + account.getBalance());

        // Create 5 threads that deposit $100 each
        Thread[] depositors = new Thread[5];
        for (int i = 0; i < 5; i++) {
            final int threadNum = i + 1;
            depositors[i] = new Thread(() -> {
                account.deposit(100.0);
            }, "Depositor-" + threadNum);
        }

        // Create 5 threads that withdraw $50 each
        Thread[] withdrawers = new Thread[5];
        for (int i = 0; i < 5; i++) {
            final int threadNum = i + 1;
            withdrawers[i] = new Thread(() -> {
                account.withdraw(50.0);
            }, "Withdrawer-" + threadNum);
        }

        // Start all threads
        for (Thread t : depositors) t.start();
        for (Thread t : withdrawers) t.start();

        // Wait for all threads to complete
        for (Thread t : depositors) t.join();
        for (Thread t : withdrawers) t.join();

        System.out.println("\\nFinal balance: $" + account.getBalance());
        System.out.println("Expected: $1250.0 (1000 + 500 - 250)");
    }
}`,
	testCode: `import static org.junit.Assert.*;
import org.junit.Test;

class BankAccount {
    private double balance;
    private final Object lock = new Object();

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        synchronized (lock) {
            double temp = balance;
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            balance = temp + amount;
        }
    }

    public void withdraw(double amount) {
        synchronized (lock) {
            double temp = balance;
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (temp >= amount) {
                balance = temp - amount;
            }
        }
    }

    public synchronized double getBalance() {
        return balance;
    }
}

// Test1: Verify initial balance is set correctly
class Test1 {
    @Test
    public void test() {
        BankAccount account = new BankAccount(1000.0);
        assertEquals("Initial balance should be 1000.0", 1000.0, account.getBalance(), 0.01);
    }
}

// Test2: Verify single deposit works
class Test2 {
    @Test
    public void test() {
        BankAccount account = new BankAccount(1000.0);
        account.deposit(100.0);
        assertEquals("Balance should be 1100.0", 1100.0, account.getBalance(), 0.01);
    }
}

// Test3: Verify single withdrawal works
class Test3 {
    @Test
    public void test() {
        BankAccount account = new BankAccount(1000.0);
        account.withdraw(50.0);
        assertEquals("Balance should be 950.0", 950.0, account.getBalance(), 0.01);
    }
}

// Test4: Verify multiple deposits are thread-safe
class Test4 {
    @Test
    public void test() throws Exception {
        BankAccount account = new BankAccount(1000.0);
        Thread[] threads = new Thread[5];

        for (int i = 0; i < 5; i++) {
            threads[i] = new Thread(() -> account.deposit(100.0));
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        assertEquals("Balance should be 1500.0 (1000 + 5*100)", 1500.0, account.getBalance(), 0.01);
    }
}

// Test5: Verify multiple withdrawals are thread-safe
class Test5 {
    @Test
    public void test() throws Exception {
        BankAccount account = new BankAccount(1000.0);
        Thread[] threads = new Thread[5];

        for (int i = 0; i < 5; i++) {
            threads[i] = new Thread(() -> account.withdraw(50.0));
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        assertEquals("Balance should be 750.0 (1000 - 5*50)", 750.0, account.getBalance(), 0.01);
    }
}

// Test6: Verify combined deposits and withdrawals
class Test6 {
    @Test
    public void test() throws Exception {
        BankAccount account = new BankAccount(1000.0);

        Thread[] depositors = new Thread[5];
        Thread[] withdrawers = new Thread[5];

        for (int i = 0; i < 5; i++) {
            depositors[i] = new Thread(() -> account.deposit(100.0));
            withdrawers[i] = new Thread(() -> account.withdraw(50.0));
        }

        for (Thread t : depositors) t.start();
        for (Thread t : withdrawers) t.start();
        for (Thread t : depositors) t.join();
        for (Thread t : withdrawers) t.join();

        assertEquals("Balance should be 1250.0 (1000 + 500 - 250)", 1250.0, account.getBalance(), 0.01);
    }
}

// Test7: Verify withdrawal does not allow negative balance
class Test7 {
    @Test
    public void test() {
        BankAccount account = new BankAccount(100.0);
        account.withdraw(150.0);
        assertEquals("Balance should remain 100.0", 100.0, account.getBalance(), 0.01);
    }
}

// Test8: Verify thread-safe balance reading
class Test8 {
    @Test
    public void test() throws Exception {
        BankAccount account = new BankAccount(1000.0);
        final boolean[] allNonNegative = {true};

        Thread writer = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                account.deposit(10.0);
                try { Thread.sleep(5); } catch (InterruptedException e) {}
            }
        });

        Thread reader = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                double balance = account.getBalance();
                if (balance < 1000.0) allNonNegative[0] = false;
                try { Thread.sleep(5); } catch (InterruptedException e) {}
            }
        });

        writer.start();
        reader.start();
        writer.join();
        reader.join();

        assertTrue("All balance readings should be non-negative", allNonNegative[0]);
    }
}

// Test9: Verify high concurrency scenario
class Test9 {
    @Test
    public void test() throws Exception {
        BankAccount account = new BankAccount(1000.0);
        Thread[] threads = new Thread[20];

        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> account.deposit(100.0));
        }
        for (int i = 10; i < 20; i++) {
            threads[i] = new Thread(() -> account.withdraw(50.0));
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        assertEquals("Balance should be 1500.0 (1000 + 1000 - 500)", 1500.0, account.getBalance(), 0.01);
    }
}

// Test10: Verify critical section prevents race conditions
class Test10 {
    @Test
    public void test() throws Exception {
        BankAccount account = new BankAccount(1000.0);
        int numThreads = 100;
        Thread[] threads = new Thread[numThreads];

        for (int i = 0; i < numThreads; i++) {
            if (i % 2 == 0) {
                threads[i] = new Thread(() -> account.deposit(10.0));
            } else {
                threads[i] = new Thread(() -> account.withdraw(10.0));
            }
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        assertEquals("Final balance should be 1000.0", 1000.0, account.getBalance(), 0.01);
    }
}`,
	hint1: `A race condition occurs when multiple threads access shared data simultaneously. Critical section is code that accesses shared resources.`,
	hint2: `Use synchronized blocks to protect critical sections. Ensure all methods that access shared state are synchronized.`,
	whyItMatters: `Thread safety is fundamental for building reliable concurrent applications.

**Why Thread Safety Matters:**

**1. Data Integrity:**
Prevents corrupted data from race conditions. Without synchronization, multiple threads can read and write simultaneously, leading to lost updates and inconsistent state.

\`\`\`java
// Without synchronization:
Thread A reads balance: 1000
Thread B reads balance: 1000
Thread A deposits 100: 1000 + 100 = 1100
Thread B deposits 100: 1000 + 100 = 1100  // Lost update!
// Expected: 1200, Got: 1100
\`\`\`

**2. Predictable Behavior:**
Ensures consistent results regardless of thread scheduling. Synchronization guarantees that critical sections execute atomically - no other thread can interfere while one thread is executing the synchronized block.

\`\`\`java
// With synchronization - atomic execution
synchronized (lock) {
    double temp = balance;  // Read
    balance = temp + amount;  // Write
    // No other thread can enter until this completes
}
\`\`\`

**3. Production Reliability:**
Critical for financial systems, e-commerce platforms, and real-time applications. Banking transactions, payment processing, and inventory management require strict thread safety to prevent financial losses and data corruption.

**Real-world examples where thread safety is critical:**
- **Banking systems:** Multiple ATMs accessing the same account
- **E-commerce:** Concurrent orders for limited inventory
- **Stock trading:** High-frequency trading systems
- **Gaming servers:** Multiplayer game state updates
- **Booking systems:** Hotel/flight reservation conflicts

**4. Real Production Example:**
E-commerce site with concurrent orders:
- **Without synchronization:** Two customers buy the last item → both orders succeed → overselling! Customer disappointment and potential legal issues.
- **With synchronization:** Second customer gets "out of stock" message → accurate inventory, happy customers.

\`\`\`java
class Inventory {
    private int stock = 1;  // Last item
    private final Object lock = new Object();

    // Thread-safe purchase
    public boolean purchase() {
        synchronized (lock) {
            if (stock > 0) {
                stock--;
                return true;  // Purchase successful
            }
            return false;  // Out of stock
        }
    }
}

// Customer A and B click "Buy" at same time
// Only one succeeds - inventory stays consistent
\`\`\`

**5. Understanding Critical Sections:**
A critical section is any code that accesses shared mutable state. Identifying and protecting critical sections is key to thread safety.

\`\`\`java
class Counter {
    private int count = 0;  // Shared mutable state
    private final Object lock = new Object();

    public void increment() {
        // Critical section - must be synchronized
        synchronized (lock) {
            count++;  // Read-Modify-Write operation
        }
    }

    public int getCount() {
        // Also needs synchronization for visibility
        synchronized (lock) {
            return count;
        }
    }
}
\`\`\`

**6. Common Thread Safety Bugs:**
\`\`\`java
// Bug #1: Inconsistent locking
public void methodA() {
    synchronized (lock) { balance++; }
}
public void methodB() {
    balance--;  // BUG: Not synchronized!
}

// Bug #2: Different lock objects
public void methodC() {
    synchronized (lockA) { balance++; }
}
public void methodD() {
    synchronized (lockB) { balance--; }  // BUG: Different lock!
}

// Bug #3: Lost updates in compound operations
public void methodE() {
    synchronized (lock) {
        int current = getCount();  // Read
    }
    // Lock released here!
    synchronized (lock) {
        setCount(current + 1);  // Write - BUG: Another thread could modify in between!
    }
}
\`\`\`

**7. Performance Considerations:**
While synchronization is necessary, it can impact performance if not used carefully:

\`\`\`java
// Bad: Lock held too long
public synchronized void processOrder(Order order) {
    validateOrder(order);  // Fast
    saveToDatabase(order);  // SLOW - I/O operation
    sendEmail(order);  // SLOW - Network operation
    // Other threads blocked for entire duration!
}

// Good: Minimize lock scope
public void processOrder(Order order) {
    validateOrder(order);  // No lock needed

    synchronized (lock) {
        // Only synchronize data modification
        orders.add(order);
        orderCount++;
    }

    saveToDatabase(order);  // No lock needed
    sendEmail(order);  // No lock needed
}
\`\`\`

**8. Alternative Approaches:**
Java provides higher-level concurrency utilities that can be easier and more efficient than manual synchronization:

\`\`\`java
// AtomicInteger for simple counters
private AtomicInteger count = new AtomicInteger(0);
public void increment() {
    count.incrementAndGet();  // Thread-safe, no explicit lock
}

// ConcurrentHashMap for thread-safe maps
private ConcurrentHashMap<String, Account> accounts = new ConcurrentHashMap<>();
public void updateAccount(String id, Account account) {
    accounts.put(id, account);  // Thread-safe
}

// ReadWriteLock for read-heavy scenarios
private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
public void write() {
    rwLock.writeLock().lock();
    try { /* modify data */ }
    finally { rwLock.writeLock().unlock(); }
}
public void read() {
    rwLock.readLock().lock();  // Multiple readers allowed
    try { /* read data */ }
    finally { rwLock.readLock().unlock(); }
}
\`\`\`

**Best Practices:**
1. Use the same lock object for related operations
2. Keep critical sections as short as possible
3. Avoid holding locks while performing I/O operations
4. Consider using concurrent collections (ConcurrentHashMap, CopyOnWriteArrayList) when applicable
5. Use synchronized methods for simple cases
6. Test with multiple threads to reveal race conditions
7. Document which lock protects which data
8. Prefer higher-level utilities (AtomicInteger, ConcurrentHashMap) over manual synchronization when possible

**Real-World Impact:**
A payment processing company had a race condition bug in their transaction handling:
- **Bug:** Multiple threads could process the same payment twice
- **Impact:** $2.3 million in duplicate charges over 3 months
- **Fix:** Added proper synchronization to payment processing
- **Result:** Zero duplicate charges, customer trust restored

**Testing for Thread Safety:**
\`\`\`java
// Stress test with many concurrent threads
@Test
public void testThreadSafety() throws InterruptedException {
    BankAccount account = new BankAccount(1000);
    int numThreads = 100;
    Thread[] threads = new Thread[numThreads];

    // Each thread deposits $10
    for (int i = 0; i < numThreads; i++) {
        threads[i] = new Thread(() -> account.deposit(10));
        threads[i].start();
    }

    // Wait for all threads
    for (Thread t : threads) t.join();

    // Should be 1000 + (100 * 10) = 2000
    assertEquals(2000.0, account.getBalance(), 0.01);
}
\`\`\`

Thread safety isn't optional in production systems - it's essential for correctness, reliability, and maintaining customer trust.`,
	order: 5,
	translations: {
		ru: {
			title: 'Потокобезопасность',
			description: `Поймите гонку данных, критические секции и как писать потокобезопасный код.

**Требования:**
1. Реализуйте потокобезопасный класс BankAccount
   1.1. Сделайте метод deposit() потокобезопасным с помощью synchronized
   1.2. Сделайте метод withdraw() потокобезопасным с помощью synchronized
   1.3. Сделайте метод getBalance() потокобезопасным
2. Создайте несколько потоков для тестирования
   2.1. Создайте 5 потоков, которые вносят по $100 каждый
   2.2. Создайте 5 потоков, которые снимают по $50 каждый
3. Выполните и проверьте
   3.1. Запустите все потоки
   3.2. Дождитесь завершения всех потоков с помощью join()
   3.3. Проверьте, что итоговый баланс равен $1250.0`,
			hint1: `Гонка данных возникает, когда несколько потоков одновременно обращаются к общим данным. Критическая секция - это код, который обращается к общим ресурсам.`,
			hint2: `Используйте синхронизированные блоки для защиты критических секций. Убедитесь, что все методы, которые обращаются к общему состоянию, синхронизированы.`,
			whyItMatters: `Потокобезопасность является основой для создания надежных параллельных приложений.

**Почему потокобезопасность важна:**

**1. Целостность данных:**
Предотвращает повреждение данных от гонки условий. Без синхронизации несколько потоков могут читать и писать одновременно, приводя к потере обновлений и несогласованному состоянию.

\`\`\`java
// Без синхронизации:
Thread A читает balance: 1000
Thread B читает balance: 1000
Thread A вносит 100: 1000 + 100 = 1100
Thread B вносит 100: 1000 + 100 = 1100  // Потерянное обновление!
// Ожидалось: 1200, Получено: 1100
\`\`\`

**2. Предсказуемое поведение:**
Обеспечивает согласованные результаты независимо от планирования потоков. Синхронизация гарантирует, что критические секции выполняются атомарно - никакой другой поток не может вмешаться, пока один поток выполняет синхронизированный блок.

\`\`\`java
// С синхронизацией - атомарное выполнение
synchronized (lock) {
    double temp = balance;  // Чтение
    balance = temp + amount;  // Запись
    // Никакой другой поток не может войти, пока это не завершится
}
\`\`\`

**3. Надежность в продакшене:**
Критично для финансовых систем, e-commerce платформ и приложений реального времени. Банковские транзакции, обработка платежей и управление инвентарем требуют строгой потокобезопасности для предотвращения финансовых потерь и повреждения данных.

**Реальные примеры, где критична потокобезопасность:**
- **Банковские системы:** Несколько банкоматов получают доступ к одному счету
- **E-commerce:** Параллельные заказы ограниченного инвентаря
- **Торговля акциями:** Высокочастотные торговые системы
- **Игровые серверы:** Обновления состояния многопользовательской игры
- **Системы бронирования:** Конфликты бронирования отелей/авиабилетов

**4. Реальный Production пример:**
E-commerce сайт с параллельными заказами:
- **Без синхронизации:** Два клиента покупают последний товар → оба заказа успешны → перепродажа! Разочарование клиентов и потенциальные юридические проблемы.
- **С синхронизацией:** Второй клиент получает сообщение "out of stock" → точный инвентарь, довольные клиенты.

\`\`\`java
class Inventory {
    private int stock = 1;  // Последний товар
    private final Object lock = new Object();

    // Потокобезопасная покупка
    public boolean purchase() {
        synchronized (lock) {
            if (stock > 0) {
                stock--;
                return true;  // Покупка успешна
            }
            return false;  // Нет в наличии
        }
    }
}

// Клиент A и B нажимают "Купить" одновременно
// Только один преуспевает - инвентарь остается согласованным
\`\`\`

**5. Понимание критических секций:**
Критическая секция - это любой код, который обращается к общему изменяемому состоянию. Идентификация и защита критических секций - ключ к потокобезопасности.

\`\`\`java
class Counter {
    private int count = 0;  // Общее изменяемое состояние
    private final Object lock = new Object();

    public void increment() {
        // Критическая секция - должна быть синхронизирована
        synchronized (lock) {
            count++;  // Операция Read-Modify-Write
        }
    }

    public int getCount() {
        // Также нужна синхронизация для видимости
        synchronized (lock) {
            return count;
        }
    }
}
\`\`\`

**6. Распространенные баги потокобезопасности:**
\`\`\`java
// Баг #1: Несогласованная блокировка
public void methodA() {
    synchronized (lock) { balance++; }
}
public void methodB() {
    balance--;  // БАГ: Не синхронизировано!
}

// Баг #2: Разные объекты блокировки
public void methodC() {
    synchronized (lockA) { balance++; }
}
public void methodD() {
    synchronized (lockB) { balance--; }  // БАГ: Другая блокировка!
}

// Баг #3: Потерянные обновления в составных операциях
public void methodE() {
    synchronized (lock) {
        int current = getCount();  // Чтение
    }
    // Блокировка освобождена здесь!
    synchronized (lock) {
        setCount(current + 1);  // Запись - БАГ: Другой поток мог изменить между ними!
    }
}
\`\`\`

**7. Соображения производительности:**
Хотя синхронизация необходима, она может повлиять на производительность, если не использовать ее осторожно:

\`\`\`java
// Плохо: Блокировка удерживается слишком долго
public synchronized void processOrder(Order order) {
    validateOrder(order);  // Быстро
    saveToDatabase(order);  // МЕДЛЕННО - I/O операция
    sendEmail(order);  // МЕДЛЕННО - Сетевая операция
    // Другие потоки блокируются на всю длительность!
}

// Хорошо: Минимизировать область блокировки
public void processOrder(Order order) {
    validateOrder(order);  // Блокировка не нужна

    synchronized (lock) {
        // Синхронизировать только изменение данных
        orders.add(order);
        orderCount++;
    }

    saveToDatabase(order);  // Блокировка не нужна
    sendEmail(order);  // Блокировка не нужна
}
\`\`\`

**8. Альтернативные подходы:**
Java предоставляет утилиты параллелизма более высокого уровня, которые могут быть проще и эффективнее, чем ручная синхронизация:

\`\`\`java
// AtomicInteger для простых счетчиков
private AtomicInteger count = new AtomicInteger(0);
public void increment() {
    count.incrementAndGet();  // Потокобезопасно, без явной блокировки
}

// ConcurrentHashMap для потокобезопасных карт
private ConcurrentHashMap<String, Account> accounts = new ConcurrentHashMap<>();
public void updateAccount(String id, Account account) {
    accounts.put(id, account);  // Потокобезопасно
}

// ReadWriteLock для сценариев с интенсивным чтением
private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
public void write() {
    rwLock.writeLock().lock();
    try { /* изменить данные */ }
    finally { rwLock.writeLock().unlock(); }
}
public void read() {
    rwLock.readLock().lock();  // Разрешено несколько читателей
    try { /* читать данные */ }
    finally { rwLock.readLock().unlock(); }
}
\`\`\`

**Лучшие практики:**
1. Используйте один и тот же объект блокировки для связанных операций
2. Держите критические секции как можно короче
3. Избегайте удержания блокировок при выполнении I/O операций
4. Рассмотрите использование параллельных коллекций (ConcurrentHashMap, CopyOnWriteArrayList) когда применимо
5. Используйте synchronized методы для простых случаев
6. Тестируйте с несколькими потоками для выявления race conditions
7. Документируйте, какая блокировка защищает какие данные
8. Предпочитайте утилиты более высокого уровня (AtomicInteger, ConcurrentHashMap) ручной синхронизации, когда это возможно

**Реальное воздействие:**
Компания по обработке платежей имела баг race condition в обработке транзакций:
- **Баг:** Несколько потоков могли обработать один и тот же платеж дважды
- **Воздействие:** $2.3 миллиона дублированных начислений за 3 месяца
- **Исправление:** Добавлена правильная синхронизация в обработку платежей
- **Результат:** Ноль дублированных начислений, доверие клиентов восстановлено

**Тестирование потокобезопасности:**
\`\`\`java
// Стресс-тест со многими параллельными потоками
@Test
public void testThreadSafety() throws InterruptedException {
    BankAccount account = new BankAccount(1000);
    int numThreads = 100;
    Thread[] threads = new Thread[numThreads];

    // Каждый поток вносит $10
    for (int i = 0; i < numThreads; i++) {
        threads[i] = new Thread(() -> account.deposit(10));
        threads[i].start();
    }

    // Ждем все потоки
    for (Thread t : threads) t.join();

    // Должно быть 1000 + (100 * 10) = 2000
    assertEquals(2000.0, account.getBalance(), 0.01);
}
\`\`\`

Потокобезопасность - это не опция в production системах - это необходимость для корректности, надежности и поддержания доверия клиентов.`,
			solutionCode: `class BankAccount {
    private double balance;
    private final Object lock = new Object();

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    // Потокобезопасный метод депозита
    public void deposit(double amount) {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " вносит $" + amount);
            double temp = balance;

            // Имитируем время обработки - критическая секция
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            balance = temp + amount;
            System.out.println(Thread.currentThread().getName() + " новый баланс: $" + balance);
        }
    }

    // Потокобезопасный метод снятия
    public void withdraw(double amount) {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " снимает $" + amount);
            double temp = balance;

            // Имитируем время обработки - критическая секция
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if (temp >= amount) {
                balance = temp - amount;
                System.out.println(Thread.currentThread().getName() + " новый баланс: $" + balance);
            } else {
                System.out.println(Thread.currentThread().getName() + " недостаточно средств!");
            }
        }
    }

    // Потокобезопасный геттер
    public synchronized double getBalance() {
        return balance;
    }
}

public class ThreadSafety {
    public static void main(String[] args) throws InterruptedException {
        BankAccount account = new BankAccount(1000.0);
        System.out.println("Начальный баланс: $" + account.getBalance());

        // Создаем 5 потоков, которые вносят по $100 каждый
        Thread[] depositors = new Thread[5];
        for (int i = 0; i < 5; i++) {
            final int threadNum = i + 1;
            depositors[i] = new Thread(() -> {
                account.deposit(100.0);
            }, "Depositor-" + threadNum);
        }

        // Создаем 5 потоков, которые снимают по $50 каждый
        Thread[] withdrawers = new Thread[5];
        for (int i = 0; i < 5; i++) {
            final int threadNum = i + 1;
            withdrawers[i] = new Thread(() -> {
                account.withdraw(50.0);
            }, "Withdrawer-" + threadNum);
        }

        // Запускаем все потоки
        for (Thread t : depositors) t.start();
        for (Thread t : withdrawers) t.start();

        // Ожидаем завершения всех потоков
        for (Thread t : depositors) t.join();
        for (Thread t : withdrawers) t.join();

        System.out.println("\\nИтоговый баланс: $" + account.getBalance());
        System.out.println("Ожидаемое значение: $1250.0 (1000 + 500 - 250)");
    }
}`
		},
		uz: {
			title: 'Oqim xavfsizligi',
			description: `Ma'lumotlar poygasi, muhim qismlar va oqim xavfsiz kodni qanday yozishni tushunish.

**Talablar:**
1. Oqim xavfsiz BankAccount klassini amalga oshiring
   1.1. deposit() metodini synchronized bilan oqim xavfsiz qiling
   1.2. withdraw() metodini synchronized bilan oqim xavfsiz qiling
   1.3. getBalance() metodini oqim xavfsiz qiling
2. Test uchun bir nechta oqim yarating
   2.1. Har biri $100 kiritadigan 5 ta oqim yarating
   2.2. Har biri $50 yechib oladigan 5 ta oqim yarating
3. Bajaring va tekshiring
   3.1. Barcha oqimlarni ishga tushiring
   3.2. join() yordamida barcha oqimlarning tugashini kuting
   3.3. Yakuniy balans $1250.0 ekanligini tekshiring`,
			hint1: `Ma'lumotlar poygasi bir nechta oqimlar umumiy ma'lumotlarga bir vaqtning o'zida kirganida yuzaga keladi. Muhim qism - bu umumiy resurslarga kiradigan kod.`,
			hint2: `Muhim qismlarni himoya qilish uchun sinxronlashtirilgan bloklardan foydalaning. Umumiy holatga kiradigan barcha metodlar sinxronlashtirilganligiga ishonch hosil qiling.`,
			whyItMatters: `Oqim xavfsizligi ishonchli parallel ilovalar yaratish uchun asosdir.

**Nima uchun oqim xavfsizligi muhim:**

**1. Ma'lumotlar yaxlitligi:**
Race conditionlardan buzilgan ma'lumotlarning oldini oladi. Sinxronizatsiyasiz bir nechta oqimlar bir vaqtning o'zida o'qishi va yozishi mumkin, bu yangilanishlarni yo'qotish va izchil bo'lmagan holatga olib keladi.

\`\`\`java
// Sinxronizatsiyasiz:
Thread A balance ni o'qiydi: 1000
Thread B balance ni o'qiydi: 1000
Thread A 100 kiritadi: 1000 + 100 = 1100
Thread B 100 kiritadi: 1000 + 100 = 1100  // Yo'qolgan yangilanish!
// Kutilgan: 1200, Olindi: 1100
\`\`\`

**2. Oldindan aytish mumkin bo'lgan xatti-harakat:**
Oqim rejalashtirish qanday bo'lishidan qat'i nazar izchil natijalarni ta'minlaydi. Sinxronizatsiya kritik qismlarning atomik bajarilishini kafolatlaydi - bir oqim sinxronlashtirilgan blokni bajarayotganda boshqa hech bir oqim aralasha olmaydi.

\`\`\`java
// Sinxronizatsiya bilan - atomik bajarilish
synchronized (lock) {
    double temp = balance;  // O'qish
    balance = temp + amount;  // Yozish
    // Boshqa hech bir oqim bu tugamaguncha kira olmaydi
}
\`\`\`

**3. Production da ishonchlilik:**
Moliyaviy tizimlar, e-commerce platformalar va real-time ilovalar uchun muhim. Bank tranzaksiyalari, to'lov qayta ishlash va inventar boshqaruvi moliyaviy yo'qotishlar va ma'lumotlar buzilishining oldini olish uchun qat'iy oqim xavfsizligini talab qiladi.

**Oqim xavfsizligi muhim bo'lgan real misollar:**
- **Bank tizimlari:** Bir hisobga bir nechta bankomat kirishadi
- **E-commerce:** Cheklangan inventar uchun parallel buyurtmalar
- **Aksiya savdosi:** Yuqori chastotali savdo tizimlari
- **O'yin serverlari:** Ko'p o'yinchi o'yin holati yangilanishlari
- **Bron qilish tizimlari:** Mehmonxona/aviachiptalarni bron qilish ziddiyatlari

**4. Real Production misol:**
E-commerce sayti parallel buyurtmalar bilan:
- **Sinxronizatsiyasiz:** Ikki mijoz oxirgi mahsulotni sotib oladi → ikkala buyurtma ham muvaffaqiyatli → ortiqcha sotish! Mijozlar noroziligi va potentsial huquqiy muammolar.
- **Sinxronizatsiya bilan:** Ikkinchi mijoz "out of stock" xabarini oladi → aniq inventar, xursand mijozlar.

\`\`\`java
class Inventory {
    private int stock = 1;  // Oxirgi mahsulot
    private final Object lock = new Object();

    // Oqim xavfsiz xarid
    public boolean purchase() {
        synchronized (lock) {
            if (stock > 0) {
                stock--;
                return true;  // Xarid muvaffaqiyatli
            }
            return false;  // Omborda yo'q
        }
    }
}

// Mijoz A va B bir vaqtning o'zida "Sotib olish" ni bosadi
// Faqat bittasi muvaffaqiyatli bo'ladi - inventar izchil qoladi
\`\`\`

**5. Kritik qismlarni tushunish:**
Kritik qism - bu umumiy o'zgaruvchan holatga kiradigan har qanday kod. Kritik qismlarni aniqlash va himoya qilish oqim xavfsizligining kalitidir.

\`\`\`java
class Counter {
    private int count = 0;  // Umumiy o'zgaruvchan holat
    private final Object lock = new Object();

    public void increment() {
        // Kritik qism - sinxronlashtirilishi kerak
        synchronized (lock) {
            count++;  // Read-Modify-Write operatsiyasi
        }
    }

    public int getCount() {
        // Ko'rinish uchun sinxronizatsiya ham kerak
        synchronized (lock) {
            return count;
        }
    }
}
\`\`\`

**6. Keng tarqalgan oqim xavfsizligi xatolari:**
\`\`\`java
// Xato #1: Izchil bo'lmagan blokirovka
public void methodA() {
    synchronized (lock) { balance++; }
}
public void methodB() {
    balance--;  // XATO: Sinxronlashtirilmagan!
}

// Xato #2: Turli lock obyektlari
public void methodC() {
    synchronized (lockA) { balance++; }
}
public void methodD() {
    synchronized (lockB) { balance--; }  // XATO: Boshqa lock!
}

// Xato #3: Murakkab operatsiyalarda yo'qolgan yangilanishlar
public void methodE() {
    synchronized (lock) {
        int current = getCount();  // O'qish
    }
    // Lock bu yerda ozod qilinadi!
    synchronized (lock) {
        setCount(current + 1);  // Yozish - XATO: Boshqa oqim ular o'rtasida o'zgartirishi mumkin!
    }
}
\`\`\`

**7. Ishlash ko'rsatkichlari:**
Sinxronizatsiya zarur bo'lsa-da, ehtiyot bo'lmasangiz ishlashga ta'sir qilishi mumkin:

\`\`\`java
// Yomon: Lock juda uzoq ushlab turiladi
public synchronized void processOrder(Order order) {
    validateOrder(order);  // Tez
    saveToDatabase(order);  // SEKIN - I/O operatsiyasi
    sendEmail(order);  // SEKIN - Tarmoq operatsiyasi
    // Boshqa oqimlar butun davomiylik uchun bloklangan!
}

// Yaxshi: Lock doirasini minimallashtirish
public void processOrder(Order order) {
    validateOrder(order);  // Lock kerak emas

    synchronized (lock) {
        // Faqat ma'lumotlar o'zgarishini sinxronlashtirish
        orders.add(order);
        orderCount++;
    }

    saveToDatabase(order);  // Lock kerak emas
    sendEmail(order);  // Lock kerak emas
}
\`\`\`

**8. Muqobil yondashuvlar:**
Java qo'lda sinxronizatsiyadan ko'ra osonroq va samaraliroq bo'lishi mumkin bo'lgan yuqori darajadagi parallellik vositalarini taqdim etadi:

\`\`\`java
// Oddiy hisoblagichlar uchun AtomicInteger
private AtomicInteger count = new AtomicInteger(0);
public void increment() {
    count.incrementAndGet();  // Oqim xavfsiz, aniq lock yo'q
}

// Oqim xavfsiz maplar uchun ConcurrentHashMap
private ConcurrentHashMap<String, Account> accounts = new ConcurrentHashMap<>();
public void updateAccount(String id, Account account) {
    accounts.put(id, account);  // Oqim xavfsiz
}

// O'qishga yo'naltirilgan stsenariylarda ReadWriteLock
private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
public void write() {
    rwLock.writeLock().lock();
    try { /* ma'lumotlarni o'zgartirish */ }
    finally { rwLock.writeLock().unlock(); }
}
public void read() {
    rwLock.readLock().lock();  // Bir nechta o'quvchilarga ruxsat berilgan
    try { /* ma'lumotlarni o'qish */ }
    finally { rwLock.readLock().unlock(); }
}
\`\`\`

**Eng yaxshi amaliyotlar:**
1. Bog'liq operatsiyalar uchun bir xil lock obyektidan foydalaning
2. Kritik qismlarni iloji boricha qisqa saqlang
3. I/O operatsiyalarini bajarish paytida lockni ushlab turishdan saqlaning
4. Qo'llanilsa parallel kolleksiyalardan (ConcurrentHashMap, CopyOnWriteArrayList) foydalanishni ko'rib chiqing
5. Oddiy holatlar uchun synchronized metodlardan foydalaning
6. Race conditionlarni aniqlash uchun bir nechta oqimlar bilan test qiling
7. Qaysi lock qaysi ma'lumotlarni himoya qilishini hujjatlashtiring
8. Mumkin bo'lganda qo'lda sinxronizatsiyaga nisbatan yuqori darajadagi vositalarni (AtomicInteger, ConcurrentHashMap) afzal ko'ring

**Real ta'sir:**
To'lovlarni qayta ishlovchi kompaniyada tranzaksiyalarni qayta ishlashda race condition xatosi bo'lgan:
- **Xato:** Bir nechta oqimlar bir xil to'lovni ikki marta qayta ishlashi mumkin edi
- **Ta'siri:** 3 oy ichida $2.3 million ikki marta to'lov
- **Tuzatish:** To'lovlarni qayta ishlashga to'g'ri sinxronizatsiya qo'shildi
- **Natija:** Nol ikki marta to'lov, mijozlar ishonchi tiklandi

**Oqim xavfsizligini test qilish:**
\`\`\`java
// Ko'p parallel oqimlar bilan stress test
@Test
public void testThreadSafety() throws InterruptedException {
    BankAccount account = new BankAccount(1000);
    int numThreads = 100;
    Thread[] threads = new Thread[numThreads];

    // Har bir oqim $10 kiritadi
    for (int i = 0; i < numThreads; i++) {
        threads[i] = new Thread(() -> account.deposit(10));
        threads[i].start();
    }

    // Barcha oqimlarni kutish
    for (Thread t : threads) t.join();

    // 1000 + (100 * 10) = 2000 bo'lishi kerak
    assertEquals(2000.0, account.getBalance(), 0.01);
}
\`\`\`

Oqim xavfsizligi production tizimlarida ixtiyoriy emas - bu to'g'rilik, ishonchlilik va mijozlar ishonchini saqlash uchun zarur.`,
			solutionCode: `class BankAccount {
    private double balance;
    private final Object lock = new Object();

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    // Oqim xavfsiz depozit metodi
    public void deposit(double amount) {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " $" + amount + " kiritmoqda");
            double temp = balance;

            // Qayta ishlash vaqtini taqlid qilamiz - muhim qism
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            balance = temp + amount;
            System.out.println(Thread.currentThread().getName() + " yangi balans: $" + balance);
        }
    }

    // Oqim xavfsiz yechib olish metodi
    public void withdraw(double amount) {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + " $" + amount + " yechib olmoqda");
            double temp = balance;

            // Qayta ishlash vaqtini taqlid qilamiz - muhim qism
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if (temp >= amount) {
                balance = temp - amount;
                System.out.println(Thread.currentThread().getName() + " yangi balans: $" + balance);
            } else {
                System.out.println(Thread.currentThread().getName() + " mablag' yetarli emas!");
            }
        }
    }

    // Oqim xavfsiz getter
    public synchronized double getBalance() {
        return balance;
    }
}

public class ThreadSafety {
    public static void main(String[] args) throws InterruptedException {
        BankAccount account = new BankAccount(1000.0);
        System.out.println("Boshlang'ich balans: $" + account.getBalance());

        // Har biri $100 kiritadigan 5 ta oqim yaratamiz
        Thread[] depositors = new Thread[5];
        for (int i = 0; i < 5; i++) {
            final int threadNum = i + 1;
            depositors[i] = new Thread(() -> {
                account.deposit(100.0);
            }, "Depositor-" + threadNum);
        }

        // Har biri $50 yechib oladigan 5 ta oqim yaratamiz
        Thread[] withdrawers = new Thread[5];
        for (int i = 0; i < 5; i++) {
            final int threadNum = i + 1;
            withdrawers[i] = new Thread(() -> {
                account.withdraw(50.0);
            }, "Withdrawer-" + threadNum);
        }

        // Barcha oqimlarni ishga tushiramiz
        for (Thread t : depositors) t.start();
        for (Thread t : withdrawers) t.start();

        // Barcha oqimlarning tugashini kutamiz
        for (Thread t : depositors) t.join();
        for (Thread t : withdrawers) t.join();

        System.out.println("\\nYakuniy balans: $" + account.getBalance());
        System.out.println("Kutilgan qiymat: $1250.0 (1000 + 500 - 250)");
    }
}`
		}
	}
};

export default task;
