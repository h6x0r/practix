import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-atomic-reference',
    title: 'AtomicReference for Objects',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'atomic', 'atomicreference'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn AtomicReference for thread-safe object references.

**Requirements:**
1. Create a User class with name and age fields
2. Create an AtomicReference<User> initialized with User("Alice", 25)
3. Print the initial user
4. Use set() to update to User("Bob", 30)
5. Use getAndSet() to change to User("Charlie", 35) and print the old value
6. Use compareAndSet() to change Charlie to David (should succeed)
7. Try compareAndSet() with wrong expected value (should fail)
8. Use updateAndGet() with a lambda to increment age by 1

AtomicReference provides atomic operations on object references, perfect for safely updating shared objects in concurrent environments.`,
    initialCode: `import java.util.concurrent.atomic.AtomicReference;

class User {
    private String name;
    private int age;

    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return name + " (" + age + ")";
    }
}

public class AtomicReferenceDemo {
    public static void main(String[] args) {
        // Create AtomicReference with User("Alice", 25)

        // Print initial user

        // Use set() to update to User("Bob", 30)

        // Use getAndSet() to change to User("Charlie", 35)

        // Use compareAndSet() to change Charlie to David

        // Try compareAndSet() with wrong expected value

        // Use updateAndGet() to increment age by 1
    }
}`,
    solutionCode: `import java.util.concurrent.atomic.AtomicReference;

class User {
    private String name;
    private int age;

    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public User withAge(int newAge) {
        return new User(this.name, newAge);
    }

    @Override
    public String toString() {
        return name + " (" + age + ")";
    }
}

public class AtomicReferenceDemo {
    public static void main(String[] args) {
        // Create AtomicReference with User("Alice", 25)
        AtomicReference<User> userRef = new AtomicReference<>(new User("Alice", 25));

        // Print initial user
        System.out.println("Initial user: " + userRef.get());

        // Use set() to update to User("Bob", 30)
        userRef.set(new User("Bob", 30));
        System.out.println("After set: " + userRef.get());

        // Use getAndSet() to change to User("Charlie", 35)
        User oldUser = userRef.getAndSet(new User("Charlie", 35));
        System.out.println("getAndSet returned: " + oldUser);
        System.out.println("Current user: " + userRef.get());

        // Use compareAndSet() to change Charlie to David
        User charlie = userRef.get();
        boolean success1 = userRef.compareAndSet(charlie, new User("David", 40));
        System.out.println("compareAndSet Charlie->David success: " + success1);
        System.out.println("Current user: " + userRef.get());

        // Try compareAndSet() with wrong expected value
        boolean success2 = userRef.compareAndSet(charlie, new User("Eve", 45));
        System.out.println("compareAndSet with wrong expected success: " + success2);

        // Use updateAndGet() to increment age by 1
        User updated = userRef.updateAndGet(user -> user.withAge(user.getAge() + 1));
        System.out.println("After updateAndGet (age+1): " + updated);
    }
}`,
    hint1: `AtomicReference works with object references. Remember that compareAndSet() compares references, not object content.`,
    hint2: `updateAndGet() takes a UnaryOperator<T> lambda that receives the current value and returns the new value.`,
    whyItMatters: `AtomicReference enables lock-free updates of object references, essential for implementing thread-safe data structures and managing shared state in concurrent applications.

**Production Pattern:**
\`\`\`java
// Thread-safe configuration management
public class ConfigManager {
    private final AtomicReference<AppConfig> config;

    public ConfigManager(AppConfig initial) {
        this.config = new AtomicReference<>(initial);
    }

    public void updateConfig(AppConfig newConfig) {
        config.set(newConfig);
    }

    public AppConfig getConfig() {
        return config.get();
    }

    // Safe update with validation
    public boolean updateIfChanged(AppConfig expected, AppConfig updated) {
        return config.compareAndSet(expected, updated);
    }
}
\`\`\`

**Practical Benefits:**
- Hot-swapping configuration without restart
- Atomic updates without locks
- Minimal latency when reading configuration`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.atomic.AtomicReference;

// Test1: Test initial value
class Test1 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("Initial");
        assertEquals("Initial", ref.get());
    }
}

// Test2: Test set method
class Test2 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("Old");
        ref.set("New");
        assertEquals("New", ref.get());
    }
}

// Test3: Test getAndSet
class Test3 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("Old");
        assertEquals("Old", ref.getAndSet("New"));
        assertEquals("New", ref.get());
    }
}

// Test4: Test compareAndSet success
class Test4 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("Expected");
        assertTrue(ref.compareAndSet("Expected", "Updated"));
        assertEquals("Updated", ref.get());
    }
}

// Test5: Test compareAndSet failure
class Test5 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("Current");
        assertFalse(ref.compareAndSet("Wrong", "Updated"));
        assertEquals("Current", ref.get());
    }
}

// Test6: Test with custom object
class Test6 {
    @Test
    public void test() {
        class User {
            String name;
            User(String name) { this.name = name; }
        }
        AtomicReference<User> ref = new AtomicReference<>(new User("Alice"));
        assertEquals("Alice", ref.get().name);
    }
}

// Test7: Test updateAndGet
class Test7 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("hello");
        assertEquals("HELLO", ref.updateAndGet(String::toUpperCase));
        assertEquals("HELLO", ref.get());
    }
}

// Test8: Test getAndUpdate
class Test8 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("hello");
        assertEquals("hello", ref.getAndUpdate(String::toUpperCase));
        assertEquals("HELLO", ref.get());
    }
}

// Test9: Test null reference
class Test9 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>(null);
        assertNull(ref.get());
        ref.set("NotNull");
        assertNotNull(ref.get());
    }
}

// Test10: Test accumulateAndGet
class Test10 {
    @Test
    public void test() {
        AtomicReference<String> ref = new AtomicReference<>("Hello");
        assertEquals("Hello World", ref.accumulateAndGet(" World", (a, b) -> a + b));
        assertEquals("Hello World", ref.get());
    }
}
`,
    translations: {
        ru: {
            title: 'AtomicReference для объектов',
            solutionCode: `import java.util.concurrent.atomic.AtomicReference;

class User {
    private String name;
    private int age;

    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public User withAge(int newAge) {
        return new User(this.name, newAge);
    }

    @Override
    public String toString() {
        return name + " (" + age + ")";
    }
}

public class AtomicReferenceDemo {
    public static void main(String[] args) {
        // Создаем AtomicReference с User("Alice", 25)
        AtomicReference<User> userRef = new AtomicReference<>(new User("Alice", 25));

        // Выводим начального пользователя
        System.out.println("Начальный пользователь: " + userRef.get());

        // Используем set() для обновления на User("Bob", 30)
        userRef.set(new User("Bob", 30));
        System.out.println("После set: " + userRef.get());

        // Используем getAndSet() для изменения на User("Charlie", 35)
        User oldUser = userRef.getAndSet(new User("Charlie", 35));
        System.out.println("getAndSet вернул: " + oldUser);
        System.out.println("Текущий пользователь: " + userRef.get());

        // Используем compareAndSet() для изменения Charlie на David
        User charlie = userRef.get();
        boolean success1 = userRef.compareAndSet(charlie, new User("David", 40));
        System.out.println("compareAndSet Charlie->David успех: " + success1);
        System.out.println("Текущий пользователь: " + userRef.get());

        // Пробуем compareAndSet() с неправильным ожидаемым значением
        boolean success2 = userRef.compareAndSet(charlie, new User("Eve", 45));
        System.out.println("compareAndSet с неправильным ожидаемым успех: " + success2);

        // Используем updateAndGet() для увеличения возраста на 1
        User updated = userRef.updateAndGet(user -> user.withAge(user.getAge() + 1));
        System.out.println("После updateAndGet (возраст+1): " + updated);
    }
}`,
            description: `Изучите AtomicReference для потокобезопасных ссылок на объекты.

**Требования:**
1. Создайте класс User с полями name и age
2. Создайте AtomicReference<User>, инициализированный User("Alice", 25)
3. Выведите начального пользователя
4. Используйте set() для обновления на User("Bob", 30)
5. Используйте getAndSet() для изменения на User("Charlie", 35) и выведите старое значение
6. Используйте compareAndSet() для изменения Charlie на David (должно успешно выполниться)
7. Попробуйте compareAndSet() с неправильным ожидаемым значением (должно не выполниться)
8. Используйте updateAndGet() с лямбдой для увеличения возраста на 1

AtomicReference предоставляет атомарные операции над ссылками на объекты, идеально подходит для безопасного обновления разделяемых объектов в многопоточных средах.`,
            hint1: `AtomicReference работает со ссылками на объекты. Помните, что compareAndSet() сравнивает ссылки, а не содержимое объектов.`,
            hint2: `updateAndGet() принимает лямбду UnaryOperator<T>, которая получает текущее значение и возвращает новое значение.`,
            whyItMatters: `AtomicReference обеспечивает неблокирующие обновления ссылок на объекты, что необходимо для реализации потокобезопасных структур данных и управления разделяемым состоянием в многопоточных приложениях.

**Продакшен паттерн:**
\`\`\`java
// Потокобезопасное управление конфигурацией
public class ConfigManager {
    private final AtomicReference<AppConfig> config;

    public ConfigManager(AppConfig initial) {
        this.config = new AtomicReference<>(initial);
    }

    public void updateConfig(AppConfig newConfig) {
        config.set(newConfig);
    }

    public AppConfig getConfig() {
        return config.get();
    }

    // Безопасное обновление с проверкой
    public boolean updateIfChanged(AppConfig expected, AppConfig updated) {
        return config.compareAndSet(expected, updated);
    }
}
\`\`\`

**Практические преимущества:**
- Горячая замена конфигурации без перезагрузки
- Атомарность обновлений без блокировок
- Минимальная задержка при чтении конфигурации`
        },
        uz: {
            title: 'Obyektlar uchun AtomicReference',
            solutionCode: `import java.util.concurrent.atomic.AtomicReference;

class User {
    private String name;
    private int age;

    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public User withAge(int newAge) {
        return new User(this.name, newAge);
    }

    @Override
    public String toString() {
        return name + " (" + age + ")";
    }
}

public class AtomicReferenceDemo {
    public static void main(String[] args) {
        // User("Alice", 25) bilan AtomicReference yaratamiz
        AtomicReference<User> userRef = new AtomicReference<>(new User("Alice", 25));

        // Boshlang'ich foydalanuvchini chiqaramiz
        System.out.println("Boshlang'ich foydalanuvchi: " + userRef.get());

        // User("Bob", 30) ga yangilash uchun set() dan foydalanamiz
        userRef.set(new User("Bob", 30));
        System.out.println("set dan keyin: " + userRef.get());

        // User("Charlie", 35) ga o'zgartirish uchun getAndSet() dan foydalanamiz
        User oldUser = userRef.getAndSet(new User("Charlie", 35));
        System.out.println("getAndSet qaytardi: " + oldUser);
        System.out.println("Hozirgi foydalanuvchi: " + userRef.get());

        // Charlie ni David ga o'zgartirish uchun compareAndSet() dan foydalanamiz
        User charlie = userRef.get();
        boolean success1 = userRef.compareAndSet(charlie, new User("David", 40));
        System.out.println("compareAndSet Charlie->David muvaffaqiyat: " + success1);
        System.out.println("Hozirgi foydalanuvchi: " + userRef.get());

        // Noto'g'ri kutilgan qiymat bilan compareAndSet() ni sinab ko'ramiz
        boolean success2 = userRef.compareAndSet(charlie, new User("Eve", 45));
        System.out.println("Noto'g'ri kutilgan bilan compareAndSet muvaffaqiyat: " + success2);

        // Yoshni 1 ga oshirish uchun updateAndGet() dan foydalanamiz
        User updated = userRef.updateAndGet(user -> user.withAge(user.getAge() + 1));
        System.out.println("updateAndGet dan keyin (yosh+1): " + updated);
    }
}`,
            description: `Thread-xavfsiz obyekt havolalari uchun AtomicReference ni o'rganing.

**Talablar:**
1. name va age maydonlari bilan User klassi yarating
2. User("Alice", 25) bilan ishga tushirilgan AtomicReference<User> yarating
3. Boshlang'ich foydalanuvchini chiqaring
4. User("Bob", 30) ga yangilash uchun set() dan foydalaning
5. User("Charlie", 35) ga o'zgartirish uchun getAndSet() dan foydalaning va eski qiymatni chiqaring
6. Charlie ni David ga o'zgartirish uchun compareAndSet() dan foydalaning (muvaffaqiyatli bo'lishi kerak)
7. Noto'g'ri kutilgan qiymat bilan compareAndSet() ni sinab ko'ring (muvaffaqiyatsiz bo'lishi kerak)
8. Yoshni 1 ga oshirish uchun lambda bilan updateAndGet() dan foydalaning

AtomicReference obyekt havolalari ustida atomik operatsiyalarni taqdim etadi, concurrent muhitlarda umumiy obyektlarni xavfsiz yangilash uchun ideal.`,
            hint1: `AtomicReference obyekt havolalari bilan ishlaydi. compareAndSet() obyekt mazmunini emas, havolalarni solishtirishini unutmang.`,
            hint2: `updateAndGet() hozirgi qiymatni qabul qiladigan va yangi qiymatni qaytaradigan UnaryOperator<T> lambda qabul qiladi.`,
            whyItMatters: `AtomicReference obyekt havolalarining lock-free yangilanishlarini ta'minlaydi, bu thread-xavfsiz ma'lumot strukturalarini amalga oshirish va concurrent ilovalarda umumiy holatni boshqarish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Thread-xavfsiz konfiguratsiya boshqaruvi
public class ConfigManager {
    private final AtomicReference<AppConfig> config;

    public ConfigManager(AppConfig initial) {
        this.config = new AtomicReference<>(initial);
    }

    public void updateConfig(AppConfig newConfig) {
        config.set(newConfig);
    }

    public AppConfig getConfig() {
        return config.get();
    }

    // Tekshirish bilan xavfsiz yangilash
    public boolean updateIfChanged(AppConfig expected, AppConfig updated) {
        return config.compareAndSet(expected, updated);
    }
}
\`\`\`

**Amaliy foydalari:**
- Qayta yuklashsiz issiq konfiguratsiya almashtirish
- Qulflarsiz yangilanishlarning atomikligi
- Konfiguratsiyani o'qishda minimal kechikish`
        }
    }
};

export default task;
