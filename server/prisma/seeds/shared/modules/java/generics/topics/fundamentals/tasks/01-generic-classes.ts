import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-generic-classes',
    title: 'Generic Classes and Type Parameters',
    difficulty: 'easy',
    tags: ['java', 'generics', 'type-parameters', 'classes'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to create and use generic classes in Java.

**Requirements:**
1. Create a generic Box<T> class that can hold any type of object
2. Add a constructor that accepts a T value
3. Add a get() method to retrieve the value
4. Add a set() method to update the value
5. Create a Pair<K, V> class with two type parameters for key and value
6. Demonstrate creating instances with different types: Box<String>, Box<Integer>, Pair<String, Integer>

Generic classes use type parameters (like T, K, V) to create reusable code that works with any object type while maintaining type safety.`,
    initialCode: `public class GenericClasses {
    // Create a generic Box<T> class
    // - Constructor that accepts T value
    // - get() method
    // - set() method

    // Create a Pair<K, V> class
    // - Constructor that accepts K key and V value
    // - getKey() and getValue() methods

    public static void main(String[] args) {
        // Create Box<String>

        // Create Box<Integer>

        // Create Pair<String, Integer>

        // Print all values
    }
}`,
    solutionCode: `public class GenericClasses {
    // Generic Box class with single type parameter
    static class Box<T> {
        private T value;

        // Constructor accepts generic type T
        public Box(T value) {
            this.value = value;
        }

        // Getter returns type T
        public T get() {
            return value;
        }

        // Setter accepts type T
        public void set(T value) {
            this.value = value;
        }
    }

    // Generic Pair class with two type parameters
    static class Pair<K, V> {
        private K key;
        private V value;

        // Constructor accepts both generic types
        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }

    public static void main(String[] args) {
        // Create Box with String type
        Box<String> stringBox = new Box<>("Hello Generics");
        System.out.println("String box: " + stringBox.get());

        // Create Box with Integer type
        Box<Integer> intBox = new Box<>(42);
        System.out.println("Integer box: " + intBox.get());

        // Update the value
        intBox.set(100);
        System.out.println("Updated integer box: " + intBox.get());

        // Create Pair with String key and Integer value
        Pair<String, Integer> pair = new Pair<>("Age", 25);
        System.out.println("Pair: " + pair.getKey() + " = " + pair.getValue());

        // Type safety: These would cause compile-time errors
        // stringBox.set(123);	// Error: incompatible types
        // intBox.set("text");	// Error: incompatible types
    }
}`,
    hint1: `Define a generic class using angle brackets after the class name: class Box<T> { ... }. T is a placeholder for any type.`,
    hint2: `For multiple type parameters, separate them with commas: class Pair<K, V> { ... }. Common conventions: T=Type, K=Key, V=Value, E=Element.`,
    whyItMatters: `Generic classes are fundamental to Java's type system. They enable you to write reusable, type-safe code that works with any object type.

**Production Pattern:**
\`\`\`java
// Reusable container for any type
class Box<T> {
    private T value;

    public T get() { return value; }
    public void set(T value) { this.value = value; }
}

// Production usage
Box<String> stringBox = new Box<>("Hello Generics");
Box<Integer> intBox = new Box<>(42);

// Compile-time type safety
// stringBox.set(123);	// Error: incompatible types
\`\`\`

**Practical Benefits:**
- Compile-time type safety
- Eliminates need for type casting
- Code reuse
- Collections like ArrayList<T>, HashMap<K,V>, and Optional<T> use generics`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify Box<String> creation and get method
class Test1 {
    @Test
    public void test() {
        GenericClasses.Box<String> box = new GenericClasses.Box<>("Hello");
        assertEquals("Hello", box.get());
    }
}

// Test2: Verify Box<Integer> creation and get method
class Test2 {
    @Test
    public void test() {
        GenericClasses.Box<Integer> box = new GenericClasses.Box<>(42);
        assertEquals(Integer.valueOf(42), box.get());
    }
}

// Test3: Verify Box set method updates value
class Test3 {
    @Test
    public void test() {
        GenericClasses.Box<String> box = new GenericClasses.Box<>("Initial");
        box.set("Updated");
        assertEquals("Updated", box.get());
    }
}

// Test4: Verify Pair<String, Integer> creation
class Test4 {
    @Test
    public void test() {
        GenericClasses.Pair<String, Integer> pair = new GenericClasses.Pair<>("Age", 25);
        assertEquals("Age", pair.getKey());
        assertEquals(Integer.valueOf(25), pair.getValue());
    }
}

// Test5: Verify Box with Double type
class Test5 {
    @Test
    public void test() {
        GenericClasses.Box<Double> box = new GenericClasses.Box<>(3.14);
        assertEquals(Double.valueOf(3.14), box.get());
    }
}

// Test6: Verify multiple set operations on Box
class Test6 {
    @Test
    public void test() {
        GenericClasses.Box<Integer> box = new GenericClasses.Box<>(10);
        box.set(20);
        box.set(30);
        assertEquals(Integer.valueOf(30), box.get());
    }
}

// Test7: Verify Pair with different types
class Test7 {
    @Test
    public void test() {
        GenericClasses.Pair<String, String> pair = new GenericClasses.Pair<>("Name", "John");
        assertEquals("Name", pair.getKey());
        assertEquals("John", pair.getValue());
    }
}

// Test8: Verify Box with Boolean type
class Test8 {
    @Test
    public void test() {
        GenericClasses.Box<Boolean> box = new GenericClasses.Box<>(true);
        assertTrue(box.get());
    }
}

// Test9: Verify Pair with Integer and Double
class Test9 {
    @Test
    public void test() {
        GenericClasses.Pair<Integer, Double> pair = new GenericClasses.Pair<>(100, 99.99);
        assertEquals(Integer.valueOf(100), pair.getKey());
        assertEquals(Double.valueOf(99.99), pair.getValue());
    }
}

// Test10: Verify Box with null value
class Test10 {
    @Test
    public void test() {
        GenericClasses.Box<String> box = new GenericClasses.Box<>(null);
        assertNull(box.get());
    }
}`,
    translations: {
        ru: {
            title: 'Обобщенные классы и параметры типа',
            solutionCode: `public class GenericClasses {
    // Обобщенный класс Box с одним параметром типа
    static class Box<T> {
        private T value;

        // Конструктор принимает обобщенный тип T
        public Box(T value) {
            this.value = value;
        }

        // Геттер возвращает тип T
        public T get() {
            return value;
        }

        // Сеттер принимает тип T
        public void set(T value) {
            this.value = value;
        }
    }

    // Обобщенный класс Pair с двумя параметрами типа
    static class Pair<K, V> {
        private K key;
        private V value;

        // Конструктор принимает оба обобщенных типа
        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }

    public static void main(String[] args) {
        // Создаем Box с типом String
        Box<String> stringBox = new Box<>("Hello Generics");
        System.out.println("String box: " + stringBox.get());

        // Создаем Box с типом Integer
        Box<Integer> intBox = new Box<>(42);
        System.out.println("Integer box: " + intBox.get());

        // Обновляем значение
        intBox.set(100);
        System.out.println("Обновленный integer box: " + intBox.get());

        // Создаем Pair с ключом String и значением Integer
        Pair<String, Integer> pair = new Pair<>("Age", 25);
        System.out.println("Pair: " + pair.getKey() + " = " + pair.getValue());

        // Безопасность типов: эти строки вызовут ошибки компиляции
        // stringBox.set(123);	// Ошибка: несовместимые типы
        // intBox.set("text");	// Ошибка: несовместимые типы
    }
}`,
            description: `Изучите создание и использование обобщенных классов в Java.

**Требования:**
1. Создайте обобщенный класс Box<T>, который может хранить объект любого типа
2. Добавьте конструктор, принимающий значение типа T
3. Добавьте метод get() для получения значения
4. Добавьте метод set() для обновления значения
5. Создайте класс Pair<K, V> с двумя параметрами типа для ключа и значения
6. Продемонстрируйте создание экземпляров с разными типами: Box<String>, Box<Integer>, Pair<String, Integer>

Обобщенные классы используют параметры типа (например, T, K, V) для создания повторно используемого кода, который работает с любым типом объекта, сохраняя при этом безопасность типов.`,
            hint1: `Определите обобщенный класс, используя угловые скобки после имени класса: class Box<T> { ... }. T - это заполнитель для любого типа.`,
            hint2: `Для нескольких параметров типа разделяйте их запятыми: class Pair<K, V> { ... }. Общепринятые соглашения: T=Type, K=Key, V=Value, E=Element.`,
            whyItMatters: `Обобщенные классы являются основой системы типов Java. Они позволяют писать повторно используемый, типобезопасный код, который работает с любым типом объекта.

**Продакшен паттерн:**
\`\`\`java
// Повторно используемый контейнер для любого типа
class Box<T> {
    private T value;

    public T get() { return value; }
    public void set(T value) { this.value = value; }
}

// Использование в продакшене
Box<String> stringBox = new Box<>("Hello Generics");
Box<Integer> intBox = new Box<>(42);

// Безопасность типов на этапе компиляции
// stringBox.set(123);	// Ошибка: несовместимые типы
\`\`\`

**Практические преимущества:**
- Безопасность типов на этапе компиляции
- Устранение необходимости приведения типов
- Повторное использование кода
- Коллекции, такие как ArrayList<T>, HashMap<K,V> и Optional<T>, используют обобщения`
        },
        uz: {
            title: 'Umumiy klasslar va tur parametrlari',
            solutionCode: `public class GenericClasses {
    // Bitta tur parametrli umumiy Box klassi
    static class Box<T> {
        private T value;

        // Konstruktor T umumiy turini qabul qiladi
        public Box(T value) {
            this.value = value;
        }

        // Getter T turini qaytaradi
        public T get() {
            return value;
        }

        // Setter T turini qabul qiladi
        public void set(T value) {
            this.value = value;
        }
    }

    // Ikki tur parametrli umumiy Pair klassi
    static class Pair<K, V> {
        private K key;
        private V value;

        // Konstruktor ikkala umumiy turni qabul qiladi
        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }

    public static void main(String[] args) {
        // String turi bilan Box yaratamiz
        Box<String> stringBox = new Box<>("Hello Generics");
        System.out.println("String box: " + stringBox.get());

        // Integer turi bilan Box yaratamiz
        Box<Integer> intBox = new Box<>(42);
        System.out.println("Integer box: " + intBox.get());

        // Qiymatni yangilaymiz
        intBox.set(100);
        System.out.println("Yangilangan integer box: " + intBox.get());

        // String kalit va Integer qiymat bilan Pair yaratamiz
        Pair<String, Integer> pair = new Pair<>("Age", 25);
        System.out.println("Pair: " + pair.getKey() + " = " + pair.getValue());

        // Tur xavfsizligi: bu qatorlar kompilyatsiya xatolariga olib keladi
        // stringBox.set(123);	// Xato: mos kelmaydigan turlar
        // intBox.set("text");	// Xato: mos kelmaydigan turlar
    }
}`,
            description: `Java da umumiy klasslarni yaratish va ishlatishni o'rganing.

**Talablar:**
1. Har qanday turdagi obyektni saqlashi mumkin bo'lgan Box<T> umumiy klassini yarating
2. T turidagi qiymatni qabul qiluvchi konstruktor qo'shing
3. Qiymatni olish uchun get() metodini qo'shing
4. Qiymatni yangilash uchun set() metodini qo'shing
5. Kalit va qiymat uchun ikkita tur parametrli Pair<K, V> klassini yarating
6. Turli turlar bilan misollar yaratishni ko'rsating: Box<String>, Box<Integer>, Pair<String, Integer>

Umumiy klasslar tur parametrlarini (masalan, T, K, V) ishlatib, tur xavfsizligini saqlab, har qanday obyekt turi bilan ishlaydigan qayta foydalaniladigan kod yaratadi.`,
            hint1: `Klass nomidan keyin burchakli qavslar yordamida umumiy klassni aniqlang: class Box<T> { ... }. T har qanday tur uchun o'rin egallaydi.`,
            hint2: `Bir nechta tur parametrlari uchun ularni vergul bilan ajrating: class Pair<K, V> { ... }. Umumiy konventsiyalar: T=Type, K=Key, V=Value, E=Element.`,
            whyItMatters: `Umumiy klasslar Java tur tizimining asosi hisoblanadi. Ular har qanday obyekt turi bilan ishlaydigan qayta foydalaniladigan, tur-xavfsiz kod yozish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Har qanday tur uchun qayta foydalaniladigan konteyner
class Box<T> {
    private T value;

    public T get() { return value; }
    public void set(T value) { this.value = value; }
}

// Ishlab chiqarishda foydalanish
Box<String> stringBox = new Box<>("Hello Generics");
Box<Integer> intBox = new Box<>(42);

// Kompilyatsiya vaqtida tur xavfsizligi
// stringBox.set(123);	// Xato: mos kelmaydigan turlar
\`\`\`

**Amaliy foydalari:**
- Kompilyatsiya vaqtida tur xavfsizligi
- Tur o'zgartirishni yo'q qilish
- Kodni qayta foydalanish
- ArrayList<T>, HashMap<K,V> va Optional<T> kabi kolleksiyalar umumiy klasslardan foydalanadi`
        }
    }
};

export default task;
