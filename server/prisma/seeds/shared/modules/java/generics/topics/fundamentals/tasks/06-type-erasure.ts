import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-type-erasure',
    title: 'Type Erasure and Runtime Limitations',
    difficulty: 'hard',
    tags: ['java', 'generics', 'type-erasure', 'runtime', 'reflection'],
    estimatedTime: '40m',
    isPremium: false,
    youtubeUrl: '',
    description: `Understand type erasure and its runtime implications in Java generics.

**Requirements:**
1. Demonstrate that generic type information is erased at runtime
2. Show why you cannot create generic arrays: new T[10] fails
3. Show why instanceof with generics doesn't work: obj instanceof List<String>
4. Implement a workaround using Class<T> for runtime type information
5. Create a generic array using Array.newInstance() and Class<T>
6. Show bridge methods created by type erasure for compatibility

Type erasure removes generic type information at runtime for backward compatibility. This creates limitations but can be worked around using Class<T> tokens.`,
    initialCode: `import java.lang.reflect.Array;
import java.util.*;

public class TypeErasure {
    // Create a generic class that stores Class<T> for runtime type access
    // - Constructor accepts Class<T>
    // - Method to create array of T
    // - Method to check instanceof using Class<T>

    // Demonstrate what doesn't work due to type erasure

    // Show workarounds using Class<T>

    public static void main(String[] args) {
        // Demonstrate type erasure limitations

        // Show workarounds
    }
}`,
    solutionCode: `import java.lang.reflect.Array;
import java.util.*;

public class TypeErasure {
    // Generic class that keeps runtime type information
    static class TypeSafeBox<T> {
        private final Class<T> type;
        private T value;

        // Constructor accepts Class<T> to preserve type info at runtime
        public TypeSafeBox(Class<T> type) {
            this.type = type;
        }

        public void setValue(T value) {
            this.value = value;
        }

        public T getValue() {
            return value;
        }

        // Can check type at runtime using Class<T>
        public boolean isInstance(Object obj) {
            return type.isInstance(obj);
        }

        // Can create array using Class<T>
        @SuppressWarnings("unchecked")
        public T[] createArray(int size) {
            return (T[]) Array.newInstance(type, size);
        }

        public Class<T> getType() {
            return type;
        }
    }

    // This demonstrates type erasure limitations
    static class ErasureLimitations<T> {
        // private T[] array;	// Cannot do: new T[10]

        // LIMITATION 1: Cannot instantiate generic type
        public void cannotInstantiate() {
            // T obj = new T();	// Compile error: cannot instantiate T
        }

        // LIMITATION 2: Cannot create generic array
        public void cannotCreateArray() {
            // T[] array = new T[10];	// Compile error
        }

        // LIMITATION 3: Cannot use instanceof with type parameter
        public boolean cannotUseInstanceof(Object obj) {
            // return obj instanceof T;	// Compile error
            return false;
        }

        // LIMITATION 4: Cannot use generic type in static context
        // private static T staticField;	// Compile error
    }

    // Workaround: Using Class<T> token
    static class GenericArrayList<T> {
        private final Class<T> type;
        private T[] array;
        private int size = 0;

        @SuppressWarnings("unchecked")
        public GenericArrayList(Class<T> type, int capacity) {
            this.type = type;
            // Use Array.newInstance to create generic array
            this.array = (T[]) Array.newInstance(type, capacity);
        }

        public void add(T element) {
            if (size >= array.length) {
                resize();
            }
            array[size++] = element;
        }

        public T get(int index) {
            if (index >= size) {
                throw new IndexOutOfBoundsException();
            }
            return array[index];
        }

        public int size() {
            return size;
        }

        @SuppressWarnings("unchecked")
        private void resize() {
            T[] newArray = (T[]) Array.newInstance(type, array.length * 2);
            System.arraycopy(array, 0, newArray, 0, array.length);
            array = newArray;
        }

        // Can perform type-safe instanceof check
        public boolean isValidElement(Object obj) {
            return type.isInstance(obj);
        }
    }

    public static void demonstrateErasure() {
        // At runtime, List<String> and List<Integer> are the same: List
        List<String> stringList = new ArrayList<>();
        List<Integer> intList = new ArrayList<>();

        // Both have the same class at runtime
        System.out.println("String list class: " + stringList.getClass());
        System.out.println("Integer list class: " + intList.getClass());
        System.out.println("Same class? " + (stringList.getClass() == intList.getClass()));

        // Cannot check parameterized type at runtime
        // if (stringList instanceof List<String>) {}  // Compile error
        // Can only check raw type
        if (stringList instanceof List) {
            System.out.println("Can check raw type List");
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Type Erasure Demonstration ===\\n");
        demonstrateErasure();

        // Using TypeSafeBox with runtime type information
        System.out.println("\\n=== TypeSafeBox with Class<T> ===");
        TypeSafeBox<String> stringBox = new TypeSafeBox<>(String.class);
        stringBox.setValue("Hello");
        System.out.println("Value: " + stringBox.getValue());
        System.out.println("Type: " + stringBox.getType().getName());

        // Can check instanceof at runtime
        System.out.println("Is 'Hello' instance? " + stringBox.isInstance("Hello"));
        System.out.println("Is 123 instance? " + stringBox.isInstance(123));

        // Can create array
        String[] strArray = stringBox.createArray(5);
        System.out.println("Created array of length: " + strArray.length);

        // Using GenericArrayList
        System.out.println("\\n=== GenericArrayList ===");
        GenericArrayList<Integer> intArrayList = new GenericArrayList<>(Integer.class, 3);
        intArrayList.add(1);
        intArrayList.add(2);
        intArrayList.add(3);
        intArrayList.add(4);	// Triggers resize

        System.out.println("Size: " + intArrayList.size());
        System.out.println("Elements:");
        for (int i = 0; i < intArrayList.size(); i++) {
            System.out.println("  [" + i + "] = " + intArrayList.get(i));
        }

        System.out.println("Is 42 valid? " + intArrayList.isValidElement(42));
        System.out.println("Is 'text' valid? " + intArrayList.isValidElement("text"));

        System.out.println("\\n=== Key Points ===");
        System.out.println("1. Generic type info is erased at runtime");
        System.out.println("2. Cannot create: new T(), new T[], T.class");
        System.out.println("3. Cannot: instanceof T, static T field");
        System.out.println("4. Workaround: Pass Class<T> for runtime access");
        System.out.println("5. Use Array.newInstance(Class<T>, size) for arrays");
    }
}`,
    hint1: `Type erasure means List<String> becomes List at runtime. Generic type parameters are replaced with their bounds (or Object if unbounded).`,
    hint2: `To preserve type information at runtime, pass Class<T> as a parameter. Use Array.newInstance(classToken, size) to create generic arrays.`,
    whyItMatters: `Understanding type erasure is crucial for advanced Java programming.

**Production Pattern:**
\`\`\`java
// Preserve type information via Class<T>
class TypeSafeBox<T> {
    private final Class<T> type;

    public TypeSafeBox(Class<T> type) {
        this.type = type;
    }

    // Can create array via Array.newInstance
    public T[] createArray(int size) {
        return (T[]) Array.newInstance(type, size);
    }
}

// Usage
TypeSafeBox<String> box = new TypeSafeBox<>(String.class);
String[] array = box.createArray(10);	// Works!
\`\`\`

**Practical Benefits:**
- Work around type erasure limitations
- Gson, Jackson, Spring use Class<T>
- Create generic arrays
- Separates junior from senior developers`,
    order: 5,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;

// Test1: Verify TypeSafeBox creation
class Test1 {
    @Test
    public void test() {
        TypeErasure.TypeSafeBox<String> box = new TypeErasure.TypeSafeBox<>(String.class);
        box.setValue("Test");
        assertEquals("Test", box.getValue());
    }
}

// Test2: Verify TypeSafeBox isInstance with correct type
class Test2 {
    @Test
    public void test() {
        TypeErasure.TypeSafeBox<String> box = new TypeErasure.TypeSafeBox<>(String.class);
        assertTrue(box.isInstance("Hello"));
    }
}

// Test3: Verify TypeSafeBox isInstance with wrong type
class Test3 {
    @Test
    public void test() {
        TypeErasure.TypeSafeBox<String> box = new TypeErasure.TypeSafeBox<>(String.class);
        assertFalse(box.isInstance(123));
    }
}

// Test4: Verify TypeSafeBox createArray method
class Test4 {
    @Test
    public void test() {
        TypeErasure.TypeSafeBox<String> box = new TypeErasure.TypeSafeBox<>(String.class);
        String[] array = box.createArray(5);
        assertEquals(5, array.length);
    }
}

// Test5: Verify GenericArrayList add and get
class Test5 {
    @Test
    public void test() {
        TypeErasure.GenericArrayList<Integer> list = new TypeErasure.GenericArrayList<>(Integer.class, 3);
        list.add(10);
        assertEquals(Integer.valueOf(10), list.get(0));
    }
}

// Test6: Verify GenericArrayList size method
class Test6 {
    @Test
    public void test() {
        TypeErasure.GenericArrayList<String> list = new TypeErasure.GenericArrayList<>(String.class, 3);
        list.add("A");
        list.add("B");
        assertEquals(2, list.size());
    }
}

// Test7: Verify GenericArrayList resize on overflow
class Test7 {
    @Test
    public void test() {
        TypeErasure.GenericArrayList<Integer> list = new TypeErasure.GenericArrayList<>(Integer.class, 2);
        list.add(1);
        list.add(2);
        list.add(3); // Should trigger resize
        assertEquals(3, list.size());
    }
}

// Test8: Verify GenericArrayList isValidElement
class Test8 {
    @Test
    public void test() {
        TypeErasure.GenericArrayList<Integer> list = new TypeErasure.GenericArrayList<>(Integer.class, 3);
        assertTrue(list.isValidElement(42));
        assertFalse(list.isValidElement("text"));
    }
}

// Test9: Verify TypeSafeBox getType method
class Test9 {
    @Test
    public void test() {
        TypeErasure.TypeSafeBox<Integer> box = new TypeErasure.TypeSafeBox<>(Integer.class);
        assertEquals(Integer.class, box.getType());
    }
}

// Test10: Verify different types have same raw class
class Test10 {
    @Test
    public void test() {
        List<String> stringList = new ArrayList<>();
        List<Integer> intList = new ArrayList<>();
        assertEquals(stringList.getClass(), intList.getClass());
    }
}`,
    translations: {
        ru: {
            title: 'Стирание типов и ограничения во время выполнения',
            solutionCode: `import java.lang.reflect.Array;
import java.util.*;

public class TypeErasure {
    // Обобщенный класс, который хранит информацию о типе во время выполнения
    static class TypeSafeBox<T> {
        private final Class<T> type;
        private T value;

        // Конструктор принимает Class<T> для сохранения информации о типе
        public TypeSafeBox(Class<T> type) {
            this.type = type;
        }

        public void setValue(T value) {
            this.value = value;
        }

        public T getValue() {
            return value;
        }

        // Можно проверить тип во время выполнения, используя Class<T>
        public boolean isInstance(Object obj) {
            return type.isInstance(obj);
        }

        // Можно создать массив, используя Class<T>
        @SuppressWarnings("unchecked")
        public T[] createArray(int size) {
            return (T[]) Array.newInstance(type, size);
        }

        public Class<T> getType() {
            return type;
        }
    }

    // Это демонстрирует ограничения стирания типов
    static class ErasureLimitations<T> {
        // private T[] array;	// Нельзя: new T[10]

        // ОГРАНИЧЕНИЕ 1: Нельзя создать экземпляр обобщенного типа
        public void cannotInstantiate() {
            // T obj = new T();	// Ошибка компиляции: нельзя создать экземпляр T
        }

        // ОГРАНИЧЕНИЕ 2: Нельзя создать обобщенный массив
        public void cannotCreateArray() {
            // T[] array = new T[10];	// Ошибка компиляции
        }

        // ОГРАНИЧЕНИЕ 3: Нельзя использовать instanceof с параметром типа
        public boolean cannotUseInstanceof(Object obj) {
            // return obj instanceof T;	// Ошибка компиляции
            return false;
        }

        // ОГРАНИЧЕНИЕ 4: Нельзя использовать обобщенный тип в статическом контексте
        // private static T staticField;	// Ошибка компиляции
    }

    // Обходной путь: Использование токена Class<T>
    static class GenericArrayList<T> {
        private final Class<T> type;
        private T[] array;
        private int size = 0;

        @SuppressWarnings("unchecked")
        public GenericArrayList(Class<T> type, int capacity) {
            this.type = type;
            // Используем Array.newInstance для создания обобщенного массива
            this.array = (T[]) Array.newInstance(type, capacity);
        }

        public void add(T element) {
            if (size >= array.length) {
                resize();
            }
            array[size++] = element;
        }

        public T get(int index) {
            if (index >= size) {
                throw new IndexOutOfBoundsException();
            }
            return array[index];
        }

        public int size() {
            return size;
        }

        @SuppressWarnings("unchecked")
        private void resize() {
            T[] newArray = (T[]) Array.newInstance(type, array.length * 2);
            System.arraycopy(array, 0, newArray, 0, array.length);
            array = newArray;
        }

        // Можно выполнить типобезопасную проверку instanceof
        public boolean isValidElement(Object obj) {
            return type.isInstance(obj);
        }
    }

    public static void demonstrateErasure() {
        // Во время выполнения List<String> и List<Integer> - это одно и то же: List
        List<String> stringList = new ArrayList<>();
        List<Integer> intList = new ArrayList<>();

        // Оба имеют одинаковый класс во время выполнения
        System.out.println("Класс списка String: " + stringList.getClass());
        System.out.println("Класс списка Integer: " + intList.getClass());
        System.out.println("Одинаковый класс? " + (stringList.getClass() == intList.getClass()));

        // Нельзя проверить параметризованный тип во время выполнения
        // if (stringList instanceof List<String>) {}  // Ошибка компиляции
        // Можно проверить только сырой тип
        if (stringList instanceof List) {
            System.out.println("Можно проверить сырой тип List");
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Демонстрация стирания типов ===\\n");
        demonstrateErasure();

        // Использование TypeSafeBox с информацией о типе во время выполнения
        System.out.println("\\n=== TypeSafeBox с Class<T> ===");
        TypeSafeBox<String> stringBox = new TypeSafeBox<>(String.class);
        stringBox.setValue("Hello");
        System.out.println("Значение: " + stringBox.getValue());
        System.out.println("Тип: " + stringBox.getType().getName());

        // Можно проверить instanceof во время выполнения
        System.out.println("'Hello' экземпляр? " + stringBox.isInstance("Hello"));
        System.out.println("123 экземпляр? " + stringBox.isInstance(123));

        // Можно создать массив
        String[] strArray = stringBox.createArray(5);
        System.out.println("Создан массив длиной: " + strArray.length);

        // Использование GenericArrayList
        System.out.println("\\n=== GenericArrayList ===");
        GenericArrayList<Integer> intArrayList = new GenericArrayList<>(Integer.class, 3);
        intArrayList.add(1);
        intArrayList.add(2);
        intArrayList.add(3);
        intArrayList.add(4);	// Вызывает resize

        System.out.println("Размер: " + intArrayList.size());
        System.out.println("Элементы:");
        for (int i = 0; i < intArrayList.size(); i++) {
            System.out.println("  [" + i + "] = " + intArrayList.get(i));
        }

        System.out.println("42 допустимо? " + intArrayList.isValidElement(42));
        System.out.println("'text' допустимо? " + intArrayList.isValidElement("text"));

        System.out.println("\\n=== Ключевые моменты ===");
        System.out.println("1. Информация об обобщенном типе стирается во время выполнения");
        System.out.println("2. Нельзя создать: new T(), new T[], T.class");
        System.out.println("3. Нельзя: instanceof T, static T field");
        System.out.println("4. Обходной путь: Передать Class<T> для доступа во время выполнения");
        System.out.println("5. Использовать Array.newInstance(Class<T>, size) для массивов");
    }
}`,
            description: `Поймите стирание типов и его влияние на выполнение в обобщениях Java.

**Требования:**
1. Продемонстрируйте, что информация об обобщенном типе стирается во время выполнения
2. Покажите, почему нельзя создать обобщенные массивы: new T[10] не работает
3. Покажите, почему instanceof с обобщениями не работает: obj instanceof List<String>
4. Реализуйте обходной путь, используя Class<T> для информации о типе во время выполнения
5. Создайте обобщенный массив, используя Array.newInstance() и Class<T>
6. Покажите мост-методы, создаваемые стиранием типов для совместимости

Стирание типов удаляет информацию об обобщенном типе во время выполнения для обратной совместимости. Это создает ограничения, но их можно обойти, используя токены Class<T>.`,
            hint1: `Стирание типов означает, что List<String> становится List во время выполнения. Параметры обобщенного типа заменяются их границами (или Object, если нет границ).`,
            hint2: `Чтобы сохранить информацию о типе во время выполнения, передайте Class<T> как параметр. Используйте Array.newInstance(classToken, size) для создания обобщенных массивов.`,
            whyItMatters: `Понимание стирания типов критически важно для продвинутого программирования на Java.

**Продакшен паттерн:**
\`\`\`java
// Сохраняем информацию о типе через Class<T>
class TypeSafeBox<T> {
    private final Class<T> type;

    public TypeSafeBox(Class<T> type) {
        this.type = type;
    }

    // Можем создать массив через Array.newInstance
    public T[] createArray(int size) {
        return (T[]) Array.newInstance(type, size);
    }
}

// Использование
TypeSafeBox<String> box = new TypeSafeBox<>(String.class);
String[] array = box.createArray(10);	// Работает!
\`\`\`

**Практические преимущества:**
- Обход ограничений стирания типов
- Gson, Jackson, Spring используют Class<T>
- Создание обобщенных массивов
- Отличает junior от senior разработчиков`
        },
        uz: {
            title: 'Tur o\'chirilishi va runtime cheklovlari',
            solutionCode: `import java.lang.reflect.Array;
import java.util.*;

public class TypeErasure {
    // Runtime da tur ma'lumotini saqlaydigan umumiy klass
    static class TypeSafeBox<T> {
        private final Class<T> type;
        private T value;

        // Konstruktor tur ma'lumotini saqlash uchun Class<T> ni qabul qiladi
        public TypeSafeBox(Class<T> type) {
            this.type = type;
        }

        public void setValue(T value) {
            this.value = value;
        }

        public T getValue() {
            return value;
        }

        // Runtime da Class<T> yordamida turni tekshirish mumkin
        public boolean isInstance(Object obj) {
            return type.isInstance(obj);
        }

        // Class<T> yordamida massiv yaratish mumkin
        @SuppressWarnings("unchecked")
        public T[] createArray(int size) {
            return (T[]) Array.newInstance(type, size);
        }

        public Class<T> getType() {
            return type;
        }
    }

    // Bu tur o'chirilishining cheklovlarini ko'rsatadi
    static class ErasureLimitations<T> {
        // private T[] array;	// Mumkin emas: new T[10]

        // CHEKLOV 1: Umumiy turdan nusxa yaratib bo'lmaydi
        public void cannotInstantiate() {
            // T obj = new T();	// Kompilyatsiya xatosi: T dan nusxa yaratib bo'lmaydi
        }

        // CHEKLOV 2: Umumiy massiv yaratib bo'lmaydi
        public void cannotCreateArray() {
            // T[] array = new T[10];	// Kompilyatsiya xatosi
        }

        // CHEKLOV 3: Tur parametri bilan instanceof ishlatib bo'lmaydi
        public boolean cannotUseInstanceof(Object obj) {
            // return obj instanceof T;	// Kompilyatsiya xatosi
            return false;
        }

        // CHEKLOV 4: Umumiy turni statik kontekstda ishlatib bo'lmaydi
        // private static T staticField;	// Kompilyatsiya xatosi
    }

    // Yechim: Class<T> tokenidan foydalanish
    static class GenericArrayList<T> {
        private final Class<T> type;
        private T[] array;
        private int size = 0;

        @SuppressWarnings("unchecked")
        public GenericArrayList(Class<T> type, int capacity) {
            this.type = type;
            // Umumiy massiv yaratish uchun Array.newInstance dan foydalanamiz
            this.array = (T[]) Array.newInstance(type, capacity);
        }

        public void add(T element) {
            if (size >= array.length) {
                resize();
            }
            array[size++] = element;
        }

        public T get(int index) {
            if (index >= size) {
                throw new IndexOutOfBoundsException();
            }
            return array[index];
        }

        public int size() {
            return size;
        }

        @SuppressWarnings("unchecked")
        private void resize() {
            T[] newArray = (T[]) Array.newInstance(type, array.length * 2);
            System.arraycopy(array, 0, newArray, 0, array.length);
            array = newArray;
        }

        // Tur-xavfsiz instanceof tekshiruvini amalga oshirish mumkin
        public boolean isValidElement(Object obj) {
            return type.isInstance(obj);
        }
    }

    public static void demonstrateErasure() {
        // Runtime da List<String> va List<Integer> bir xil: List
        List<String> stringList = new ArrayList<>();
        List<Integer> intList = new ArrayList<>();

        // Ikkalasi ham runtime da bir xil klassga ega
        System.out.println("String ro'yxat klassi: " + stringList.getClass());
        System.out.println("Integer ro'yxat klassi: " + intList.getClass());
        System.out.println("Bir xil klass? " + (stringList.getClass() == intList.getClass()));

        // Runtime da parametrlangan turni tekshirib bo'lmaydi
        // if (stringList instanceof List<String>) {}  // Kompilyatsiya xatosi
        // Faqat xom turni tekshirish mumkin
        if (stringList instanceof List) {
            System.out.println("Xom List turini tekshirish mumkin");
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Tur o'chirilishi ko'rsatmasi ===\\n");
        demonstrateErasure();

        // Runtime tur ma'lumotli TypeSafeBox dan foydalanish
        System.out.println("\\n=== Class<T> bilan TypeSafeBox ===");
        TypeSafeBox<String> stringBox = new TypeSafeBox<>(String.class);
        stringBox.setValue("Hello");
        System.out.println("Qiymat: " + stringBox.getValue());
        System.out.println("Tur: " + stringBox.getType().getName());

        // Runtime da instanceof tekshirish mumkin
        System.out.println("'Hello' nusxa? " + stringBox.isInstance("Hello"));
        System.out.println("123 nusxa? " + stringBox.isInstance(123));

        // Massiv yaratish mumkin
        String[] strArray = stringBox.createArray(5);
        System.out.println("Yaratilgan massiv uzunligi: " + strArray.length);

        // GenericArrayList dan foydalanish
        System.out.println("\\n=== GenericArrayList ===");
        GenericArrayList<Integer> intArrayList = new GenericArrayList<>(Integer.class, 3);
        intArrayList.add(1);
        intArrayList.add(2);
        intArrayList.add(3);
        intArrayList.add(4);	// resize ni chaqiradi

        System.out.println("O'lcham: " + intArrayList.size());
        System.out.println("Elementlar:");
        for (int i = 0; i < intArrayList.size(); i++) {
            System.out.println("  [" + i + "] = " + intArrayList.get(i));
        }

        System.out.println("42 to'g'ri? " + intArrayList.isValidElement(42));
        System.out.println("'text' to'g'ri? " + intArrayList.isValidElement("text"));

        System.out.println("\\n=== Asosiy fikrlar ===");
        System.out.println("1. Umumiy tur ma'lumoti runtime da o'chiriladi");
        System.out.println("2. Yaratib bo'lmaydi: new T(), new T[], T.class");
        System.out.println("3. Mumkin emas: instanceof T, static T field");
        System.out.println("4. Yechim: Runtime kirish uchun Class<T> ni uzating");
        System.out.println("5. Massivlar uchun Array.newInstance(Class<T>, size) dan foydalaning");
    }
}`,
            description: `Java generics da tur o'chirilishi va uning runtime ta'sirlarini tushunib oling.

**Talablar:**
1. Umumiy tur ma'lumoti runtime da o'chirilishini ko'rsating
2. Nima uchun umumiy massivlar yaratib bo'lmasligini ko'rsating: new T[10] ishlamaydi
3. Nima uchun generics bilan instanceof ishlamasligini ko'rsating: obj instanceof List<String>
4. Runtime tur ma'lumoti uchun Class<T> dan foydalangan holda yechim yarating
5. Array.newInstance() va Class<T> dan foydalanib umumiy massiv yarating
6. Muvofiqlik uchun tur o'chirilishi tomonidan yaratilgan ko'prik metodlarini ko'rsating

Tur o'chirilishi orqaga muvofiqlik uchun runtime da umumiy tur ma'lumotini olib tashlaydi. Bu cheklovlar yaratadi, lekin Class<T> tokenlaridan foydalanib yechish mumkin.`,
            hint1: `Tur o'chirilishi degani List<String> runtime da List ga aylanadi. Umumiy tur parametrlari chegaralari bilan almashtiriladi (yoki chegarasiz bo'lsa Object bilan).`,
            hint2: `Runtime da tur ma'lumotini saqlash uchun Class<T> ni parametr sifatida uzating. Umumiy massivlar yaratish uchun Array.newInstance(classToken, size) dan foydalaning.`,
            whyItMatters: `Tur o'chirilishini tushunish ilg'or Java dasturlash uchun juda muhim.

**Ishlab chiqarish patterni:**
\`\`\`java
// Class<T> orqali tur ma'lumotini saqlaymiz
class TypeSafeBox<T> {
    private final Class<T> type;

    public TypeSafeBox(Class<T> type) {
        this.type = type;
    }

    // Array.newInstance orqali massiv yaratish mumkin
    public T[] createArray(int size) {
        return (T[]) Array.newInstance(type, size);
    }
}

// Foydalanish
TypeSafeBox<String> box = new TypeSafeBox<>(String.class);
String[] array = box.createArray(10);	// Ishlaydi!
\`\`\`

**Amaliy foydalari:**
- Tur o'chirilishi cheklovlarini yechish
- Gson, Jackson, Spring Class<T> dan foydalanadi
- Umumiy massivlar yaratish
- Junior va senior dasturchilari orasidagi farq`
        }
    }
};

export default task;
