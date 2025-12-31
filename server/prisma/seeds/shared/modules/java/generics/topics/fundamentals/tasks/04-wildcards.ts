import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-wildcards',
    title: 'Wildcards and Unknown Types',
    difficulty: 'medium',
    tags: ['java', 'generics', 'wildcards', 'unknown-types'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to use wildcards (?) for flexible generic types in Java.

**Requirements:**
1. Create a printList method that accepts List<?> to print any type of list
2. Create a getSize method that returns size of Collection<?>
3. Demonstrate that you can read from List<?> but not add (except null)
4. Create a method that works with List<? extends Object> (same as List<?>)
5. Show when to use wildcards vs type parameters
6. Compare List<?>, List<Object>, and List<T>

Wildcards allow you to write methods that work with collections of unknown types, providing flexibility when you don't need to know the exact type.`,
    initialCode: `import java.util.*;

public class Wildcards {
    // Create printList method accepting List<?>
    // - Print each element

    // Create getSize method accepting Collection<?>
    // - Return size

    // Create hasElements method accepting Collection<?>
    // - Return true if not empty

    // Demonstrate wildcard limitations

    public static void main(String[] args) {
        // Create different types of lists

        // Test printList with Integer list

        // Test printList with String list

        // Test printList with Double list

        // Demonstrate what you can and cannot do with wildcards
    }
}`,
    solutionCode: `import java.util.*;

public class Wildcards {
    // Wildcard <?> represents unknown type
    // Can read as Object, but cannot add (except null)
    public static void printList(List<?> list) {
        System.out.print("[");
        for (int i = 0; i < list.size(); i++) {
            System.out.print(list.get(i));
            if (i < list.size() - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    // Wildcards work with any Collection type
    public static int getSize(Collection<?> collection) {
        return collection.size();
    }

    // Can call methods that don't depend on type parameter
    public static boolean hasElements(Collection<?> collection) {
        return !collection.isEmpty();
    }

    // Can read from wildcard collections
    public static void processElements(List<?> list) {
        for (Object element : list) {
            // Elements are read as Object
            System.out.println("Processing: " + element);
        }
    }

    // This shows wildcard limitations
    public static void demonstrateWildcardLimitations(List<?> list) {
        // CAN do:
        int size = list.size();
        boolean isEmpty = list.isEmpty();
        Object first = list.isEmpty() ? null : list.get(0);

        // CANNOT do (won't compile):
        // list.add("text");	// Error: cannot add to List<?>
        // list.add(123);	// Error: cannot add to List<?>
        // list.add(new Object());	// Error: cannot add to List<?>

        // CAN add null (null is member of every type)
        // list.add(null);	// This compiles but often not useful
    }

    public static void main(String[] args) {
        // Create different types of lists
        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        List<String> strList = Arrays.asList("Java", "Python", "Go");
        List<Double> doubleList = Arrays.asList(1.1, 2.2, 3.3);

        // printList works with any list type
        System.out.println("Integer list:");
        printList(intList);

        System.out.println("\\nString list:");
        printList(strList);

        System.out.println("\\nDouble list:");
        printList(doubleList);

        // getSize works with any collection
        System.out.println("\\nSizes:");
        System.out.println("Integer list size: " + getSize(intList));
        System.out.println("String list size: " + getSize(strList));

        Set<String> stringSet = new HashSet<>(Arrays.asList("A", "B", "C"));
        System.out.println("String set size: " + getSize(stringSet));

        // hasElements check
        System.out.println("\\nHas elements:");
        System.out.println("Integer list has elements: " + hasElements(intList));
        System.out.println("Empty list has elements: " + hasElements(new ArrayList<>()));

        // Process elements
        System.out.println("\\nProcessing string list:");
        processElements(strList);

        // Demonstrate differences
        System.out.println("\\nKey differences:");
        System.out.println("List<?> - Unknown type, cannot add");
        System.out.println("List<Object> - Specifically Object type");
        System.out.println("List<T> - Type parameter, more flexible");
    }
}`,
    hint1: `The unbounded wildcard <?> means "a collection of unknown type". You can read from it (as Object) but cannot add to it (except null).`,
    hint2: `Use wildcards when you only need to read from a collection or call methods that don't depend on the type parameter. Use type parameters <T> when you need to write to the collection or return values of that type.`,
    whyItMatters: `Wildcards are essential for writing flexible APIs that work with any generic type.

**Production Pattern:**
\`\`\`java
// Works with any list type
public static void printList(List<?> list) {
    for (Object element : list) {
        System.out.println(element);	// Read as Object
    }
    // list.add("text");	// Error: cannot add
}

// Usage
printList(Arrays.asList(1, 2, 3));	// List<Integer>
printList(Arrays.asList("A", "B"));	// List<String>
printList(Arrays.asList(1.1, 2.2));	// List<Double>
\`\`\`

**Practical Benefits:**
- One method works with any list type
- Used in Java Collections API
- Key to understanding covariance
- Avoids code duplication`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;

// Test1: Verify printList with Integer list outputs correctly
class Test1 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            List<Integer> list = Arrays.asList(1, 2, 3);
            Wildcards.printList(list);
            String output = out.toString().trim();
            assertTrue(output.contains("1") && output.contains("2") && output.contains("3"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test2: Verify getSize with List
class Test2 {
    @Test
    public void test() {
        List<String> list = Arrays.asList("A", "B", "C");
        int size = Wildcards.getSize(list);
        assertEquals(3, size);
    }
}

// Test3: Verify hasElements returns true for non-empty collection
class Test3 {
    @Test
    public void test() {
        List<Integer> list = Arrays.asList(1, 2, 3);
        boolean result = Wildcards.hasElements(list);
        assertTrue(result);
    }
}

// Test4: Verify hasElements returns false for empty collection
class Test4 {
    @Test
    public void test() {
        List<String> list = new ArrayList<>();
        boolean result = Wildcards.hasElements(list);
        assertFalse(result);
    }
}

// Test5: Verify printList with String list outputs correctly
class Test5 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            List<String> list = Arrays.asList("Java", "Python");
            Wildcards.printList(list);
            String output = out.toString().trim();
            assertTrue(output.contains("Java") && output.contains("Python"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test6: Verify getSize with Set
class Test6 {
    @Test
    public void test() {
        Set<String> set = new HashSet<>(Arrays.asList("A", "B", "C"));
        int size = Wildcards.getSize(set);
        assertEquals(3, size);
    }
}

// Test7: Verify processElements outputs each element
class Test7 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            List<String> list = Arrays.asList("Alpha", "Beta", "Gamma");
            Wildcards.processElements(list);
            String output = out.toString();
            assertTrue(output.contains("Alpha") && output.contains("Beta") && output.contains("Gamma"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test8: Verify printList with Double list outputs correctly
class Test8 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            List<Double> list = Arrays.asList(1.1, 2.2, 3.3);
            Wildcards.printList(list);
            String output = out.toString().trim();
            assertTrue(output.contains("1.1") && output.contains("2.2") && output.contains("3.3"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test9: Verify getSize with empty collection
class Test9 {
    @Test
    public void test() {
        List<Integer> list = new ArrayList<>();
        int size = Wildcards.getSize(list);
        assertEquals(0, size);
    }
}

// Test10: Verify processElements with Integer list outputs correctly
class Test10 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            List<Integer> list = Arrays.asList(100, 200, 300);
            Wildcards.processElements(list);
            String output = out.toString();
            assertTrue(output.contains("100") && output.contains("200") && output.contains("300"));
        } finally {
            System.setOut(originalOut);
        }
    }
}`,
    translations: {
        ru: {
            title: 'Подстановочные знаки и неизвестные типы',
            solutionCode: `import java.util.*;

public class Wildcards {
    // Подстановочный знак <?> представляет неизвестный тип
    // Можно читать как Object, но нельзя добавлять (кроме null)
    public static void printList(List<?> list) {
        System.out.print("[");
        for (int i = 0; i < list.size(); i++) {
            System.out.print(list.get(i));
            if (i < list.size() - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    // Подстановочные знаки работают с любым типом Collection
    public static int getSize(Collection<?> collection) {
        return collection.size();
    }

    // Можно вызывать методы, которые не зависят от параметра типа
    public static boolean hasElements(Collection<?> collection) {
        return !collection.isEmpty();
    }

    // Можно читать из коллекций с подстановочными знаками
    public static void processElements(List<?> list) {
        for (Object element : list) {
            // Элементы читаются как Object
            System.out.println("Обработка: " + element);
        }
    }

    // Это показывает ограничения подстановочных знаков
    public static void demonstrateWildcardLimitations(List<?> list) {
        // МОЖНО делать:
        int size = list.size();
        boolean isEmpty = list.isEmpty();
        Object first = list.isEmpty() ? null : list.get(0);

        // НЕЛЬЗЯ делать (не скомпилируется):
        // list.add("text");	// Ошибка: нельзя добавлять в List<?>
        // list.add(123);	// Ошибка: нельзя добавлять в List<?>
        // list.add(new Object());	// Ошибка: нельзя добавлять в List<?>

        // МОЖНО добавить null (null является членом каждого типа)
        // list.add(null);	// Это компилируется, но часто бесполезно
    }

    public static void main(String[] args) {
        // Создаем списки разных типов
        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        List<String> strList = Arrays.asList("Java", "Python", "Go");
        List<Double> doubleList = Arrays.asList(1.1, 2.2, 3.3);

        // printList работает с любым типом списка
        System.out.println("Список Integer:");
        printList(intList);

        System.out.println("\\nСписок String:");
        printList(strList);

        System.out.println("\\nСписок Double:");
        printList(doubleList);

        // getSize работает с любой коллекцией
        System.out.println("\\nРазмеры:");
        System.out.println("Размер списка Integer: " + getSize(intList));
        System.out.println("Размер списка String: " + getSize(strList));

        Set<String> stringSet = new HashSet<>(Arrays.asList("A", "B", "C"));
        System.out.println("Размер множества String: " + getSize(stringSet));

        // Проверка hasElements
        System.out.println("\\nЕсть элементы:");
        System.out.println("В списке Integer есть элементы: " + hasElements(intList));
        System.out.println("В пустом списке есть элементы: " + hasElements(new ArrayList<>()));

        // Обработка элементов
        System.out.println("\\nОбработка списка строк:");
        processElements(strList);

        // Демонстрация различий
        System.out.println("\\nКлючевые различия:");
        System.out.println("List<?> - Неизвестный тип, нельзя добавлять");
        System.out.println("List<Object> - Конкретно тип Object");
        System.out.println("List<T> - Параметр типа, более гибкий");
    }
}`,
            description: `Изучите использование подстановочных знаков (?) для гибких обобщенных типов в Java.

**Требования:**
1. Создайте метод printList, принимающий List<?> для печати списка любого типа
2. Создайте метод getSize, возвращающий размер Collection<?>
3. Продемонстрируйте, что из List<?> можно читать, но нельзя добавлять (кроме null)
4. Создайте метод, работающий с List<? extends Object> (то же, что List<?>)
5. Покажите, когда использовать подстановочные знаки вместо параметров типа
6. Сравните List<?>, List<Object> и List<T>

Подстановочные знаки позволяют писать методы, работающие с коллекциями неизвестных типов, обеспечивая гибкость, когда вам не нужно знать точный тип.`,
            hint1: `Неограниченный подстановочный знак <?> означает "коллекция неизвестного типа". Вы можете читать из нее (как Object), но не можете добавлять в нее (кроме null).`,
            hint2: `Используйте подстановочные знаки, когда нужно только читать из коллекции или вызывать методы, не зависящие от параметра типа. Используйте параметры типа <T>, когда нужно писать в коллекцию или возвращать значения этого типа.`,
            whyItMatters: `Подстановочные знаки необходимы для написания гибких API, работающих с любым обобщенным типом.

**Продакшен паттерн:**
\`\`\`java
// Работает с любым типом списка
public static void printList(List<?> list) {
    for (Object element : list) {
        System.out.println(element);	// Читаем как Object
    }
    // list.add("text");	// Ошибка: нельзя добавлять
}

// Использование
printList(Arrays.asList(1, 2, 3));	// List<Integer>
printList(Arrays.asList("A", "B"));	// List<String>
printList(Arrays.asList(1.1, 2.2));	// List<Double>
\`\`\`

**Практические преимущества:**
- Один метод работает с любым типом списка
- Используется в Java Collections API
- Ключ к пониманию ковариантности
- Избегает дублирования кода`
        },
        uz: {
            title: 'Noma\'lum turlar va wildcard belgilari',
            solutionCode: `import java.util.*;

public class Wildcards {
    // Wildcard <?> noma'lum turni bildiradi
    // Object sifatida o'qish mumkin, lekin qo'shish mumkin emas (null bundan mustasno)
    public static void printList(List<?> list) {
        System.out.print("[");
        for (int i = 0; i < list.size(); i++) {
            System.out.print(list.get(i));
            if (i < list.size() - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    // Wildcard har qanday Collection turi bilan ishlaydi
    public static int getSize(Collection<?> collection) {
        return collection.size();
    }

    // Tur parametriga bog'liq bo'lmagan metodlarni chaqirish mumkin
    public static boolean hasElements(Collection<?> collection) {
        return !collection.isEmpty();
    }

    // Wildcard kolleksiyalaridan o'qish mumkin
    public static void processElements(List<?> list) {
        for (Object element : list) {
            // Elementlar Object sifatida o'qiladi
            System.out.println("Qayta ishlash: " + element);
        }
    }

    // Bu wildcard cheklovlarini ko'rsatadi
    public static void demonstrateWildcardLimitations(List<?> list) {
        // MUMKIN:
        int size = list.size();
        boolean isEmpty = list.isEmpty();
        Object first = list.isEmpty() ? null : list.get(0);

        // MUMKIN EMAS (kompilyatsiya qilinmaydi):
        // list.add("text");	// Xato: List<?> ga qo'shish mumkin emas
        // list.add(123);	// Xato: List<?> ga qo'shish mumkin emas
        // list.add(new Object());	// Xato: List<?> ga qo'shish mumkin emas

        // null qo'shish MUMKIN (null har bir turning a'zosi)
        // list.add(null);	// Bu kompilyatsiya qilinadi, lekin ko'pincha foydali emas
    }

    public static void main(String[] args) {
        // Turli turdagi ro'yxatlarni yaratamiz
        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        List<String> strList = Arrays.asList("Java", "Python", "Go");
        List<Double> doubleList = Arrays.asList(1.1, 2.2, 3.3);

        // printList har qanday ro'yxat turi bilan ishlaydi
        System.out.println("Integer ro'yxati:");
        printList(intList);

        System.out.println("\\nString ro'yxati:");
        printList(strList);

        System.out.println("\\nDouble ro'yxati:");
        printList(doubleList);

        // getSize har qanday kolleksiya bilan ishlaydi
        System.out.println("\\nO'lchamlar:");
        System.out.println("Integer ro'yxat o'lchami: " + getSize(intList));
        System.out.println("String ro'yxat o'lchami: " + getSize(strList));

        Set<String> stringSet = new HashSet<>(Arrays.asList("A", "B", "C"));
        System.out.println("String to'plam o'lchami: " + getSize(stringSet));

        // hasElements tekshiruvi
        System.out.println("\\nElementlar bor:");
        System.out.println("Integer ro'yxatda elementlar bor: " + hasElements(intList));
        System.out.println("Bo'sh ro'yxatda elementlar bor: " + hasElements(new ArrayList<>()));

        // Elementlarni qayta ishlash
        System.out.println("\\nString ro'yxatni qayta ishlash:");
        processElements(strList);

        // Farqlarni ko'rsatish
        System.out.println("\\nAsosiy farqlar:");
        System.out.println("List<?> - Noma'lum tur, qo'shish mumkin emas");
        System.out.println("List<Object> - Aniq Object turi");
        System.out.println("List<T> - Tur parametri, ko'proq moslashuvchan");
    }
}`,
            description: `Java da moslashuvchan umumiy turlar uchun wildcard (?) dan foydalanishni o'rganing.

**Talablar:**
1. Har qanday turdagi ro'yxatni chiqarish uchun List<?> qabul qiluvchi printList metodini yarating
2. Collection<?> o'lchamini qaytaruvchi getSize metodini yarating
3. List<?> dan o'qish mumkin, lekin qo'shish mumkin emasligini (null bundan mustasno) ko'rsating
4. List<? extends Object> bilan ishlaydigan metod yarating (List<?> bilan bir xil)
5. Qachon wildcard va qachon tur parametrlaridan foydalanishni ko'rsating
6. List<?>, List<Object> va List<T> ni solishtiring

Wildcard belgilari aniq turni bilmasangiz ham, noma'lum turdagi kolleksiyalar bilan ishlaydigan metodlar yozish imkonini beradi va moslashuvchanlikni ta'minlaydi.`,
            hint1: `Cheklanmagan wildcard <?> "noma'lum turdagi kolleksiya" degan ma'noni anglatadi. Undan o'qish mumkin (Object sifatida), lekin qo'shish mumkin emas (null bundan mustasno).`,
            hint2: `Faqat kolleksiyadan o'qish kerak bo'lganda yoki tur parametriga bog'liq bo'lmagan metodlarni chaqirish kerak bo'lganda wildcard dan foydalaning. Kolleksiyaga yozish yoki shu turdagi qiymatlarni qaytarish kerak bo'lganda <T> tur parametrlaridan foydalaning.`,
            whyItMatters: `Wildcard belgilari har qanday umumiy tur bilan ishlaydigan moslashuvchan API lar yozish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Har qanday ro'yxat turi bilan ishlaydi
public static void printList(List<?> list) {
    for (Object element : list) {
        System.out.println(element);	// Object sifatida o'qiymiz
    }
    // list.add("text");	// Xato: qo'shish mumkin emas
}

// Foydalanish
printList(Arrays.asList(1, 2, 3));	// List<Integer>
printList(Arrays.asList("A", "B"));	// List<String>
printList(Arrays.asList(1.1, 2.2));	// List<Double>
\`\`\`

**Amaliy foydalari:**
- Bitta metod har qanday ro'yxat turi bilan ishlaydi
- Java Collections API da ishlatiladi
- Kovariantlikni tushunish kaliti
- Kod takrorlanishini oldini oladi`
        }
    }
};

export default task;
