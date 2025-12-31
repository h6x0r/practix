import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-generic-methods',
    title: 'Generic Methods',
    difficulty: 'easy',
    tags: ['java', 'generics', 'methods', 'type-inference'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to create and use generic methods in Java.

**Requirements:**
1. Create a static generic method printArray<T> that prints any array
2. Create a generic method swap<T> that swaps two elements in an array
3. Create a generic method getMiddle<T> that returns the middle element
4. Demonstrate type inference - calling methods without explicit type arguments
5. Create a generic method comparePairs<T> that compares two values and returns a boolean
6. Test all methods with different types: Integer[], String[], Double[]

Generic methods allow you to write a single method that works with different types, providing type safety and code reuse.`,
    initialCode: `public class GenericMethods {
    // Create static generic method printArray<T>
    // - Accept T[] array parameter
    // - Print each element

    // Create generic method swap<T>
    // - Accept T[] array and two indices
    // - Swap elements at those indices

    // Create generic method getMiddle<T>
    // - Accept T[] array
    // - Return middle element

    // Create generic method comparePairs<T>
    // - Accept two T values
    // - Return true if they are equal

    public static void main(String[] args) {
        // Test with Integer array

        // Test with String array

        // Test with Double array

        // Demonstrate type inference
    }
}`,
    solutionCode: `public class GenericMethods {
    // Generic method to print any array
    public static <T> void printArray(T[] array) {
        System.out.print("[");
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i]);
            if (i < array.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    // Generic method to swap elements in an array
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    // Generic method to get middle element
    public static <T> T getMiddle(T[] array) {
        return array[array.length / 2];
    }

    // Generic method to compare two values
    public static <T> boolean comparePairs(T a, T b) {
        return a.equals(b);
    }

    public static void main(String[] args) {
        // Test with Integer array
        Integer[] intArray = {1, 2, 3, 4, 5};
        System.out.print("Integer array: ");
        printArray(intArray);

        swap(intArray, 0, 4);
        System.out.print("After swap(0, 4): ");
        printArray(intArray);

        Integer middle = getMiddle(intArray);
        System.out.println("Middle element: " + middle);

        // Test with String array
        String[] strArray = {"Java", "Python", "JavaScript", "Go"};
        System.out.print("\\nString array: ");
        printArray(strArray);

        String strMiddle = getMiddle(strArray);
        System.out.println("Middle element: " + strMiddle);

        // Test with Double array
        Double[] doubleArray = {1.1, 2.2, 3.3};
        System.out.print("\\nDouble array: ");
        printArray(doubleArray);

        // Demonstrate type inference - compiler infers types
        System.out.println("\\nCompare pairs:");
        System.out.println("5 == 5: " + comparePairs(5, 5));	// Integer inferred
        System.out.println("\\"Hi\\" == \\"Hi\\": " + comparePairs("Hi", "Hi"));	// String inferred
        System.out.println("5 == 10: " + comparePairs(5, 10));
    }
}`,
    hint1: `Generic method syntax: <T> returnType methodName(T parameter). The <T> goes before the return type.`,
    hint2: `Type inference allows the compiler to determine T from the arguments you pass. You don't need to write GenericMethods.<Integer>printArray(intArray).`,
    whyItMatters: `Generic methods are essential for writing flexible utility functions. The Java Collections API uses them extensively (Collections.sort(), Arrays.asList(), etc.).

**Production Pattern:**
\`\`\`java
// Universal method to print any array
public static <T> void printArray(T[] array) {
    for (T element : array) {
        System.out.print(element + " ");
    }
}

// Type inference - compiler determines types automatically
printArray(new Integer[]{1, 2, 3});	// T = Integer
printArray(new String[]{"A", "B"});	// T = String
\`\`\`

**Practical Benefits:**
- Type safety without code duplication
- Automatic type inference by compiler
- Used in Collections.sort(), Arrays.asList()
- Avoids need to create classes for each type`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify printArray with Integer array outputs correctly
class Test1 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Integer[] arr = {1, 2, 3};
            GenericMethods.printArray(arr);
            String output = out.toString().trim();
            assertTrue(output.contains("1") && output.contains("2") && output.contains("3"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test2: Verify swap method swaps elements
class Test2 {
    @Test
    public void test() {
        Integer[] arr = {1, 2, 3};
        GenericMethods.swap(arr, 0, 2);
        assertEquals(Integer.valueOf(3), arr[0]);
        assertEquals(Integer.valueOf(1), arr[2]);
    }
}

// Test3: Verify getMiddle returns middle element
class Test3 {
    @Test
    public void test() {
        Integer[] arr = {1, 2, 3, 4, 5};
        Integer middle = GenericMethods.getMiddle(arr);
        assertEquals(Integer.valueOf(3), middle);
    }
}

// Test4: Verify comparePairs with equal values
class Test4 {
    @Test
    public void test() {
        boolean result = GenericMethods.comparePairs(5, 5);
        assertTrue(result);
    }
}

// Test5: Verify comparePairs with unequal values
class Test5 {
    @Test
    public void test() {
        boolean result = GenericMethods.comparePairs("hello", "world");
        assertFalse(result);
    }
}

// Test6: Verify printArray with String array outputs correctly
class Test6 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            String[] arr = {"Java", "Python"};
            GenericMethods.printArray(arr);
            String output = out.toString().trim();
            assertTrue(output.contains("Java") && output.contains("Python"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test7: Verify swap with String array
class Test7 {
    @Test
    public void test() {
        String[] arr = {"A", "B", "C"};
        GenericMethods.swap(arr, 0, 2);
        assertEquals("C", arr[0]);
        assertEquals("A", arr[2]);
    }
}

// Test8: Verify getMiddle with even-length array
class Test8 {
    @Test
    public void test() {
        String[] arr = {"A", "B", "C", "D"};
        String middle = GenericMethods.getMiddle(arr);
        assertEquals("C", middle);
    }
}

// Test9: Verify getMiddle with single element array
class Test9 {
    @Test
    public void test() {
        Integer[] arr = {42};
        Integer middle = GenericMethods.getMiddle(arr);
        assertEquals(Integer.valueOf(42), middle);
    }
}

// Test10: Verify printArray with Double array outputs correctly
class Test10 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Double[] arr = {1.1, 2.2, 3.3};
            GenericMethods.printArray(arr);
            String output = out.toString().trim();
            assertTrue(output.contains("1.1") && output.contains("2.2") && output.contains("3.3"));
        } finally {
            System.setOut(originalOut);
        }
    }
}`,
    translations: {
        ru: {
            title: 'Обобщенные методы',
            solutionCode: `public class GenericMethods {
    // Обобщенный метод для печати любого массива
    public static <T> void printArray(T[] array) {
        System.out.print("[");
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i]);
            if (i < array.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    // Обобщенный метод для обмена элементов в массиве
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    // Обобщенный метод для получения среднего элемента
    public static <T> T getMiddle(T[] array) {
        return array[array.length / 2];
    }

    // Обобщенный метод для сравнения двух значений
    public static <T> boolean comparePairs(T a, T b) {
        return a.equals(b);
    }

    public static void main(String[] args) {
        // Тестируем с массивом Integer
        Integer[] intArray = {1, 2, 3, 4, 5};
        System.out.print("Массив Integer: ");
        printArray(intArray);

        swap(intArray, 0, 4);
        System.out.print("После swap(0, 4): ");
        printArray(intArray);

        Integer middle = getMiddle(intArray);
        System.out.println("Средний элемент: " + middle);

        // Тестируем с массивом String
        String[] strArray = {"Java", "Python", "JavaScript", "Go"};
        System.out.print("\\nМассив String: ");
        printArray(strArray);

        String strMiddle = getMiddle(strArray);
        System.out.println("Средний элемент: " + strMiddle);

        // Тестируем с массивом Double
        Double[] doubleArray = {1.1, 2.2, 3.3};
        System.out.print("\\nМассив Double: ");
        printArray(doubleArray);

        // Демонстрируем вывод типов - компилятор определяет типы
        System.out.println("\\nСравнение пар:");
        System.out.println("5 == 5: " + comparePairs(5, 5));	// Integer выведен
        System.out.println("\\"Hi\\" == \\"Hi\\": " + comparePairs("Hi", "Hi"));	// String выведен
        System.out.println("5 == 10: " + comparePairs(5, 10));
    }
}`,
            description: `Изучите создание и использование обобщенных методов в Java.

**Требования:**
1. Создайте статический обобщенный метод printArray<T>, который печатает любой массив
2. Создайте обобщенный метод swap<T>, который меняет местами два элемента в массиве
3. Создайте обобщенный метод getMiddle<T>, который возвращает средний элемент
4. Продемонстрируйте вывод типов - вызов методов без явных аргументов типа
5. Создайте обобщенный метод comparePairs<T>, который сравнивает два значения и возвращает boolean
6. Протестируйте все методы с разными типами: Integer[], String[], Double[]

Обобщенные методы позволяют писать один метод, который работает с разными типами, обеспечивая безопасность типов и повторное использование кода.`,
            hint1: `Синтаксис обобщенного метода: <T> returnType methodName(T parameter). <T> идет перед типом возвращаемого значения.`,
            hint2: `Вывод типов позволяет компилятору определить T из переданных аргументов. Не нужно писать GenericMethods.<Integer>printArray(intArray).`,
            whyItMatters: `Обобщенные методы необходимы для написания гибких вспомогательных функций. Java Collections API активно их использует (Collections.sort(), Arrays.asList() и т.д.).

**Продакшен паттерн:**
\`\`\`java
// Универсальный метод печати любого массива
public static <T> void printArray(T[] array) {
    for (T element : array) {
        System.out.print(element + " ");
    }
}

// Вывод типов - компилятор определяет типы автоматически
printArray(new Integer[]{1, 2, 3});	// T = Integer
printArray(new String[]{"A", "B"});	// T = String
\`\`\`

**Практические преимущества:**
- Безопасность типов без дублирования кода
- Автоматический вывод типов компилятором
- Используется в Collections.sort(), Arrays.asList()
- Избегает необходимости создания классов для каждого типа`
        },
        uz: {
            title: 'Umumiy metodlar',
            solutionCode: `public class GenericMethods {
    // Har qanday massivni chiqarish uchun umumiy metod
    public static <T> void printArray(T[] array) {
        System.out.print("[");
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i]);
            if (i < array.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    // Massivdagi elementlarni almashtirish uchun umumiy metod
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    // O'rta elementni olish uchun umumiy metod
    public static <T> T getMiddle(T[] array) {
        return array[array.length / 2];
    }

    // Ikki qiymatni solishtirish uchun umumiy metod
    public static <T> boolean comparePairs(T a, T b) {
        return a.equals(b);
    }

    public static void main(String[] args) {
        // Integer massiv bilan test qilamiz
        Integer[] intArray = {1, 2, 3, 4, 5};
        System.out.print("Integer massiv: ");
        printArray(intArray);

        swap(intArray, 0, 4);
        System.out.print("swap(0, 4) dan keyin: ");
        printArray(intArray);

        Integer middle = getMiddle(intArray);
        System.out.println("O'rta element: " + middle);

        // String massiv bilan test qilamiz
        String[] strArray = {"Java", "Python", "JavaScript", "Go"};
        System.out.print("\\nString massiv: ");
        printArray(strArray);

        String strMiddle = getMiddle(strArray);
        System.out.println("O'rta element: " + strMiddle);

        // Double massiv bilan test qilamiz
        Double[] doubleArray = {1.1, 2.2, 3.3};
        System.out.print("\\nDouble massiv: ");
        printArray(doubleArray);

        // Tur xulosasini ko'rsatamiz - kompilyator turlarni aniqlaydi
        System.out.println("\\nJuftliklarni solishtirish:");
        System.out.println("5 == 5: " + comparePairs(5, 5));	// Integer aniqlanadi
        System.out.println("\\"Hi\\" == \\"Hi\\": " + comparePairs("Hi", "Hi"));	// String aniqlanadi
        System.out.println("5 == 10: " + comparePairs(5, 10));
    }
}`,
            description: `Java da umumiy metodlarni yaratish va ishlatishni o'rganing.

**Talablar:**
1. Har qanday massivni chiqaradigan printArray<T> statik umumiy metodini yarating
2. Massivdagi ikki elementni almashtiradigan swap<T> umumiy metodini yarating
3. O'rta elementni qaytaradigan getMiddle<T> umumiy metodini yarating
4. Tur xulosasini ko'rsating - metodlarni aniq tur argumentlarisiz chaqirish
5. Ikki qiymatni solishtiruvchi va boolean qaytaruvchi comparePairs<T> umumiy metodini yarating
6. Barcha metodlarni turli turlar bilan sinab ko'ring: Integer[], String[], Double[]

Umumiy metodlar tur xavfsizligini ta'minlab va kodni qayta ishlatishga imkon berib, turli turlar bilan ishlaydigan yagona metod yozish imkonini beradi.`,
            hint1: `Umumiy metod sintaksisi: <T> returnType methodName(T parameter). <T> qaytish turidan oldin keladi.`,
            hint2: `Tur xulosasi kompilyatorga siz uzatgan argumentlardan T ni aniqlash imkonini beradi. GenericMethods.<Integer>printArray(intArray) deb yozish shart emas.`,
            whyItMatters: `Umumiy metodlar moslashuvchan yordamchi funksiyalarni yozish uchun zarur. Java Collections API ularni keng qo'llaydi (Collections.sort(), Arrays.asList() va boshqalar).

**Ishlab chiqarish patterni:**
\`\`\`java
// Har qanday massivni chiqarish uchun universal metod
public static <T> void printArray(T[] array) {
    for (T element : array) {
        System.out.print(element + " ");
    }
}

// Tur xulosasi - kompilyator turlarni avtomatik aniqlaydi
printArray(new Integer[]{1, 2, 3});	// T = Integer
printArray(new String[]{"A", "B"});	// T = String
\`\`\`

**Amaliy foydalari:**
- Kod takrorlanishisiz tur xavfsizligi
- Kompilyator tomonidan avtomatik tur xulosasi
- Collections.sort(), Arrays.asList() da ishlatiladi
- Har bir tur uchun klasslar yaratish zarurati yo'q`
        }
    }
};

export default task;
