import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-bounded-types',
    title: 'Bounded Type Parameters',
    difficulty: 'medium',
    tags: ['java', 'generics', 'bounds', 'constraints'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to restrict generic types using bounds in Java.

**Requirements:**
1. Create a method findMax<T extends Comparable<T>> that finds the maximum in an array
2. Create a NumberBox<T extends Number> class that only accepts numeric types
3. Add methods to NumberBox: doubleValue() and intValue()
4. Create a method sumNumbers<T extends Number> that sums numeric values
5. Create a class with multiple bounds: <T extends Comparable<T> & Cloneable>
6. Demonstrate that only types matching the bounds can be used

Bounded type parameters restrict which types can be used with generics, ensuring the type has certain methods or implements certain interfaces.`,
    initialCode: `public class BoundedTypes {
    // Create findMax<T extends Comparable<T>> method
    // - Accept T[] array
    // - Return maximum element

    // Create NumberBox<T extends Number> class
    // - Store a T value
    // - Implement doubleValue() method
    // - Implement intValue() method

    // Create sumNumbers<T extends Number> method
    // - Accept array of T extends Number
    // - Return sum as double

    // Create method with multiple bounds
    // <T extends Comparable<T> & Cloneable>

    public static void main(String[] args) {
        // Test findMax with Integer, String, Double

        // Test NumberBox with Integer, Double

        // Test sumNumbers

        // Demonstrate compile-time safety
    }
}`,
    solutionCode: `public class BoundedTypes {
    // Generic method with upper bound - T must implement Comparable
    public static <T extends Comparable<T>> T findMax(T[] array) {
        if (array == null || array.length == 0) {
            return null;
        }

        T max = array[0];
        for (int i = 1; i < array.length; i++) {
            // Can call compareTo because T extends Comparable<T>
            if (array[i].compareTo(max) > 0) {
                max = array[i];
            }
        }
        return max;
    }

    // Generic class with upper bound - T must extend Number
    static class NumberBox<T extends Number> {
        private T value;

        public NumberBox(T value) {
            this.value = value;
        }

        // Can call Number methods because T extends Number
        public double doubleValue() {
            return value.doubleValue();
        }

        public int intValue() {
            return value.intValue();
        }

        public T getValue() {
            return value;
        }
    }

    // Generic method that sums numeric values
    public static <T extends Number> double sumNumbers(T[] numbers) {
        double sum = 0;
        for (T num : numbers) {
            // Can call doubleValue() because T extends Number
            sum += num.doubleValue();
        }
        return sum;
    }

    // Multiple bounds: must implement both Comparable and Cloneable
    public static <T extends Comparable<T> & Cloneable> T cloneAndCompare(T obj1, T obj2) throws CloneNotSupportedException {
        // Can use both Comparable and Cloneable methods
        return obj1.compareTo(obj2) > 0 ? obj1 : obj2;
    }

    public static void main(String[] args) throws Exception {
        // Test findMax with different types
        Integer[] intArray = {3, 7, 2, 9, 1};
        System.out.println("Max integer: " + findMax(intArray));

        String[] strArray = {"apple", "zebra", "banana"};
        System.out.println("Max string: " + findMax(strArray));

        Double[] doubleArray = {3.14, 2.71, 1.41, 9.81};
        System.out.println("Max double: " + findMax(doubleArray));

        // Test NumberBox - only numeric types allowed
        NumberBox<Integer> intBox = new NumberBox<>(42);
        System.out.println("\\nInteger box: " + intBox.getValue());
        System.out.println("As double: " + intBox.doubleValue());
        System.out.println("As int: " + intBox.intValue());

        NumberBox<Double> doubleBox = new NumberBox<>(3.14159);
        System.out.println("\\nDouble box: " + doubleBox.getValue());
        System.out.println("As int: " + doubleBox.intValue());

        // Test sumNumbers
        Integer[] nums = {1, 2, 3, 4, 5};
        System.out.println("\\nSum of integers: " + sumNumbers(nums));

        Double[] decimals = {1.5, 2.5, 3.5};
        System.out.println("Sum of doubles: " + sumNumbers(decimals));

        // These would cause compile-time errors:
        // NumberBox<String> strBox = new NumberBox<>("text");	// Error: String is not a Number
        // findMax(new Object[]{});	// Error: Object doesn't implement Comparable
    }
}`,
    hint1: `Use 'extends' keyword for bounds: <T extends SomeClass>. For classes that implement interfaces, you still use 'extends', not 'implements'.`,
    hint2: `Multiple bounds are separated by '&': <T extends Class & Interface1 & Interface2>. The class (if any) must come first.`,
    whyItMatters: `Bounded type parameters are crucial for writing generic code that needs to call specific methods.

**Production Pattern:**
\`\`\`java
// Only Number and its subtypes
class NumberBox<T extends Number> {
    private T value;

    public double doubleValue() {
        return value.doubleValue();	// Can call Number methods
    }
}

// Usage
NumberBox<Integer> intBox = new NumberBox<>(42);
NumberBox<Double> doubleBox = new NumberBox<>(3.14);
// NumberBox<String> strBox = new NumberBox<>();	// Compile error!
\`\`\`

**Practical Benefits:**
- Guarantees presence of specific methods
- Compile-time type safety
- Collections.sort() requires Comparable
- Avoids ClassCastException errors`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify findMax with Integer array
class Test1 {
    @Test
    public void test() {
        Integer[] arr = {3, 7, 2, 9, 1};
        Integer max = BoundedTypes.findMax(arr);
        assertEquals(Integer.valueOf(9), max);
    }
}

// Test2: Verify findMax with String array
class Test2 {
    @Test
    public void test() {
        String[] arr = {"apple", "zebra", "banana"};
        String max = BoundedTypes.findMax(arr);
        assertEquals("zebra", max);
    }
}

// Test3: Verify NumberBox with Integer
class Test3 {
    @Test
    public void test() {
        BoundedTypes.NumberBox<Integer> box = new BoundedTypes.NumberBox<>(42);
        assertEquals(42.0, box.doubleValue(), 0.001);
    }
}

// Test4: Verify NumberBox intValue method
class Test4 {
    @Test
    public void test() {
        BoundedTypes.NumberBox<Double> box = new BoundedTypes.NumberBox<>(3.14);
        assertEquals(3, box.intValue());
    }
}

// Test5: Verify sumNumbers with Integer array
class Test5 {
    @Test
    public void test() {
        Integer[] nums = {1, 2, 3, 4, 5};
        double sum = BoundedTypes.sumNumbers(nums);
        assertEquals(15.0, sum, 0.001);
    }
}

// Test6: Verify sumNumbers with Double array
class Test6 {
    @Test
    public void test() {
        Double[] nums = {1.5, 2.5, 3.5};
        double sum = BoundedTypes.sumNumbers(nums);
        assertEquals(7.5, sum, 0.001);
    }
}

// Test7: Verify findMax with Double array
class Test7 {
    @Test
    public void test() {
        Double[] arr = {3.14, 2.71, 9.81, 1.41};
        Double max = BoundedTypes.findMax(arr);
        assertEquals(9.81, max, 0.001);
    }
}

// Test8: Verify NumberBox getValue method
class Test8 {
    @Test
    public void test() {
        BoundedTypes.NumberBox<Integer> box = new BoundedTypes.NumberBox<>(100);
        assertEquals(Integer.valueOf(100), box.getValue());
    }
}

// Test9: Verify sumNumbers with mixed Number types
class Test9 {
    @Test
    public void test() {
        Number[] nums = {1, 2.5, 3};
        double sum = BoundedTypes.sumNumbers(nums);
        assertEquals(6.5, sum, 0.001);
    }
}

// Test10: Verify findMax with empty array returns null
class Test10 {
    @Test
    public void test() {
        Integer[] arr = {};
        Integer max = BoundedTypes.findMax(arr);
        assertNull(max);
    }
}`,
    translations: {
        ru: {
            title: 'Ограниченные параметры типа',
            solutionCode: `public class BoundedTypes {
    // Обобщенный метод с верхней границей - T должен реализовывать Comparable
    public static <T extends Comparable<T>> T findMax(T[] array) {
        if (array == null || array.length == 0) {
            return null;
        }

        T max = array[0];
        for (int i = 1; i < array.length; i++) {
            // Можем вызвать compareTo, потому что T extends Comparable<T>
            if (array[i].compareTo(max) > 0) {
                max = array[i];
            }
        }
        return max;
    }

    // Обобщенный класс с верхней границей - T должен расширять Number
    static class NumberBox<T extends Number> {
        private T value;

        public NumberBox(T value) {
            this.value = value;
        }

        // Можем вызывать методы Number, потому что T extends Number
        public double doubleValue() {
            return value.doubleValue();
        }

        public int intValue() {
            return value.intValue();
        }

        public T getValue() {
            return value;
        }
    }

    // Обобщенный метод, который суммирует числовые значения
    public static <T extends Number> double sumNumbers(T[] numbers) {
        double sum = 0;
        for (T num : numbers) {
            // Можем вызвать doubleValue(), потому что T extends Number
            sum += num.doubleValue();
        }
        return sum;
    }

    // Множественные границы: должен реализовывать и Comparable, и Cloneable
    public static <T extends Comparable<T> & Cloneable> T cloneAndCompare(T obj1, T obj2) throws CloneNotSupportedException {
        // Можем использовать методы и Comparable, и Cloneable
        return obj1.compareTo(obj2) > 0 ? obj1 : obj2;
    }

    public static void main(String[] args) throws Exception {
        // Тестируем findMax с разными типами
        Integer[] intArray = {3, 7, 2, 9, 1};
        System.out.println("Максимальное целое: " + findMax(intArray));

        String[] strArray = {"apple", "zebra", "banana"};
        System.out.println("Максимальная строка: " + findMax(strArray));

        Double[] doubleArray = {3.14, 2.71, 1.41, 9.81};
        System.out.println("Максимальное double: " + findMax(doubleArray));

        // Тестируем NumberBox - разрешены только числовые типы
        NumberBox<Integer> intBox = new NumberBox<>(42);
        System.out.println("\\nInteger box: " + intBox.getValue());
        System.out.println("Как double: " + intBox.doubleValue());
        System.out.println("Как int: " + intBox.intValue());

        NumberBox<Double> doubleBox = new NumberBox<>(3.14159);
        System.out.println("\\nDouble box: " + doubleBox.getValue());
        System.out.println("Как int: " + doubleBox.intValue());

        // Тестируем sumNumbers
        Integer[] nums = {1, 2, 3, 4, 5};
        System.out.println("\\nСумма целых: " + sumNumbers(nums));

        Double[] decimals = {1.5, 2.5, 3.5};
        System.out.println("Сумма double: " + sumNumbers(decimals));

        // Это вызовет ошибки компиляции:
        // NumberBox<String> strBox = new NumberBox<>("text");	// Ошибка: String не Number
        // findMax(new Object[]{});	// Ошибка: Object не реализует Comparable
    }
}`,
            description: `Изучите ограничение обобщенных типов с помощью границ в Java.

**Требования:**
1. Создайте метод findMax<T extends Comparable<T>>, который находит максимум в массиве
2. Создайте класс NumberBox<T extends Number>, который принимает только числовые типы
3. Добавьте методы в NumberBox: doubleValue() и intValue()
4. Создайте метод sumNumbers<T extends Number>, который суммирует числовые значения
5. Создайте класс с множественными границами: <T extends Comparable<T> & Cloneable>
6. Продемонстрируйте, что можно использовать только типы, соответствующие границам

Ограниченные параметры типа ограничивают, какие типы можно использовать с обобщениями, обеспечивая наличие у типа определенных методов или реализацию определенных интерфейсов.`,
            hint1: `Используйте ключевое слово 'extends' для границ: <T extends SomeClass>. Для классов, реализующих интерфейсы, вы все равно используете 'extends', а не 'implements'.`,
            hint2: `Множественные границы разделяются '&': <T extends Class & Interface1 & Interface2>. Класс (если есть) должен идти первым.`,
            whyItMatters: `Ограниченные параметры типа критически важны для написания обобщенного кода, которому нужно вызывать определенные методы.

**Продакшен паттерн:**
\`\`\`java
// Только Number и его подтипы
class NumberBox<T extends Number> {
    private T value;

    public double doubleValue() {
        return value.doubleValue();	// Можем вызвать методы Number
    }
}

// Использование
NumberBox<Integer> intBox = new NumberBox<>(42);
NumberBox<Double> doubleBox = new NumberBox<>(3.14);
// NumberBox<String> strBox = new NumberBox<>();	// Ошибка компиляции!
\`\`\`

**Практические преимущества:**
- Гарантирует наличие определенных методов
- Безопасность типов во время компиляции
- Collections.sort() требует Comparable
- Избегает ошибок ClassCastException`
        },
        uz: {
            title: 'Chegaralangan tur parametrlari',
            solutionCode: `public class BoundedTypes {
    // Yuqori chegara bilan umumiy metod - T Comparable ni amalga oshirishi kerak
    public static <T extends Comparable<T>> T findMax(T[] array) {
        if (array == null || array.length == 0) {
            return null;
        }

        T max = array[0];
        for (int i = 1; i < array.length; i++) {
            // compareTo ni chaqira olamiz, chunki T extends Comparable<T>
            if (array[i].compareTo(max) > 0) {
                max = array[i];
            }
        }
        return max;
    }

    // Yuqori chegara bilan umumiy klass - T Number ni kengaytirishi kerak
    static class NumberBox<T extends Number> {
        private T value;

        public NumberBox(T value) {
            this.value = value;
        }

        // Number metodlarini chaqira olamiz, chunki T extends Number
        public double doubleValue() {
            return value.doubleValue();
        }

        public int intValue() {
            return value.intValue();
        }

        public T getValue() {
            return value;
        }
    }

    // Raqamli qiymatlarni yig'adigan umumiy metod
    public static <T extends Number> double sumNumbers(T[] numbers) {
        double sum = 0;
        for (T num : numbers) {
            // doubleValue() ni chaqira olamiz, chunki T extends Number
            sum += num.doubleValue();
        }
        return sum;
    }

    // Ko'p chegaralar: Comparable va Cloneable ni amalga oshirishi kerak
    public static <T extends Comparable<T> & Cloneable> T cloneAndCompare(T obj1, T obj2) throws CloneNotSupportedException {
        // Comparable va Cloneable metodlarini ishlatishimiz mumkin
        return obj1.compareTo(obj2) > 0 ? obj1 : obj2;
    }

    public static void main(String[] args) throws Exception {
        // findMax ni turli turlar bilan sinaymiz
        Integer[] intArray = {3, 7, 2, 9, 1};
        System.out.println("Maksimal butun son: " + findMax(intArray));

        String[] strArray = {"apple", "zebra", "banana"};
        System.out.println("Maksimal satr: " + findMax(strArray));

        Double[] doubleArray = {3.14, 2.71, 1.41, 9.81};
        System.out.println("Maksimal double: " + findMax(doubleArray));

        // NumberBox ni sinaymiz - faqat raqamli turlar ruxsat etilgan
        NumberBox<Integer> intBox = new NumberBox<>(42);
        System.out.println("\\nInteger box: " + intBox.getValue());
        System.out.println("Double sifatida: " + intBox.doubleValue());
        System.out.println("Int sifatida: " + intBox.intValue());

        NumberBox<Double> doubleBox = new NumberBox<>(3.14159);
        System.out.println("\\nDouble box: " + doubleBox.getValue());
        System.out.println("Int sifatida: " + doubleBox.intValue());

        // sumNumbers ni sinaymiz
        Integer[] nums = {1, 2, 3, 4, 5};
        System.out.println("\\nButun sonlar yig'indisi: " + sumNumbers(nums));

        Double[] decimals = {1.5, 2.5, 3.5};
        System.out.println("Double sonlar yig'indisi: " + sumNumbers(decimals));

        // Bu qatorlar kompilyatsiya xatolariga olib keladi:
        // NumberBox<String> strBox = new NumberBox<>("text");	// Xato: String Number emas
        // findMax(new Object[]{});	// Xato: Object Comparable ni amalga oshirmaydi
    }
}`,
            description: `Java da chegaralar yordamida umumiy turlarni cheklashni o'rganing.

**Talablar:**
1. Massivdagi maksimumni topadigan findMax<T extends Comparable<T>> metodini yarating
2. Faqat raqamli turlarni qabul qiluvchi NumberBox<T extends Number> klassini yarating
3. NumberBox ga metodlar qo'shing: doubleValue() va intValue()
4. Raqamli qiymatlarni yig'uvchi sumNumbers<T extends Number> metodini yarating
5. Ko'p chegarali klass yarating: <T extends Comparable<T> & Cloneable>
6. Faqat chegara talablariga mos keladigan turlardan foydalanish mumkinligini ko'rsating

Chegaralangan tur parametrlari qaysi turlarni umumiy tiplar bilan ishlatish mumkinligini cheklaydi va tur ma'lum metodlarga ega yoki ma'lum interfeyslarni amalga oshirishini ta'minlaydi.`,
            hint1: `Chegaralar uchun 'extends' kalit so'zidan foydalaning: <T extends SomeClass>. Interfeyslarni amalga oshiruvchi klasslar uchun ham 'extends' ishlatiladi, 'implements' emas.`,
            hint2: `Ko'p chegaralar '&' bilan ajratiladi: <T extends Class & Interface1 & Interface2>. Klass (agar bor bo'lsa) birinchi bo'lishi kerak.`,
            whyItMatters: `Chegaralangan tur parametrlari ma'lum metodlarni chaqirishi kerak bo'lgan umumiy kod yozish uchun juda muhim.

**Ishlab chiqarish patterni:**
\`\`\`java
// Faqat Number va uning kichik turlari
class NumberBox<T extends Number> {
    private T value;

    public double doubleValue() {
        return value.doubleValue();	// Number metodlarini chaqira olamiz
    }
}

// Foydalanish
NumberBox<Integer> intBox = new NumberBox<>(42);
NumberBox<Double> doubleBox = new NumberBox<>(3.14);
// NumberBox<String> strBox = new NumberBox<>();	// Kompilyatsiya xatosi!
\`\`\`

**Amaliy foydalari:**
- Ma'lum metodlarning mavjudligini kafolatlaydi
- Kompilyatsiya vaqtida tur xavfsizligi
- Collections.sort() Comparable ni talab qiladi
- ClassCastException xatolarini oldini oladi`
        }
    }
};

export default task;
