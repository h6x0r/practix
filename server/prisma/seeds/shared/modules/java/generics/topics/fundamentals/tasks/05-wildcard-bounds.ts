import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-wildcard-bounds',
    title: 'Upper and Lower Bounded Wildcards',
    difficulty: 'medium',
    tags: ['java', 'generics', 'wildcards', 'bounds', 'covariance', 'contravariance'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master upper and lower bounded wildcards and the PECS principle in Java.

**Requirements:**
1. Create sumNumbers method using <? extends Number> (upper bound)
2. Create addIntegers method using <? super Integer> (lower bound)
3. Demonstrate PECS: Producer Extends, Consumer Super
4. Show covariance with <? extends T> - can read but not write
5. Show contravariance with <? super T> - can write but not read (as specific type)
6. Create copyElements method demonstrating PECS principle

Upper bounded wildcards (<? extends T>) allow reading as T. Lower bounded wildcards (<? super T>) allow writing T values. PECS helps you choose the right bound.`,
    initialCode: `import java.util.*;

public class WildcardBounds {
    // Create sumNumbers with upper bound <? extends Number>
    // - Sum all numeric values

    // Create addIntegers with lower bound <? super Integer>
    // - Add integers to the list

    // Create copyElements demonstrating PECS
    // - Source: <? extends T> (producer)
    // - Destination: <? super T> (consumer)

    // Demonstrate what you can/cannot do with each bound

    public static void main(String[] args) {
        // Test upper bound with different numeric types

        // Test lower bound with supertype lists

        // Demonstrate PECS principle with copy
    }
}`,
    solutionCode: `import java.util.*;

public class WildcardBounds {
    // UPPER BOUND: <? extends Number>
    // Producer - can READ as Number, cannot WRITE
    public static double sumNumbers(List<? extends Number> numbers) {
        double sum = 0;
        for (Number num : numbers) {
            // Can read as Number (or its methods)
            sum += num.doubleValue();
        }
        // Cannot add: numbers.add(5);	// Compile error!
        // Cannot add: numbers.add(5.0);	// Compile error!
        return sum;
    }

    // LOWER BOUND: <? super Integer>
    // Consumer - can WRITE Integer, cannot READ (as specific type)
    public static void addIntegers(List<? super Integer> list) {
        // Can write Integer values
        list.add(1);
        list.add(2);
        list.add(3);

        // Can read as Object only
        Object obj = list.get(0);
        // Cannot read as Integer: Integer num = list.get(0);	// Compile error!
    }

    // PECS Principle: Producer Extends, Consumer Super
    // Source is producer (extends), Destination is consumer (super)
    public static <T> void copyElements(
            List<? extends T> source,     // Producer: extends
            List<? super T> destination   // Consumer: super
    ) {
        for (T element : source) {
            destination.add(element);
        }
    }

    // Upper bound allows any subtype of Number
    public static void processNumbers(List<? extends Number> numbers) {
        System.out.print("Numbers: [");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i));
            if (i < numbers.size() - 1) System.out.print(", ");
        }
        System.out.println("]");

        // Can call Number methods
        if (!numbers.isEmpty()) {
            System.out.println("First as double: " + numbers.get(0).doubleValue());
        }
    }

    // Lower bound allows any supertype of Integer
    public static void fillWithNumbers(List<? super Integer> list) {
        // Can add Integer and its subtypes
        for (int i = 1; i <= 5; i++) {
            list.add(i);
        }
    }

    public static void main(String[] args) {
        // UPPER BOUND EXAMPLES
        System.out.println("=== Upper Bound: <? extends Number> ===");

        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        System.out.println("Sum of integers: " + sumNumbers(intList));

        List<Double> doubleList = Arrays.asList(1.5, 2.5, 3.5);
        System.out.println("Sum of doubles: " + sumNumbers(doubleList));

        List<Number> numList = Arrays.asList(1, 2.5, 3, 4.5);
        System.out.println("Sum of mixed numbers: " + sumNumbers(numList));

        processNumbers(intList);
        processNumbers(doubleList);

        // LOWER BOUND EXAMPLES
        System.out.println("\\n=== Lower Bound: <? super Integer> ===");

        List<Integer> integerList = new ArrayList<>();
        addIntegers(integerList);
        System.out.println("Integer list: " + integerList);

        List<Number> numberList = new ArrayList<>();
        addIntegers(numberList);	// Number is supertype of Integer
        System.out.println("Number list: " + numberList);

        List<Object> objectList = new ArrayList<>();
        addIntegers(objectList);	// Object is supertype of Integer
        System.out.println("Object list: " + objectList);

        // PECS PRINCIPLE
        System.out.println("\\n=== PECS: Producer Extends, Consumer Super ===");

        List<Integer> source = Arrays.asList(10, 20, 30);
        List<Number> dest1 = new ArrayList<>();
        copyElements(source, dest1);	// Integer -> Number (OK)
        System.out.println("Copied to Number list: " + dest1);

        List<Object> dest2 = new ArrayList<>();
        copyElements(source, dest2);	// Integer -> Object (OK)
        System.out.println("Copied to Object list: " + dest2);

        // This demonstrates the key principle:
        // - Use <? extends T> when you GET values (producer)
        // - Use <? super T> when you PUT values (consumer)
        // - Use T when you do BOTH

        System.out.println("\\n=== Summary ===");
        System.out.println("Upper bound <? extends T>: Read as T, cannot write");
        System.out.println("Lower bound <? super T>: Write T, read as Object");
        System.out.println("PECS: Producer Extends, Consumer Super");
    }
}`,
    hint1: `Upper bound <? extends T>: Use when you're READING from a structure (producer). You can read as T but cannot add elements (except null).`,
    hint2: `Lower bound <? super T>: Use when you're WRITING to a structure (consumer). You can add T values but can only read as Object.`,
    whyItMatters: `Bounded wildcards and PECS are fundamental to understanding variance in Java generics.

**Production Pattern:**
\`\`\`java
// Producer Extends: read Number and subtypes
public static double sumNumbers(List<? extends Number> numbers) {
    double sum = 0;
    for (Number num : numbers) {
        sum += num.doubleValue();	// Can read
    }
    return sum;
}

// Consumer Super: write Integer to List<Number> or List<Object>
public static void addIntegers(List<? super Integer> list) {
    list.add(1);
    list.add(2);	// Can write Integer
}
\`\`\`

**Practical Benefits:**
- PECS: Producer Extends, Consumer Super
- Collections.copy() uses this pattern
- Flexible, type-safe APIs
- Key skill for senior developers`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;

// Test1: Verify sumNumbers with Integer list
class Test1 {
    @Test
    public void test() {
        List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
        double sum = WildcardBounds.sumNumbers(list);
        assertEquals(15.0, sum, 0.001);
    }
}

// Test2: Verify sumNumbers with Double list
class Test2 {
    @Test
    public void test() {
        List<Double> list = Arrays.asList(1.5, 2.5, 3.5);
        double sum = WildcardBounds.sumNumbers(list);
        assertEquals(7.5, sum, 0.001);
    }
}

// Test3: Verify addIntegers adds values to list
class Test3 {
    @Test
    public void test() {
        List<Integer> list = new ArrayList<>();
        WildcardBounds.addIntegers(list);
        assertEquals(3, list.size());
        assertTrue(list.contains(1));
    }
}

// Test4: Verify addIntegers with Number list
class Test4 {
    @Test
    public void test() {
        List<Number> list = new ArrayList<>();
        WildcardBounds.addIntegers(list);
        assertEquals(3, list.size());
    }
}

// Test5: Verify copyElements from Integer to Number
class Test5 {
    @Test
    public void test() {
        List<Integer> source = Arrays.asList(10, 20, 30);
        List<Number> dest = new ArrayList<>();
        WildcardBounds.copyElements(source, dest);
        assertEquals(3, dest.size());
        assertEquals(10, dest.get(0));
    }
}

// Test6: Verify processNumbers with Integer list outputs correctly
class Test6 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            List<Integer> list = Arrays.asList(1, 2, 3);
            WildcardBounds.processNumbers(list);
            String output = out.toString();
            assertTrue(output.contains("1") && output.contains("2") && output.contains("3"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test7: Verify fillWithNumbers populates list
class Test7 {
    @Test
    public void test() {
        List<Integer> list = new ArrayList<>();
        WildcardBounds.fillWithNumbers(list);
        assertEquals(5, list.size());
        assertEquals(Integer.valueOf(1), list.get(0));
    }
}

// Test8: Verify sumNumbers with Number list
class Test8 {
    @Test
    public void test() {
        List<Number> list = Arrays.asList(1, 2.5, 3, 4.5);
        double sum = WildcardBounds.sumNumbers(list);
        assertEquals(11.0, sum, 0.001);
    }
}

// Test9: Verify copyElements from Integer to Object
class Test9 {
    @Test
    public void test() {
        List<Integer> source = Arrays.asList(1, 2);
        List<Object> dest = new ArrayList<>();
        WildcardBounds.copyElements(source, dest);
        assertEquals(2, dest.size());
    }
}

// Test10: Verify addIntegers with Object list
class Test10 {
    @Test
    public void test() {
        List<Object> list = new ArrayList<>();
        WildcardBounds.addIntegers(list);
        assertEquals(3, list.size());
    }
}`,
    translations: {
        ru: {
            title: 'Ограниченные подстановочные знаки',
            solutionCode: `import java.util.*;

public class WildcardBounds {
    // ВЕРХНЯЯ ГРАНИЦА: <? extends Number>
    // Производитель - можно ЧИТАТЬ как Number, нельзя ПИСАТЬ
    public static double sumNumbers(List<? extends Number> numbers) {
        double sum = 0;
        for (Number num : numbers) {
            // Можно читать как Number (или его методы)
            sum += num.doubleValue();
        }
        // Нельзя добавлять: numbers.add(5);	// Ошибка компиляции!
        // Нельзя добавлять: numbers.add(5.0);	// Ошибка компиляции!
        return sum;
    }

    // НИЖНЯЯ ГРАНИЦА: <? super Integer>
    // Потребитель - можно ПИСАТЬ Integer, нельзя ЧИТАТЬ (как конкретный тип)
    public static void addIntegers(List<? super Integer> list) {
        // Можно писать значения Integer
        list.add(1);
        list.add(2);
        list.add(3);

        // Можно читать только как Object
        Object obj = list.get(0);
        // Нельзя читать как Integer: Integer num = list.get(0);	// Ошибка компиляции!
    }

    // Принцип PECS: Producer Extends, Consumer Super
    // Источник - производитель (extends), Назначение - потребитель (super)
    public static <T> void copyElements(
            List<? extends T> source,     // Производитель: extends
            List<? super T> destination   // Потребитель: super
    ) {
        for (T element : source) {
            destination.add(element);
        }
    }

    // Верхняя граница позволяет любой подтип Number
    public static void processNumbers(List<? extends Number> numbers) {
        System.out.print("Числа: [");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i));
            if (i < numbers.size() - 1) System.out.print(", ");
        }
        System.out.println("]");

        // Можно вызывать методы Number
        if (!numbers.isEmpty()) {
            System.out.println("Первое как double: " + numbers.get(0).doubleValue());
        }
    }

    // Нижняя граница позволяет любой супертип Integer
    public static void fillWithNumbers(List<? super Integer> list) {
        // Можно добавлять Integer и его подтипы
        for (int i = 1; i <= 5; i++) {
            list.add(i);
        }
    }

    public static void main(String[] args) {
        // ПРИМЕРЫ ВЕРХНЕЙ ГРАНИЦЫ
        System.out.println("=== Верхняя граница: <? extends Number> ===");

        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        System.out.println("Сумма целых: " + sumNumbers(intList));

        List<Double> doubleList = Arrays.asList(1.5, 2.5, 3.5);
        System.out.println("Сумма double: " + sumNumbers(doubleList));

        List<Number> numList = Arrays.asList(1, 2.5, 3, 4.5);
        System.out.println("Сумма смешанных чисел: " + sumNumbers(numList));

        processNumbers(intList);
        processNumbers(doubleList);

        // ПРИМЕРЫ НИЖНЕЙ ГРАНИЦЫ
        System.out.println("\\n=== Нижняя граница: <? super Integer> ===");

        List<Integer> integerList = new ArrayList<>();
        addIntegers(integerList);
        System.out.println("Список Integer: " + integerList);

        List<Number> numberList = new ArrayList<>();
        addIntegers(numberList);	// Number - супертип Integer
        System.out.println("Список Number: " + numberList);

        List<Object> objectList = new ArrayList<>();
        addIntegers(objectList);	// Object - супертип Integer
        System.out.println("Список Object: " + objectList);

        // ПРИНЦИП PECS
        System.out.println("\\n=== PECS: Producer Extends, Consumer Super ===");

        List<Integer> source = Arrays.asList(10, 20, 30);
        List<Number> dest1 = new ArrayList<>();
        copyElements(source, dest1);	// Integer -> Number (OK)
        System.out.println("Скопировано в список Number: " + dest1);

        List<Object> dest2 = new ArrayList<>();
        copyElements(source, dest2);	// Integer -> Object (OK)
        System.out.println("Скопировано в список Object: " + dest2);

        // Это демонстрирует ключевой принцип:
        // - Используйте <? extends T>, когда ПОЛУЧАЕТЕ значения (производитель)
        // - Используйте <? super T>, когда ПОМЕЩАЕТЕ значения (потребитель)
        // - Используйте T, когда делаете ОБА действия

        System.out.println("\\n=== Резюме ===");
        System.out.println("Верхняя граница <? extends T>: Читать как T, нельзя писать");
        System.out.println("Нижняя граница <? super T>: Писать T, читать как Object");
        System.out.println("PECS: Producer Extends, Consumer Super");
    }
}`,
            description: `Освойте ограниченные подстановочные знаки и принцип PECS в Java.

**Требования:**
1. Создайте метод sumNumbers, используя <? extends Number> (верхняя граница)
2. Создайте метод addIntegers, используя <? super Integer> (нижняя граница)
3. Продемонстрируйте PECS: Producer Extends, Consumer Super
4. Покажите ковариантность с <? extends T> - можно читать, но не писать
5. Покажите контравариантность с <? super T> - можно писать, но не читать (как конкретный тип)
6. Создайте метод copyElements, демонстрирующий принцип PECS

Верхние ограниченные подстановочные знаки (<? extends T>) позволяют читать как T. Нижние ограниченные подстановочные знаки (<? super T>) позволяют писать значения T. PECS помогает выбрать правильную границу.`,
            hint1: `Верхняя граница <? extends T>: Используйте, когда ЧИТАЕТЕ из структуры (производитель). Можно читать как T, но нельзя добавлять элементы (кроме null).`,
            hint2: `Нижняя граница <? super T>: Используйте, когда ПИШЕТЕ в структуру (потребитель). Можно добавлять значения T, но можно читать только как Object.`,
            whyItMatters: `Ограниченные подстановочные знаки и PECS - основа понимания вариантности в обобщениях Java.

**Продакшен паттерн:**
\`\`\`java
// Producer Extends: читаем Number и подтипы
public static double sumNumbers(List<? extends Number> numbers) {
    double sum = 0;
    for (Number num : numbers) {
        sum += num.doubleValue();	// Можем читать
    }
    return sum;
}

// Consumer Super: пишем Integer в List<Number> или List<Object>
public static void addIntegers(List<? super Integer> list) {
    list.add(1);
    list.add(2);	// Можем писать Integer
}
\`\`\`

**Практические преимущества:**
- PECS: Producer Extends, Consumer Super
- Collections.copy() использует этот паттерн
- Гибкие, типобезопасные API
- Ключевой навык для senior разработчиков`
        },
        uz: {
            title: 'Yuqori va quyi chegarali wildcard belgilari',
            solutionCode: `import java.util.*;

public class WildcardBounds {
    // YUQORI CHEGARA: <? extends Number>
    // Ishlab chiqaruvchi - Number sifatida O'QISH mumkin, YOZISH mumkin emas
    public static double sumNumbers(List<? extends Number> numbers) {
        double sum = 0;
        for (Number num : numbers) {
            // Number sifatida o'qish mumkin (yoki uning metodlari)
            sum += num.doubleValue();
        }
        // Qo'shish mumkin emas: numbers.add(5);	// Kompilyatsiya xatosi!
        // Qo'shish mumkin emas: numbers.add(5.0);	// Kompilyatsiya xatosi!
        return sum;
    }

    // QUYI CHEGARA: <? super Integer>
    // Iste'molchi - Integer YOZISH mumkin, O'QISH mumkin emas (aniq tur sifatida)
    public static void addIntegers(List<? super Integer> list) {
        // Integer qiymatlarini yozish mumkin
        list.add(1);
        list.add(2);
        list.add(3);

        // Faqat Object sifatida o'qish mumkin
        Object obj = list.get(0);
        // Integer sifatida o'qib bo'lmaydi: Integer num = list.get(0);	// Kompilyatsiya xatosi!
    }

    // PECS printsipi: Producer Extends, Consumer Super
    // Manba - ishlab chiqaruvchi (extends), Maqsad - iste'molchi (super)
    public static <T> void copyElements(
            List<? extends T> source,     // Ishlab chiqaruvchi: extends
            List<? super T> destination   // Iste'molchi: super
    ) {
        for (T element : source) {
            destination.add(element);
        }
    }

    // Yuqori chegara Number ning har qanday kichik turini qabul qiladi
    public static void processNumbers(List<? extends Number> numbers) {
        System.out.print("Raqamlar: [");
        for (int i = 0; i < numbers.size(); i++) {
            System.out.print(numbers.get(i));
            if (i < numbers.size() - 1) System.out.print(", ");
        }
        System.out.println("]");

        // Number metodlarini chaqirish mumkin
        if (!numbers.isEmpty()) {
            System.out.println("Birinchisi double sifatida: " + numbers.get(0).doubleValue());
        }
    }

    // Quyi chegara Integer ning har qanday katta turini qabul qiladi
    public static void fillWithNumbers(List<? super Integer> list) {
        // Integer va uning kichik turlarini qo'shish mumkin
        for (int i = 1; i <= 5; i++) {
            list.add(i);
        }
    }

    public static void main(String[] args) {
        // YUQORI CHEGARA MISOLLARI
        System.out.println("=== Yuqori chegara: <? extends Number> ===");

        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        System.out.println("Butun sonlar yig'indisi: " + sumNumbers(intList));

        List<Double> doubleList = Arrays.asList(1.5, 2.5, 3.5);
        System.out.println("Double sonlar yig'indisi: " + sumNumbers(doubleList));

        List<Number> numList = Arrays.asList(1, 2.5, 3, 4.5);
        System.out.println("Aralash sonlar yig'indisi: " + sumNumbers(numList));

        processNumbers(intList);
        processNumbers(doubleList);

        // QUYI CHEGARA MISOLLARI
        System.out.println("\\n=== Quyi chegara: <? super Integer> ===");

        List<Integer> integerList = new ArrayList<>();
        addIntegers(integerList);
        System.out.println("Integer ro'yxati: " + integerList);

        List<Number> numberList = new ArrayList<>();
        addIntegers(numberList);	// Number Integer ning katta turi
        System.out.println("Number ro'yxati: " + numberList);

        List<Object> objectList = new ArrayList<>();
        addIntegers(objectList);	// Object Integer ning katta turi
        System.out.println("Object ro'yxati: " + objectList);

        // PECS PRINTSIPI
        System.out.println("\\n=== PECS: Producer Extends, Consumer Super ===");

        List<Integer> source = Arrays.asList(10, 20, 30);
        List<Number> dest1 = new ArrayList<>();
        copyElements(source, dest1);	// Integer -> Number (OK)
        System.out.println("Number ro'yxatiga nusxalandi: " + dest1);

        List<Object> dest2 = new ArrayList<>();
        copyElements(source, dest2);	// Integer -> Object (OK)
        System.out.println("Object ro'yxatiga nusxalandi: " + dest2);

        // Bu asosiy printsipni ko'rsatadi:
        // - Qiymatlarni OLGANINGIZDA <? extends T> dan foydalaning (ishlab chiqaruvchi)
        // - Qiymatlarni QOYSANGIZ <? super T> dan foydalaning (iste'molchi)
        // - IKKALA ishni ham qilsangiz T dan foydalaning

        System.out.println("\\n=== Xulosa ===");
        System.out.println("Yuqori chegara <? extends T>: T sifatida o'qish, yozish mumkin emas");
        System.out.println("Quyi chegara <? super T>: T yozish, Object sifatida o'qish");
        System.out.println("PECS: Producer Extends, Consumer Super");
    }
}`,
            description: `Java da yuqori va quyi chegarali wildcard va PECS prinsipini o'zlashtirang.

**Talablar:**
1. <? extends Number> (yuqori chegara) dan foydalangan holda sumNumbers metodini yarating
2. <? super Integer> (quyi chegara) dan foydalangan holda addIntegers metodini yarating
3. PECS ni ko'rsating: Producer Extends, Consumer Super
4. <? extends T> bilan kovariantlikni ko'rsating - o'qish mumkin, yozish mumkin emas
5. <? super T> bilan kontravariantlikni ko'rsating - yozish mumkin, o'qish mumkin emas (aniq tur sifatida)
6. PECS prinsipini ko'rsatuvchi copyElements metodini yarating

Yuqori chegarali wildcard belgilari (<? extends T>) T sifatida o'qish imkonini beradi. Quyi chegarali wildcard belgilari (<? super T>) T qiymatlarini yozish imkonini beradi. PECS to'g'ri chegarani tanlashga yordam beradi.`,
            hint1: `Yuqori chegara <? extends T>: Strukturadan O'QIYOTGANINGIZDA foydalaning (ishlab chiqaruvchi). T sifatida o'qish mumkin, lekin elementlar qo'shib bo'lmaydi (null bundan mustasno).`,
            hint2: `Quyi chegara <? super T>: Strukturaga YOZAYOTGANINGIZDA foydalaning (iste'molchi). T qiymatlarini qo'shish mumkin, lekin faqat Object sifatida o'qish mumkin.`,
            whyItMatters: `Chegaralangan wildcard belgilari va PECS Java generics da variantlikni tushunishning asosi hisoblanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Producer Extends: Number va kichik turlarni o'qiymiz
public static double sumNumbers(List<? extends Number> numbers) {
    double sum = 0;
    for (Number num : numbers) {
        sum += num.doubleValue();	// O'qish mumkin
    }
    return sum;
}

// Consumer Super: Integer ni List<Number> yoki List<Object> ga yozamiz
public static void addIntegers(List<? super Integer> list) {
    list.add(1);
    list.add(2);	// Integer yozish mumkin
}
\`\`\`

**Amaliy foydalari:**
- PECS: Producer Extends, Consumer Super
- Collections.copy() bu patterndan foydalanadi
- Moslashuvchan, tur-xavfsiz API lar
- Senior dasturchilari uchun kalit ko'nikma`
        }
    }
};

export default task;
