import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-junit-assertions',
    title: 'JUnit Assertions',
    difficulty: 'easy',
    tags: ['java', 'testing', 'junit', 'assertions'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master different types of assertions in **JUnit 5** to thoroughly test a StringUtils class.

**Requirements:**
1. Create a StringUtils class with methods:
   1.1. reverse(String str) - reverses a string
   1.2. isPalindrome(String str) - checks if palindrome
   1.3. toUpperCase(String str) - converts to uppercase
   1.4. contains(String str, String substring) - checks if contains substring
   1.5. split(String str, String delimiter) - splits string into array

2. Create StringUtilsTest with comprehensive tests using:
   2.1. assertEquals() - for exact value comparison
   2.2. assertTrue() / assertFalse() - for boolean conditions
   2.3. assertNull() / assertNotNull() - for null checks
   2.4. assertThrows() - for exception testing
   2.5. assertAll() - for grouped assertions
   2.6. assertArrayEquals() - for array comparison

3. Test edge cases:
   3.1. Null inputs
   3.2. Empty strings
   3.3. Special characters

**Learning Goals:**
- Master different assertion types
- Learn when to use each assertion
- Practice grouped assertions with assertAll()
- Understand assertion error messages`,
    initialCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class StringUtils {
    // TODO: Implement reverse method

    // TODO: Implement isPalindrome method

    // TODO: Implement toUpperCase method

    // TODO: Implement contains method

    // TODO: Implement split method
}

class StringUtilsTest {
    // TODO: Write tests using different assertion types
}`,
    solutionCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class StringUtils {
    // Reverse a string
    public String reverse(String str) {
        if (str == null) {
            throw new IllegalArgumentException("String cannot be null");
        }
        return new StringBuilder(str).reverse().toString();
    }

    // Check if string is palindrome
    public boolean isPalindrome(String str) {
        if (str == null) {
            return false;
        }
        String reversed = reverse(str);
        return str.equals(reversed);
    }

    // Convert to uppercase
    public String toUpperCase(String str) {
        return str == null ? null : str.toUpperCase();
    }

    // Check if string contains substring
    public boolean contains(String str, String substring) {
        if (str == null || substring == null) {
            return false;
        }
        return str.contains(substring);
    }

    // Split string by delimiter
    public String[] split(String str, String delimiter) {
        if (str == null) {
            return null;
        }
        return str.split(delimiter);
    }
}

class StringUtilsTest {
    // Test reverse with assertEquals
    @Test
    void testReverse() {
        StringUtils utils = new StringUtils();
        assertEquals("olleh", utils.reverse("hello"));
        assertEquals("12321", utils.reverse("12321"));
    }

    // Test reverse with null throws exception
    @Test
    void testReverseWithNull() {
        StringUtils utils = new StringUtils();
        assertThrows(IllegalArgumentException.class, () -> {
            utils.reverse(null);
        });
    }

    // Test palindrome with assertTrue/assertFalse
    @Test
    void testIsPalindrome() {
        StringUtils utils = new StringUtils();
        assertTrue(utils.isPalindrome("racecar"));
        assertTrue(utils.isPalindrome("12321"));
        assertFalse(utils.isPalindrome("hello"));
        assertFalse(utils.isPalindrome(null));
    }

    // Test toUpperCase with assertNull/assertNotNull
    @Test
    void testToUpperCase() {
        StringUtils utils = new StringUtils();
        assertEquals("HELLO", utils.toUpperCase("hello"));
        assertNull(utils.toUpperCase(null));
        assertNotNull(utils.toUpperCase("test"));
    }

    // Test contains method
    @Test
    void testContains() {
        StringUtils utils = new StringUtils();
        assertTrue(utils.contains("hello world", "world"));
        assertFalse(utils.contains("hello world", "java"));
        assertFalse(utils.contains(null, "test"));
    }

    // Test split with assertArrayEquals
    @Test
    void testSplit() {
        StringUtils utils = new StringUtils();
        String[] expected = {"hello", "world", "java"};
        assertArrayEquals(expected, utils.split("hello,world,java", ","));
    }

    // Test split with null returns null
    @Test
    void testSplitWithNull() {
        StringUtils utils = new StringUtils();
        assertNull(utils.split(null, ","));
    }

    // Group multiple assertions with assertAll
    @Test
    void testMultipleAssertions() {
        StringUtils utils = new StringUtils();
        String input = "hello";

        assertAll("String operations",
            () -> assertEquals(5, input.length()),
            () -> assertEquals("HELLO", utils.toUpperCase(input)),
            () -> assertEquals("olleh", utils.reverse(input)),
            () -> assertTrue(utils.contains(input, "ell"))
        );
    }
}`,
    hint1: `Start by implementing each StringUtils method with proper null checks. Remember to throw exceptions for invalid inputs where appropriate.`,
    hint2: `Use assertEquals for comparing values, assertTrue/assertFalse for boolean results, and assertThrows for testing exceptions. Use assertAll to group related assertions.`,
    whyItMatters: `Different assertion types help you write more expressive and maintainable tests. Using the right assertion makes test failures clearer and helps you quickly identify what went wrong. Grouped assertions with assertAll() ensure all checks run even if one fails, giving you complete test feedback.

**Production Pattern:**
\`\`\`java
@Test
void testUserValidation() {
    User user = new User("john@example.com");
    assertAll("User validation",
        () -> assertNotNull(user.getEmail()),
        () -> assertTrue(user.isValidEmail()),
        () -> assertFalse(user.getEmail().isEmpty())
    );
}
\`\`\`

**Practical Benefits:**
- Complete diagnostics when test fails
- Improved readability and maintainability of tests`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Test assertEquals with message
class Test1 {
    @Test
    public void test() {
        assertEquals("Values should be equal", 10, 10);
    }
}

// Test2: Test assertTrue with condition
class Test2 {
    @Test
    public void test() {
        int value = 15;
        assertTrue("Value should be greater than 10", value > 10);
    }
}

// Test3: Test assertFalse with condition
class Test3 {
    @Test
    public void test() {
        String text = "Hello";
        assertFalse("Text should not be empty", text.isEmpty());
    }
}

// Test4: Test assertNotEquals
class Test4 {
    @Test
    public void test() {
        assertNotEquals(5, 10);
    }
}

// Test5: Test assertSame for object identity
class Test5 {
    @Test
    public void test() {
        String str = "test";
        assertSame(str, str);
    }
}

// Test6: Test assertNotSame for different objects
class Test6 {
    @Test
    public void test() {
        String str1 = new String("test");
        String str2 = new String("test");
        assertNotSame(str1, str2);
    }
}

// Test7: Test assertArrayEquals with custom message
class Test7 {
    @Test
    public void test() {
        int[] expected = {1, 2, 3};
        int[] actual = {1, 2, 3};
        assertArrayEquals("Arrays should match", expected, actual);
    }
}

// Test8: Test assertEquals with delta for doubles
class Test8 {
    @Test
    public void test() {
        double expected = 10.5;
        double actual = 10.50001;
        assertEquals(expected, actual, 0.001);
    }
}

// Test9: Test fail method
class Test9 {
    @Test
    public void test() {
        boolean condition = true;
        if (!condition) {
            fail("This should not be reached");
        }
        assertTrue(condition);
    }
}

// Test10: Test combined assertions
class Test10 {
    @Test
    public void test() {
        String value = "JUnit";
        assertNotNull("Value should not be null", value);
        assertEquals("Should equal JUnit", "JUnit", value);
        assertTrue("Length should be 5", value.length() == 5);
    }
}
`,
    translations: {
        ru: {
            title: 'Утверждения JUnit',
            solutionCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class StringUtils {
    // Переворачиваем строку
    public String reverse(String str) {
        if (str == null) {
            throw new IllegalArgumentException("Строка не может быть null");
        }
        return new StringBuilder(str).reverse().toString();
    }

    // Проверяем, является ли строка палиндромом
    public boolean isPalindrome(String str) {
        if (str == null) {
            return false;
        }
        String reversed = reverse(str);
        return str.equals(reversed);
    }

    // Преобразуем в верхний регистр
    public String toUpperCase(String str) {
        return str == null ? null : str.toUpperCase();
    }

    // Проверяем, содержит ли строка подстроку
    public boolean contains(String str, String substring) {
        if (str == null || substring == null) {
            return false;
        }
        return str.contains(substring);
    }

    // Разбиваем строку по разделителю
    public String[] split(String str, String delimiter) {
        if (str == null) {
            return null;
        }
        return str.split(delimiter);
    }
}

class StringUtilsTest {
    // Тест переворота с assertEquals
    @Test
    void testReverse() {
        StringUtils utils = new StringUtils();
        assertEquals("olleh", utils.reverse("hello"));
        assertEquals("12321", utils.reverse("12321"));
    }

    // Тест переворота с null выбрасывает исключение
    @Test
    void testReverseWithNull() {
        StringUtils utils = new StringUtils();
        assertThrows(IllegalArgumentException.class, () -> {
            utils.reverse(null);
        });
    }

    // Тест палиндрома с assertTrue/assertFalse
    @Test
    void testIsPalindrome() {
        StringUtils utils = new StringUtils();
        assertTrue(utils.isPalindrome("racecar"));
        assertTrue(utils.isPalindrome("12321"));
        assertFalse(utils.isPalindrome("hello"));
        assertFalse(utils.isPalindrome(null));
    }

    // Тест toUpperCase с assertNull/assertNotNull
    @Test
    void testToUpperCase() {
        StringUtils utils = new StringUtils();
        assertEquals("HELLO", utils.toUpperCase("hello"));
        assertNull(utils.toUpperCase(null));
        assertNotNull(utils.toUpperCase("test"));
    }

    // Тест метода contains
    @Test
    void testContains() {
        StringUtils utils = new StringUtils();
        assertTrue(utils.contains("hello world", "world"));
        assertFalse(utils.contains("hello world", "java"));
        assertFalse(utils.contains(null, "test"));
    }

    // Тест split с assertArrayEquals
    @Test
    void testSplit() {
        StringUtils utils = new StringUtils();
        String[] expected = {"hello", "world", "java"};
        assertArrayEquals(expected, utils.split("hello,world,java", ","));
    }

    // Тест split с null возвращает null
    @Test
    void testSplitWithNull() {
        StringUtils utils = new StringUtils();
        assertNull(utils.split(null, ","));
    }

    // Группируем несколько утверждений с assertAll
    @Test
    void testMultipleAssertions() {
        StringUtils utils = new StringUtils();
        String input = "hello";

        assertAll("Операции со строками",
            () -> assertEquals(5, input.length()),
            () -> assertEquals("HELLO", utils.toUpperCase(input)),
            () -> assertEquals("olleh", utils.reverse(input)),
            () -> assertTrue(utils.contains(input, "ell"))
        );
    }
}`,
            description: `Освойте различные типы утверждений в **JUnit 5** для тщательного тестирования класса StringUtils.

**Требования:**
1. Создайте класс StringUtils с методами:
   1.1. reverse(String str) - переворачивает строку
   1.2. isPalindrome(String str) - проверяет, является ли палиндромом
   1.3. toUpperCase(String str) - преобразует в верхний регистр
   1.4. contains(String str, String substring) - проверяет, содержит ли подстроку
   1.5. split(String str, String delimiter) - разбивает строку на массив

2. Создайте StringUtilsTest с комплексными тестами используя:
   2.1. assertEquals() - для точного сравнения значений
   2.2. assertTrue() / assertFalse() - для булевых условий
   2.3. assertNull() / assertNotNull() - для проверки на null
   2.4. assertThrows() - для тестирования исключений
   2.5. assertAll() - для группировки утверждений
   2.6. assertArrayEquals() - для сравнения массивов

3. Протестируйте граничные случаи:
   3.1. Null входные данные
   3.2. Пустые строки
   3.3. Специальные символы

**Цели обучения:**
- Освоить различные типы утверждений
- Научиться, когда использовать каждое утверждение
- Практиковаться в группировке утверждений с assertAll()
- Понять сообщения об ошибках утверждений`,
            hint1: `Начните с реализации каждого метода StringUtils с правильными проверками на null. Не забудьте выбрасывать исключения для недопустимых входных данных там, где это уместно.`,
            hint2: `Используйте assertEquals для сравнения значений, assertTrue/assertFalse для булевых результатов и assertThrows для тестирования исключений. Используйте assertAll для группировки связанных утверждений.`,
            whyItMatters: `Различные типы утверждений помогают писать более выразительные и поддерживаемые тесты. Использование правильного утверждения делает сбои тестов более понятными и помогает быстро определить, что пошло не так. Группировка утверждений с помощью assertAll() гарантирует, что все проверки выполняются, даже если одна не удалась, давая вам полную обратную связь по тестам.

**Продакшен паттерн:**
\`\`\`java
@Test
void testUserValidation() {
    User user = new User("john@example.com");
    assertAll("User validation",
        () -> assertNotNull(user.getEmail()),
        () -> assertTrue(user.isValidEmail()),
        () -> assertFalse(user.getEmail().isEmpty())
    );
}
\`\`\`

**Практические преимущества:**
- Полная диагностика проблем при провале теста
- Улучшенная читаемость и поддерживаемость тестов`
        },
        uz: {
            title: 'JUnit Tasdiqlashlari',
            solutionCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class StringUtils {
    // Stringni teskari o'girish
    public String reverse(String str) {
        if (str == null) {
            throw new IllegalArgumentException("String null bo'lishi mumkin emas");
        }
        return new StringBuilder(str).reverse().toString();
    }

    // Stringning palindrom ekanligini tekshirish
    public boolean isPalindrome(String str) {
        if (str == null) {
            return false;
        }
        String reversed = reverse(str);
        return str.equals(reversed);
    }

    // Katta harflarga o'tkazish
    public String toUpperCase(String str) {
        return str == null ? null : str.toUpperCase();
    }

    // String ichida substring borligini tekshirish
    public boolean contains(String str, String substring) {
        if (str == null || substring == null) {
            return false;
        }
        return str.contains(substring);
    }

    // Stringni ajratuvchi bo'yicha bo'lish
    public String[] split(String str, String delimiter) {
        if (str == null) {
            return null;
        }
        return str.split(delimiter);
    }
}

class StringUtilsTest {
    // Teskari o'girish testi assertEquals bilan
    @Test
    void testReverse() {
        StringUtils utils = new StringUtils();
        assertEquals("olleh", utils.reverse("hello"));
        assertEquals("12321", utils.reverse("12321"));
    }

    // Null bilan teskari o'girish istisno chiqaradi
    @Test
    void testReverseWithNull() {
        StringUtils utils = new StringUtils();
        assertThrows(IllegalArgumentException.class, () -> {
            utils.reverse(null);
        });
    }

    // Palindrom testi assertTrue/assertFalse bilan
    @Test
    void testIsPalindrome() {
        StringUtils utils = new StringUtils();
        assertTrue(utils.isPalindrome("racecar"));
        assertTrue(utils.isPalindrome("12321"));
        assertFalse(utils.isPalindrome("hello"));
        assertFalse(utils.isPalindrome(null));
    }

    // ToUpperCase testi assertNull/assertNotNull bilan
    @Test
    void testToUpperCase() {
        StringUtils utils = new StringUtils();
        assertEquals("HELLO", utils.toUpperCase("hello"));
        assertNull(utils.toUpperCase(null));
        assertNotNull(utils.toUpperCase("test"));
    }

    // Contains metodi testi
    @Test
    void testContains() {
        StringUtils utils = new StringUtils();
        assertTrue(utils.contains("hello world", "world"));
        assertFalse(utils.contains("hello world", "java"));
        assertFalse(utils.contains(null, "test"));
    }

    // Split testi assertArrayEquals bilan
    @Test
    void testSplit() {
        StringUtils utils = new StringUtils();
        String[] expected = {"hello", "world", "java"};
        assertArrayEquals(expected, utils.split("hello,world,java", ","));
    }

    // Null bilan split null qaytaradi
    @Test
    void testSplitWithNull() {
        StringUtils utils = new StringUtils();
        assertNull(utils.split(null, ","));
    }

    // Bir nechta tasdiqlashlarni assertAll bilan guruhlash
    @Test
    void testMultipleAssertions() {
        StringUtils utils = new StringUtils();
        String input = "hello";

        assertAll("String operatsiyalari",
            () -> assertEquals(5, input.length()),
            () -> assertEquals("HELLO", utils.toUpperCase(input)),
            () -> assertEquals("olleh", utils.reverse(input)),
            () -> assertTrue(utils.contains(input, "ell"))
        );
    }
}`,
            description: `StringUtils sinfini to'liq testlash uchun **JUnit 5** dagi turli xil tasdiqlashlarni o'rganing.

**Talablar:**
1. Quyidagi metodlarga ega StringUtils sinfini yarating:
   1.1. reverse(String str) - stringni teskari o'giradi
   1.2. isPalindrome(String str) - palindrom ekanligini tekshiradi
   1.3. toUpperCase(String str) - katta harflarga o'tkazadi
   1.4. contains(String str, String substring) - substring mavjudligini tekshiradi
   1.5. split(String str, String delimiter) - stringni massivga bo'ladi

2. Quyidagilardan foydalangan holda StringUtilsTest yarating:
   2.1. assertEquals() - aniq qiymat taqqoslash uchun
   2.2. assertTrue() / assertFalse() - mantiqiy shartlar uchun
   2.3. assertNull() / assertNotNull() - null tekshirish uchun
   2.4. assertThrows() - istisno testlash uchun
   2.5. assertAll() - tasdiqlashlarni guruhlash uchun
   2.6. assertArrayEquals() - massivlarni taqqoslash uchun

3. Chegara holatlarini sinab ko'ring:
   3.1. Null kiritishlar
   3.2. Bo'sh stringlar
   3.3. Maxsus belgilar

**O'rganish maqsadlari:**
- Turli xil tasdiqlash turlarini o'zlashtirish
- Har bir tasdiqlashdan qachon foydalanishni o'rganish
- assertAll() bilan guruhlangan tasdiqlashlarda amaliyot
- Tasdiqlash xato xabarlarini tushunish`,
            hint1: `Har bir StringUtils metodini to'g'ri null tekshiruvlari bilan amalga oshirishdan boshlang. Kerak bo'lganda noto'g'ri kiritishlar uchun istisnolar chiqarishni unutmang.`,
            hint2: `Qiymatlarni taqqoslash uchun assertEquals dan, mantiqiy natijalar uchun assertTrue/assertFalse dan va istisnolarni testlash uchun assertThrows dan foydalaning. Bog'liq tasdiqlashlarni guruhlash uchun assertAll dan foydalaning.`,
            whyItMatters: `Turli xil tasdiqlash turlari yanada ifodali va boshqariladigan testlar yozishga yordam beradi. To'g'ri tasdiqlashdan foydalanish test muvaffaqiyatsizliklarini aniqroq qiladi va nima noto'g'ri ketganini tezda aniqlashga yordam beradi. assertAll() bilan tasdiqlashlarni guruhlash barcha tekshiruvlar bittasi muvaffaqiyatsiz bo'lsa ham bajarilishini ta'minlaydi va sizga to'liq test fikrlari beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Test
void testUserValidation() {
    User user = new User("john@example.com");
    assertAll("User validation",
        () -> assertNotNull(user.getEmail()),
        () -> assertTrue(user.isValidEmail()),
        () -> assertFalse(user.getEmail().isEmpty())
    );
}
\`\`\`

**Amaliy foydalari:**
- Test muvaffaqiyatsiz bo'lganda to'liq diagnostika
- Testlarning o'qilishi va boshqarilishini yaxshilash`
        }
    }
};

export default task;
