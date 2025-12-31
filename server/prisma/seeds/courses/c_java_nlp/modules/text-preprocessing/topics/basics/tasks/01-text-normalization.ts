import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-text-normalization',
	title: 'Text Normalization',
	difficulty: 'easy',
	tags: ['nlp', 'preprocessing', 'normalization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Text Normalization

Normalize text for consistent NLP processing.

## Task

Implement text normalization:
- Convert to lowercase
- Remove punctuation
- Normalize whitespace
- Remove special characters

## Example

\`\`\`java
String text = "Hello, World! How are you?";
String normalized = TextNormalizer.normalize(text);
// Result: "hello world how are you"
\`\`\``,

	initialCode: `import java.util.regex.Pattern;

public class TextNormalizer {

    /**
     */
    public static String toLowerCase(String text) {
        return null;
    }

    /**
     */
    public static String removePunctuation(String text) {
        return null;
    }

    /**
     */
    public static String normalizeWhitespace(String text) {
        return null;
    }

    /**
     */
    public static String normalize(String text) {
        return null;
    }
}`,

	solutionCode: `import java.util.regex.Pattern;

public class TextNormalizer {

    private static final Pattern PUNCTUATION = Pattern.compile("[\\\\p{Punct}]");
    private static final Pattern WHITESPACE = Pattern.compile("\\\\s+");
    private static final Pattern SPECIAL_CHARS = Pattern.compile("[^a-zA-Z0-9\\\\s]");

    /**
     * Convert text to lowercase.
     */
    public static String toLowerCase(String text) {
        if (text == null) return "";
        return text.toLowerCase();
    }

    /**
     * Remove punctuation from text.
     */
    public static String removePunctuation(String text) {
        if (text == null) return "";
        return PUNCTUATION.matcher(text).replaceAll("");
    }

    /**
     * Normalize whitespace (multiple spaces to single).
     */
    public static String normalizeWhitespace(String text) {
        if (text == null) return "";
        return WHITESPACE.matcher(text.trim()).replaceAll(" ");
    }

    /**
     * Remove special characters (keep only alphanumeric and spaces).
     */
    public static String removeSpecialChars(String text) {
        if (text == null) return "";
        return SPECIAL_CHARS.matcher(text).replaceAll("");
    }

    /**
     * Full normalization pipeline.
     */
    public static String normalize(String text) {
        if (text == null) return "";
        String result = toLowerCase(text);
        result = removePunctuation(result);
        result = normalizeWhitespace(result);
        return result;
    }

    /**
     * Remove numbers from text.
     */
    public static String removeNumbers(String text) {
        if (text == null) return "";
        return text.replaceAll("\\\\d+", "");
    }

    /**
     * Remove URLs from text.
     */
    public static String removeUrls(String text) {
        if (text == null) return "";
        return text.replaceAll("https?://\\\\S+", "");
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TextNormalizerTest {

    @Test
    void testToLowerCase() {
        assertEquals("hello world", TextNormalizer.toLowerCase("Hello World"));
        assertEquals("", TextNormalizer.toLowerCase(null));
    }

    @Test
    void testRemovePunctuation() {
        assertEquals("Hello World", TextNormalizer.removePunctuation("Hello, World!"));
    }

    @Test
    void testNormalizeWhitespace() {
        assertEquals("hello world", TextNormalizer.normalizeWhitespace("  hello   world  "));
    }

    @Test
    void testNormalize() {
        String result = TextNormalizer.normalize("Hello, World! How are you?");
        assertEquals("hello world how are you", result);
    }

    @Test
    void testRemoveUrls() {
        String text = "Check https://example.com for more";
        assertEquals("Check  for more", TextNormalizer.removeUrls(text));
    }

    @Test
    void testRemoveNumbers() {
        assertEquals("Hello", TextNormalizer.removeNumbers("Hello123"));
        assertEquals("abc", TextNormalizer.removeNumbers("a1b2c3"));
    }

    @Test
    void testRemoveSpecialChars() {
        assertEquals("Hello World", TextNormalizer.removeSpecialChars("Hello@World!"));
    }

    @Test
    void testNullHandling() {
        assertEquals("", TextNormalizer.removePunctuation(null));
        assertEquals("", TextNormalizer.normalizeWhitespace(null));
        assertEquals("", TextNormalizer.normalize(null));
    }

    @Test
    void testEmptyString() {
        assertEquals("", TextNormalizer.normalize(""));
        assertEquals("", TextNormalizer.toLowerCase(""));
    }

    @Test
    void testMixedContent() {
        String result = TextNormalizer.normalize("Hello123, World! @test");
        assertTrue(result.contains("hello"));
        assertTrue(result.contains("world"));
    }
}`,

	hint1: 'Use Pattern and Matcher for efficient regex operations',
	hint2: 'Always handle null inputs to prevent NullPointerException',

	whyItMatters: `Text normalization is the foundation of NLP:

- **Consistency**: Same word appears in same form
- **Reduced vocabulary**: Fewer unique tokens to process
- **Better matching**: "Hello" and "hello" become identical
- **Noise removal**: Punctuation often not needed for meaning

Clean text leads to better NLP model performance.`,

	translations: {
		ru: {
			title: 'Нормализация текста',
			description: `# Нормализация текста

Нормализуйте текст для консистентной NLP обработки.

## Задача

Реализуйте нормализацию текста:
- Преобразование в нижний регистр
- Удаление пунктуации
- Нормализация пробелов
- Удаление специальных символов

## Пример

\`\`\`java
String text = "Hello, World! How are you?";
String normalized = TextNormalizer.normalize(text);
// Result: "hello world how are you"
\`\`\``,
			hint1: 'Используйте Pattern и Matcher для эффективных regex операций',
			hint2: 'Всегда обрабатывайте null для предотвращения NullPointerException',
			whyItMatters: `Нормализация текста - основа NLP:

- **Консистентность**: Одно слово появляется в одной форме
- **Уменьшение словаря**: Меньше уникальных токенов для обработки
- **Лучшее сопоставление**: "Hello" и "hello" становятся идентичны
- **Удаление шума**: Пунктуация часто не нужна для смысла`,
		},
		uz: {
			title: 'Matn normalizatsiyasi',
			description: `# Matn normalizatsiyasi

Izchil NLP qayta ishlash uchun matnni normallang.

## Topshiriq

Matn normalizatsiyasini amalga oshiring:
- Kichik harfga aylantirish
- Tinish belgilarini olib tashlash
- Bo'shliqlarni normallashtirish
- Maxsus belgilarni olib tashlash

## Misol

\`\`\`java
String text = "Hello, World! How are you?";
String normalized = TextNormalizer.normalize(text);
// Result: "hello world how are you"
\`\`\``,
			hint1: "Samarali regex operatsiyalari uchun Pattern va Matcher dan foydalaning",
			hint2: "NullPointerException ni oldini olish uchun doimo null ni tekshiring",
			whyItMatters: `Matn normalizatsiyasi NLP ning asosi:

- **Izchillik**: Bir xil so'z bir xil shaklda ko'rinadi
- **Kamaytirilgan lug'at**: Qayta ishlash uchun kamroq noyob tokenlar
- **Yaxshiroq moslik**: "Hello" va "hello" bir xil bo'ladi
- **Shovqinni olib tashlash**: Tinish belgilari ko'pincha ma'no uchun kerak emas`,
		},
	},
};

export default task;
