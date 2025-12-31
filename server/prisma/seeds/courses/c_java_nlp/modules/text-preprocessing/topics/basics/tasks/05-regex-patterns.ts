import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-regex-patterns',
	title: 'Regex for Text Cleaning',
	difficulty: 'medium',
	tags: ['nlp', 'preprocessing', 'regex', 'patterns'],
	estimatedTime: '20m',
	isPremium: false,
	order: 5,
	description: `# Regex for Text Cleaning

Use regular expressions to clean and extract text patterns.

## Task

Implement regex-based text cleaning:
- Extract emails and URLs
- Remove HTML tags
- Clean social media text

## Example

\`\`\`java
TextCleaner cleaner = new TextCleaner();
String clean = cleaner.removeHtmlTags("<p>Hello</p>");
// Result: "Hello"
\`\`\``,

	initialCode: `import java.util.regex.*;
import java.util.*;

public class TextCleaner {

    /**
     */
    public static String removeHtmlTags(String text) {
        return null;
    }

    /**
     */
    public static List<String> extractEmails(String text) {
        return null;
    }

    /**
     */
    public static List<String> extractUrls(String text) {
        return null;
    }

    /**
     */
    public static String removeMentions(String text) {
        return null;
    }

    /**
     */
    public static String removeHashtags(String text) {
        return null;
    }
}`,

	solutionCode: `import java.util.regex.*;
import java.util.*;

public class TextCleaner {

    private static final Pattern HTML_TAG = Pattern.compile("<[^>]+>");
    private static final Pattern EMAIL = Pattern.compile(
        "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}"
    );
    private static final Pattern URL = Pattern.compile(
        "https?://[\\\\w\\\\-._~:/?#\\\\[\\\\]@!$&'()*+,;=%]+"
    );
    private static final Pattern MENTION = Pattern.compile("@\\\\w+");
    private static final Pattern HASHTAG = Pattern.compile("#\\\\w+");
    private static final Pattern EMOJI = Pattern.compile(
        "[\\\\x{1F600}-\\\\x{1F64F}\\\\x{1F300}-\\\\x{1F5FF}\\\\x{1F680}-\\\\x{1F6FF}]"
    );

    /**
     * Remove HTML tags from text.
     */
    public static String removeHtmlTags(String text) {
        if (text == null) return "";
        return HTML_TAG.matcher(text).replaceAll("");
    }

    /**
     * Extract all email addresses.
     */
    public static List<String> extractEmails(String text) {
        List<String> emails = new ArrayList<>();
        if (text == null) return emails;

        Matcher matcher = EMAIL.matcher(text);
        while (matcher.find()) {
            emails.add(matcher.group());
        }
        return emails;
    }

    /**
     * Extract all URLs.
     */
    public static List<String> extractUrls(String text) {
        List<String> urls = new ArrayList<>();
        if (text == null) return urls;

        Matcher matcher = URL.matcher(text);
        while (matcher.find()) {
            urls.add(matcher.group());
        }
        return urls;
    }

    /**
     * Remove mentions (@username).
     */
    public static String removeMentions(String text) {
        if (text == null) return "";
        return MENTION.matcher(text).replaceAll("");
    }

    /**
     * Remove hashtags (#topic).
     */
    public static String removeHashtags(String text) {
        if (text == null) return "";
        return HASHTAG.matcher(text).replaceAll("");
    }

    /**
     * Extract hashtags from text.
     */
    public static List<String> extractHashtags(String text) {
        List<String> hashtags = new ArrayList<>();
        if (text == null) return hashtags;

        Matcher matcher = HASHTAG.matcher(text);
        while (matcher.find()) {
            hashtags.add(matcher.group().substring(1)); // Remove #
        }
        return hashtags;
    }

    /**
     * Clean social media text (remove mentions, hashtags, URLs).
     */
    public static String cleanSocialMedia(String text) {
        if (text == null) return "";
        String result = removeMentions(text);
        result = removeHashtags(result);
        result = URL.matcher(result).replaceAll("");
        result = result.replaceAll("\\\\s+", " ").trim();
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
     * Replace multiple whitespace with single space.
     */
    public static String normalizeSpaces(String text) {
        if (text == null) return "";
        return text.replaceAll("\\\\s+", " ").trim();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class TextCleanerTest {

    @Test
    void testRemoveHtmlTags() {
        assertEquals("Hello", TextCleaner.removeHtmlTags("<p>Hello</p>"));
        assertEquals("Hello World", TextCleaner.removeHtmlTags("<b>Hello</b> World"));
    }

    @Test
    void testExtractEmails() {
        String text = "Contact us at test@example.com or support@company.org";
        List<String> emails = TextCleaner.extractEmails(text);

        assertEquals(2, emails.size());
        assertTrue(emails.contains("test@example.com"));
    }

    @Test
    void testExtractUrls() {
        String text = "Visit https://example.com and http://test.org";
        List<String> urls = TextCleaner.extractUrls(text);

        assertEquals(2, urls.size());
        assertTrue(urls.contains("https://example.com"));
    }

    @Test
    void testRemoveMentions() {
        String result = TextCleaner.removeMentions("Hello @user how are you?");
        assertEquals("Hello  how are you?", result);
    }

    @Test
    void testCleanSocialMedia() {
        String text = "Great post @user! #java https://t.co/abc";
        String result = TextCleaner.cleanSocialMedia(text);
        assertEquals("Great post ! t.co/abc", result.trim());
    }

    @Test
    void testRemoveHashtags() {
        String result = TextCleaner.removeHashtags("Check #java and #python");
        assertEquals("Check  and ", result);
    }

    @Test
    void testExtractHashtags() {
        List<String> hashtags = TextCleaner.extractHashtags("Trending #ai and #ml");
        assertEquals(2, hashtags.size());
        assertTrue(hashtags.contains("ai"));
    }

    @Test
    void testRemoveNumbers() {
        assertEquals("Price is  dollars", TextCleaner.removeNumbers("Price is 100 dollars"));
    }

    @Test
    void testNormalizeSpaces() {
        assertEquals("Hello World", TextCleaner.normalizeSpaces("Hello   World"));
    }

    @Test
    void testNullHandling() {
        assertEquals("", TextCleaner.removeHtmlTags(null));
        assertTrue(TextCleaner.extractEmails(null).isEmpty());
        assertTrue(TextCleaner.extractUrls(null).isEmpty());
    }
}`,

	hint1: 'Compile patterns once and reuse them for performance',
	hint2: 'Use non-greedy quantifiers to avoid over-matching',

	whyItMatters: `Regex is essential for text preprocessing:

- **Pattern extraction**: Find emails, URLs, dates in text
- **Noise removal**: Clean HTML, special characters
- **Data preparation**: Standardize text format
- **Efficiency**: Process large texts quickly

Mastering regex is crucial for NLP data pipelines.`,

	translations: {
		ru: {
			title: 'Regex для очистки текста',
			description: `# Regex для очистки текста

Используйте регулярные выражения для очистки и извлечения паттернов текста.

## Задача

Реализуйте regex-очистку текста:
- Извлечение email и URL
- Удаление HTML тегов
- Очистка текста соцсетей

## Пример

\`\`\`java
TextCleaner cleaner = new TextCleaner();
String clean = cleaner.removeHtmlTags("<p>Hello</p>");
// Result: "Hello"
\`\`\``,
			hint1: 'Компилируйте паттерны один раз и переиспользуйте для производительности',
			hint2: 'Используйте нежадные квантификаторы для избежания лишних захватов',
			whyItMatters: `Regex необходим для предобработки текста:

- **Извлечение паттернов**: Находите email, URL, даты в тексте
- **Удаление шума**: Очистка HTML, специальных символов
- **Подготовка данных**: Стандартизация формата текста
- **Эффективность**: Быстрая обработка больших текстов`,
		},
		uz: {
			title: 'Matn tozalash uchun Regex',
			description: `# Matn tozalash uchun Regex

Matn patternlarini tozalash va ajratib olish uchun regulyar ifodalardan foydalaning.

## Topshiriq

Regex asosida matn tozalashni amalga oshiring:
- Email va URL larni ajratib olish
- HTML teglarni olib tashlash
- Ijtimoiy tarmoq matnini tozalash

## Misol

\`\`\`java
TextCleaner cleaner = new TextCleaner();
String clean = cleaner.removeHtmlTags("<p>Hello</p>");
// Result: "Hello"
\`\`\``,
			hint1: "Samaradorlik uchun patternlarni bir marta kompilyatsiya qiling va qayta foydalaning",
			hint2: "Ortiqcha mos kelishning oldini olish uchun nochid kvantifikatorlardan foydalaning",
			whyItMatters: `Regex matn oldindan qayta ishlash uchun zarur:

- **Pattern ajratib olish**: Matndagi email, URL, sanalarni toping
- **Shovqinni olib tashlash**: HTML, maxsus belgilarni tozalash
- **Ma'lumot tayyorlash**: Matn formatini standartlashtirish
- **Samaradorlik**: Katta matnlarni tez qayta ishlash`,
		},
	},
};

export default task;
