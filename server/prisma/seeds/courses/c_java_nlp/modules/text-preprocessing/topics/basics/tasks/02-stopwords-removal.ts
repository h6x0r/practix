import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-stopwords-removal',
	title: 'Stop Words Removal',
	difficulty: 'easy',
	tags: ['nlp', 'preprocessing', 'stopwords'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Stop Words Removal

Remove common words that add little meaning to text.

## Task

Implement stop word filtering:
- Load stop word list
- Filter text tokens
- Custom stop word lists

## Example

\`\`\`java
List<String> tokens = Arrays.asList("the", "quick", "brown", "fox");
List<String> filtered = StopWordRemover.remove(tokens);
// Result: ["quick", "brown", "fox"]
\`\`\``,

	initialCode: `import java.util.*;

public class StopWordRemover {

    private Set<String> stopWords;

    /**
     */
    public StopWordRemover() {
    }

    /**
     */
    public void addStopWords(Collection<String> words) {
    }

    /**
     */
    public boolean isStopWord(String word) {
        return false;
    }

    /**
     */
    public List<String> remove(List<String> tokens) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class StopWordRemover {

    private Set<String> stopWords;

    private static final String[] DEFAULT_STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "what", "which", "who", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "own", "same", "so", "than", "too", "very", "just"
    };

    /**
     * Initialize with default English stop words.
     */
    public StopWordRemover() {
        this.stopWords = new HashSet<>(Arrays.asList(DEFAULT_STOP_WORDS));
    }

    /**
     * Initialize with custom stop words.
     */
    public StopWordRemover(Collection<String> customStopWords) {
        this.stopWords = new HashSet<>(customStopWords);
    }

    /**
     * Add custom stop words.
     */
    public void addStopWords(Collection<String> words) {
        stopWords.addAll(words.stream()
            .map(String::toLowerCase)
            .collect(Collectors.toList()));
    }

    /**
     * Check if word is a stop word.
     */
    public boolean isStopWord(String word) {
        if (word == null) return false;
        return stopWords.contains(word.toLowerCase());
    }

    /**
     * Remove stop words from token list.
     */
    public List<String> remove(List<String> tokens) {
        if (tokens == null) return Collections.emptyList();
        return tokens.stream()
            .filter(token -> !isStopWord(token))
            .collect(Collectors.toList());
    }

    /**
     * Remove stop words from text.
     */
    public String removeFromText(String text) {
        if (text == null) return "";
        String[] words = text.split("\\\\s+");
        return Arrays.stream(words)
            .filter(word -> !isStopWord(word))
            .collect(Collectors.joining(" "));
    }

    /**
     * Get the current stop words set.
     */
    public Set<String> getStopWords() {
        return Collections.unmodifiableSet(stopWords);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class StopWordRemoverTest {

    @Test
    void testIsStopWord() {
        StopWordRemover remover = new StopWordRemover();
        assertTrue(remover.isStopWord("the"));
        assertTrue(remover.isStopWord("The"));
        assertFalse(remover.isStopWord("quick"));
    }

    @Test
    void testRemove() {
        StopWordRemover remover = new StopWordRemover();
        List<String> tokens = Arrays.asList("the", "quick", "brown", "fox");
        List<String> filtered = remover.remove(tokens);

        assertEquals(3, filtered.size());
        assertFalse(filtered.contains("the"));
        assertTrue(filtered.contains("quick"));
    }

    @Test
    void testAddStopWords() {
        StopWordRemover remover = new StopWordRemover();
        remover.addStopWords(Arrays.asList("custom", "words"));

        assertTrue(remover.isStopWord("custom"));
        assertTrue(remover.isStopWord("words"));
    }

    @Test
    void testRemoveFromText() {
        StopWordRemover remover = new StopWordRemover();
        String result = remover.removeFromText("the quick brown fox");
        assertEquals("quick brown fox", result);
    }

    @Test
    void testGetStopWords() {
        StopWordRemover remover = new StopWordRemover();
        Set<String> stopWords = remover.getStopWords();
        assertNotNull(stopWords);
        assertTrue(stopWords.contains("the"));
    }

    @Test
    void testRemoveNullInput() {
        StopWordRemover remover = new StopWordRemover();
        List<String> result = remover.remove(null);
        assertTrue(result.isEmpty());
    }

    @Test
    void testIsStopWordNull() {
        StopWordRemover remover = new StopWordRemover();
        assertFalse(remover.isStopWord(null));
    }

    @Test
    void testRemoveFromTextNull() {
        StopWordRemover remover = new StopWordRemover();
        assertEquals("", remover.removeFromText(null));
    }

    @Test
    void testRemoveEmptyList() {
        StopWordRemover remover = new StopWordRemover();
        List<String> result = remover.remove(Collections.emptyList());
        assertTrue(result.isEmpty());
    }

    @Test
    void testCustomStopWordsConstructor() {
        StopWordRemover remover = new StopWordRemover(Arrays.asList("foo", "bar"));
        assertTrue(remover.isStopWord("foo"));
        assertFalse(remover.isStopWord("the"));
    }
}`,

	hint1: 'Use HashSet for O(1) lookup performance',
	hint2: 'Convert words to lowercase before checking',

	whyItMatters: `Stop word removal improves NLP efficiency:

- **Focus on meaning**: Remove words that do not carry meaning
- **Reduced dimensionality**: Smaller vocabulary for models
- **Better similarity**: "The quick fox" matches "quick fox"
- **Faster processing**: Fewer tokens to process

Essential for search, classification, and topic modeling.`,

	translations: {
		ru: {
			title: 'Удаление стоп-слов',
			description: `# Удаление стоп-слов

Удаляйте распространенные слова, не несущие смысловой нагрузки.

## Задача

Реализуйте фильтрацию стоп-слов:
- Загрузка списка стоп-слов
- Фильтрация токенов текста
- Кастомные списки стоп-слов

## Пример

\`\`\`java
List<String> tokens = Arrays.asList("the", "quick", "brown", "fox");
List<String> filtered = StopWordRemover.remove(tokens);
// Result: ["quick", "brown", "fox"]
\`\`\``,
			hint1: 'Используйте HashSet для O(1) производительности поиска',
			hint2: 'Преобразуйте слова в нижний регистр перед проверкой',
			whyItMatters: `Удаление стоп-слов улучшает эффективность NLP:

- **Фокус на смысле**: Удаление слов не несущих значения
- **Уменьшение размерности**: Меньший словарь для моделей
- **Лучшее сходство**: "The quick fox" совпадает с "quick fox"
- **Быстрая обработка**: Меньше токенов для обработки`,
		},
		uz: {
			title: "Stop so'zlarni olib tashlash",
			description: `# Stop so'zlarni olib tashlash

Ma'noga kam qo'shadigan umumiy so'zlarni olib tashlang.

## Topshiriq

Stop so'z filtrlashni amalga oshiring:
- Stop so'zlar ro'yxatini yuklash
- Matn tokenlarini filtrlash
- Maxsus stop so'zlar ro'yxatlari

## Misol

\`\`\`java
List<String> tokens = Arrays.asList("the", "quick", "brown", "fox");
List<String> filtered = StopWordRemover.remove(tokens);
// Result: ["quick", "brown", "fox"]
\`\`\``,
			hint1: "O(1) qidiruv samaradorligi uchun HashSet dan foydalaning",
			hint2: "Tekshirishdan oldin so'zlarni kichik harfga aylantiring",
			whyItMatters: `Stop so'zlarni olib tashlash NLP samaradorligini yaxshilaydi:

- **Ma'noga e'tibor**: Ma'no tashimaydigan so'zlarni olib tashlash
- **Kamaytirilgan o'lcham**: Modellar uchun kichikroq lug'at
- **Yaxshiroq o'xshashlik**: "The quick fox" "quick fox" bilan mos keladi
- **Tezroq qayta ishlash**: Kamroq tokenlarni qayta ishlash`,
		},
	},
};

export default task;
