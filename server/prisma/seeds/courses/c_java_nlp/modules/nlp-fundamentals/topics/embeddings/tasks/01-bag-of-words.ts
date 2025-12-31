import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-bag-of-words',
	title: 'Bag of Words',
	difficulty: 'easy',
	tags: ['nlp', 'embeddings', 'bow', 'vectorization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Bag of Words

Convert text to numerical vectors using Bag of Words.

## Task

Implement BoW vectorization:
- Build vocabulary from corpus
- Convert documents to vectors
- Handle term frequencies

## Example

\`\`\`java
BagOfWords bow = new BagOfWords();
bow.fit(documents);
int[] vector = bow.transform("hello world");
\`\`\``,

	initialCode: `import java.util.*;

public class BagOfWords {

    private Map<String, Integer> vocabulary;
    private int vocabSize;

    /**
     */
    public void fit(List<String> documents) {
    }

    /**
     */
    public int[] transform(String document) {
        return null;
    }

    /**
     */
    public int[][] fitTransform(List<String> documents) {
        return null;
    }

    /**
     */
    public int getVocabSize() {
        return 0;
    }
}`,

	solutionCode: `import java.util.*;

public class BagOfWords {

    private Map<String, Integer> vocabulary;
    private int vocabSize;

    public BagOfWords() {
        this.vocabulary = new LinkedHashMap<>();
        this.vocabSize = 0;
    }

    /**
     * Build vocabulary from documents.
     */
    public void fit(List<String> documents) {
        vocabulary.clear();
        int index = 0;

        for (String doc : documents) {
            String[] words = tokenize(doc);
            for (String word : words) {
                if (!vocabulary.containsKey(word)) {
                    vocabulary.put(word, index++);
                }
            }
        }
        vocabSize = vocabulary.size();
    }

    /**
     * Transform document to vector.
     */
    public int[] transform(String document) {
        int[] vector = new int[vocabSize];
        String[] words = tokenize(document);

        for (String word : words) {
            if (vocabulary.containsKey(word)) {
                int index = vocabulary.get(word);
                vector[index]++;
            }
        }
        return vector;
    }

    /**
     * Fit and transform in one step.
     */
    public int[][] fitTransform(List<String> documents) {
        fit(documents);
        int[][] vectors = new int[documents.size()][vocabSize];

        for (int i = 0; i < documents.size(); i++) {
            vectors[i] = transform(documents.get(i));
        }
        return vectors;
    }

    /**
     * Get vocabulary size.
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Get word index in vocabulary.
     */
    public int getWordIndex(String word) {
        return vocabulary.getOrDefault(word, -1);
    }

    /**
     * Simple tokenization.
     */
    private String[] tokenize(String text) {
        return text.toLowerCase()
            .replaceAll("[^a-zA-Z\\\\s]", "")
            .split("\\\\s+");
    }

    /**
     * Get vocabulary as list.
     */
    public List<String> getVocabulary() {
        return new ArrayList<>(vocabulary.keySet());
    }

    /**
     * Convert vector back to terms (for debugging).
     */
    public Map<String, Integer> vectorToTerms(int[] vector) {
        Map<String, Integer> terms = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
            int count = vector[entry.getValue()];
            if (count > 0) {
                terms.put(entry.getKey(), count);
            }
        }
        return terms;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class BagOfWordsTest {

    @Test
    void testFit() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("hello world", "world peace"));

        assertTrue(bow.getVocabSize() >= 3);
    }

    @Test
    void testTransform() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("hello world", "hello there"));

        int[] vector = bow.transform("hello");
        assertTrue(vector[bow.getWordIndex("hello")] > 0);
    }

    @Test
    void testFitTransform() {
        BagOfWords bow = new BagOfWords();
        int[][] vectors = bow.fitTransform(Arrays.asList("a b", "b c"));

        assertEquals(2, vectors.length);
        assertEquals(bow.getVocabSize(), vectors[0].length);
    }

    @Test
    void testWordCount() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("hello hello world"));

        int[] vector = bow.transform("hello hello world");
        assertEquals(2, vector[bow.getWordIndex("hello")]);
        assertEquals(1, vector[bow.getWordIndex("world")]);
    }

    @Test
    void testGetVocabulary() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("hello world"));
        List<String> vocab = bow.getVocabulary();
        assertTrue(vocab.contains("hello"));
        assertTrue(vocab.contains("world"));
    }

    @Test
    void testGetWordIndex() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("apple banana"));
        assertEquals(-1, bow.getWordIndex("unknown"));
        assertTrue(bow.getWordIndex("apple") >= 0);
    }

    @Test
    void testVectorToTerms() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("test word"));
        int[] vector = bow.transform("test test");
        Map<String, Integer> terms = bow.vectorToTerms(vector);
        assertEquals(2, terms.get("test").intValue());
    }

    @Test
    void testTransformUnknownWords() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("known word"));
        int[] vector = bow.transform("unknown word");
        assertEquals(1, vector[bow.getWordIndex("word")]);
    }

    @Test
    void testEmptyDocument() {
        BagOfWords bow = new BagOfWords();
        bow.fit(Arrays.asList("hello"));
        int[] vector = bow.transform("");
        assertEquals(bow.getVocabSize(), vector.length);
    }

    @Test
    void testMultipleDocuments() {
        BagOfWords bow = new BagOfWords();
        int[][] vectors = bow.fitTransform(Arrays.asList("doc one", "doc two", "doc three"));
        assertEquals(3, vectors.length);
    }
}`,

	hint1: 'Use LinkedHashMap to maintain vocabulary order',
	hint2: 'Increment counts for repeated words in documents',

	whyItMatters: `Bag of Words is a foundational NLP technique:

- **Simplicity**: Easy to understand and implement
- **Baseline**: Often used as baseline for comparison
- **Feature extraction**: Create features for ML classifiers
- **Sparse representation**: Efficient for large vocabularies

Understanding BoW is essential before learning advanced embeddings.`,

	translations: {
		ru: {
			title: 'Мешок слов',
			description: `# Мешок слов

Преобразуйте текст в числовые векторы используя Bag of Words.

## Задача

Реализуйте BoW векторизацию:
- Построение словаря из корпуса
- Преобразование документов в векторы
- Обработка частот терминов

## Пример

\`\`\`java
BagOfWords bow = new BagOfWords();
bow.fit(documents);
int[] vector = bow.transform("hello world");
\`\`\``,
			hint1: 'Используйте LinkedHashMap для сохранения порядка словаря',
			hint2: 'Увеличивайте счетчики для повторяющихся слов в документах',
			whyItMatters: `Bag of Words - базовая техника NLP:

- **Простота**: Легко понять и реализовать
- **Базовая линия**: Часто используется для сравнения
- **Извлечение признаков**: Создание признаков для ML классификаторов
- **Разреженное представление**: Эффективно для больших словарей`,
		},
		uz: {
			title: "So'zlar xaltasi",
			description: `# So'zlar xaltasi

Bag of Words yordamida matnni raqamli vektorlarga aylantiring.

## Topshiriq

BoW vektorizatsiyasini amalga oshiring:
- Korpusdan lug'at qurish
- Hujjatlarni vektorlarga aylantirish
- Term chastotalarini boshqarish

## Misol

\`\`\`java
BagOfWords bow = new BagOfWords();
bow.fit(documents);
int[] vector = bow.transform("hello world");
\`\`\``,
			hint1: "Lug'at tartibini saqlash uchun LinkedHashMap dan foydalaning",
			hint2: "Hujjatlarda takrorlanadigan so'zlar uchun hisoblagichlarni oshiring",
			whyItMatters: `Bag of Words asosiy NLP texnikasi:

- **Oddiylik**: Tushunish va amalga oshirish oson
- **Asosiy chiziq**: Ko'pincha taqqoslash uchun ishlatiladi
- **Xususiyat ajratib olish**: ML klassifikatorlari uchun xususiyatlar yaratish
- **Siyrak representatsiya**: Katta lug'atlar uchun samarali`,
		},
	},
};

export default task;
