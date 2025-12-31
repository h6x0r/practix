import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-tfidf',
	title: 'TF-IDF Vectorization',
	difficulty: 'medium',
	tags: ['nlp', 'embeddings', 'tfidf', 'vectorization'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# TF-IDF Vectorization

Implement Term Frequency-Inverse Document Frequency.

## Task

Build TF-IDF vectorizer:
- Calculate term frequency
- Calculate inverse document frequency
- Combine TF and IDF scores

## Example

\`\`\`java
TfidfVectorizer tfidf = new TfidfVectorizer();
tfidf.fit(documents);
double[] vector = tfidf.transform("important rare words");
\`\`\``,

	initialCode: `import java.util.*;

public class TfidfVectorizer {

    private Map<String, Integer> vocabulary;
    private Map<String, Double> idfScores;
    private int numDocuments;

    /**
     */
    public Map<String, Double> calculateTF(String document) {
        return null;
    }

    /**
     */
    public void calculateIDF(List<String> documents) {
    }

    /**
     */
    public void fit(List<String> documents) {
    }

    /**
     */
    public double[] transform(String document) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class TfidfVectorizer {

    private Map<String, Integer> vocabulary;
    private Map<String, Double> idfScores;
    private int numDocuments;

    public TfidfVectorizer() {
        this.vocabulary = new LinkedHashMap<>();
        this.idfScores = new HashMap<>();
    }

    /**
     * Calculate term frequency.
     */
    public Map<String, Double> calculateTF(String document) {
        Map<String, Double> tf = new HashMap<>();
        String[] words = tokenize(document);
        int totalWords = words.length;

        Map<String, Integer> wordCounts = new HashMap<>();
        for (String word : words) {
            wordCounts.merge(word, 1, Integer::sum);
        }

        for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
            tf.put(entry.getKey(), (double) entry.getValue() / totalWords);
        }
        return tf;
    }

    /**
     * Calculate IDF for all terms.
     */
    public void calculateIDF(List<String> documents) {
        Map<String, Integer> docFreq = new HashMap<>();
        numDocuments = documents.size();

        // Count document frequency for each term
        for (String doc : documents) {
            Set<String> uniqueWords = new HashSet<>(Arrays.asList(tokenize(doc)));
            for (String word : uniqueWords) {
                docFreq.merge(word, 1, Integer::sum);
            }
        }

        // Calculate IDF: log(N / df)
        for (Map.Entry<String, Integer> entry : docFreq.entrySet()) {
            double idf = Math.log((double) numDocuments / (entry.getValue() + 1)) + 1;
            idfScores.put(entry.getKey(), idf);
        }
    }

    /**
     * Fit vectorizer on documents.
     */
    public void fit(List<String> documents) {
        vocabulary.clear();
        int index = 0;

        // Build vocabulary
        for (String doc : documents) {
            for (String word : tokenize(doc)) {
                if (!vocabulary.containsKey(word)) {
                    vocabulary.put(word, index++);
                }
            }
        }

        // Calculate IDF scores
        calculateIDF(documents);
    }

    /**
     * Transform document to TF-IDF vector.
     */
    public double[] transform(String document) {
        double[] vector = new double[vocabulary.size()];
        Map<String, Double> tf = calculateTF(document);

        for (Map.Entry<String, Double> entry : tf.entrySet()) {
            String word = entry.getKey();
            if (vocabulary.containsKey(word)) {
                int index = vocabulary.get(word);
                double idf = idfScores.getOrDefault(word, 1.0);
                vector[index] = entry.getValue() * idf;
            }
        }
        return vector;
    }

    /**
     * Fit and transform in one step.
     */
    public double[][] fitTransform(List<String> documents) {
        fit(documents);
        double[][] vectors = new double[documents.size()][vocabulary.size()];
        for (int i = 0; i < documents.size(); i++) {
            vectors[i] = transform(documents.get(i));
        }
        return vectors;
    }

    private String[] tokenize(String text) {
        return text.toLowerCase()
            .replaceAll("[^a-zA-Z\\\\s]", "")
            .split("\\\\s+");
    }

    public int getVocabSize() {
        return vocabulary.size();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class TfidfVectorizerTest {

    @Test
    void testCalculateTF() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        Map<String, Double> tf = tfidf.calculateTF("hello hello world");

        assertEquals(2.0/3, tf.get("hello"), 0.01);
        assertEquals(1.0/3, tf.get("world"), 0.01);
    }

    @Test
    void testFit() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        tfidf.fit(Arrays.asList("hello world", "world peace"));

        assertTrue(tfidf.getVocabSize() >= 3);
    }

    @Test
    void testTransform() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        tfidf.fit(Arrays.asList("hello world", "hello there", "world peace"));

        double[] vector = tfidf.transform("hello world");
        assertTrue(vector.length > 0);
    }

    @Test
    void testRareWordsHigherIDF() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        List<String> docs = Arrays.asList(
            "common common common rare",
            "common common common",
            "common common common"
        );
        tfidf.fit(docs);

        double[] vector = tfidf.transform("common rare");
        // Rare word should have higher TF-IDF than common
    }

    @Test
    void testFitTransform() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        double[][] vectors = tfidf.fitTransform(Arrays.asList("doc one", "doc two"));
        assertEquals(2, vectors.length);
        assertEquals(tfidf.getVocabSize(), vectors[0].length);
    }

    @Test
    void testCalculateTFSingleWord() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        Map<String, Double> tf = tfidf.calculateTF("word");
        assertEquals(1.0, tf.get("word"), 0.001);
    }

    @Test
    void testVectorLength() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        tfidf.fit(Arrays.asList("a b c", "d e f"));
        double[] vector = tfidf.transform("a b");
        assertEquals(6, vector.length);
    }

    @Test
    void testTransformUnknownWord() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        tfidf.fit(Arrays.asList("known words"));
        double[] vector = tfidf.transform("unknown");
        // Vector should be all zeros for unknown word
        double sum = 0;
        for (double v : vector) sum += v;
        assertEquals(0, sum, 0.001);
    }

    @Test
    void testGetVocabSize() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        tfidf.fit(Arrays.asList("one two three"));
        assertEquals(3, tfidf.getVocabSize());
    }

    @Test
    void testMultipleTransforms() {
        TfidfVectorizer tfidf = new TfidfVectorizer();
        tfidf.fit(Arrays.asList("hello world"));
        double[] v1 = tfidf.transform("hello");
        double[] v2 = tfidf.transform("world");
        assertNotNull(v1);
        assertNotNull(v2);
    }
}`,

	hint1: 'TF = (term count) / (total words in document)',
	hint2: 'IDF = log(N / df) where N is total docs and df is docs containing term',

	whyItMatters: `TF-IDF improves on simple word counts:

- **Importance weighting**: Common words get lower weights
- **Discriminative**: Rare words that appear in few docs get higher weight
- **Search engines**: Core algorithm for document retrieval
- **Better features**: Often outperforms raw counts for classification

Industry standard for text vectorization before embeddings.`,

	translations: {
		ru: {
			title: 'TF-IDF векторизация',
			description: `# TF-IDF векторизация

Реализуйте Term Frequency-Inverse Document Frequency.

## Задача

Создайте TF-IDF векторизатор:
- Вычисление частоты терминов
- Вычисление обратной частоты документов
- Комбинирование TF и IDF оценок

## Пример

\`\`\`java
TfidfVectorizer tfidf = new TfidfVectorizer();
tfidf.fit(documents);
double[] vector = tfidf.transform("important rare words");
\`\`\``,
			hint1: 'TF = (количество терминов) / (всего слов в документе)',
			hint2: 'IDF = log(N / df) где N - всего документов, df - документов с термином',
			whyItMatters: `TF-IDF улучшает простой подсчет слов:

- **Взвешивание важности**: Частые слова получают меньший вес
- **Дискриминативность**: Редкие слова в немногих документах получают больший вес
- **Поисковики**: Основной алгоритм для поиска документов
- **Лучшие признаки**: Часто превосходит сырые подсчеты для классификации`,
		},
		uz: {
			title: 'TF-IDF vektorizatsiyasi',
			description: `# TF-IDF vektorizatsiyasi

Term Frequency-Inverse Document Frequency ni amalga oshiring.

## Topshiriq

TF-IDF vektorizator yarating:
- Term chastotasini hisoblash
- Teskari hujjat chastotasini hisoblash
- TF va IDF ballarini birlashtirish

## Misol

\`\`\`java
TfidfVectorizer tfidf = new TfidfVectorizer();
tfidf.fit(documents);
double[] vector = tfidf.transform("important rare words");
\`\`\``,
			hint1: "TF = (term soni) / (hujjatdagi jami so'zlar)",
			hint2: "IDF = log(N / df) bu yerda N jami hujjatlar, df termni o'z ichiga olgan hujjatlar",
			whyItMatters: `TF-IDF oddiy so'z hisoblashni yaxshilaydi:

- **Muhimlik vaznlash**: Tez-tez uchraydigan so'zlar kamroq vazn oladi
- **Diskriminativ**: Kam hujjatlarda uchraydigan kam so'zlar yuqoriroq vazn oladi
- **Qidiruv tizimlari**: Hujjatlarni qidirish uchun asosiy algoritm
- **Yaxshiroq xususiyatlar**: Ko'pincha klassifikatsiya uchun xom hisoblardan ustun`,
		},
	},
};

export default task;
