import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-document-similarity',
	title: 'Document Similarity',
	difficulty: 'medium',
	tags: ['nlp', 'similarity', 'cosine'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Document Similarity

Calculate similarity between documents using various metrics.

## Task

Implement document similarity:
- Cosine similarity
- Jaccard similarity
- Find similar documents

## Example

\`\`\`java
DocumentSimilarity sim = new DocumentSimilarity();
double score = sim.cosineSimilarity(doc1, doc2);
List<String> similar = sim.findSimilar(query, docs, 5);
\`\`\``,

	initialCode: `import java.util.*;

public class DocumentSimilarity {

    /**
     * Calculate Jaccard similarity between two texts.
     */
    public double jaccardSimilarity(String text1, String text2) {
        return 0.0;
    }

    /**
     * Calculate cosine similarity using TF vectors.
     */
    public double cosineSimilarity(String text1, String text2) {
        return 0.0;
    }

    /**
     * Find top N similar documents.
     */
    public List<String> findSimilar(String query, List<String> documents, int n) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class DocumentSimilarity {

    /**
     * Calculate Jaccard similarity between two texts.
     */
    public double jaccardSimilarity(String text1, String text2) {
        Set<String> set1 = new HashSet<>(Arrays.asList(tokenize(text1)));
        Set<String> set2 = new HashSet<>(Arrays.asList(tokenize(text2)));

        Set<String> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);

        Set<String> union = new HashSet<>(set1);
        union.addAll(set2);

        if (union.isEmpty()) return 0.0;
        return (double) intersection.size() / union.size();
    }

    /**
     * Calculate cosine similarity using TF vectors.
     */
    public double cosineSimilarity(String text1, String text2) {
        Map<String, Integer> freq1 = getTermFrequency(text1);
        Map<String, Integer> freq2 = getTermFrequency(text2);

        Set<String> allTerms = new HashSet<>();
        allTerms.addAll(freq1.keySet());
        allTerms.addAll(freq2.keySet());

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (String term : allTerms) {
            int count1 = freq1.getOrDefault(term, 0);
            int count2 = freq2.getOrDefault(term, 0);

            dotProduct += count1 * count2;
            norm1 += count1 * count1;
            norm2 += count2 * count2;
        }

        if (norm1 == 0 || norm2 == 0) return 0.0;
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Find top N similar documents.
     */
    public List<String> findSimilar(String query, List<String> documents, int n) {
        Map<String, Double> scores = new HashMap<>();

        for (String doc : documents) {
            double score = cosineSimilarity(query, doc);
            scores.put(doc, score);
        }

        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(n)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

    /**
     * Get term frequency map.
     */
    private Map<String, Integer> getTermFrequency(String text) {
        Map<String, Integer> freq = new HashMap<>();
        for (String word : tokenize(text)) {
            freq.merge(word, 1, Integer::sum);
        }
        return freq;
    }

    /**
     * Calculate overlap coefficient.
     */
    public double overlapCoefficient(String text1, String text2) {
        Set<String> set1 = new HashSet<>(Arrays.asList(tokenize(text1)));
        Set<String> set2 = new HashSet<>(Arrays.asList(tokenize(text2)));

        Set<String> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);

        int minSize = Math.min(set1.size(), set2.size());
        if (minSize == 0) return 0.0;

        return (double) intersection.size() / minSize;
    }

    /**
     * Rank all documents by similarity.
     */
    public List<Map.Entry<String, Double>> rankDocuments(String query, List<String> documents) {
        Map<String, Double> scores = new HashMap<>();
        for (String doc : documents) {
            scores.put(doc, cosineSimilarity(query, doc));
        }

        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .collect(Collectors.toList());
    }

    private String[] tokenize(String text) {
        return text.toLowerCase()
            .replaceAll("[^a-zA-Z\\\\s]", "")
            .split("\\\\s+");
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class DocumentSimilarityTest {

    @Test
    void testJaccardSimilarity() {
        DocumentSimilarity sim = new DocumentSimilarity();

        double score = sim.jaccardSimilarity("hello world", "hello there");
        assertTrue(score > 0 && score < 1);

        double identical = sim.jaccardSimilarity("hello", "hello");
        assertEquals(1.0, identical, 0.001);
    }

    @Test
    void testCosineSimilarity() {
        DocumentSimilarity sim = new DocumentSimilarity();

        double identical = sim.cosineSimilarity("hello world", "hello world");
        assertEquals(1.0, identical, 0.001);

        double different = sim.cosineSimilarity("hello", "goodbye");
        assertEquals(0.0, different, 0.001);
    }

    @Test
    void testFindSimilar() {
        DocumentSimilarity sim = new DocumentSimilarity();
        List<String> docs = Arrays.asList(
            "machine learning is great",
            "deep learning works well",
            "cooking recipes food"
        );

        List<String> similar = sim.findSimilar("machine learning algorithms", docs, 2);
        assertEquals(2, similar.size());
        assertTrue(similar.get(0).contains("learning"));
    }

    @Test
    void testJaccardEmptyText() {
        DocumentSimilarity sim = new DocumentSimilarity();
        double score = sim.jaccardSimilarity("", "");
        assertEquals(0.0, score, 0.001);
    }

    @Test
    void testCosineEmptyText() {
        DocumentSimilarity sim = new DocumentSimilarity();
        double score = sim.cosineSimilarity("", "hello");
        assertEquals(0.0, score, 0.001);
    }

    @Test
    void testOverlapCoefficient() {
        DocumentSimilarity sim = new DocumentSimilarity();
        double score = sim.overlapCoefficient("hello world", "hello there world");
        assertEquals(1.0, score, 0.001);
    }

    @Test
    void testRankDocuments() {
        DocumentSimilarity sim = new DocumentSimilarity();
        List<String> docs = Arrays.asList("hello world", "hello there", "goodbye");
        List<Map.Entry<String, Double>> ranked = sim.rankDocuments("hello", docs);
        assertEquals(3, ranked.size());
    }

    @Test
    void testFindSimilarEmptyList() {
        DocumentSimilarity sim = new DocumentSimilarity();
        List<String> similar = sim.findSimilar("query", Arrays.asList(), 5);
        assertTrue(similar.isEmpty());
    }

    @Test
    void testJaccardNoOverlap() {
        DocumentSimilarity sim = new DocumentSimilarity();
        double score = sim.jaccardSimilarity("hello world", "goodbye there");
        assertEquals(0.0, score, 0.001);
    }

    @Test
    void testCosinePartialMatch() {
        DocumentSimilarity sim = new DocumentSimilarity();
        double score = sim.cosineSimilarity("hello world", "hello");
        assertTrue(score > 0 && score < 1);
    }
}`,

	hint1: 'Jaccard = |intersection| / |union| of word sets',
	hint2: 'Cosine similarity works on term frequency vectors',

	whyItMatters: `Document similarity enables many applications:

- **Search**: Find relevant documents for a query
- **Deduplication**: Detect near-duplicate content
- **Recommendations**: Suggest similar articles
- **Clustering**: Group related documents together

Core technique for information retrieval systems.`,

	translations: {
		ru: {
			title: 'Сходство документов',
			description: `# Сходство документов

Вычисляйте сходство между документами используя различные метрики.

## Задача

Реализуйте сходство документов:
- Косинусное сходство
- Сходство Жаккара
- Поиск похожих документов

## Пример

\`\`\`java
DocumentSimilarity sim = new DocumentSimilarity();
double score = sim.cosineSimilarity(doc1, doc2);
List<String> similar = sim.findSimilar(query, docs, 5);
\`\`\``,
			hint1: 'Жаккар = |пересечение| / |объединение| множеств слов',
			hint2: 'Косинусное сходство работает на векторах частот терминов',
			whyItMatters: `Сходство документов позволяет многие приложения:

- **Поиск**: Находите релевантные документы для запроса
- **Дедупликация**: Обнаружение почти-дубликатов
- **Рекомендации**: Предложение похожих статей
- **Кластеризация**: Группировка связанных документов`,
		},
		uz: {
			title: "Hujjat o'xshashligi",
			description: `# Hujjat o'xshashligi

Turli metrikalar yordamida hujjatlar o'rtasidagi o'xshashlikni hisoblang.

## Topshiriq

Hujjat o'xshashligini amalga oshiring:
- Kosinus o'xshashligi
- Jakkar o'xshashligi
- O'xshash hujjatlarni topish

## Misol

\`\`\`java
DocumentSimilarity sim = new DocumentSimilarity();
double score = sim.cosineSimilarity(doc1, doc2);
List<String> similar = sim.findSimilar(query, docs, 5);
\`\`\``,
			hint1: "Jakkar = |kesishuv| / |birlashuv| so'zlar to'plamlari",
			hint2: "Kosinus o'xshashligi term chastota vektorlarida ishlaydi",
			whyItMatters: `Hujjat o'xshashligi ko'p ilovalarni yoqadi:

- **Qidiruv**: So'rov uchun tegishli hujjatlarni topish
- **Duplikatsiyani olib tashlash**: Deyarli-dublikat kontentni aniqlash
- **Tavsiyalar**: O'xshash maqolalarni taklif qilish
- **Klasterlash**: Bog'liq hujjatlarni birga guruhlash`,
		},
	},
};

export default task;
