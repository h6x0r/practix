import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-sentence-embeddings',
	title: 'Sentence Embeddings',
	difficulty: 'medium',
	tags: ['nlp', 'embeddings', 'sentences', 'averaging'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# Sentence Embeddings

Create vector representations for entire sentences.

## Task

Implement sentence embedding methods:
- Average word vectors
- Weighted averaging with TF-IDF
- Calculate sentence similarity

## Example

\`\`\`java
SentenceEmbedder embedder = new SentenceEmbedder(word2vec);
double[] vector = embedder.embed("This is a sentence");
double similarity = embedder.similarity(sent1, sent2);
\`\`\``,

	initialCode: `import java.util.*;

public class SentenceEmbedder {

    private Map<String, double[]> wordVectors;
    private int vectorSize;

    /**
     */
    public SentenceEmbedder(Map<String, double[]> wordVectors, int vectorSize) {
    }

    /**
     */
    public double[] embedAverage(String sentence) {
        return null;
    }

    /**
     */
    public double similarity(String sent1, String sent2) {
        return 0.0;
    }

    /**
     */
    public String mostSimilar(String query, List<String> sentences) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class SentenceEmbedder {

    private Map<String, double[]> wordVectors;
    private int vectorSize;

    /**
     * Initialize with word vectors.
     */
    public SentenceEmbedder(Map<String, double[]> wordVectors, int vectorSize) {
        this.wordVectors = wordVectors;
        this.vectorSize = vectorSize;
    }

    /**
     * Average word vectors to get sentence vector.
     */
    public double[] embedAverage(String sentence) {
        String[] words = tokenize(sentence);
        double[] embedding = new double[vectorSize];
        int count = 0;

        for (String word : words) {
            double[] wordVec = wordVectors.get(word.toLowerCase());
            if (wordVec != null) {
                for (int i = 0; i < vectorSize; i++) {
                    embedding[i] += wordVec[i];
                }
                count++;
            }
        }

        if (count > 0) {
            for (int i = 0; i < vectorSize; i++) {
                embedding[i] /= count;
            }
        }
        return embedding;
    }

    /**
     * Weighted average with custom weights.
     */
    public double[] embedWeighted(String sentence, Map<String, Double> weights) {
        String[] words = tokenize(sentence);
        double[] embedding = new double[vectorSize];
        double totalWeight = 0;

        for (String word : words) {
            double[] wordVec = wordVectors.get(word.toLowerCase());
            double weight = weights.getOrDefault(word.toLowerCase(), 1.0);

            if (wordVec != null) {
                for (int i = 0; i < vectorSize; i++) {
                    embedding[i] += wordVec[i] * weight;
                }
                totalWeight += weight;
            }
        }

        if (totalWeight > 0) {
            for (int i = 0; i < vectorSize; i++) {
                embedding[i] /= totalWeight;
            }
        }
        return embedding;
    }

    /**
     * Calculate sentence similarity.
     */
    public double similarity(String sent1, String sent2) {
        double[] vec1 = embedAverage(sent1);
        double[] vec2 = embedAverage(sent2);
        return cosineSimilarity(vec1, vec2);
    }

    /**
     * Find most similar sentence from list.
     */
    public String mostSimilar(String query, List<String> sentences) {
        double[] queryVec = embedAverage(query);
        String best = null;
        double bestSim = -1;

        for (String sent : sentences) {
            double[] sentVec = embedAverage(sent);
            double sim = cosineSimilarity(queryVec, sentVec);
            if (sim > bestSim) {
                bestSim = sim;
                best = sent;
            }
        }
        return best;
    }

    /**
     * Rank sentences by similarity to query.
     */
    public List<String> rankBySimilarity(String query, List<String> sentences) {
        double[] queryVec = embedAverage(query);
        Map<String, Double> scores = new HashMap<>();

        for (String sent : sentences) {
            double[] sentVec = embedAverage(sent);
            scores.put(sent, cosineSimilarity(queryVec, sentVec));
        }

        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

    private double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0 || normB == 0) return 0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private String[] tokenize(String text) {
        return text.toLowerCase()
            .replaceAll("[^a-zA-Z\\\\s]", "")
            .split("\\\\s+");
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SentenceEmbedderTest {

    private SentenceEmbedder embedder;

    @BeforeEach
    void setup() {
        Map<String, double[]> vectors = new HashMap<>();
        vectors.put("hello", new double[]{1.0, 0.0, 0.0});
        vectors.put("world", new double[]{0.0, 1.0, 0.0});
        vectors.put("hi", new double[]{0.9, 0.1, 0.0});
        vectors.put("earth", new double[]{0.1, 0.9, 0.0});

        embedder = new SentenceEmbedder(vectors, 3);
    }

    @Test
    void testEmbedAverage() {
        double[] vec = embedder.embedAverage("hello world");
        assertEquals(0.5, vec[0], 0.01);
        assertEquals(0.5, vec[1], 0.01);
    }

    @Test
    void testSimilarity() {
        double sim = embedder.similarity("hello world", "hi earth");
        assertTrue(sim > 0.5);
    }

    @Test
    void testMostSimilar() {
        List<String> sentences = Arrays.asList(
            "hello world",
            "hi earth",
            "different text"
        );
        String similar = embedder.mostSimilar("hi world", sentences);
        assertNotNull(similar);
    }

    @Test
    void testEmbedAverageUnknownWords() {
        double[] vec = embedder.embedAverage("unknown words only");
        assertEquals(0.0, vec[0], 0.001);
    }

    @Test
    void testEmbedSingleWord() {
        double[] vec = embedder.embedAverage("hello");
        assertEquals(1.0, vec[0], 0.01);
        assertEquals(0.0, vec[1], 0.01);
    }

    @Test
    void testSimilaritySameText() {
        double sim = embedder.similarity("hello world", "hello world");
        assertEquals(1.0, sim, 0.01);
    }

    @Test
    void testRankBySimilarity() {
        List<String> sentences = Arrays.asList("hello", "world", "hi");
        List<String> ranked = embedder.rankBySimilarity("hello", sentences);
        assertEquals(3, ranked.size());
        assertEquals("hello", ranked.get(0));
    }

    @Test
    void testEmbedWeighted() {
        Map<String, Double> weights = new HashMap<>();
        weights.put("hello", 2.0);
        weights.put("world", 1.0);
        double[] vec = embedder.embedWeighted("hello world", weights);
        assertNotNull(vec);
    }

    @Test
    void testMostSimilarEmptyList() {
        List<String> sentences = Arrays.asList();
        String result = embedder.mostSimilar("hello", sentences);
        assertNull(result);
    }

    @Test
    void testVectorLength() {
        double[] vec = embedder.embedAverage("hello world");
        assertEquals(3, vec.length);
    }
}`,

	hint1: 'Average word vectors element-wise, dividing by word count',
	hint2: 'Handle unknown words by skipping them in the average',

	whyItMatters: `Sentence embeddings enable text comparison:

- **Semantic search**: Find similar documents by meaning
- **Clustering**: Group similar sentences together
- **Paraphrase detection**: Identify same meaning in different words
- **Simple baseline**: Often works surprisingly well

Foundation for document-level NLP applications.`,

	translations: {
		ru: {
			title: 'Эмбеддинги предложений',
			description: `# Эмбеддинги предложений

Создавайте векторные представления для целых предложений.

## Задача

Реализуйте методы эмбеддингов предложений:
- Усреднение векторов слов
- Взвешенное усреднение с TF-IDF
- Вычисление сходства предложений

## Пример

\`\`\`java
SentenceEmbedder embedder = new SentenceEmbedder(word2vec);
double[] vector = embedder.embed("This is a sentence");
double similarity = embedder.similarity(sent1, sent2);
\`\`\``,
			hint1: 'Усредняйте векторы слов поэлементно, деля на количество слов',
			hint2: 'Обрабатывайте неизвестные слова пропуская их при усреднении',
			whyItMatters: `Эмбеддинги предложений позволяют сравнивать тексты:

- **Семантический поиск**: Находите похожие документы по смыслу
- **Кластеризация**: Группируйте похожие предложения вместе
- **Обнаружение парафраз**: Определяйте одинаковый смысл в разных словах
- **Простой baseline**: Часто работает удивительно хорошо`,
		},
		uz: {
			title: 'Gap embeddinglar',
			description: `# Gap embeddinglar

Butun gaplar uchun vektor representatsiyalarini yarating.

## Topshiriq

Gap embedding metodlarini amalga oshiring:
- So'z vektorlarini o'rtacha qilish
- TF-IDF bilan vaznli o'rtacha
- Gap o'xshashligini hisoblash

## Misol

\`\`\`java
SentenceEmbedder embedder = new SentenceEmbedder(word2vec);
double[] vector = embedder.embed("This is a sentence");
double similarity = embedder.similarity(sent1, sent2);
\`\`\``,
			hint1: "So'z vektorlarini element bo'yicha o'rtacha qiling, so'zlar soniga bo'ling",
			hint2: "Noma'lum so'zlarni o'rtacha olishda o'tkazib yuboring",
			whyItMatters: `Gap embeddinglar matn taqqoslashni yoqadi:

- **Semantik qidiruv**: Ma'no bo'yicha o'xshash hujjatlarni toping
- **Klasterlash**: O'xshash gaplarni birga guruhlang
- **Parafraz aniqlash**: Turli so'zlarda bir xil ma'noni aniqlang
- **Oddiy baseline**: Ko'pincha hayratlanarli darajada yaxshi ishlaydi`,
		},
	},
};

export default task;
