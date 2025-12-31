import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-word2vec',
	title: 'Word2Vec Embeddings',
	difficulty: 'medium',
	tags: ['nlp', 'embeddings', 'word2vec', 'dl4j'],
	estimatedTime: '25m',
	isPremium: false,
	order: 3,
	description: `# Word2Vec Embeddings

Train and use Word2Vec word embeddings with DL4J.

## Task

Implement Word2Vec operations:
- Train Word2Vec model
- Get word vectors
- Find similar words

## Example

\`\`\`java
Word2Vec vec = new Word2Vec.Builder()
    .minWordFrequency(5)
    .iterations(1)
    .layerSize(100)
    .build();
vec.fit(sentences);
double[] vector = vec.getWordVector("king");
\`\`\``,

	initialCode: `import java.util.*;

public class Word2VecWrapper {

    private Map<String, double[]> wordVectors;
    private int vectorSize;

    /**
     */
    public Word2VecWrapper(int vectorSize) {
    }

    /**
     */
    public void addWord(String word, double[] vector) {
    }

    /**
     */
    public double[] getWordVector(String word) {
        return null;
    }

    /**
     */
    public double similarity(String word1, String word2) {
        return 0.0;
    }

    /**
     */
    public List<String> mostSimilar(String word, int n) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class Word2VecWrapper {

    private Map<String, double[]> wordVectors;
    private int vectorSize;

    /**
     * Initialize with vector size.
     */
    public Word2VecWrapper(int vectorSize) {
        this.vectorSize = vectorSize;
        this.wordVectors = new HashMap<>();
    }

    /**
     * Add word vector.
     */
    public void addWord(String word, double[] vector) {
        if (vector.length != vectorSize) {
            throw new IllegalArgumentException("Vector size mismatch");
        }
        wordVectors.put(word.toLowerCase(), vector);
    }

    /**
     * Get vector for word.
     */
    public double[] getWordVector(String word) {
        return wordVectors.get(word.toLowerCase());
    }

    /**
     * Check if word exists.
     */
    public boolean hasWord(String word) {
        return wordVectors.containsKey(word.toLowerCase());
    }

    /**
     * Calculate cosine similarity between two words.
     */
    public double similarity(String word1, String word2) {
        double[] v1 = getWordVector(word1);
        double[] v2 = getWordVector(word2);

        if (v1 == null || v2 == null) return 0.0;
        return cosineSimilarity(v1, v2);
    }

    /**
     * Cosine similarity between two vectors.
     */
    public static double cosineSimilarity(double[] a, double[] b) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0) return 0.0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Find most similar words.
     */
    public List<String> mostSimilar(String word, int n) {
        double[] targetVector = getWordVector(word);
        if (targetVector == null) return Collections.emptyList();

        Map<String, Double> similarities = new HashMap<>();
        for (String w : wordVectors.keySet()) {
            if (!w.equals(word.toLowerCase())) {
                similarities.put(w, similarity(word, w));
            }
        }

        return similarities.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(n)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

    /**
     * Word analogy: king - man + woman = queen
     */
    public String analogy(String word1, String word2, String word3) {
        double[] v1 = getWordVector(word1);
        double[] v2 = getWordVector(word2);
        double[] v3 = getWordVector(word3);

        if (v1 == null || v2 == null || v3 == null) return null;

        // target = v1 - v2 + v3
        double[] target = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            target[i] = v1[i] - v2[i] + v3[i];
        }

        // Find closest word
        String bestWord = null;
        double bestSim = -1;
        Set<String> exclude = new HashSet<>(Arrays.asList(
            word1.toLowerCase(), word2.toLowerCase(), word3.toLowerCase()
        ));

        for (Map.Entry<String, double[]> entry : wordVectors.entrySet()) {
            if (!exclude.contains(entry.getKey())) {
                double sim = cosineSimilarity(target, entry.getValue());
                if (sim > bestSim) {
                    bestSim = sim;
                    bestWord = entry.getKey();
                }
            }
        }
        return bestWord;
    }

    /**
     * Get vocabulary size.
     */
    public int getVocabSize() {
        return wordVectors.size();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class Word2VecWrapperTest {

    private Word2VecWrapper w2v;

    @BeforeEach
    void setup() {
        w2v = new Word2VecWrapper(3);
        w2v.addWord("king", new double[]{1.0, 0.5, 0.0});
        w2v.addWord("queen", new double[]{0.9, 0.6, 0.1});
        w2v.addWord("man", new double[]{0.8, 0.0, 0.0});
        w2v.addWord("woman", new double[]{0.7, 0.1, 0.1});
    }

    @Test
    void testGetWordVector() {
        double[] vector = w2v.getWordVector("king");
        assertNotNull(vector);
        assertEquals(3, vector.length);
    }

    @Test
    void testSimilarity() {
        double sim = w2v.similarity("king", "queen");
        assertTrue(sim > 0.9); // Should be highly similar
    }

    @Test
    void testMostSimilar() {
        List<String> similar = w2v.mostSimilar("king", 2);
        assertEquals(2, similar.size());
        assertTrue(similar.contains("queen"));
    }

    @Test
    void testCosineSimilarity() {
        double sim = Word2VecWrapper.cosineSimilarity(
            new double[]{1, 0, 0},
            new double[]{1, 0, 0}
        );
        assertEquals(1.0, sim, 0.001);
    }

    @Test
    void testHasWord() {
        assertTrue(w2v.hasWord("king"));
        assertFalse(w2v.hasWord("unknown"));
    }

    @Test
    void testGetVocabSize() {
        assertEquals(4, w2v.getVocabSize());
    }

    @Test
    void testSimilarityUnknownWord() {
        double sim = w2v.similarity("king", "unknown");
        assertEquals(0.0, sim, 0.001);
    }

    @Test
    void testMostSimilarUnknownWord() {
        List<String> similar = w2v.mostSimilar("unknown", 2);
        assertTrue(similar.isEmpty());
    }

    @Test
    void testAnalogy() {
        String result = w2v.analogy("king", "man", "woman");
        assertNotNull(result);
    }

    @Test
    void testOrthogonalVectors() {
        double sim = Word2VecWrapper.cosineSimilarity(
            new double[]{1, 0, 0},
            new double[]{0, 1, 0}
        );
        assertEquals(0.0, sim, 0.001);
    }
}`,

	hint1: 'Cosine similarity = dot(a,b) / (norm(a) * norm(b))',
	hint2: 'Word analogies work by vector arithmetic: king - man + woman ≈ queen',

	whyItMatters: `Word2Vec revolutionized NLP:

- **Semantic meaning**: Similar words have similar vectors
- **Analogies**: Captures relationships like king-man+woman=queen
- **Transfer learning**: Pre-trained vectors improve downstream tasks
- **Foundation**: Basis for modern contextual embeddings

Understanding word vectors is key to modern NLP.`,

	translations: {
		ru: {
			title: 'Word2Vec эмбеддинги',
			description: `# Word2Vec эмбеддинги

Обучайте и используйте Word2Vec эмбеддинги слов с DL4J.

## Задача

Реализуйте операции Word2Vec:
- Обучение модели Word2Vec
- Получение векторов слов
- Поиск похожих слов

## Пример

\`\`\`java
Word2Vec vec = new Word2Vec.Builder()
    .minWordFrequency(5)
    .iterations(1)
    .layerSize(100)
    .build();
vec.fit(sentences);
double[] vector = vec.getWordVector("king");
\`\`\``,
			hint1: 'Косинусное сходство = dot(a,b) / (norm(a) * norm(b))',
			hint2: 'Аналогии работают через векторную арифметику: king - man + woman ≈ queen',
			whyItMatters: `Word2Vec произвел революцию в NLP:

- **Семантическое значение**: Похожие слова имеют похожие векторы
- **Аналогии**: Захватывает отношения вроде king-man+woman=queen
- **Transfer learning**: Предобученные векторы улучшают последующие задачи
- **Основа**: База для современных контекстных эмбеддингов`,
		},
		uz: {
			title: 'Word2Vec embeddinglar',
			description: `# Word2Vec embeddinglar

DL4J bilan Word2Vec so'z embeddinglarini o'rgating va foydalaning.

## Topshiriq

Word2Vec operatsiyalarini amalga oshiring:
- Word2Vec modelini o'qitish
- So'z vektorlarini olish
- O'xshash so'zlarni topish

## Misol

\`\`\`java
Word2Vec vec = new Word2Vec.Builder()
    .minWordFrequency(5)
    .iterations(1)
    .layerSize(100)
    .build();
vec.fit(sentences);
double[] vector = vec.getWordVector("king");
\`\`\``,
			hint1: "Kosinus o'xshashligi = dot(a,b) / (norm(a) * norm(b))",
			hint2: "So'z analogiyalari vektor arifmetikasi orqali ishlaydi: king - man + woman ≈ queen",
			whyItMatters: `Word2Vec NLP da inqilob qildi:

- **Semantik ma'no**: O'xshash so'zlar o'xshash vektorlarga ega
- **Analogiyalar**: king-man+woman=queen kabi munosabatlarni oladi
- **Transfer learning**: Oldindan o'qitilgan vektorlar keyingi vazifalarni yaxshilaydi
- **Asos**: Zamonaviy kontekstual embeddinglar uchun asos`,
		},
	},
};

export default task;
