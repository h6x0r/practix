import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-bert-embeddings',
	title: 'BERT Embeddings',
	difficulty: 'hard',
	tags: ['nlp', 'bert', 'embeddings', 'transformers'],
	estimatedTime: '30m',
	isPremium: true,
	order: 3,
	description: `# BERT Embeddings

Use BERT model for contextual word embeddings.

## Task

Implement BERT embedding extraction:
- Load pre-trained BERT
- Extract contextual embeddings
- Pool for sentence representations

## Example

\`\`\`java
BertEmbeddings bert = new BertEmbeddings();
double[] embedding = bert.encode("Hello, how are you?");
double similarity = bert.similarity(sent1, sent2);
\`\`\``,

	initialCode: `import java.util.*;

public class BertEmbeddings {

    /**
     */
    public double[] encode(String text) {
        return null;
    }

    /**
     */
    public double[] meanPooling(double[][] tokenEmbeddings) {
        return null;
    }

    /**
     */
    public double[] clsPooling(double[][] tokenEmbeddings) {
        return null;
    }

    /**
     */
    public double similarity(String text1, String text2) {
        return 0.0;
    }
}`,

	solutionCode: `import java.util.*;

public class BertEmbeddings {

    private int embeddingDim;
    private Map<String, double[]> wordEmbeddings;
    private Random random;

    public BertEmbeddings() {
        this.embeddingDim = 768;  // BERT base dimension
        this.wordEmbeddings = new HashMap<>();
        this.random = new Random(42);
    }

    /**
     * Encode text to embedding vector.
     */
    public double[] encode(String text) {
        // Simulate BERT encoding (real impl would use DJL/ONNX)
        double[][] tokenEmbeddings = getTokenEmbeddings(text);
        return meanPooling(tokenEmbeddings);
    }

    /**
     * Get simulated token embeddings.
     */
    private double[][] getTokenEmbeddings(String text) {
        String[] tokens = text.toLowerCase().split("\\\\s+");
        double[][] embeddings = new double[tokens.length + 2][embeddingDim];  // +2 for [CLS] and [SEP]

        // [CLS] token
        embeddings[0] = getOrCreateEmbedding("[CLS]");

        // Word tokens
        for (int i = 0; i < tokens.length; i++) {
            embeddings[i + 1] = getOrCreateEmbedding(tokens[i]);
        }

        // [SEP] token
        embeddings[tokens.length + 1] = getOrCreateEmbedding("[SEP]");

        return embeddings;
    }

    private double[] getOrCreateEmbedding(String token) {
        if (!wordEmbeddings.containsKey(token)) {
            double[] embedding = new double[embeddingDim];
            for (int i = 0; i < embeddingDim; i++) {
                embedding[i] = random.nextGaussian() * 0.02;
            }
            wordEmbeddings.put(token, embedding);
        }
        return wordEmbeddings.get(token).clone();
    }

    /**
     * Mean pooling over token embeddings.
     */
    public double[] meanPooling(double[][] tokenEmbeddings) {
        int numTokens = tokenEmbeddings.length;
        int dim = tokenEmbeddings[0].length;
        double[] pooled = new double[dim];

        for (int i = 0; i < dim; i++) {
            double sum = 0;
            for (int t = 0; t < numTokens; t++) {
                sum += tokenEmbeddings[t][i];
            }
            pooled[i] = sum / numTokens;
        }
        return pooled;
    }

    /**
     * CLS pooling (use first token).
     */
    public double[] clsPooling(double[][] tokenEmbeddings) {
        return tokenEmbeddings[0].clone();
    }

    /**
     * Max pooling over token embeddings.
     */
    public double[] maxPooling(double[][] tokenEmbeddings) {
        int numTokens = tokenEmbeddings.length;
        int dim = tokenEmbeddings[0].length;
        double[] pooled = new double[dim];

        for (int i = 0; i < dim; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < numTokens; t++) {
                max = Math.max(max, tokenEmbeddings[t][i]);
            }
            pooled[i] = max;
        }
        return pooled;
    }

    /**
     * Calculate similarity between two texts.
     */
    public double similarity(String text1, String text2) {
        double[] emb1 = encode(text1);
        double[] emb2 = encode(text2);
        return cosineSimilarity(emb1, emb2);
    }

    /**
     * Cosine similarity.
     */
    private double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Encode multiple texts (batched).
     */
    public double[][] encodeBatch(List<String> texts) {
        double[][] embeddings = new double[texts.size()][embeddingDim];
        for (int i = 0; i < texts.size(); i++) {
            embeddings[i] = encode(texts.get(i));
        }
        return embeddings;
    }

    /**
     * Find most similar text from candidates.
     */
    public String findMostSimilar(String query, List<String> candidates) {
        double[] queryEmb = encode(query);
        String best = null;
        double bestSim = -1;

        for (String cand : candidates) {
            double sim = cosineSimilarity(queryEmb, encode(cand));
            if (sim > bestSim) {
                bestSim = sim;
                best = cand;
            }
        }
        return best;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class BertEmbeddingsTest {

    @Test
    void testEncode() {
        BertEmbeddings bert = new BertEmbeddings();
        double[] embedding = bert.encode("Hello world");

        assertEquals(768, embedding.length);
    }

    @Test
    void testMeanPooling() {
        BertEmbeddings bert = new BertEmbeddings();
        double[][] tokens = {{1, 2}, {3, 4}, {5, 6}};
        double[] pooled = bert.meanPooling(tokens);

        assertEquals(3.0, pooled[0], 0.001);
        assertEquals(4.0, pooled[1], 0.001);
    }

    @Test
    void testClsPooling() {
        BertEmbeddings bert = new BertEmbeddings();
        double[][] tokens = {{1, 2}, {3, 4}, {5, 6}};
        double[] pooled = bert.clsPooling(tokens);

        assertEquals(1.0, pooled[0], 0.001);
        assertEquals(2.0, pooled[1], 0.001);
    }

    @Test
    void testSimilarity() {
        BertEmbeddings bert = new BertEmbeddings();

        // Same text should have high similarity
        double sameSim = bert.similarity("hello world", "hello world");
        assertTrue(sameSim > 0.9);
    }

    @Test
    void testFindMostSimilar() {
        BertEmbeddings bert = new BertEmbeddings();
        List<String> candidates = Arrays.asList(
            "machine learning",
            "deep learning neural networks",
            "cooking recipes"
        );

        String similar = bert.findMostSimilar("artificial intelligence", candidates);
        assertNotNull(similar);
    }

    @Test
    void testMaxPooling() {
        BertEmbeddings bert = new BertEmbeddings();
        double[][] tokens = {{1, 2}, {3, 4}, {5, 6}};
        double[] pooled = bert.maxPooling(tokens);
        assertEquals(5.0, pooled[0], 0.001);
        assertEquals(6.0, pooled[1], 0.001);
    }

    @Test
    void testEncodeBatch() {
        BertEmbeddings bert = new BertEmbeddings();
        List<String> texts = Arrays.asList("hello", "world");
        double[][] embeddings = bert.encodeBatch(texts);
        assertEquals(2, embeddings.length);
        assertEquals(768, embeddings[0].length);
    }

    @Test
    void testSimilarityRange() {
        BertEmbeddings bert = new BertEmbeddings();
        double sim = bert.similarity("hello", "world");
        assertTrue(sim >= -1 && sim <= 1);
    }

    @Test
    void testEncodeEmpty() {
        BertEmbeddings bert = new BertEmbeddings();
        double[] embedding = bert.encode("");
        assertNotNull(embedding);
    }

    @Test
    void testFindMostSimilarEmptyList() {
        BertEmbeddings bert = new BertEmbeddings();
        String result = bert.findMostSimilar("query", Arrays.asList());
        assertNull(result);
    }
}`,

	hint1: 'Mean pooling averages all token embeddings for sentence representation',
	hint2: 'CLS token embedding is trained to represent the whole sentence',

	whyItMatters: `BERT embeddings revolutionized NLP:

- **Contextual**: Same word gets different embeddings in different contexts
- **Pre-trained**: Transfer learning from massive text corpora
- **Multi-purpose**: One model for many NLP tasks
- **State-of-the-art**: Powers modern search, QA, and classification

Understanding BERT is essential for modern NLP practitioners.`,

	translations: {
		ru: {
			title: 'BERT эмбеддинги',
			description: `# BERT эмбеддинги

Используйте модель BERT для контекстных эмбеддингов слов.

## Задача

Реализуйте извлечение BERT эмбеддингов:
- Загрузка предобученного BERT
- Извлечение контекстных эмбеддингов
- Пулинг для представления предложений

## Пример

\`\`\`java
BertEmbeddings bert = new BertEmbeddings();
double[] embedding = bert.encode("Hello, how are you?");
double similarity = bert.similarity(sent1, sent2);
\`\`\``,
			hint1: 'Mean pooling усредняет все эмбеддинги токенов для представления предложения',
			hint2: 'Эмбеддинг CLS токена обучен представлять все предложение',
			whyItMatters: `BERT эмбеддинги произвели революцию в NLP:

- **Контекстуальные**: Одно слово получает разные эмбеддинги в разных контекстах
- **Предобученные**: Transfer learning из массивных текстовых корпусов
- **Универсальные**: Одна модель для многих NLP задач
- **State-of-the-art**: Питает современный поиск, QA и классификацию`,
		},
		uz: {
			title: 'BERT embeddinglar',
			description: `# BERT embeddinglar

Kontekstli so'z embeddinglar uchun BERT modelidan foydalaning.

## Topshiriq

BERT embedding ajratishni amalga oshiring:
- Oldindan o'qitilgan BERT ni yuklash
- Kontekstli embeddinglarni ajratib olish
- Gap representatsiyalari uchun pooling

## Misol

\`\`\`java
BertEmbeddings bert = new BertEmbeddings();
double[] embedding = bert.encode("Hello, how are you?");
double similarity = bert.similarity(sent1, sent2);
\`\`\``,
			hint1: "Mean pooling gap representatsiyasi uchun barcha token embeddinglarni o'rtacha qiladi",
			hint2: "CLS token embedding butun gapni ifodalash uchun o'qitilgan",
			whyItMatters: `BERT embeddinglar NLP da inqilob qildi:

- **Kontekstli**: Bir xil so'z turli kontekstlarda turli embeddinglar oladi
- **Oldindan o'qitilgan**: Ulkan matn korpuslaridan transfer learning
- **Ko'p maqsadli**: Ko'p NLP vazifalari uchun bitta model
- **State-of-the-art**: Zamonaviy qidiruv, QA va klassifikatsiyani quvvatlaydi`,
		},
	},
};

export default task;
