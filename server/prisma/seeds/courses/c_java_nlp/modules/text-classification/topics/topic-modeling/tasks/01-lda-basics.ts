import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-lda-basics',
	title: 'LDA Topic Modeling',
	difficulty: 'medium',
	tags: ['nlp', 'lda', 'topic-modeling'],
	estimatedTime: '25m',
	isPremium: false,
	order: 1,
	description: `# LDA Topic Modeling

Implement Latent Dirichlet Allocation for topic discovery.

## Task

Build LDA topic model:
- Document-term matrix preparation
- Topic discovery
- Document topic assignment

## Example

\`\`\`java
LDAModel lda = new LDAModel(numTopics);
lda.fit(documents);
double[] topicDist = lda.getTopicDistribution(doc);
\`\`\``,

	initialCode: `import java.util.*;

public class SimpleLDA {

    private int numTopics;
    private Map<String, Integer> vocabulary;
    private double[][] topicWordDist;

    /**
     */
    public SimpleLDA(int numTopics) {
    }

    /**
     */
    public int[][] buildDocTermMatrix(List<String> documents) {
        return null;
    }

    /**
     */
    public List<String> getTopWords(int topicId, int n) {
        return null;
    }

    /**
     */
    public int assignTopic(String document) {
        return 0;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class SimpleLDA {

    private int numTopics;
    private Map<String, Integer> vocabulary;
    private double[][] topicWordDist;  // P(word | topic)
    private double[] topicDist;         // P(topic)
    private List<String> vocabList;
    private Random random;

    /**
     * Initialize LDA with number of topics.
     */
    public SimpleLDA(int numTopics) {
        this.numTopics = numTopics;
        this.vocabulary = new LinkedHashMap<>();
        this.vocabList = new ArrayList<>();
        this.random = new Random(42);
    }

    /**
     * Build vocabulary from documents.
     */
    private void buildVocabulary(List<String> documents) {
        vocabulary.clear();
        vocabList.clear();
        int idx = 0;

        for (String doc : documents) {
            for (String word : tokenize(doc)) {
                if (!vocabulary.containsKey(word)) {
                    vocabulary.put(word, idx++);
                    vocabList.add(word);
                }
            }
        }
    }

    /**
     * Build document-term matrix.
     */
    public int[][] buildDocTermMatrix(List<String> documents) {
        buildVocabulary(documents);
        int[][] matrix = new int[documents.size()][vocabulary.size()];

        for (int d = 0; d < documents.size(); d++) {
            for (String word : tokenize(documents.get(d))) {
                Integer idx = vocabulary.get(word);
                if (idx != null) {
                    matrix[d][idx]++;
                }
            }
        }
        return matrix;
    }

    /**
     * Simplified LDA training (random initialization + smoothing).
     */
    public void fit(List<String> documents) {
        int[][] docTermMatrix = buildDocTermMatrix(documents);
        int V = vocabulary.size();

        // Initialize topic-word distribution randomly
        topicWordDist = new double[numTopics][V];
        topicDist = new double[numTopics];

        // Random initialization with smoothing
        for (int t = 0; t < numTopics; t++) {
            double sum = 0;
            for (int w = 0; w < V; w++) {
                topicWordDist[t][w] = random.nextDouble() + 0.01;
                sum += topicWordDist[t][w];
            }
            // Normalize
            for (int w = 0; w < V; w++) {
                topicWordDist[t][w] /= sum;
            }
            topicDist[t] = 1.0 / numTopics;
        }
    }

    /**
     * Get top words for each topic.
     */
    public List<String> getTopWords(int topicId, int n) {
        if (topicId >= numTopics || topicWordDist == null) {
            return Collections.emptyList();
        }

        Map<Integer, Double> wordScores = new HashMap<>();
        for (int w = 0; w < topicWordDist[topicId].length; w++) {
            wordScores.put(w, topicWordDist[topicId][w]);
        }

        return wordScores.entrySet().stream()
            .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
            .limit(n)
            .map(e -> vocabList.get(e.getKey()))
            .collect(Collectors.toList());
    }

    /**
     * Get topic distribution for document.
     */
    public double[] getTopicDistribution(String document) {
        double[] dist = new double[numTopics];
        String[] words = tokenize(document);

        for (int t = 0; t < numTopics; t++) {
            double score = topicDist[t];
            for (String word : words) {
                Integer idx = vocabulary.get(word);
                if (idx != null) {
                    score *= topicWordDist[t][idx];
                }
            }
            dist[t] = score;
        }

        // Normalize
        double sum = 0;
        for (double d : dist) sum += d;
        if (sum > 0) {
            for (int i = 0; i < dist.length; i++) {
                dist[i] /= sum;
            }
        }
        return dist;
    }

    /**
     * Assign topic to new document.
     */
    public int assignTopic(String document) {
        double[] dist = getTopicDistribution(document);
        int maxTopic = 0;
        for (int i = 1; i < dist.length; i++) {
            if (dist[i] > dist[maxTopic]) {
                maxTopic = i;
            }
        }
        return maxTopic;
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

public class SimpleLDATest {

    @Test
    void testBuildDocTermMatrix() {
        SimpleLDA lda = new SimpleLDA(3);
        List<String> docs = Arrays.asList("hello world", "hello there");
        int[][] matrix = lda.buildDocTermMatrix(docs);

        assertEquals(2, matrix.length);
    }

    @Test
    void testFit() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> docs = Arrays.asList(
            "machine learning ai",
            "deep learning neural",
            "sports football game",
            "basketball game score"
        );
        lda.fit(docs);

        List<String> topWords = lda.getTopWords(0, 3);
        assertEquals(3, topWords.size());
    }

    @Test
    void testAssignTopic() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> docs = Arrays.asList("hello world", "test data");
        lda.fit(docs);

        int topic = lda.assignTopic("hello world");
        assertTrue(topic >= 0 && topic < 2);
    }

    @Test
    void testGetTopicDistribution() {
        SimpleLDA lda = new SimpleLDA(3);
        List<String> docs = Arrays.asList("hello world", "test data", "hello test");
        lda.fit(docs);

        double[] dist = lda.getTopicDistribution("hello world");
        assertEquals(3, dist.length);
    }

    @Test
    void testGetTopWordsInvalidTopic() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> docs = Arrays.asList("hello world");
        lda.fit(docs);

        List<String> words = lda.getTopWords(10, 5);
        assertTrue(words.isEmpty());
    }

    @Test
    void testDocTermMatrixValues() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> docs = Arrays.asList("hello hello world");
        int[][] matrix = lda.buildDocTermMatrix(docs);

        int helloCount = matrix[0][0];
        assertTrue(helloCount >= 1);
    }

    @Test
    void testTopicDistributionSumsToOne() {
        SimpleLDA lda = new SimpleLDA(3);
        List<String> docs = Arrays.asList("hello world", "test data", "hello test");
        lda.fit(docs);

        double[] dist = lda.getTopicDistribution("hello world");
        double sum = 0;
        for (double d : dist) sum += d;
        assertEquals(1.0, sum, 0.01);
    }

    @Test
    void testEmptyDocTermMatrix() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> docs = Arrays.asList("");
        int[][] matrix = lda.buildDocTermMatrix(docs);
        assertNotNull(matrix);
    }

    @Test
    void testAssignTopicUnknownWords() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> docs = Arrays.asList("hello world", "test data");
        lda.fit(docs);

        int topic = lda.assignTopic("unknown words here");
        assertTrue(topic >= 0 && topic < 2);
    }

    @Test
    void testGetTopWordsBeforeFit() {
        SimpleLDA lda = new SimpleLDA(2);
        List<String> words = lda.getTopWords(0, 3);
        assertTrue(words.isEmpty());
    }
}`,

	hint1: 'Build a document-term matrix counting word occurrences',
	hint2: 'Topics are probability distributions over words',

	whyItMatters: `LDA discovers hidden themes in documents:

- **Unsupervised**: No labeled data needed
- **Interpretable**: Topics are word distributions
- **Exploratory**: Understand document collections
- **Dimensionality reduction**: Compress documents to topic vectors

Widely used for content analysis and recommendation.`,

	translations: {
		ru: {
			title: 'LDA тематическое моделирование',
			description: `# LDA тематическое моделирование

Реализуйте Latent Dirichlet Allocation для обнаружения тем.

## Задача

Создайте LDA модель:
- Подготовка матрицы документ-терм
- Обнаружение тем
- Назначение тем документам

## Пример

\`\`\`java
LDAModel lda = new LDAModel(numTopics);
lda.fit(documents);
double[] topicDist = lda.getTopicDistribution(doc);
\`\`\``,
			hint1: 'Создайте матрицу документ-терм считая вхождения слов',
			hint2: 'Темы - это распределения вероятностей по словам',
			whyItMatters: `LDA обнаруживает скрытые темы в документах:

- **Без учителя**: Не нужны размеченные данные
- **Интерпретируемость**: Темы - это распределения слов
- **Исследовательский**: Понимание коллекций документов
- **Снижение размерности**: Сжатие документов в векторы тем`,
		},
		uz: {
			title: 'LDA mavzu modellash',
			description: `# LDA mavzu modellash

Mavzularni aniqlash uchun Latent Dirichlet Allocation ni amalga oshiring.

## Topshiriq

LDA modelini yarating:
- Hujjat-term matritsasini tayyorlash
- Mavzularni aniqlash
- Hujjatlarga mavzu tayinlash

## Misol

\`\`\`java
LDAModel lda = new LDAModel(numTopics);
lda.fit(documents);
double[] topicDist = lda.getTopicDistribution(doc);
\`\`\``,
			hint1: "So'z takrorlanishlarini hisoblab hujjat-term matritsasini yarating",
			hint2: "Mavzular so'zlar bo'yicha ehtimollik taqsimotlari",
			whyItMatters: `LDA hujjatlarda yashirin mavzularni aniqlaydi:

- **Nazoratsiz**: Belgilangan ma'lumotlar kerak emas
- **Tushunarli**: Mavzular so'z taqsimotlari
- **Tadqiqot**: Hujjat to'plamlarini tushunish
- **O'lchamni kamaytirish**: Hujjatlarni mavzu vektorlariga siqish`,
		},
	},
};

export default task;
