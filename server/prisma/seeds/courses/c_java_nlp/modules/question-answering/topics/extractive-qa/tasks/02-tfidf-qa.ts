import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-tfidf-qa',
	title: 'TF-IDF Question Answering',
	difficulty: 'medium',
	tags: ['nlp', 'qa', 'tfidf', 'information-retrieval'],
	estimatedTime: '30m',
	isPremium: true,
	order: 2,
	description: `# TF-IDF Question Answering

Implement a question answering system using TF-IDF for passage retrieval.

## Task

Build a TF-IDF based QA system that:
- Computes TF-IDF vectors for passages
- Finds most similar passage to question
- Extracts answer spans from relevant passages

## Example

\`\`\`java
TfidfQA qa = new TfidfQA();
qa.indexDocuments(passages);
String answer = qa.answer("What is machine learning?");
\`\`\``,

	initialCode: `import java.util.*;

public class TfidfQA {

    /**
     */
    public void indexDocuments(List<String> documents) {
    }

    /**
     */
    public String answer(String question) {
        return null;
    }

    /**
     */
    public Map<String, Double> getTfidfVector(String text) {
        return null;
    }

    /**
     */
    public double cosineSimilarity(Map<String, Double> v1, Map<String, Double> v2) {
        return 0.0;
    }
}`,

	solutionCode: `import java.util.*;

public class TfidfQA {

    private List<String> documents;
    private Map<String, Double> idfScores;
    private List<Map<String, Double>> documentVectors;
    private Set<String> vocabulary;

    public TfidfQA() {
        this.documents = new ArrayList<>();
        this.idfScores = new HashMap<>();
        this.documentVectors = new ArrayList<>();
        this.vocabulary = new HashSet<>();
    }

    /**
     * Tokenize text into words.
     */
    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String[] words = text.toLowerCase().split("\\\\W+");
        for (String word : words) {
            if (word.length() > 1) {
                tokens.add(word);
                vocabulary.add(word);
            }
        }
        return tokens;
    }

    /**
     * Calculate term frequency.
     */
    private Map<String, Double> calculateTF(List<String> tokens) {
        Map<String, Double> tf = new HashMap<>();
        int total = tokens.size();

        for (String token : tokens) {
            tf.merge(token, 1.0, Double::sum);
        }

        for (String token : tf.keySet()) {
            tf.put(token, tf.get(token) / total);
        }

        return tf;
    }

    /**
     * Index documents for retrieval.
     */
    public void indexDocuments(List<String> docs) {
        this.documents = new ArrayList<>(docs);
        this.idfScores.clear();
        this.documentVectors.clear();

        // Calculate document frequencies
        Map<String, Integer> docFreq = new HashMap<>();
        for (String doc : documents) {
            Set<String> seen = new HashSet<>();
            for (String token : tokenize(doc)) {
                if (!seen.contains(token)) {
                    docFreq.merge(token, 1, Integer::sum);
                    seen.add(token);
                }
            }
        }

        // Calculate IDF scores
        int N = documents.size();
        for (Map.Entry<String, Integer> entry : docFreq.entrySet()) {
            double idf = Math.log((double) N / (1 + entry.getValue())) + 1;
            idfScores.put(entry.getKey(), idf);
        }

        // Calculate TF-IDF vectors for all documents
        for (String doc : documents) {
            documentVectors.add(getTfidfVector(doc));
        }
    }

    /**
     * Calculate TF-IDF vector for text.
     */
    public Map<String, Double> getTfidfVector(String text) {
        List<String> tokens = tokenize(text);
        Map<String, Double> tf = calculateTF(tokens);
        Map<String, Double> tfidf = new HashMap<>();

        for (Map.Entry<String, Double> entry : tf.entrySet()) {
            double idf = idfScores.getOrDefault(entry.getKey(), 1.0);
            tfidf.put(entry.getKey(), entry.getValue() * idf);
        }

        return tfidf;
    }

    /**
     * Calculate cosine similarity between two vectors.
     */
    public double cosineSimilarity(Map<String, Double> v1, Map<String, Double> v2) {
        Set<String> allTerms = new HashSet<>();
        allTerms.addAll(v1.keySet());
        allTerms.addAll(v2.keySet());

        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;

        for (String term : allTerms) {
            double val1 = v1.getOrDefault(term, 0.0);
            double val2 = v2.getOrDefault(term, 0.0);

            dotProduct += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }

        if (norm1 == 0 || norm2 == 0) return 0;

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Answer a question using TF-IDF retrieval.
     */
    public String answer(String question) {
        if (documents.isEmpty()) return null;

        Map<String, Double> questionVector = getTfidfVector(question);

        int bestIdx = 0;
        double bestScore = -1;

        for (int i = 0; i < documentVectors.size(); i++) {
            double score = cosineSimilarity(questionVector, documentVectors.get(i));
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        return documents.get(bestIdx);
    }

    /**
     * Get top K most relevant passages.
     */
    public List<RankedPassage> getTopPassages(String question, int k) {
        Map<String, Double> questionVector = getTfidfVector(question);
        List<RankedPassage> ranked = new ArrayList<>();

        for (int i = 0; i < documents.size(); i++) {
            double score = cosineSimilarity(questionVector, documentVectors.get(i));
            ranked.add(new RankedPassage(documents.get(i), score));
        }

        ranked.sort((a, b) -> Double.compare(b.score, a.score));

        return ranked.subList(0, Math.min(k, ranked.size()));
    }

    public static class RankedPassage {
        public String passage;
        public double score;

        public RankedPassage(String passage, double score) {
            this.passage = passage;
            this.score = score;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class TfidfQATest {

    @Test
    void testIndexDocuments() {
        TfidfQA qa = new TfidfQA();
        List<String> docs = Arrays.asList(
            "Machine learning is a branch of artificial intelligence.",
            "Deep learning uses neural networks.",
            "Natural language processing deals with text."
        );
        qa.indexDocuments(docs);

        String answer = qa.answer("What is machine learning?");
        assertTrue(answer.contains("Machine learning"));
    }

    @Test
    void testCosineSimilarity() {
        TfidfQA qa = new TfidfQA();
        qa.indexDocuments(Arrays.asList("test document"));

        Map<String, Double> v1 = new HashMap<>();
        v1.put("a", 1.0);
        v1.put("b", 2.0);

        Map<String, Double> v2 = new HashMap<>();
        v2.put("a", 1.0);
        v2.put("b", 2.0);

        double sim = qa.cosineSimilarity(v1, v2);
        assertEquals(1.0, sim, 0.001);
    }

    @Test
    void testGetTfidfVector() {
        TfidfQA qa = new TfidfQA();
        qa.indexDocuments(Arrays.asList("hello world", "hello there"));

        Map<String, Double> vector = qa.getTfidfVector("hello world");
        assertTrue(vector.containsKey("hello"));
        assertTrue(vector.containsKey("world"));
    }

    @Test
    void testGetTopPassages() {
        TfidfQA qa = new TfidfQA();
        List<String> docs = Arrays.asList(
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Natural language processing"
        );
        qa.indexDocuments(docs);

        List<TfidfQA.RankedPassage> top = qa.getTopPassages("machine learning", 2);
        assertEquals(2, top.size());
    }

    @Test
    void testRankedPassageClass() {
        TfidfQA.RankedPassage rp = new TfidfQA.RankedPassage("test passage", 0.75);
        assertEquals("test passage", rp.passage);
        assertEquals(0.75, rp.score, 0.001);
    }

    @Test
    void testEmptyDocuments() {
        TfidfQA qa = new TfidfQA();
        String answer = qa.answer("some question");
        assertNull(answer);
    }

    @Test
    void testOrthogonalVectors() {
        TfidfQA qa = new TfidfQA();
        qa.indexDocuments(Arrays.asList("test"));

        Map<String, Double> v1 = new HashMap<>();
        v1.put("a", 1.0);

        Map<String, Double> v2 = new HashMap<>();
        v2.put("b", 1.0);

        double sim = qa.cosineSimilarity(v1, v2);
        assertEquals(0.0, sim, 0.001);
    }

    @Test
    void testZeroNormVector() {
        TfidfQA qa = new TfidfQA();
        qa.indexDocuments(Arrays.asList("test"));

        Map<String, Double> v1 = new HashMap<>();
        Map<String, Double> v2 = new HashMap<>();
        v2.put("a", 1.0);

        double sim = qa.cosineSimilarity(v1, v2);
        assertEquals(0.0, sim, 0.001);
    }

    @Test
    void testSingleDocument() {
        TfidfQA qa = new TfidfQA();
        qa.indexDocuments(Arrays.asList("The only document"));
        String answer = qa.answer("any question");
        assertEquals("The only document", answer);
    }

    @Test
    void testTopPassagesScore() {
        TfidfQA qa = new TfidfQA();
        qa.indexDocuments(Arrays.asList("machine learning", "deep learning"));
        List<TfidfQA.RankedPassage> top = qa.getTopPassages("machine", 2);
        assertTrue(top.get(0).score >= top.get(1).score);
    }
}`,

	hint1: 'TF-IDF = Term Frequency × Inverse Document Frequency',
	hint2: 'Cosine similarity measures angle between vectors (dot product / norms)',

	whyItMatters: `TF-IDF QA is the industry standard for document retrieval:

- **Efficient**: Sparse vectors enable fast similarity search
- **Scalable**: Works with millions of documents
- **Baseline**: Standard benchmark for neural retrievers
- **Practical**: Powers many production search systems

TF-IDF forms the foundation of modern information retrieval.`,

	translations: {
		ru: {
			title: 'TF-IDF QA',
			description: `# TF-IDF QA

Реализуйте систему ответов на вопросы с использованием TF-IDF для поиска пассажей.

## Задача

Создайте QA-систему на основе TF-IDF:
- Вычисление TF-IDF векторов для пассажей
- Поиск наиболее похожего пассажа на вопрос
- Извлечение ответов из релевантных пассажей

## Пример

\`\`\`java
TfidfQA qa = new TfidfQA();
qa.indexDocuments(passages);
String answer = qa.answer("What is machine learning?");
\`\`\``,
			hint1: 'TF-IDF = Частота терма × Обратная частота документа',
			hint2: 'Косинусное сходство измеряет угол между векторами',
			whyItMatters: `TF-IDF QA - отраслевой стандарт поиска документов:

- **Эффективность**: Разреженные векторы для быстрого поиска
- **Масштабируемость**: Работает с миллионами документов
- **Baseline**: Стандартный бенчмарк для нейронных retriever-ов
- **Практичность**: Питает многие production системы поиска`,
		},
		uz: {
			title: 'TF-IDF QA',
			description: `# TF-IDF QA

Passage qidirish uchun TF-IDF dan foydalanib savollarga javob tizimini amalga oshiring.

## Topshiriq

TF-IDF asosidagi QA tizimini yarating:
- Passagelar uchun TF-IDF vektorlarini hisoblash
- Savolga eng o'xshash passage ni topish
- Tegishli passagelardan javoblarni ajratib olish

## Misol

\`\`\`java
TfidfQA qa = new TfidfQA();
qa.indexDocuments(passages);
String answer = qa.answer("What is machine learning?");
\`\`\``,
			hint1: 'TF-IDF = Term Frequency × Inverse Document Frequency',
			hint2: "Kosinus o'xshashligi vektorlar orasidagi burchakni o'lchaydi",
			whyItMatters: `TF-IDF QA hujjat qidirish uchun sanoat standarti:

- **Samarali**: Siyrak vektorlar tez o'xshashlik qidirishni ta'minlaydi
- **Masshtablanuvchi**: Millionlab hujjatlar bilan ishlaydi
- **Baseline**: Neyron retrieverlar uchun standart benchmark
- **Amaliy**: Ko'plab ishlab chiqarish qidiruv tizimlarini quvvatlaydi`,
		},
	},
};

export default task;
