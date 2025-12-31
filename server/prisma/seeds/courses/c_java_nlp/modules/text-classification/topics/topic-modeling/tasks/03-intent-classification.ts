import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-intent-classification',
	title: 'Intent Classification',
	difficulty: 'medium',
	tags: ['nlp', 'classification', 'intent', 'chatbot'],
	estimatedTime: '25m',
	isPremium: false,
	order: 3,
	description: `# Intent Classification

Build an intent classifier for chatbot applications.

## Task

Implement intent classification that:
- Classifies user utterances into intents
- Uses keyword patterns and TF-IDF
- Handles multiple intent patterns
- Returns intent with confidence score

## Example

\`\`\`java
IntentClassifier classifier = new IntentClassifier();
classifier.addIntent("greeting", Arrays.asList("hello", "hi", "hey"));
Intent result = classifier.classify("Hello there!");
// Intent: greeting, confidence: 0.95
\`\`\``,

	initialCode: `import java.util.*;

public class IntentClassifier {

    /**
     * Add intent with example phrases.
     */
    public void addIntent(String intent, List<String> examples) {
    }

    /**
     * Classify user utterance.
     */
    public Intent classify(String utterance) {
        return null;
    }

    /**
     * Get top N intent candidates.
     */
    public List<Intent> getTopIntents(String utterance, int n) {
        return null;
    }

    public static class Intent {
        public String name;
        public double confidence;

        public Intent(String name, double confidence) {
            this.name = name;
            this.confidence = confidence;
        }
    }
}`,

	solutionCode: `import java.util.*;

public class IntentClassifier {

    private Map<String, List<String>> intentExamples;
    private Map<String, Set<String>> intentKeywords;
    private Map<String, Map<String, Double>> intentTfidf;
    private Map<String, Double> idfScores;
    private Set<String> vocabulary;

    public IntentClassifier() {
        this.intentExamples = new HashMap<>();
        this.intentKeywords = new HashMap<>();
        this.intentTfidf = new HashMap<>();
        this.idfScores = new HashMap<>();
        this.vocabulary = new HashSet<>();
        initializeDefaultIntents();
    }

    private void initializeDefaultIntents() {
        addIntent("greeting", Arrays.asList(
            "hello", "hi", "hey", "good morning", "good evening",
            "howdy", "greetings", "what's up"
        ));
        addIntent("goodbye", Arrays.asList(
            "bye", "goodbye", "see you", "farewell", "later",
            "take care", "good night"
        ));
        addIntent("help", Arrays.asList(
            "help", "help me", "need help", "assist", "support",
            "how do i", "can you help", "what should i do"
        ));
        addIntent("order_status", Arrays.asList(
            "where is my order", "order status", "track order",
            "delivery status", "when will it arrive", "shipping"
        ));
        addIntent("cancel", Arrays.asList(
            "cancel", "cancel order", "stop", "nevermind",
            "forget it", "don't want"
        ));
        addIntent("thanks", Arrays.asList(
            "thank you", "thanks", "appreciate", "grateful",
            "that's helpful", "perfect"
        ));
    }

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
     * Add intent with example phrases.
     */
    public void addIntent(String intent, List<String> examples) {
        intentExamples.put(intent, new ArrayList<>(examples));

        // Extract keywords
        Set<String> keywords = new HashSet<>();
        for (String example : examples) {
            keywords.addAll(tokenize(example));
        }
        intentKeywords.put(intent, keywords);

        // Rebuild TF-IDF
        rebuildTfidf();
    }

    private void rebuildTfidf() {
        // Calculate document frequencies
        Map<String, Integer> docFreq = new HashMap<>();
        for (String intent : intentExamples.keySet()) {
            Set<String> seen = new HashSet<>();
            for (String example : intentExamples.get(intent)) {
                for (String token : tokenize(example)) {
                    if (!seen.contains(token)) {
                        docFreq.merge(token, 1, Integer::sum);
                        seen.add(token);
                    }
                }
            }
        }

        // Calculate IDF
        int N = intentExamples.size();
        for (String word : vocabulary) {
            double idf = Math.log((double) N / (1 + docFreq.getOrDefault(word, 0))) + 1;
            idfScores.put(word, idf);
        }

        // Calculate TF-IDF for each intent
        for (String intent : intentExamples.keySet()) {
            Map<String, Double> tfidf = new HashMap<>();
            Map<String, Integer> tf = new HashMap<>();
            int total = 0;

            for (String example : intentExamples.get(intent)) {
                for (String token : tokenize(example)) {
                    tf.merge(token, 1, Integer::sum);
                    total++;
                }
            }

            for (Map.Entry<String, Integer> entry : tf.entrySet()) {
                double tfVal = (double) entry.getValue() / total;
                double idf = idfScores.getOrDefault(entry.getKey(), 1.0);
                tfidf.put(entry.getKey(), tfVal * idf);
            }

            intentTfidf.put(intent, tfidf);
        }
    }

    /**
     * Classify user utterance.
     */
    public Intent classify(String utterance) {
        List<Intent> intents = getTopIntents(utterance, 1);
        return intents.isEmpty() ? new Intent("unknown", 0.0) : intents.get(0);
    }

    /**
     * Get top N intent candidates.
     */
    public List<Intent> getTopIntents(String utterance, int n) {
        List<String> tokens = tokenize(utterance);
        List<Intent> results = new ArrayList<>();

        for (String intent : intentExamples.keySet()) {
            double score = calculateScore(tokens, intent);
            if (score > 0) {
                results.add(new Intent(intent, score));
            }
        }

        // Normalize scores
        if (!results.isEmpty()) {
            double maxScore = results.stream().mapToDouble(i -> i.confidence).max().orElse(1.0);
            for (Intent intent : results) {
                intent.confidence = intent.confidence / maxScore;
            }
        }

        results.sort((a, b) -> Double.compare(b.confidence, a.confidence));
        return results.subList(0, Math.min(n, results.size()));
    }

    private double calculateScore(List<String> tokens, String intent) {
        double score = 0;

        // Keyword matching
        Set<String> keywords = intentKeywords.get(intent);
        int keywordMatches = 0;
        for (String token : tokens) {
            if (keywords.contains(token)) {
                keywordMatches++;
            }
        }
        score += keywordMatches * 0.5;

        // TF-IDF similarity
        Map<String, Double> intentVector = intentTfidf.get(intent);
        if (intentVector != null) {
            for (String token : tokens) {
                if (intentVector.containsKey(token)) {
                    score += intentVector.get(token);
                }
            }
        }

        // Exact phrase matching bonus
        String utterance = String.join(" ", tokens);
        for (String example : intentExamples.get(intent)) {
            if (utterance.contains(example.toLowerCase()) ||
                example.toLowerCase().contains(utterance)) {
                score += 2.0;
                break;
            }
        }

        return score;
    }

    public static class Intent {
        public String name;
        public double confidence;

        public Intent(String name, double confidence) {
            this.name = name;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return name + " (" + String.format("%.2f", confidence) + ")";
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class IntentClassifierTest {

    @Test
    void testClassifyGreeting() {
        IntentClassifier classifier = new IntentClassifier();
        IntentClassifier.Intent result = classifier.classify("Hello there!");

        assertEquals("greeting", result.name);
        assertTrue(result.confidence > 0.5);
    }

    @Test
    void testClassifyGoodbye() {
        IntentClassifier classifier = new IntentClassifier();
        IntentClassifier.Intent result = classifier.classify("Goodbye, see you later");

        assertEquals("goodbye", result.name);
    }

    @Test
    void testGetTopIntents() {
        IntentClassifier classifier = new IntentClassifier();
        List<IntentClassifier.Intent> intents = classifier.getTopIntents("hi, can you help me?", 3);

        assertTrue(intents.size() >= 2);
    }

    @Test
    void testAddIntent() {
        IntentClassifier classifier = new IntentClassifier();
        classifier.addIntent("weather", Arrays.asList("weather", "forecast", "temperature", "rain"));

        IntentClassifier.Intent result = classifier.classify("What's the weather today?");
        assertEquals("weather", result.name);
    }

    @Test
    void testUnknownIntent() {
        IntentClassifier classifier = new IntentClassifier();
        IntentClassifier.Intent result = classifier.classify("xyzabc123");

        assertEquals("unknown", result.name);
    }

    @Test
    void testClassifyHelp() {
        IntentClassifier classifier = new IntentClassifier();
        IntentClassifier.Intent result = classifier.classify("I need help please");
        assertEquals("help", result.name);
    }

    @Test
    void testClassifyThanks() {
        IntentClassifier classifier = new IntentClassifier();
        IntentClassifier.Intent result = classifier.classify("Thank you so much!");
        assertEquals("thanks", result.name);
    }

    @Test
    void testIntentToString() {
        IntentClassifier.Intent intent = new IntentClassifier.Intent("test", 0.95);
        String str = intent.toString();
        assertTrue(str.contains("test"));
        assertTrue(str.contains("0.95"));
    }

    @Test
    void testConfidenceRange() {
        IntentClassifier classifier = new IntentClassifier();
        IntentClassifier.Intent result = classifier.classify("hello");
        assertTrue(result.confidence >= 0 && result.confidence <= 1);
    }

    @Test
    void testGetTopIntentsEmpty() {
        IntentClassifier classifier = new IntentClassifier();
        List<IntentClassifier.Intent> intents = classifier.getTopIntents("xyzabc123", 5);
        assertTrue(intents.isEmpty());
    }
}`,

	hint1: 'Combine keyword matching and TF-IDF similarity for scoring',
	hint2: 'Add bonus score for exact phrase matches from training examples',

	whyItMatters: `Intent classification is core to conversational AI:

- **Chatbots**: Route user requests to correct handlers
- **Virtual assistants**: Understand what users want to do
- **Customer service**: Automate support ticket routing
- **Voice interfaces**: Handle spoken commands

Intent classification is the first step in dialogue systems.`,

	translations: {
		ru: {
			title: 'Классификация интентов',
			description: `# Классификация интентов

Создайте классификатор интентов для чат-бот приложений.

## Задача

Реализуйте классификацию интентов:
- Классификация пользовательских фраз по интентам
- Использование паттернов ключевых слов и TF-IDF
- Обработка множественных паттернов интентов
- Возврат интента с оценкой уверенности

## Пример

\`\`\`java
IntentClassifier classifier = new IntentClassifier();
classifier.addIntent("greeting", Arrays.asList("hello", "hi", "hey"));
Intent result = classifier.classify("Hello there!");
// Intent: greeting, confidence: 0.95
\`\`\``,
			hint1: 'Комбинируйте сопоставление ключевых слов и TF-IDF сходство для оценки',
			hint2: 'Добавляйте бонусный балл за точное совпадение фраз из примеров',
			whyItMatters: `Классификация интентов - ядро разговорного ИИ:

- **Чатботы**: Направление запросов пользователей к нужным обработчикам
- **Виртуальные ассистенты**: Понимание что хочет сделать пользователь
- **Служба поддержки**: Автоматизация маршрутизации тикетов
- **Голосовые интерфейсы**: Обработка голосовых команд`,
		},
		uz: {
			title: 'Intent klassifikatsiyasi',
			description: `# Intent klassifikatsiyasi

Chatbot ilovalari uchun intent klassifikatorini yarating.

## Topshiriq

Intent klassifikatsiyasini amalga oshiring:
- Foydalanuvchi iboralarini intentlarga klassifikatsiya qilish
- Kalit so'z patternlari va TF-IDF dan foydalanish
- Bir nechta intent patternlarini qayta ishlash
- Ishonch balli bilan intentni qaytarish

## Misol

\`\`\`java
IntentClassifier classifier = new IntentClassifier();
classifier.addIntent("greeting", Arrays.asList("hello", "hi", "hey"));
Intent result = classifier.classify("Hello there!");
// Intent: greeting, confidence: 0.95
\`\`\``,
			hint1: "Baholash uchun kalit so'z mosligi va TF-IDF o'xshashligini birlashtiring",
			hint2: "O'qitish misollaridan aniq ibora mosliklari uchun bonus ball qo'shing",
			whyItMatters: `Intent klassifikatsiyasi suhbat AI ning yadrosi:

- **Chatbotlar**: Foydalanuvchi so'rovlarini to'g'ri handlerlarga yo'naltirish
- **Virtual yordamchilar**: Foydalanuvchilar nima qilmoqchi ekanini tushunish
- **Mijozlarga xizmat ko'rsatish**: Yordam tiketlarini avtomatik yo'naltirish
- **Ovozli interfeyslar**: Gaplashuv buyruqlarini qayta ishlash`,
		},
	},
};

export default task;
