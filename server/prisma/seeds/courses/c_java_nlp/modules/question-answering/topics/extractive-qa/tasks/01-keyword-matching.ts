import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-keyword-matching',
	title: 'Keyword Matching QA',
	difficulty: 'medium',
	tags: ['nlp', 'qa', 'keyword-matching', 'information-retrieval'],
	estimatedTime: '25m',
	isPremium: true,
	order: 1,
	description: `# Keyword Matching QA

Implement a simple question answering system using keyword matching.

## Task

Build a QA system that:
- Extracts keywords from questions
- Finds sentences containing those keywords
- Ranks sentences by relevance score
- Returns the best matching answer

## Example

\`\`\`java
KeywordQA qa = new KeywordQA();
qa.addDocument("The Eiffel Tower is in Paris. It was built in 1889.");
String answer = qa.answer("When was the Eiffel Tower built?");
// Returns: "It was built in 1889."
\`\`\``,

	initialCode: `import java.util.*;

public class KeywordQA {

    /**
     * Add a document to the knowledge base.
     */
    public void addDocument(String document) {
    }

    /**
     * Answer a question based on the knowledge base.
     */
    public String answer(String question) {
        return null;
    }

    /**
     * Calculate relevance score between question and sentence.
     */
    public double calculateScore(String question, String sentence) {
        return 0.0;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class KeywordQA {

    private List<String> sentences;
    private Set<String> stopwords;

    public KeywordQA() {
        this.sentences = new ArrayList<>();
        this.stopwords = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "or", "if",
            "because", "until", "while", "what", "which", "who", "whom"
        ));
    }

    /**
     * Add a document to the knowledge base.
     */
    public void addDocument(String document) {
        if (document == null || document.isEmpty()) return;

        // Split into sentences
        String[] parts = document.split("[.!?]+");
        for (String part : parts) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty()) {
                sentences.add(trimmed);
            }
        }
    }

    /**
     * Extract keywords from text.
     */
    private Set<String> extractKeywords(String text) {
        Set<String> keywords = new HashSet<>();
        String[] words = text.toLowerCase().split("\\\\W+");

        for (String word : words) {
            if (word.length() > 2 && !stopwords.contains(word)) {
                keywords.add(word);
            }
        }

        return keywords;
    }

    /**
     * Answer a question based on the knowledge base.
     */
    public String answer(String question) {
        if (question == null || sentences.isEmpty()) {
            return null;
        }

        Set<String> questionKeywords = extractKeywords(question);

        String bestAnswer = null;
        double bestScore = 0;

        for (String sentence : sentences) {
            double score = calculateScore(question, sentence);
            if (score > bestScore) {
                bestScore = score;
                bestAnswer = sentence;
            }
        }

        return bestAnswer;
    }

    /**
     * Calculate relevance score between question and sentence.
     */
    public double calculateScore(String question, String sentence) {
        Set<String> questionKeywords = extractKeywords(question);
        Set<String> sentenceKeywords = extractKeywords(sentence);

        if (questionKeywords.isEmpty()) return 0;

        // Count keyword overlap
        int overlap = 0;
        for (String keyword : questionKeywords) {
            if (sentenceKeywords.contains(keyword)) {
                overlap++;
            }
        }

        // Jaccard-like similarity
        double score = (double) overlap / questionKeywords.size();

        // Boost for question word patterns
        String lowerQuestion = question.toLowerCase();
        String lowerSentence = sentence.toLowerCase();

        if (lowerQuestion.contains("when") &&
            (lowerSentence.contains("in ") || lowerSentence.matches(".*\\\\d{4}.*"))) {
            score *= 1.5;
        }
        if (lowerQuestion.contains("where") &&
            (lowerSentence.contains("in ") || lowerSentence.contains("at "))) {
            score *= 1.5;
        }
        if (lowerQuestion.contains("who") && sentenceKeywords.stream()
                .anyMatch(w -> Character.isUpperCase(w.charAt(0)))) {
            score *= 1.3;
        }

        return score;
    }

    /**
     * Get top N answers with scores.
     */
    public List<ScoredAnswer> getTopAnswers(String question, int n) {
        List<ScoredAnswer> scored = new ArrayList<>();

        for (String sentence : sentences) {
            double score = calculateScore(question, sentence);
            if (score > 0) {
                scored.add(new ScoredAnswer(sentence, score));
            }
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));

        return scored.subList(0, Math.min(n, scored.size()));
    }

    public static class ScoredAnswer {
        public String answer;
        public double score;

        public ScoredAnswer(String answer, double score) {
            this.answer = answer;
            this.score = score;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class KeywordQATest {

    @Test
    void testAddDocument() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("The Eiffel Tower is in Paris. It was built in 1889.");

        String answer = qa.answer("Eiffel Tower");
        assertNotNull(answer);
    }

    @Test
    void testAnswerWhen() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("The Eiffel Tower is in Paris. It was built in 1889.");

        String answer = qa.answer("When was the Eiffel Tower built?");
        assertTrue(answer.contains("1889"));
    }

    @Test
    void testAnswerWhere() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("The Eiffel Tower is in Paris. It was built in 1889.");

        String answer = qa.answer("Where is the Eiffel Tower?");
        assertTrue(answer.contains("Paris"));
    }

    @Test
    void testCalculateScore() {
        KeywordQA qa = new KeywordQA();
        double score = qa.calculateScore("Eiffel Tower Paris", "The Eiffel Tower is in Paris");

        assertTrue(score > 0);
    }

    @Test
    void testEmptyKnowledgeBase() {
        KeywordQA qa = new KeywordQA();
        String answer = qa.answer("What is the capital of France?");
        assertNull(answer);
    }

    @Test
    void testGetTopAnswers() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("Paris is the capital of France. London is the capital of England.");
        List<KeywordQA.ScoredAnswer> answers = qa.getTopAnswers("What is the capital of France?", 2);
        assertFalse(answers.isEmpty());
    }

    @Test
    void testScoredAnswerClass() {
        KeywordQA.ScoredAnswer sa = new KeywordQA.ScoredAnswer("Test answer", 0.85);
        assertEquals("Test answer", sa.answer);
        assertEquals(0.85, sa.score, 0.001);
    }

    @Test
    void testNullQuestion() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("Some content here.");
        String answer = qa.answer(null);
        assertNull(answer);
    }

    @Test
    void testAddEmptyDocument() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("");
        String answer = qa.answer("test");
        assertNull(answer);
    }

    @Test
    void testMultipleDocuments() {
        KeywordQA qa = new KeywordQA();
        qa.addDocument("The sky is blue.");
        qa.addDocument("Grass is green.");
        String answer = qa.answer("What color is the sky?");
        assertTrue(answer.contains("blue"));
    }
}`,

	hint1: 'Extract keywords by removing stopwords and punctuation',
	hint2: 'Boost scores for sentences containing date patterns when answering "when" questions',

	whyItMatters: `Keyword matching QA is foundational for search:

- **Simple baseline**: No ML training required
- **Interpretable**: Easy to understand why an answer was selected
- **Fast**: No model inference needed
- **Building block**: Forms basis for more advanced QA systems

Understanding keyword matching helps you appreciate neural QA improvements.`,

	translations: {
		ru: {
			title: 'QA на ключевых словах',
			description: `# QA на ключевых словах

Реализуйте простую систему ответов на вопросы с использованием сопоставления ключевых слов.

## Задача

Создайте QA-систему, которая:
- Извлекает ключевые слова из вопросов
- Находит предложения с этими ключевыми словами
- Ранжирует предложения по релевантности
- Возвращает лучший ответ

## Пример

\`\`\`java
KeywordQA qa = new KeywordQA();
qa.addDocument("The Eiffel Tower is in Paris. It was built in 1889.");
String answer = qa.answer("When was the Eiffel Tower built?");
// Returns: "It was built in 1889."
\`\`\``,
			hint1: 'Извлекайте ключевые слова, удаляя стоп-слова и пунктуацию',
			hint2: 'Повышайте оценку для предложений с датами при ответе на вопросы "когда"',
			whyItMatters: `QA на ключевых словах - основа поиска:

- **Простой baseline**: Не требуется обучение ML
- **Интерпретируемость**: Легко понять, почему выбран ответ
- **Быстрота**: Не нужен инференс модели
- **Строительный блок**: Основа для продвинутых QA-систем`,
		},
		uz: {
			title: "Kalit so'zlar bo'yicha QA",
			description: `# Kalit so'zlar bo'yicha QA

Kalit so'zlar mosligidan foydalanib oddiy savollarga javob tizimini amalga oshiring.

## Topshiriq

QA tizimini yarating:
- Savollardan kalit so'zlarni ajratib olish
- O'sha kalit so'zlarni o'z ichiga olgan gaplarni topish
- Gaplarni relevantlik bo'yicha tartiblash
- Eng yaxshi javobni qaytarish

## Misol

\`\`\`java
KeywordQA qa = new KeywordQA();
qa.addDocument("The Eiffel Tower is in Paris. It was built in 1889.");
String answer = qa.answer("When was the Eiffel Tower built?");
// Returns: "It was built in 1889."
\`\`\``,
			hint1: "Stop-so'zlar va tinish belgilarini olib tashlab kalit so'zlarni ajratib oling",
			hint2: '"Qachon" savollariga javob berayotganda sana patternlari bor gaplar uchun ballni oshiring',
			whyItMatters: `Kalit so'zlar bo'yicha QA qidiruvning asosi:

- **Oddiy baseline**: ML o'qitish kerak emas
- **Tushunarli**: Nima uchun javob tanlangangini tushunish oson
- **Tez**: Model inference kerak emas
- **Qurilish bloki**: Ilg'or QA tizimlari uchun asos`,
		},
	},
};

export default task;
