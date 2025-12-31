import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-span-extraction',
	title: 'Answer Span Extraction',
	difficulty: 'hard',
	tags: ['nlp', 'qa', 'span-extraction', 'bert'],
	estimatedTime: '35m',
	isPremium: true,
	order: 3,
	description: `# Answer Span Extraction

Implement answer span extraction from a context passage.

## Task

Build a span extractor that:
- Identifies potential answer spans in context
- Scores spans based on question overlap
- Handles question type patterns (who, what, when, where)
- Returns the most likely answer span

## Example

\`\`\`java
SpanExtractor extractor = new SpanExtractor();
String context = "Albert Einstein was born in 1879 in Germany.";
String question = "When was Einstein born?";
String answer = extractor.extractAnswer(context, question); // "1879"
\`\`\``,

	initialCode: `import java.util.*;

public class SpanExtractor {

    /**
     * Extract answer span from context given a question.
     */
    public String extractAnswer(String context, String question) {
        return null;
    }

    /**
     * Get all candidate answer spans.
     */
    public List<Span> getCandidateSpans(String context) {
        return null;
    }

    /**
     * Score a candidate span for a given question.
     */
    public double scoreSpan(Span span, String question, String context) {
        return 0.0;
    }

    public static class Span {
        public String text;
        public int start;
        public int end;
        public String type;

        public Span(String text, int start, int end, String type) {
            this.text = text;
            this.start = start;
            this.end = end;
            this.type = type;
        }
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class SpanExtractor {

    private static final Pattern DATE_PATTERN = Pattern.compile("\\\\b(\\\\d{4}|\\\\d{1,2}/\\\\d{1,2}/\\\\d{2,4}|\\\\w+ \\\\d{1,2},? \\\\d{4})\\\\b");
    private static final Pattern NUMBER_PATTERN = Pattern.compile("\\\\b\\\\d+(?:\\\\.\\\\d+)?(?:\\\\s*(?:million|billion|thousand|hundred|percent|%|km|miles|meters))?\\\\b");
    private static final Pattern PROPER_NOUN_PATTERN = Pattern.compile("\\\\b[A-Z][a-z]+(?:\\\\s+[A-Z][a-z]+)*\\\\b");
    private static final Pattern LOCATION_PATTERN = Pattern.compile("\\\\b(?:in|at|from|to)\\\\s+([A-Z][a-z]+(?:\\\\s+[A-Z][a-z]+)*)\\\\b");

    private Set<String> stopwords;

    public SpanExtractor() {
        this.stopwords = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "and", "or", "but"
        ));
    }

    /**
     * Detect question type.
     */
    private String detectQuestionType(String question) {
        String lower = question.toLowerCase();
        if (lower.startsWith("when") || lower.contains("what year") || lower.contains("what date")) {
            return "DATE";
        }
        if (lower.startsWith("where") || lower.contains("what place") || lower.contains("what country")) {
            return "LOCATION";
        }
        if (lower.startsWith("who") || lower.contains("what person")) {
            return "PERSON";
        }
        if (lower.startsWith("how many") || lower.startsWith("how much") || lower.contains("what number")) {
            return "NUMBER";
        }
        return "GENERAL";
    }

    /**
     * Get all candidate answer spans.
     */
    public List<Span> getCandidateSpans(String context) {
        List<Span> spans = new ArrayList<>();

        // Extract dates
        Matcher dateMatcher = DATE_PATTERN.matcher(context);
        while (dateMatcher.find()) {
            spans.add(new Span(dateMatcher.group(), dateMatcher.start(), dateMatcher.end(), "DATE"));
        }

        // Extract numbers
        Matcher numMatcher = NUMBER_PATTERN.matcher(context);
        while (numMatcher.find()) {
            spans.add(new Span(numMatcher.group(), numMatcher.start(), numMatcher.end(), "NUMBER"));
        }

        // Extract proper nouns (potential persons/locations)
        Matcher propMatcher = PROPER_NOUN_PATTERN.matcher(context);
        while (propMatcher.find()) {
            spans.add(new Span(propMatcher.group(), propMatcher.start(), propMatcher.end(), "ENTITY"));
        }

        // Extract locations after prepositions
        Matcher locMatcher = LOCATION_PATTERN.matcher(context);
        while (locMatcher.find()) {
            spans.add(new Span(locMatcher.group(1), locMatcher.start(1), locMatcher.end(1), "LOCATION"));
        }

        // Extract noun phrases (simplified)
        String[] sentences = context.split("[.!?]");
        for (String sent : sentences) {
            String[] words = sent.trim().split("\\\\s+");
            for (int i = 0; i < words.length; i++) {
                // Single important words
                String word = words[i].replaceAll("[^a-zA-Z0-9]", "");
                if (word.length() > 2 && !stopwords.contains(word.toLowerCase())) {
                    int start = context.indexOf(word);
                    if (start >= 0) {
                        spans.add(new Span(word, start, start + word.length(), "WORD"));
                    }
                }
            }
        }

        return spans;
    }

    /**
     * Extract keywords from question.
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
     * Score a candidate span for a given question.
     */
    public double scoreSpan(Span span, String question, String context) {
        double score = 0.0;
        String questionType = detectQuestionType(question);

        // Type match bonus
        if (questionType.equals(span.type)) {
            score += 2.0;
        } else if (questionType.equals("PERSON") && span.type.equals("ENTITY")) {
            score += 1.5;
        } else if (questionType.equals("LOCATION") && span.type.equals("ENTITY")) {
            score += 1.5;
        }

        // Keyword proximity
        Set<String> questionKeywords = extractKeywords(question);
        String[] contextWords = context.toLowerCase().split("\\\\W+");

        int spanPos = span.start;
        int minDist = Integer.MAX_VALUE;

        for (String keyword : questionKeywords) {
            int keywordPos = context.toLowerCase().indexOf(keyword);
            if (keywordPos >= 0) {
                int dist = Math.abs(spanPos - keywordPos);
                minDist = Math.min(minDist, dist);
            }
        }

        if (minDist < Integer.MAX_VALUE) {
            score += 1.0 / (1 + minDist / 50.0);
        }

        // Context overlap
        String spanContext = getSpanContext(context, span, 30);
        for (String keyword : questionKeywords) {
            if (spanContext.toLowerCase().contains(keyword)) {
                score += 0.3;
            }
        }

        return score;
    }

    private String getSpanContext(String context, Span span, int window) {
        int start = Math.max(0, span.start - window);
        int end = Math.min(context.length(), span.end + window);
        return context.substring(start, end);
    }

    /**
     * Extract answer span from context given a question.
     */
    public String extractAnswer(String context, String question) {
        List<Span> candidates = getCandidateSpans(context);

        if (candidates.isEmpty()) {
            return null;
        }

        Span bestSpan = null;
        double bestScore = -1;

        for (Span span : candidates) {
            double score = scoreSpan(span, question, context);
            if (score > bestScore) {
                bestScore = score;
                bestSpan = span;
            }
        }

        return bestSpan != null ? bestSpan.text : null;
    }

    /**
     * Get top N answer candidates with scores.
     */
    public List<ScoredSpan> getTopAnswers(String context, String question, int n) {
        List<Span> candidates = getCandidateSpans(context);
        List<ScoredSpan> scored = new ArrayList<>();

        for (Span span : candidates) {
            double score = scoreSpan(span, question, context);
            scored.add(new ScoredSpan(span, score));
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));

        return scored.subList(0, Math.min(n, scored.size()));
    }

    public static class Span {
        public String text;
        public int start;
        public int end;
        public String type;

        public Span(String text, int start, int end, String type) {
            this.text = text;
            this.start = start;
            this.end = end;
            this.type = type;
        }
    }

    public static class ScoredSpan {
        public Span span;
        public double score;

        public ScoredSpan(Span span, double score) {
            this.span = span;
            this.score = score;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SpanExtractorTest {

    @Test
    void testExtractAnswerWhen() {
        SpanExtractor extractor = new SpanExtractor();
        String context = "Albert Einstein was born in 1879 in Germany.";
        String question = "When was Einstein born?";

        String answer = extractor.extractAnswer(context, question);
        assertEquals("1879", answer);
    }

    @Test
    void testExtractAnswerWhere() {
        SpanExtractor extractor = new SpanExtractor();
        String context = "Albert Einstein was born in 1879 in Germany.";
        String question = "Where was Einstein born?";

        String answer = extractor.extractAnswer(context, question);
        assertEquals("Germany", answer);
    }

    @Test
    void testGetCandidateSpans() {
        SpanExtractor extractor = new SpanExtractor();
        String context = "Einstein was born in 1879 in Germany.";

        List<SpanExtractor.Span> spans = extractor.getCandidateSpans(context);
        assertTrue(spans.size() >= 2);  // At least date and location
    }

    @Test
    void testScoreSpan() {
        SpanExtractor extractor = new SpanExtractor();
        SpanExtractor.Span dateSpan = new SpanExtractor.Span("1879", 0, 4, "DATE");

        double score = extractor.scoreSpan(dateSpan, "When was Einstein born?", "Einstein born in 1879");
        assertTrue(score > 0);
    }

    @Test
    void testGetTopAnswers() {
        SpanExtractor extractor = new SpanExtractor();
        String context = "Einstein was born in 1879 in Germany.";

        List<SpanExtractor.ScoredSpan> top = extractor.getTopAnswers(context, "When was Einstein born?", 3);
        assertTrue(top.size() >= 1);
    }

    @Test
    void testSpanClass() {
        SpanExtractor.Span span = new SpanExtractor.Span("test", 0, 4, "DATE");
        assertEquals("test", span.text);
        assertEquals(0, span.start);
        assertEquals(4, span.end);
        assertEquals("DATE", span.type);
    }

    @Test
    void testScoredSpanClass() {
        SpanExtractor.Span span = new SpanExtractor.Span("test", 0, 4, "DATE");
        SpanExtractor.ScoredSpan ss = new SpanExtractor.ScoredSpan(span, 0.85);
        assertEquals(span, ss.span);
        assertEquals(0.85, ss.score, 0.001);
    }

    @Test
    void testEmptyContext() {
        SpanExtractor extractor = new SpanExtractor();
        List<SpanExtractor.Span> spans = extractor.getCandidateSpans("");
        assertTrue(spans.isEmpty());
    }

    @Test
    void testExtractNumbers() {
        SpanExtractor extractor = new SpanExtractor();
        String context = "The population is 5 million people.";
        List<SpanExtractor.Span> spans = extractor.getCandidateSpans(context);
        assertTrue(spans.stream().anyMatch(s -> s.type.equals("NUMBER")));
    }

    @Test
    void testHowManyQuestion() {
        SpanExtractor extractor = new SpanExtractor();
        String context = "The team has 25 players.";
        String question = "How many players does the team have?";
        String answer = extractor.extractAnswer(context, question);
        assertNotNull(answer);
    }
}`,

	hint1: 'Use regex patterns to identify candidate spans (dates, numbers, proper nouns)',
	hint2: 'Score spans higher if their type matches the question type (when=DATE, who=PERSON)',

	whyItMatters: `Span extraction is core to modern QA:

- **BERT-style QA**: Neural models predict start/end token positions
- **Reading comprehension**: SQuAD benchmark uses span extraction
- **Factoid QA**: Precise answers for factual questions
- **Information extraction**: Finding specific information in documents

Understanding span extraction prepares you for transformer-based QA.`,

	translations: {
		ru: {
			title: 'Извлечение спанов ответов',
			description: `# Извлечение спанов ответов

Реализуйте извлечение спанов ответов из контекстного пассажа.

## Задача

Создайте экстрактор спанов:
- Определение потенциальных спанов ответов
- Оценка спанов по пересечению с вопросом
- Обработка паттернов типов вопросов
- Возврат наиболее вероятного спана

## Пример

\`\`\`java
SpanExtractor extractor = new SpanExtractor();
String context = "Albert Einstein was born in 1879 in Germany.";
String question = "When was Einstein born?";
String answer = extractor.extractAnswer(context, question); // "1879"
\`\`\``,
			hint1: 'Используйте regex для определения кандидатов (даты, числа, имена собственные)',
			hint2: 'Повышайте оценку если тип спана совпадает с типом вопроса (когда=DATE, кто=PERSON)',
			whyItMatters: `Извлечение спанов - ядро современных QA:

- **BERT-style QA**: Нейронные модели предсказывают позиции начала/конца
- **Reading comprehension**: Бенчмарк SQuAD использует извлечение спанов
- **Factoid QA**: Точные ответы на фактические вопросы
- **Извлечение информации**: Поиск конкретной информации в документах`,
		},
		uz: {
			title: 'Javob spanlarini ajratib olish',
			description: `# Javob spanlarini ajratib olish

Kontekst passagedan javob spanlarini ajratib olishni amalga oshiring.

## Topshiriq

Span ekstraktorini yarating:
- Potensial javob spanlarini aniqlash
- Spanlarni savol bilan o'xshashlik bo'yicha baholash
- Savol turi patternlarini qayta ishlash
- Eng ehtimoliy span ni qaytarish

## Misol

\`\`\`java
SpanExtractor extractor = new SpanExtractor();
String context = "Albert Einstein was born in 1879 in Germany.";
String question = "When was Einstein born?";
String answer = extractor.extractAnswer(context, question); // "1879"
\`\`\``,
			hint1: "Kandidatlarni aniqlash uchun regex patternlaridan foydalaning (sanalar, raqamlar, xos otlar)",
			hint2: "Agar span turi savol turiga mos kelsa ballni oshiring (qachon=DATE, kim=PERSON)",
			whyItMatters: `Span ajratib olish zamonaviy QA ning yadrosi:

- **BERT-style QA**: Neyron modellar boshlash/tugash token pozitsiyalarini bashorat qiladi
- **Reading comprehension**: SQuAD benchmark span ajratib olishdan foydalanadi
- **Factoid QA**: Haqiqiy savollar uchun aniq javoblar
- **Ma'lumot ajratib olish**: Hujjatlarda aniq ma'lumotlarni topish`,
		},
	},
};

export default task;
