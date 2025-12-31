import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-chunking',
	title: 'Text Chunking',
	difficulty: 'hard',
	tags: ['nlp', 'chunking', 'phrases', 'parsing'],
	estimatedTime: '30m',
	isPremium: true,
	order: 4,
	description: `# Text Chunking

Implement shallow parsing to extract noun phrases and verb phrases.

## Task

Build a chunker that:
- Identifies noun phrases (NP)
- Identifies verb phrases (VP)
- Uses IOB tagging scheme
- Extracts phrase spans from text

## Example

\`\`\`java
TextChunker chunker = new TextChunker();
List<Chunk> chunks = chunker.chunk("The quick brown fox jumps over the lazy dog");
// [NP: The quick brown fox, VP: jumps, PP: over the lazy dog]
\`\`\``,

	initialCode: `import java.util.*;

public class TextChunker {

    /**
     * Chunk text into phrases.
     */
    public List<Chunk> chunk(String text) {
        return null;
    }

    /**
     * Extract only noun phrases.
     */
    public List<String> extractNounPhrases(String text) {
        return null;
    }

    /**
     * Extract only verb phrases.
     */
    public List<String> extractVerbPhrases(String text) {
        return null;
    }

    public static class Chunk {
        public String type;  // NP, VP, PP
        public String text;
        public int start;
        public int end;

        public Chunk(String type, String text, int start, int end) {
            this.type = type;
            this.text = text;
            this.start = start;
            this.end = end;
        }
    }
}`,

	solutionCode: `import java.util.*;

public class TextChunker {

    private Map<String, String> tagToChunk;
    private Set<String> npStartTags;
    private Set<String> npContinueTags;
    private Set<String> vpStartTags;
    private Set<String> vpContinueTags;

    public TextChunker() {
        initializePatterns();
    }

    private void initializePatterns() {
        // Tags that can start a noun phrase
        npStartTags = new HashSet<>(Arrays.asList("DT", "JJ", "NN", "NNS", "NNP", "NNPS", "PRP"));

        // Tags that can continue a noun phrase
        npContinueTags = new HashSet<>(Arrays.asList("JJ", "NN", "NNS", "NNP", "NNPS"));

        // Tags that can start a verb phrase
        vpStartTags = new HashSet<>(Arrays.asList("VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"));

        // Tags that can continue a verb phrase
        vpContinueTags = new HashSet<>(Arrays.asList("VB", "VBD", "VBG", "VBN", "RB"));
    }

    /**
     * Simple POS tagger for chunking.
     */
    private List<TaggedWord> posTag(String text) {
        List<TaggedWord> result = new ArrayList<>();
        Map<String, String> wordTags = new HashMap<>();

        // Simple word-to-tag mapping
        wordTags.put("the", "DT");
        wordTags.put("a", "DT");
        wordTags.put("an", "DT");
        wordTags.put("quick", "JJ");
        wordTags.put("brown", "JJ");
        wordTags.put("lazy", "JJ");
        wordTags.put("big", "JJ");
        wordTags.put("small", "JJ");
        wordTags.put("fox", "NN");
        wordTags.put("dog", "NN");
        wordTags.put("cat", "NN");
        wordTags.put("jumps", "VBZ");
        wordTags.put("runs", "VBZ");
        wordTags.put("sat", "VBD");
        wordTags.put("over", "IN");
        wordTags.put("under", "IN");
        wordTags.put("in", "IN");
        wordTags.put("on", "IN");

        String[] words = text.split("\\\\s+");
        int pos = 0;
        for (String word : words) {
            String clean = word.replaceAll("[^a-zA-Z]", "").toLowerCase();
            String tag = wordTags.getOrDefault(clean, guessTag(clean));
            result.add(new TaggedWord(word, tag, pos, pos + word.length()));
            pos += word.length() + 1;
        }

        return result;
    }

    private String guessTag(String word) {
        if (word.endsWith("ing")) return "VBG";
        if (word.endsWith("ed")) return "VBD";
        if (word.endsWith("ly")) return "RB";
        if (word.endsWith("s") && word.length() > 3) return "NNS";
        return "NN";
    }

    /**
     * Chunk text into phrases.
     */
    public List<Chunk> chunk(String text) {
        List<Chunk> chunks = new ArrayList<>();
        List<TaggedWord> tagged = posTag(text);

        StringBuilder currentPhrase = new StringBuilder();
        String currentType = null;
        int startPos = 0;

        for (int i = 0; i < tagged.size(); i++) {
            TaggedWord tw = tagged.get(i);
            String tag = tw.tag;

            // Determine chunk type for current word
            String wordChunkType = null;
            if (npStartTags.contains(tag) || npContinueTags.contains(tag)) {
                wordChunkType = "NP";
            } else if (vpStartTags.contains(tag) || vpContinueTags.contains(tag)) {
                wordChunkType = "VP";
            } else if (tag.equals("IN")) {
                wordChunkType = "PP";
            }

            // Handle chunk boundaries
            if (wordChunkType == null || !wordChunkType.equals(currentType)) {
                // Save current chunk if exists
                if (currentPhrase.length() > 0 && currentType != null) {
                    chunks.add(new Chunk(currentType, currentPhrase.toString().trim(),
                                        startPos, tw.start - 1));
                }

                // Start new chunk
                if (wordChunkType != null) {
                    currentType = wordChunkType;
                    currentPhrase = new StringBuilder(tw.word);
                    startPos = tw.start;
                } else {
                    currentType = null;
                    currentPhrase = new StringBuilder();
                }
            } else {
                // Continue current chunk
                currentPhrase.append(" ").append(tw.word);
            }
        }

        // Add final chunk
        if (currentPhrase.length() > 0 && currentType != null) {
            chunks.add(new Chunk(currentType, currentPhrase.toString().trim(),
                                startPos, text.length()));
        }

        return chunks;
    }

    /**
     * Extract only noun phrases.
     */
    public List<String> extractNounPhrases(String text) {
        List<String> nps = new ArrayList<>();
        for (Chunk chunk : chunk(text)) {
            if ("NP".equals(chunk.type)) {
                nps.add(chunk.text);
            }
        }
        return nps;
    }

    /**
     * Extract only verb phrases.
     */
    public List<String> extractVerbPhrases(String text) {
        List<String> vps = new ArrayList<>();
        for (Chunk chunk : chunk(text)) {
            if ("VP".equals(chunk.type)) {
                vps.add(chunk.text);
            }
        }
        return vps;
    }

    /**
     * Get IOB tags for tokens.
     */
    public List<String> getIOBTags(String text) {
        List<String> iobTags = new ArrayList<>();
        List<TaggedWord> tagged = posTag(text);
        List<Chunk> chunks = chunk(text);

        for (TaggedWord tw : tagged) {
            String iob = "O";  // Outside by default

            for (Chunk chunk : chunks) {
                if (tw.start >= chunk.start && tw.end <= chunk.end) {
                    if (tw.start == chunk.start) {
                        iob = "B-" + chunk.type;  // Beginning
                    } else {
                        iob = "I-" + chunk.type;  // Inside
                    }
                    break;
                }
            }

            iobTags.add(iob);
        }

        return iobTags;
    }

    private static class TaggedWord {
        String word;
        String tag;
        int start;
        int end;

        TaggedWord(String word, String tag, int start, int end) {
            this.word = word;
            this.tag = tag;
            this.start = start;
            this.end = end;
        }
    }

    public static class Chunk {
        public String type;
        public String text;
        public int start;
        public int end;

        public Chunk(String type, String text, int start, int end) {
            this.type = type;
            this.text = text;
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return "[" + type + ": " + text + "]";
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class TextChunkerTest {

    @Test
    void testChunk() {
        TextChunker chunker = new TextChunker();
        List<TextChunker.Chunk> chunks = chunker.chunk("The quick brown fox jumps");

        assertTrue(chunks.size() >= 2);
    }

    @Test
    void testExtractNounPhrases() {
        TextChunker chunker = new TextChunker();
        List<String> nps = chunker.extractNounPhrases("The quick brown fox");

        assertEquals(1, nps.size());
        assertTrue(nps.get(0).contains("fox"));
    }

    @Test
    void testExtractVerbPhrases() {
        TextChunker chunker = new TextChunker();
        List<String> vps = chunker.extractVerbPhrases("The fox jumps quickly");

        assertTrue(vps.size() >= 1);
    }

    @Test
    void testIOBTags() {
        TextChunker chunker = new TextChunker();
        List<String> iob = chunker.getIOBTags("The fox jumps");

        assertEquals(3, iob.size());
        assertTrue(iob.get(0).startsWith("B-") || iob.get(0).equals("O"));
    }

    @Test
    void testEmptyInput() {
        TextChunker chunker = new TextChunker();
        List<TextChunker.Chunk> chunks = chunker.chunk("");
        assertTrue(chunks.isEmpty());
    }

    @Test
    void testChunkToString() {
        TextChunker.Chunk chunk = new TextChunker.Chunk("NP", "the fox", 0, 7);
        String str = chunk.toString();
        assertTrue(str.contains("NP"));
        assertTrue(str.contains("the fox"));
    }

    @Test
    void testChunkHasType() {
        TextChunker chunker = new TextChunker();
        List<TextChunker.Chunk> chunks = chunker.chunk("The fox jumps");
        boolean hasNP = chunks.stream().anyMatch(c -> "NP".equals(c.type));
        assertTrue(hasNP);
    }

    @Test
    void testPrepositionPhrase() {
        TextChunker chunker = new TextChunker();
        List<TextChunker.Chunk> chunks = chunker.chunk("jumps over the dog");
        boolean hasPP = chunks.stream().anyMatch(c -> "PP".equals(c.type));
        assertTrue(hasPP);
    }

    @Test
    void testChunkPositions() {
        TextChunker.Chunk chunk = new TextChunker.Chunk("NP", "test", 5, 9);
        assertEquals(5, chunk.start);
        assertEquals(9, chunk.end);
    }

    @Test
    void testExtractNounPhrasesEmpty() {
        TextChunker chunker = new TextChunker();
        List<String> nps = chunker.extractNounPhrases("runs quickly");
        assertTrue(nps.isEmpty());
    }
}`,

	hint1: 'Use POS tag patterns to identify phrase boundaries (DT JJ* NN+ for NP)',
	hint2: 'IOB tagging marks Beginning, Inside, and Outside of chunks',

	whyItMatters: `Text chunking enables information extraction:

- **Noun phrases**: Key entities and concepts in text
- **Verb phrases**: Actions and events in text
- **Information extraction**: Who did what to whom
- **Question answering**: Finding relevant phrases for queries

Chunking is a crucial step between POS tagging and full parsing.`,

	translations: {
		ru: {
			title: 'Чанкинг текста',
			description: `# Чанкинг текста

Реализуйте поверхностный парсинг для извлечения именных и глагольных групп.

## Задача

Создайте чанкер, который:
- Определяет именные группы (NP)
- Определяет глагольные группы (VP)
- Использует схему IOB-разметки
- Извлекает границы фраз из текста

## Пример

\`\`\`java
TextChunker chunker = new TextChunker();
List<Chunk> chunks = chunker.chunk("The quick brown fox jumps over the lazy dog");
// [NP: The quick brown fox, VP: jumps, PP: over the lazy dog]
\`\`\``,
			hint1: 'Используйте паттерны POS-тегов для определения границ фраз (DT JJ* NN+ для NP)',
			hint2: 'IOB-разметка отмечает начало, внутренность и внешность чанков',
			whyItMatters: `Чанкинг текста обеспечивает извлечение информации:

- **Именные группы**: Ключевые сущности и концепции
- **Глагольные группы**: Действия и события в тексте
- **Извлечение информации**: Кто что сделал кому
- **Ответы на вопросы**: Поиск релевантных фраз`,
		},
		uz: {
			title: 'Matn chunking',
			description: `# Matn chunking

Ot va fe'l iboralarini ajratib olish uchun sayoz tahlilni amalga oshiring.

## Topshiriq

Chunker yarating:
- Ot iboralarini (NP) aniqlash
- Fe'l iboralarini (VP) aniqlash
- IOB teglash sxemasidan foydalanish
- Matndan ibora chegaralarini ajratib olish

## Misol

\`\`\`java
TextChunker chunker = new TextChunker();
List<Chunk> chunks = chunker.chunk("The quick brown fox jumps over the lazy dog");
// [NP: The quick brown fox, VP: jumps, PP: over the lazy dog]
\`\`\``,
			hint1: "Ibora chegaralarini aniqlash uchun POS teg patternlaridan foydalaning (NP uchun DT JJ* NN+)",
			hint2: "IOB teglash chunklarning Boshlanishi, Ichkarisi va Tashqarisini belgilaydi",
			whyItMatters: `Matn chunking ma'lumot ajratib olishni ta'minlaydi:

- **Ot iboralari**: Matndagi asosiy ob'ektlar va tushunchalar
- **Fe'l iboralari**: Matndagi harakatlar va voqealar
- **Ma'lumot ajratib olish**: Kim nimani kimga qildi
- **Savollarga javob**: So'rovlar uchun tegishli iboralarni topish`,
		},
	},
};

export default task;
