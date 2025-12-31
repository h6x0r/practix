import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-subword-tokenization',
	title: 'Subword Tokenization',
	difficulty: 'medium',
	tags: ['nlp', 'tokenization', 'bpe', 'subword'],
	estimatedTime: '25m',
	isPremium: false,
	order: 3,
	description: `# Subword Tokenization

Implement BPE (Byte Pair Encoding) style tokenization.

## Task

Build subword tokenizer:
- Character-level tokenization
- Merge frequent pairs
- Handle unknown words

## Example

\`\`\`java
BPETokenizer bpe = new BPETokenizer();
bpe.train(corpus);
List<String> tokens = bpe.tokenize("unhappiness");
// Result: ["un", "happi", "ness"]
\`\`\``,

	initialCode: `import java.util.*;

public class SimpleBPE {

    private Map<String, Integer> vocab;
    private List<String[]> merges;

    /**
     */
    public SimpleBPE() {
    }

    /**
     */
    public List<String> charTokenize(String word) {
        return null;
    }

    /**
     */
    public Map<String, Integer> countPairs(List<List<String>> tokenizedWords) {
        return null;
    }

    /**
     */
    public void mergePair(List<List<String>> tokenizedWords, String pair) {
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class SimpleBPE {

    private Map<String, Integer> vocab;
    private List<String[]> merges;
    private int vocabSize;

    /**
     * Initialize tokenizer.
     */
    public SimpleBPE() {
        this.vocab = new HashMap<>();
        this.merges = new ArrayList<>();
        this.vocabSize = 1000;
    }

    /**
     * Get character-level tokens.
     */
    public List<String> charTokenize(String word) {
        List<String> chars = new ArrayList<>();
        for (char c : word.toCharArray()) {
            chars.add(String.valueOf(c));
        }
        chars.add("</w>"); // End of word marker
        return chars;
    }

    /**
     * Count pair frequencies in tokens.
     */
    public Map<String, Integer> countPairs(List<List<String>> tokenizedWords) {
        Map<String, Integer> pairs = new HashMap<>();

        for (List<String> tokens : tokenizedWords) {
            for (int i = 0; i < tokens.size() - 1; i++) {
                String pair = tokens.get(i) + " " + tokens.get(i + 1);
                pairs.merge(pair, 1, Integer::sum);
            }
        }
        return pairs;
    }

    /**
     * Merge most frequent pair.
     */
    public void mergePair(List<List<String>> tokenizedWords, String pair) {
        String[] parts = pair.split(" ");
        String merged = parts[0] + parts[1];

        for (List<String> tokens : tokenizedWords) {
            for (int i = 0; i < tokens.size() - 1; i++) {
                if (tokens.get(i).equals(parts[0]) &&
                    tokens.get(i + 1).equals(parts[1])) {
                    tokens.set(i, merged);
                    tokens.remove(i + 1);
                }
            }
        }
    }

    /**
     * Train BPE on corpus.
     */
    public void train(List<String> corpus, int numMerges) {
        // Initialize with character vocabulary
        List<List<String>> tokenizedWords = new ArrayList<>();
        for (String word : corpus) {
            tokenizedWords.add(new ArrayList<>(charTokenize(word)));
        }

        for (int i = 0; i < numMerges; i++) {
            Map<String, Integer> pairs = countPairs(tokenizedWords);
            if (pairs.isEmpty()) break;

            // Find most frequent pair
            String bestPair = pairs.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);

            if (bestPair == null) break;

            merges.add(bestPair.split(" "));
            mergePair(tokenizedWords, bestPair);

            // Add merged token to vocab
            String merged = bestPair.replace(" ", "");
            vocab.put(merged, i);
        }
    }

    /**
     * Tokenize a word using learned merges.
     */
    public List<String> tokenize(String word) {
        List<String> tokens = charTokenize(word);

        for (String[] merge : merges) {
            List<String> newTokens = new ArrayList<>();
            int i = 0;
            while (i < tokens.size()) {
                if (i < tokens.size() - 1 &&
                    tokens.get(i).equals(merge[0]) &&
                    tokens.get(i + 1).equals(merge[1])) {
                    newTokens.add(merge[0] + merge[1]);
                    i += 2;
                } else {
                    newTokens.add(tokens.get(i));
                    i++;
                }
            }
            tokens = newTokens;
        }
        return tokens;
    }

    /**
     * Get vocabulary size.
     */
    public int getVocabSize() {
        return vocab.size();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SimpleBPETest {

    @Test
    void testCharTokenize() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> chars = bpe.charTokenize("hello");

        assertEquals(6, chars.size()); // h, e, l, l, o, </w>
        assertEquals("h", chars.get(0));
        assertEquals("</w>", chars.get(5));
    }

    @Test
    void testCountPairs() {
        SimpleBPE bpe = new SimpleBPE();
        List<List<String>> words = new ArrayList<>();
        words.add(new ArrayList<>(Arrays.asList("h", "e", "l", "l", "o")));
        words.add(new ArrayList<>(Arrays.asList("h", "e", "l", "p")));

        Map<String, Integer> pairs = bpe.countPairs(words);
        assertEquals(2, pairs.get("h e").intValue());
        assertEquals(2, pairs.get("e l").intValue());
    }

    @Test
    void testTrainAndTokenize() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> corpus = Arrays.asList("low", "lower", "lowest", "new", "newer");
        bpe.train(corpus, 10);

        List<String> tokens = bpe.tokenize("low");
        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    void testGetVocabSize() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> corpus = Arrays.asList("hello", "help", "held");
        bpe.train(corpus, 5);
        assertTrue(bpe.getVocabSize() > 0);
    }

    @Test
    void testMergePair() {
        SimpleBPE bpe = new SimpleBPE();
        List<List<String>> words = new ArrayList<>();
        words.add(new ArrayList<>(Arrays.asList("a", "b", "c")));
        bpe.mergePair(words, "a b");
        assertEquals(2, words.get(0).size());
        assertEquals("ab", words.get(0).get(0));
    }

    @Test
    void testCharTokenizeEmpty() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> chars = bpe.charTokenize("");
        assertEquals(1, chars.size()); // only </w>
    }

    @Test
    void testCountPairsEmpty() {
        SimpleBPE bpe = new SimpleBPE();
        List<List<String>> words = new ArrayList<>();
        Map<String, Integer> pairs = bpe.countPairs(words);
        assertTrue(pairs.isEmpty());
    }

    @Test
    void testTokenizeUntrainedBPE() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> tokens = bpe.tokenize("test");
        assertEquals(5, tokens.size()); // t, e, s, t, </w>
    }

    @Test
    void testTrainWithSingleWord() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> corpus = Arrays.asList("hello");
        bpe.train(corpus, 3);
        assertTrue(bpe.getVocabSize() >= 0);
    }

    @Test
    void testCharTokenizeSingleChar() {
        SimpleBPE bpe = new SimpleBPE();
        List<String> chars = bpe.charTokenize("a");
        assertEquals(2, chars.size());
        assertEquals("a", chars.get(0));
    }
}`,

	hint1: 'Start with character-level tokens and iteratively merge',
	hint2: 'Track merge operations to apply them during tokenization',

	whyItMatters: `Subword tokenization powers modern NLP:

- **Open vocabulary**: Handle any word, even unseen ones
- **Efficiency**: Smaller vocabulary than word-level
- **Morphology**: Capture meaningful word parts
- **Used everywhere**: GPT, BERT, and all modern transformers use it

Understanding BPE is essential for working with modern LLMs.`,

	translations: {
		ru: {
			title: 'Subword токенизация',
			description: `# Subword токенизация

Реализуйте токенизацию в стиле BPE (Byte Pair Encoding).

## Задача

Создайте subword токенизатор:
- Токенизация на уровне символов
- Слияние частых пар
- Обработка неизвестных слов

## Пример

\`\`\`java
BPETokenizer bpe = new BPETokenizer();
bpe.train(corpus);
List<String> tokens = bpe.tokenize("unhappiness");
// Result: ["un", "happi", "ness"]
\`\`\``,
			hint1: 'Начните с токенов уровня символов и итеративно объединяйте',
			hint2: 'Отслеживайте операции слияния для применения при токенизации',
			whyItMatters: `Subword токенизация - основа современного NLP:

- **Открытый словарь**: Обработка любого слова, даже невиденного
- **Эффективность**: Меньший словарь чем на уровне слов
- **Морфология**: Захват значимых частей слов
- **Используется везде**: GPT, BERT и все современные трансформеры`,
		},
		uz: {
			title: 'Subword tokenizatsiyasi',
			description: `# Subword tokenizatsiyasi

BPE (Byte Pair Encoding) uslubida tokenizatsiyani amalga oshiring.

## Topshiriq

Subword tokenizator yarating:
- Belgi darajasida tokenizatsiya
- Tez-tez uchraydigan juftlarni birlashtirish
- Noma'lum so'zlarni boshqarish

## Misol

\`\`\`java
BPETokenizer bpe = new BPETokenizer();
bpe.train(corpus);
List<String> tokens = bpe.tokenize("unhappiness");
// Result: ["un", "happi", "ness"]
\`\`\``,
			hint1: "Belgi darajasidagi tokenlardan boshlang va iterativ ravishda birlashtiring",
			hint2: "Tokenizatsiya paytida qo'llash uchun birlashtirish operatsiyalarini kuzating",
			whyItMatters: `Subword tokenizatsiya zamonaviy NLP ni quvvatlaydi:

- **Ochiq lug'at**: Har qanday so'zni, hatto ko'rilmaganlarni ham boshqarish
- **Samaradorlik**: So'z darajasidan kichikroq lug'at
- **Morfologiya**: Ma'noli so'z qismlarini olish
- **Hamma joyda ishlatiladi**: GPT, BERT va barcha zamonaviy transformerlar`,
		},
	},
};

export default task;
