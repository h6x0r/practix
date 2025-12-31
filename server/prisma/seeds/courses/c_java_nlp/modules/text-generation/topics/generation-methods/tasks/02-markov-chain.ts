import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-markov-chain',
	title: 'Markov Chain Text Generator',
	difficulty: 'medium',
	tags: ['nlp', 'text-generation', 'markov-chain', 'probabilistic'],
	estimatedTime: '25m',
	isPremium: true,
	order: 2,
	description: `# Markov Chain Text Generator

Build a Markov chain for text generation with variable order.

## Task

Implement a Markov chain generator that:
- Builds transition matrix from text
- Supports variable chain order
- Generates coherent text sequences
- Handles sentence boundaries

## Example

\`\`\`java
MarkovChain chain = new MarkovChain(2);
chain.train(text);
String generated = chain.generate("The quick", 50);
\`\`\``,

	initialCode: `import java.util.*;

public class MarkovChain {

    private int order;

    public MarkovChain(int order) {
        this.order = order;
    }

    /**
     * Train on text corpus.
     */
    public void train(String corpus) {
    }

    /**
     * Generate text starting with seed.
     */
    public String generate(String seed, int maxLength) {
        return null;
    }

    /**
     * Get transition probabilities from state.
     */
    public Map<String, Double> getTransitions(String state) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class MarkovChain {

    private int order;
    private Map<String, Map<String, Integer>> transitions;
    private Map<String, Integer> stateCounts;
    private List<String> startStates;
    private Random random;

    public MarkovChain(int order) {
        this.order = order;
        this.transitions = new HashMap<>();
        this.stateCounts = new HashMap<>();
        this.startStates = new ArrayList<>();
        this.random = new Random(42);
    }

    /**
     * Train on text corpus.
     */
    public void train(String corpus) {
        String[] sentences = corpus.split("[.!?]+");

        for (String sentence : sentences) {
            String[] words = sentence.trim().toLowerCase().split("\\\\s+");
            if (words.length < order + 1) continue;

            // Record start state
            StringBuilder startBuilder = new StringBuilder();
            for (int i = 0; i < order; i++) {
                if (i > 0) startBuilder.append(" ");
                startBuilder.append(words[i]);
            }
            startStates.add(startBuilder.toString());

            // Build transitions
            for (int i = 0; i <= words.length - order - 1; i++) {
                StringBuilder stateBuilder = new StringBuilder();
                for (int j = 0; j < order; j++) {
                    if (j > 0) stateBuilder.append(" ");
                    stateBuilder.append(words[i + j]);
                }
                String state = stateBuilder.toString();
                String nextWord = words[i + order];

                transitions.computeIfAbsent(state, k -> new HashMap<>())
                    .merge(nextWord, 1, Integer::sum);
                stateCounts.merge(state, 1, Integer::sum);
            }
        }
    }

    /**
     * Get transition probabilities from state.
     */
    public Map<String, Double> getTransitions(String state) {
        Map<String, Double> probs = new HashMap<>();
        Map<String, Integer> counts = transitions.get(state.toLowerCase());

        if (counts == null) {
            return probs;
        }

        int total = stateCounts.get(state.toLowerCase());
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            probs.put(entry.getKey(), (double) entry.getValue() / total);
        }

        return probs;
    }

    /**
     * Sample next word from state.
     */
    private String sampleNext(String state) {
        Map<String, Integer> counts = transitions.get(state.toLowerCase());
        if (counts == null || counts.isEmpty()) {
            return null;
        }

        int total = stateCounts.get(state.toLowerCase());
        double rand = random.nextDouble() * total;
        double cumulative = 0;

        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            cumulative += entry.getValue();
            if (cumulative >= rand) {
                return entry.getKey();
            }
        }

        return counts.keySet().iterator().next();
    }

    /**
     * Generate text starting with seed.
     */
    public String generate(String seed, int maxLength) {
        String currentState;

        if (seed == null || seed.isEmpty()) {
            // Use random start state
            if (startStates.isEmpty()) return "";
            currentState = startStates.get(random.nextInt(startStates.size()));
        } else {
            currentState = seed.toLowerCase();
        }

        StringBuilder result = new StringBuilder(currentState);
        String[] stateWords = currentState.split("\\\\s+");

        if (stateWords.length != order) {
            // Try to find matching state
            for (String state : transitions.keySet()) {
                if (state.contains(seed.toLowerCase())) {
                    currentState = state;
                    stateWords = currentState.split("\\\\s+");
                    result = new StringBuilder(currentState);
                    break;
                }
            }
        }

        int generated = stateWords.length;

        while (generated < maxLength) {
            String nextWord = sampleNext(currentState);
            if (nextWord == null) break;

            result.append(" ").append(nextWord);
            generated++;

            // Update state
            String[] currentWords = currentState.split("\\\\s+");
            StringBuilder newState = new StringBuilder();
            for (int i = 1; i < currentWords.length; i++) {
                if (newState.length() > 0) newState.append(" ");
                newState.append(currentWords[i]);
            }
            newState.append(" ").append(nextWord);
            currentState = newState.toString().trim();
        }

        return result.toString();
    }

    /**
     * Get most likely next words.
     */
    public List<String> getMostLikelyNext(String state, int n) {
        Map<String, Double> probs = getTransitions(state);
        List<Map.Entry<String, Double>> sorted = new ArrayList<>(probs.entrySet());
        sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(n, sorted.size()); i++) {
            result.add(sorted.get(i).getKey());
        }
        return result;
    }

    /**
     * Generate multiple variations.
     */
    public List<String> generateVariations(String seed, int maxLength, int count) {
        List<String> variations = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            variations.add(generate(seed, maxLength));
        }
        return variations;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class MarkovChainTest {

    @Test
    void testTrain() {
        MarkovChain chain = new MarkovChain(2);
        chain.train("The quick brown fox. The quick red dog.");

        Map<String, Double> trans = chain.getTransitions("the quick");
        assertFalse(trans.isEmpty());
    }

    @Test
    void testGenerate() {
        MarkovChain chain = new MarkovChain(2);
        chain.train("Hello world and hello there and hello friend.");

        String generated = chain.generate("hello", 10);
        assertNotNull(generated);
        assertTrue(generated.toLowerCase().contains("hello"));
    }

    @Test
    void testGetTransitions() {
        MarkovChain chain = new MarkovChain(1);
        chain.train("the cat the dog the bird");

        Map<String, Double> trans = chain.getTransitions("the");
        assertTrue(trans.size() >= 1);
    }

    @Test
    void testGenerateVariations() {
        MarkovChain chain = new MarkovChain(2);
        chain.train("The quick brown fox jumps. The quick red fox runs.");

        List<String> variations = chain.generateVariations("the quick", 10, 3);
        assertEquals(3, variations.size());
    }

    @Test
    void testMostLikelyNext() {
        MarkovChain chain = new MarkovChain(1);
        chain.train("I love cats I love dogs I love birds I hate rain");

        List<String> next = chain.getMostLikelyNext("love", 2);
        assertTrue(next.size() >= 1);
    }

    @Test
    void testEmptySeed() {
        MarkovChain chain = new MarkovChain(2);
        chain.train("Hello world hello there.");
        String generated = chain.generate("", 10);
        assertNotNull(generated);
    }

    @Test
    void testGetTransitionsEmpty() {
        MarkovChain chain = new MarkovChain(2);
        chain.train("hello world");
        Map<String, Double> trans = chain.getTransitions("unknown state");
        assertTrue(trans.isEmpty());
    }

    @Test
    void testTransitionProbabilitiesSum() {
        MarkovChain chain = new MarkovChain(1);
        chain.train("the cat the dog the bird");
        Map<String, Double> trans = chain.getTransitions("the");
        double sum = trans.values().stream().mapToDouble(Double::doubleValue).sum();
        assertEquals(1.0, sum, 0.01);
    }

    @Test
    void testNullSeed() {
        MarkovChain chain = new MarkovChain(2);
        chain.train("Hello world hello there.");
        String generated = chain.generate(null, 10);
        assertNotNull(generated);
    }

    @Test
    void testGenerateMaxLength() {
        MarkovChain chain = new MarkovChain(1);
        chain.train("a b c d e f g h i j k l m n o");
        String generated = chain.generate("a", 5);
        String[] words = generated.split("\\\\s+");
        assertTrue(words.length <= 5);
    }
}`,

	hint1: 'State is the last n words; transition is the probability of the next word',
	hint2: 'Handle end of sentences by checking for null transitions',

	whyItMatters: `Markov chains demonstrate key generative concepts:

- **State transitions**: Foundation for sequence modeling
- **Memory**: Higher order chains capture more context
- **Probabilistic generation**: Sampling from distributions
- **Simplicity**: No gradient descent or training loops

Markov chains remain useful for simple generation tasks and prototyping.`,

	translations: {
		ru: {
			title: 'Генератор текста на цепях Маркова',
			description: `# Генератор текста на цепях Маркова

Создайте цепь Маркова для генерации текста с переменным порядком.

## Задача

Реализуйте генератор на цепях Маркова:
- Построение матрицы переходов из текста
- Поддержка переменного порядка цепи
- Генерация связных текстовых последовательностей
- Обработка границ предложений

## Пример

\`\`\`java
MarkovChain chain = new MarkovChain(2);
chain.train(text);
String generated = chain.generate("The quick", 50);
\`\`\``,
			hint1: 'Состояние - последние n слов; переход - вероятность следующего слова',
			hint2: 'Обрабатывайте конец предложений проверяя null переходы',
			whyItMatters: `Цепи Маркова демонстрируют ключевые концепции генерации:

- **Переходы состояний**: Основа для моделирования последовательностей
- **Память**: Цепи высшего порядка захватывают больше контекста
- **Вероятностная генерация**: Выборка из распределений
- **Простота**: Без градиентного спуска или циклов обучения`,
		},
		uz: {
			title: "Markov zanjiri matn generatori",
			description: `# Markov zanjiri matn generatori

O'zgaruvchan tartibli matn generatsiyasi uchun Markov zanjirini yarating.

## Topshiriq

Markov zanjiri generatorini amalga oshiring:
- Matndan o'tish matritsasini qurish
- O'zgaruvchan zanjir tartibini qo'llab-quvvatlash
- Bog'langan matn ketma-ketliklarini generatsiya qilish
- Gap chegaralarini qayta ishlash

## Misol

\`\`\`java
MarkovChain chain = new MarkovChain(2);
chain.train(text);
String generated = chain.generate("The quick", 50);
\`\`\``,
			hint1: "Holat - oxirgi n so'z; o'tish - keyingi so'z ehtimolligi",
			hint2: "Null o'tishlarni tekshirib gap oxirini qayta ishlang",
			whyItMatters: `Markov zanjirlari asosiy generativ konseptlarni namoyish etadi:

- **Holat o'tishlari**: Ketma-ketlikni modellashtirish uchun asos
- **Xotira**: Yuqori tartibli zanjirlar ko'proq kontekst ushlaydi
- **Ehtimoliy generatsiya**: Taqsimotlardan sampling
- **Soddalik**: Gradient descent yoki o'qitish sikllari yo'q`,
		},
	},
};

export default task;
