import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-model-comparison',
	title: 'Model Comparison',
	difficulty: 'medium',
	tags: ['metrics', 'comparison', 'evaluation', 'selection'],
	estimatedTime: '20m',
	isPremium: false,
	order: 5,
	description: `# Model Comparison

Compare multiple models and select the best performer.

## Task

Implement model comparison:
- Compare models on same test set
- Statistical significance testing
- Select best model based on criteria

## Example

\`\`\`java
ModelComparer comparer = new ModelComparer();
comparer.addModel("model1", accuracy1, f1_1);
comparer.addModel("model2", accuracy2, f1_2);
String best = comparer.selectBest("f1");
\`\`\``,

	initialCode: `import java.util.*;

public class ModelComparer {

    private Map<String, Map<String, Double>> modelMetrics = new HashMap<>();

    /**
     */
    public void addModel(String name, double accuracy, double f1, double auc) {
    }

    /**
     */
    public String selectBest(String metric) {
        return null;
    }

    /**
     */
    public List<String> rankModels(String metric) {
        return null;
    }

    /**
     */
    public double getImprovement(String model, String baseline, String metric) {
        return 0.0;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class ModelComparer {

    private Map<String, Map<String, Double>> modelMetrics = new HashMap<>();

    /**
     * Add model with its metrics.
     */
    public void addModel(String name, double accuracy, double f1, double auc) {
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("accuracy", accuracy);
        metrics.put("f1", f1);
        metrics.put("auc", auc);
        modelMetrics.put(name, metrics);
    }

    /**
     * Get best model by specific metric.
     */
    public String selectBest(String metric) {
        return modelMetrics.entrySet().stream()
            .max(Comparator.comparingDouble(e -> e.getValue().getOrDefault(metric, 0.0)))
            .map(Map.Entry::getKey)
            .orElse(null);
    }

    /**
     * Rank all models by metric.
     */
    public List<String> rankModels(String metric) {
        return modelMetrics.entrySet().stream()
            .sorted((a, b) -> Double.compare(
                b.getValue().getOrDefault(metric, 0.0),
                a.getValue().getOrDefault(metric, 0.0)))
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

    /**
     * Get improvement over baseline.
     */
    public double getImprovement(String model, String baseline, String metric) {
        double modelScore = modelMetrics.getOrDefault(model, Collections.emptyMap())
            .getOrDefault(metric, 0.0);
        double baselineScore = modelMetrics.getOrDefault(baseline, Collections.emptyMap())
            .getOrDefault(metric, 0.0);

        if (baselineScore == 0) return 0.0;
        return (modelScore - baselineScore) / baselineScore * 100;
    }

    /**
     * Get metric value for a model.
     */
    public double getMetric(String model, String metric) {
        return modelMetrics.getOrDefault(model, Collections.emptyMap())
            .getOrDefault(metric, 0.0);
    }

    /**
     * Compare two models across all metrics.
     */
    public Map<String, Integer> compareModels(String model1, String model2) {
        Map<String, Integer> comparison = new HashMap<>();
        Set<String> metrics = new HashSet<>();

        modelMetrics.values().forEach(m -> metrics.addAll(m.keySet()));

        for (String metric : metrics) {
            double score1 = getMetric(model1, metric);
            double score2 = getMetric(model2, metric);

            if (score1 > score2) comparison.put(metric, 1);
            else if (score2 > score1) comparison.put(metric, -1);
            else comparison.put(metric, 0);
        }

        return comparison;
    }

    /**
     * Generate comparison report.
     */
    public String generateReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("Model Comparison Report\\n");
        sb.append("=".repeat(50)).append("\\n");

        for (Map.Entry<String, Map<String, Double>> entry : modelMetrics.entrySet()) {
            sb.append(entry.getKey()).append(":\\n");
            for (Map.Entry<String, Double> metric : entry.getValue().entrySet()) {
                sb.append(String.format("  %s: %.4f\\n", metric.getKey(), metric.getValue()));
            }
        }

        return sb.toString();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class ModelComparerTest {

    private ModelComparer comparer;

    @BeforeEach
    void setup() {
        comparer = new ModelComparer();
        comparer.addModel("model_a", 0.85, 0.82, 0.90);
        comparer.addModel("model_b", 0.88, 0.85, 0.92);
        comparer.addModel("model_c", 0.80, 0.78, 0.85);
    }

    @Test
    void testSelectBest() {
        assertEquals("model_b", comparer.selectBest("accuracy"));
        assertEquals("model_b", comparer.selectBest("f1"));
    }

    @Test
    void testRankModels() {
        List<String> ranking = comparer.rankModels("accuracy");
        assertEquals("model_b", ranking.get(0));
        assertEquals("model_a", ranking.get(1));
        assertEquals("model_c", ranking.get(2));
    }

    @Test
    void testGetImprovement() {
        double improvement = comparer.getImprovement("model_b", "model_c", "accuracy");
        assertEquals(10.0, improvement, 0.1);
    }

    @Test
    void testGetMetric() {
        assertEquals(0.85, comparer.getMetric("model_a", "accuracy"), 0.001);
    }

    @Test
    void testSelectBestByAUC() {
        assertEquals("model_b", comparer.selectBest("auc"));
    }

    @Test
    void testRankModelsByF1() {
        List<String> ranking = comparer.rankModels("f1");
        assertEquals(3, ranking.size());
        assertEquals("model_b", ranking.get(0));
    }

    @Test
    void testCompareModels() {
        var comparison = comparer.compareModels("model_b", "model_a");
        assertNotNull(comparison);
        assertEquals(1, comparison.get("accuracy"));
    }

    @Test
    void testGenerateReport() {
        String report = comparer.generateReport();
        assertNotNull(report);
        assertTrue(report.contains("model_a"));
    }

    @Test
    void testGetMetricForNonexistent() {
        assertEquals(0.0, comparer.getMetric("nonexistent", "accuracy"), 0.001);
    }

    @Test
    void testImprovementZeroBaseline() {
        ModelComparer newComparer = new ModelComparer();
        newComparer.addModel("baseline", 0.0, 0.0, 0.0);
        newComparer.addModel("model", 0.8, 0.8, 0.8);
        assertEquals(0.0, newComparer.getImprovement("model", "baseline", "accuracy"), 0.001);
    }
}`,

	hint1: 'Use stream operations to find max values efficiently',
	hint2: 'Consider multiple metrics when comparing models',

	whyItMatters: `Model comparison enables informed decisions:

- **Objective selection**: Choose models based on data not intuition
- **Trade-off analysis**: Understand accuracy vs speed vs complexity
- **Baseline comparison**: Measure improvement over simpler models
- **Reproducibility**: Document why a model was selected

Systematic comparison is essential for production ML systems.`,

	translations: {
		ru: {
			title: 'Сравнение моделей',
			description: `# Сравнение моделей

Сравнивайте несколько моделей и выбирайте лучшую.

## Задача

Реализуйте сравнение моделей:
- Сравнение моделей на одном тестовом наборе
- Тестирование статистической значимости
- Выбор лучшей модели по критериям

## Пример

\`\`\`java
ModelComparer comparer = new ModelComparer();
comparer.addModel("model1", accuracy1, f1_1);
comparer.addModel("model2", accuracy2, f1_2);
String best = comparer.selectBest("f1");
\`\`\``,
			hint1: 'Используйте stream операции для эффективного поиска максимума',
			hint2: 'Учитывайте несколько метрик при сравнении моделей',
			whyItMatters: `Сравнение моделей позволяет принимать обоснованные решения:

- **Объективный выбор**: Выбор моделей на основе данных, не интуиции
- **Анализ компромиссов**: Понимание точности vs скорости vs сложности
- **Сравнение с базой**: Измерение улучшения над простыми моделями
- **Воспроизводимость**: Документация почему модель была выбрана`,
		},
		uz: {
			title: 'Modellarni taqqoslash',
			description: `# Modellarni taqqoslash

Bir nechta modellarni taqqoslang va eng yaxshisini tanlang.

## Topshiriq

Model taqqoslashni amalga oshiring:
- Modellarni bir xil test to'plamida taqqoslash
- Statistik ahamiyatni tekshirish
- Mezonlar asosida eng yaxshi modelni tanlash

## Misol

\`\`\`java
ModelComparer comparer = new ModelComparer();
comparer.addModel("model1", accuracy1, f1_1);
comparer.addModel("model2", accuracy2, f1_2);
String best = comparer.selectBest("f1");
\`\`\``,
			hint1: "Maksimal qiymatlarni samarali topish uchun stream operatsiyalaridan foydalaning",
			hint2: "Modellarni taqqoslashda bir nechta metrikalarni hisobga oling",
			whyItMatters: `Model taqqoslash asosli qarorlar qabul qilishga imkon beradi:

- **Ob'ektiv tanlash**: Modellarni intuitsiya emas ma'lumotlar asosida tanlash
- **Kelishuv tahlili**: Aniqlik vs tezlik vs murakkablikni tushunish
- **Bazaviy taqqoslash**: Oddiyroq modellardan yaxshilanishni o'lchash
- **Takrorlanish**: Model nima uchun tanlangani hujjatlash`,
		},
	},
};

export default task;
