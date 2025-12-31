import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-classification-metrics',
	title: 'Classification Metrics',
	difficulty: 'easy',
	tags: ['metrics', 'accuracy', 'precision', 'recall', 'f1'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Classification Metrics

Calculate accuracy, precision, recall, and F1-score for classification models.

## Task

Implement classification metrics:
- Accuracy calculation
- Precision and Recall
- F1-score (harmonic mean)

## Example

\`\`\`java
Evaluation eval = new Evaluation(numClasses);
eval.eval(labels, predictions);
double accuracy = eval.accuracy();
\`\`\``,

	initialCode: `import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ClassificationMetrics {

    /**
     */
    public static double calculateAccuracy(Evaluation eval) {
        return 0.0;
    }

    /**
     */
    public static double calculatePrecision(Evaluation eval, int classIndex) {
        return 0.0;
    }

    /**
     */
    public static double calculateRecall(Evaluation eval, int classIndex) {
        return 0.0;
    }

    /**
     */
    public static double calculateF1(Evaluation eval, int classIndex) {
        return 0.0;
    }
}`,

	solutionCode: `import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ClassificationMetrics {

    /**
     * Calculate accuracy from evaluation.
     */
    public static double calculateAccuracy(Evaluation eval) {
        return eval.accuracy();
    }

    /**
     * Calculate precision for a class.
     */
    public static double calculatePrecision(Evaluation eval, int classIndex) {
        return eval.precision(classIndex);
    }

    /**
     * Calculate recall for a class.
     */
    public static double calculateRecall(Evaluation eval, int classIndex) {
        return eval.recall(classIndex);
    }

    /**
     * Calculate F1 score for a class.
     */
    public static double calculateF1(Evaluation eval, int classIndex) {
        return eval.f1(classIndex);
    }

    /**
     * Calculate macro F1 (average across all classes).
     */
    public static double calculateMacroF1(Evaluation eval, int numClasses) {
        double sum = 0.0;
        for (int i = 0; i < numClasses; i++) {
            sum += eval.f1(i);
        }
        return sum / numClasses;
    }

    /**
     * Manual F1 calculation from precision and recall.
     */
    public static double manualF1(double precision, double recall) {
        if (precision + recall == 0) return 0.0;
        return 2 * (precision * recall) / (precision + recall);
    }

    /**
     * Create evaluation and compute metrics.
     */
    public static Evaluation evaluate(INDArray labels, INDArray predictions, int numClasses) {
        Evaluation eval = new Evaluation(numClasses);
        eval.eval(labels, predictions);
        return eval;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class ClassificationMetricsTest {

    @Test
    void testManualF1() {
        double f1 = ClassificationMetrics.manualF1(0.8, 0.6);
        assertEquals(0.6857, f1, 0.001);
    }

    @Test
    void testManualF1ZeroCase() {
        double f1 = ClassificationMetrics.manualF1(0.0, 0.0);
        assertEquals(0.0, f1, 0.001);
    }

    @Test
    void testEvaluate() {
        Evaluation eval = ClassificationMetrics.evaluate(
            Nd4j.create(new double[][]{{1, 0}, {0, 1}, {1, 0}}),
            Nd4j.create(new double[][]{{1, 0}, {0, 1}, {1, 0}}),
            2
        );
        assertEquals(1.0, ClassificationMetrics.calculateAccuracy(eval), 0.001);
    }

    @Test
    void testManualF1Perfect() {
        double f1 = ClassificationMetrics.manualF1(1.0, 1.0);
        assertEquals(1.0, f1, 0.001);
    }

    @Test
    void testAccuracyRange() {
        Evaluation eval = ClassificationMetrics.evaluate(
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            2
        );
        double acc = ClassificationMetrics.calculateAccuracy(eval);
        assertTrue(acc >= 0 && acc <= 1);
    }

    @Test
    void testEvaluateReturnsType() {
        Evaluation eval = ClassificationMetrics.evaluate(
            Nd4j.create(new double[][]{{1, 0}}),
            Nd4j.create(new double[][]{{1, 0}}),
            2
        );
        assertInstanceOf(Evaluation.class, eval);
    }

    @Test
    void testPrecisionRange() {
        Evaluation eval = ClassificationMetrics.evaluate(
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            2
        );
        double precision = ClassificationMetrics.calculatePrecision(eval, 0);
        assertTrue(precision >= 0 && precision <= 1);
    }

    @Test
    void testRecallRange() {
        Evaluation eval = ClassificationMetrics.evaluate(
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            2
        );
        double recall = ClassificationMetrics.calculateRecall(eval, 0);
        assertTrue(recall >= 0 && recall <= 1);
    }

    @Test
    void testF1Range() {
        Evaluation eval = ClassificationMetrics.evaluate(
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            Nd4j.create(new double[][]{{1, 0}, {0, 1}}),
            2
        );
        double f1 = ClassificationMetrics.calculateF1(eval, 0);
        assertTrue(f1 >= 0 && f1 <= 1);
    }

    @Test
    void testManualF1EqualInputs() {
        double f1 = ClassificationMetrics.manualF1(0.5, 0.5);
        assertEquals(0.5, f1, 0.001);
    }
}`,

	hint1: 'Use Evaluation class methods like accuracy(), precision(), recall()',
	hint2: 'F1 = 2 * (precision * recall) / (precision + recall)',

	whyItMatters: `Classification metrics measure model quality:

- **Accuracy**: Overall correctness but can be misleading with imbalanced data
- **Precision**: How many predicted positives are actually positive
- **Recall**: How many actual positives are correctly identified
- **F1-score**: Balance between precision and recall

Choosing the right metric depends on your business problem.`,

	translations: {
		ru: {
			title: 'Метрики классификации',
			description: `# Метрики классификации

Вычисляйте accuracy, precision, recall и F1-score для моделей классификации.

## Задача

Реализуйте метрики классификации:
- Вычисление accuracy
- Precision и Recall
- F1-score (гармоническое среднее)

## Пример

\`\`\`java
Evaluation eval = new Evaluation(numClasses);
eval.eval(labels, predictions);
double accuracy = eval.accuracy();
\`\`\``,
			hint1: 'Используйте методы класса Evaluation: accuracy(), precision(), recall()',
			hint2: 'F1 = 2 * (precision * recall) / (precision + recall)',
			whyItMatters: `Метрики классификации измеряют качество модели:

- **Accuracy**: Общая правильность, но может вводить в заблуждение при дисбалансе
- **Precision**: Сколько предсказанных положительных действительно положительны
- **Recall**: Сколько реальных положительных правильно определено
- **F1-score**: Баланс между precision и recall`,
		},
		uz: {
			title: 'Klassifikatsiya metrikalari',
			description: `# Klassifikatsiya metrikalari

Klassifikatsiya modellari uchun accuracy, precision, recall va F1-score ni hisoblang.

## Topshiriq

Klassifikatsiya metrikalarini amalga oshiring:
- Accuracy hisoblash
- Precision va Recall
- F1-score (garmonik o'rtacha)

## Misol

\`\`\`java
Evaluation eval = new Evaluation(numClasses);
eval.eval(labels, predictions);
double accuracy = eval.accuracy();
\`\`\``,
			hint1: "Evaluation class metodlaridan foydalaning: accuracy(), precision(), recall()",
			hint2: 'F1 = 2 * (precision * recall) / (precision + recall)',
			whyItMatters: `Klassifikatsiya metrikalari model sifatini o'lchaydi:

- **Accuracy**: Umumiy to'g'rilik, lekin muvozanatsiz ma'lumotlarda chalg'itishi mumkin
- **Precision**: Bashorat qilingan ijobiylardan qanchasi haqiqatda ijobiy
- **Recall**: Haqiqiy ijobiylardan qanchasi to'g'ri aniqlangan
- **F1-score**: Precision va recall o'rtasidagi muvozanat`,
		},
	},
};

export default task;
