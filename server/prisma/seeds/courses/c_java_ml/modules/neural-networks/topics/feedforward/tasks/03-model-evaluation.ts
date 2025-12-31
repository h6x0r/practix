import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-model-evaluation',
	title: 'Model Evaluation',
	difficulty: 'medium',
	tags: ['dl4j', 'evaluation', 'metrics'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Model Evaluation

Evaluate model performance using various metrics in DL4J.

## Task

Implement evaluation utilities:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- ROC curves and AUC

## Example

\`\`\`java
Evaluation eval = model.evaluate(testIterator);

System.out.println("Accuracy: " + eval.accuracy());
System.out.println("Precision: " + eval.precision());
System.out.println("Recall: " + eval.recall());
System.out.println("F1: " + eval.f1());
System.out.println(eval.confusionToString());
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ModelEvaluator {

    /**
     * Get classification metrics.
     */
    public static EvaluationResult evaluateClassification(MultiLayerNetwork model,
                                                          DataSetIterator testData) {
        return null;
    }

    /**
     * Get per-class metrics.
     */
    public static ClassMetrics[] getPerClassMetrics(Evaluation eval, int numClasses) {
        return null;
    }

    /**
     * Calculate ROC and AUC for binary classification.
     */
    public static double calculateAUC(MultiLayerNetwork model,
                                      DataSetIterator testData) {
        return 0.0;
    }

    public static class EvaluationResult {
        public double accuracy;
        public double precision;
        public double recall;
        public double f1;
        public String confusionMatrix;
    }

    public static class ClassMetrics {
        public int classIndex;
        public double precision;
        public double recall;
        public double f1;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ModelEvaluator {

    /**
     * Get classification metrics.
     */
    public static EvaluationResult evaluateClassification(MultiLayerNetwork model,
                                                          DataSetIterator testData) {
        testData.reset();
        Evaluation eval = model.evaluate(testData);

        EvaluationResult result = new EvaluationResult();
        result.accuracy = eval.accuracy();
        result.precision = eval.precision();
        result.recall = eval.recall();
        result.f1 = eval.f1();
        result.confusionMatrix = eval.confusionToString();

        return result;
    }

    /**
     * Get per-class metrics.
     */
    public static ClassMetrics[] getPerClassMetrics(Evaluation eval, int numClasses) {
        ClassMetrics[] metrics = new ClassMetrics[numClasses];

        for (int i = 0; i < numClasses; i++) {
            metrics[i] = new ClassMetrics();
            metrics[i].classIndex = i;
            metrics[i].precision = eval.precision(i);
            metrics[i].recall = eval.recall(i);
            metrics[i].f1 = eval.f1(i);
        }

        return metrics;
    }

    /**
     * Calculate ROC and AUC for binary classification.
     */
    public static double calculateAUC(MultiLayerNetwork model,
                                      DataSetIterator testData) {
        testData.reset();
        ROC roc = new ROC();

        while (testData.hasNext()) {
            DataSet batch = testData.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            INDArray predictions = model.output(features);

            roc.eval(labels, predictions);
        }

        return roc.calculateAUC();
    }

    public static class EvaluationResult {
        public double accuracy;
        public double precision;
        public double recall;
        public double f1;
        public String confusionMatrix;

        @Override
        public String toString() {
            return String.format(
                "Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                accuracy, precision, recall, f1
            );
        }
    }

    public static class ClassMetrics {
        public int classIndex;
        public double precision;
        public double recall;
        public double f1;

        @Override
        public String toString() {
            return String.format(
                "Class %d - Precision: %.4f, Recall: %.4f, F1: %.4f",
                classIndex, precision, recall, f1
            );
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.eval.Evaluation;
import static org.junit.jupiter.api.Assertions.*;

public class ModelEvaluatorTest {

    @Test
    void testEvaluationResultToString() {
        ModelEvaluator.EvaluationResult result = new ModelEvaluator.EvaluationResult();
        result.accuracy = 0.95;
        result.precision = 0.94;
        result.recall = 0.93;
        result.f1 = 0.935;

        String str = result.toString();
        assertTrue(str.contains("0.95"));
        assertTrue(str.contains("Accuracy"));
    }

    @Test
    void testClassMetricsToString() {
        ModelEvaluator.ClassMetrics metrics = new ModelEvaluator.ClassMetrics();
        metrics.classIndex = 0;
        metrics.precision = 0.9;
        metrics.recall = 0.85;
        metrics.f1 = 0.875;

        String str = metrics.toString();
        assertTrue(str.contains("Class 0"));
        assertTrue(str.contains("Precision"));
    }

    @Test
    void testGetPerClassMetrics() {
        Evaluation eval = new Evaluation(3);
        // Add some fake predictions
        ModelEvaluator.ClassMetrics[] metrics = ModelEvaluator.getPerClassMetrics(eval, 3);
        assertEquals(3, metrics.length);
    }

    @Test
    void testEvaluationResultFieldsExist() {
        ModelEvaluator.EvaluationResult result = new ModelEvaluator.EvaluationResult();
        result.accuracy = 0.8;
        assertEquals(0.8, result.accuracy, 0.001);
    }

    @Test
    void testClassMetricsFieldsExist() {
        ModelEvaluator.ClassMetrics metrics = new ModelEvaluator.ClassMetrics();
        metrics.classIndex = 1;
        metrics.precision = 0.9;
        assertEquals(1, metrics.classIndex);
        assertEquals(0.9, metrics.precision, 0.001);
    }

    @Test
    void testPerClassMetricsLength() {
        Evaluation eval = new Evaluation(5);
        ModelEvaluator.ClassMetrics[] metrics = ModelEvaluator.getPerClassMetrics(eval, 5);
        assertEquals(5, metrics.length);
    }

    @Test
    void testPerClassMetricsClassIndex() {
        Evaluation eval = new Evaluation(2);
        ModelEvaluator.ClassMetrics[] metrics = ModelEvaluator.getPerClassMetrics(eval, 2);
        assertEquals(0, metrics[0].classIndex);
        assertEquals(1, metrics[1].classIndex);
    }

    @Test
    void testEvaluationResultPrecision() {
        ModelEvaluator.EvaluationResult result = new ModelEvaluator.EvaluationResult();
        result.precision = 0.92;
        assertEquals(0.92, result.precision, 0.001);
    }

    @Test
    void testEvaluationResultRecall() {
        ModelEvaluator.EvaluationResult result = new ModelEvaluator.EvaluationResult();
        result.recall = 0.88;
        assertEquals(0.88, result.recall, 0.001);
    }

    @Test
    void testClassMetricsRecall() {
        ModelEvaluator.ClassMetrics metrics = new ModelEvaluator.ClassMetrics();
        metrics.recall = 0.75;
        assertEquals(0.75, metrics.recall, 0.001);
    }
}`,

	hint1: 'Use Evaluation class for multi-class, ROC for binary classification',
	hint2: 'eval.precision(i) gives precision for class i',

	whyItMatters: `Proper evaluation is crucial for understanding model performance:

- **Accuracy**: Overall correctness
- **Precision/Recall**: Important for imbalanced datasets
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

Metrics guide model improvement decisions.`,

	translations: {
		ru: {
			title: 'Оценка модели',
			description: `# Оценка модели

Оценивайте производительность модели с использованием различных метрик в DL4J.

## Задача

Реализуйте утилиты оценки:
- Метрики классификации (accuracy, precision, recall, F1)
- Матрица ошибок
- ROC кривые и AUC

## Пример

\`\`\`java
Evaluation eval = model.evaluate(testIterator);

System.out.println("Accuracy: " + eval.accuracy());
System.out.println("Precision: " + eval.precision());
System.out.println("Recall: " + eval.recall());
System.out.println("F1: " + eval.f1());
System.out.println(eval.confusionToString());
\`\`\``,
			hint1: 'Используйте Evaluation для многоклассовой, ROC для бинарной классификации',
			hint2: 'eval.precision(i) дает precision для класса i',
			whyItMatters: `Правильная оценка критична для понимания качества модели:

- **Accuracy**: Общая правильность
- **Precision/Recall**: Важны для несбалансированных датасетов
- **F1 Score**: Гармоническое среднее precision и recall
- **Матрица ошибок**: Детальный анализ ошибок`,
		},
		uz: {
			title: 'Modelni baholash',
			description: `# Modelni baholash

DL4J da turli metrikalar yordamida model samaradorligini baholang.

## Topshiriq

Baholash yordamchilarini amalga oshiring:
- Klassifikatsiya metrikalari (accuracy, precision, recall, F1)
- Confusion matritsasi
- ROC egri chiziqlari va AUC

## Misol

\`\`\`java
Evaluation eval = model.evaluate(testIterator);

System.out.println("Accuracy: " + eval.accuracy());
System.out.println("Precision: " + eval.precision());
System.out.println("Recall: " + eval.recall());
System.out.println("F1: " + eval.f1());
System.out.println(eval.confusionToString());
\`\`\``,
			hint1: "Ko'p sinfli uchun Evaluation, ikkilik klassifikatsiya uchun ROC dan foydalaning",
			hint2: "eval.precision(i) i sinfi uchun precision beradi",
			whyItMatters: `To'g'ri baholash model samaradorligini tushunish uchun muhim:

- **Accuracy**: Umumiy to'g'rilik
- **Precision/Recall**: Muvozanatsiz datasetlar uchun muhim
- **F1 Score**: Precision va recall ning garmonik o'rtachasi
- **Confusion Matrix**: Batafsil xato tahlili`,
		},
	},
};

export default task;
