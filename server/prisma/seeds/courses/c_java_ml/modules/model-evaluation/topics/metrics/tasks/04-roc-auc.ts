import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-roc-auc',
	title: 'ROC Curve and AUC',
	difficulty: 'medium',
	tags: ['metrics', 'roc', 'auc', 'classification'],
	estimatedTime: '25m',
	isPremium: false,
	order: 4,
	description: `# ROC Curve and AUC

Calculate ROC curves and Area Under Curve for binary classification.

## Task

Implement ROC-AUC analysis:
- Calculate ROC points at different thresholds
- Compute AUC score
- Interpret ROC curve

## Example

\`\`\`java
ROC roc = new ROC();
roc.eval(labels, predictions);
double auc = roc.calculateAUC();
\`\`\``,

	initialCode: `import org.deeplearning4j.eval.ROC;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;

public class RocAucCalculator {

    /**
     */
    public static double calculateAUC(INDArray labels, INDArray predictions) {
        return 0.0;
    }

    /**
     */
    public static double calculateTPR(int tp, int fn) {
        return 0.0;
    }

    /**
     */
    public static double calculateFPR(int fp, int tn) {
        return 0.0;
    }

    /**
     */
    public static String interpretAUC(double auc) {
        return "";
    }
}`,

	solutionCode: `import org.deeplearning4j.eval.ROC;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;
import java.util.ArrayList;

public class RocAucCalculator {

    /**
     * Calculate AUC for binary classification.
     */
    public static double calculateAUC(INDArray labels, INDArray predictions) {
        ROC roc = new ROC();
        roc.eval(labels, predictions);
        return roc.calculateAUC();
    }

    /**
     * Calculate True Positive Rate (Recall/Sensitivity).
     */
    public static double calculateTPR(int tp, int fn) {
        if (tp + fn == 0) return 0.0;
        return (double) tp / (tp + fn);
    }

    /**
     * Calculate False Positive Rate.
     */
    public static double calculateFPR(int fp, int tn) {
        if (fp + tn == 0) return 0.0;
        return (double) fp / (fp + tn);
    }

    /**
     * Interpret AUC score.
     */
    public static String interpretAUC(double auc) {
        if (auc >= 0.9) return "Excellent";
        if (auc >= 0.8) return "Good";
        if (auc >= 0.7) return "Fair";
        if (auc >= 0.6) return "Poor";
        return "Fail (worse than random)";
    }

    /**
     * Calculate ROC points at different thresholds.
     */
    public static List<double[]> calculateROCPoints(
            double[] scores, int[] labels, double[] thresholds) {
        List<double[]> points = new ArrayList<>();

        for (double threshold : thresholds) {
            int tp = 0, fp = 0, tn = 0, fn = 0;

            for (int i = 0; i < scores.length; i++) {
                int predicted = scores[i] >= threshold ? 1 : 0;
                if (labels[i] == 1 && predicted == 1) tp++;
                else if (labels[i] == 0 && predicted == 1) fp++;
                else if (labels[i] == 0 && predicted == 0) tn++;
                else fn++;
            }

            double fpr = calculateFPR(fp, tn);
            double tpr = calculateTPR(tp, fn);
            points.add(new double[] {fpr, tpr});
        }

        return points;
    }

    /**
     * Calculate AUC using trapezoidal rule.
     */
    public static double trapezoidalAUC(List<double[]> rocPoints) {
        double auc = 0.0;
        for (int i = 1; i < rocPoints.size(); i++) {
            double[] prev = rocPoints.get(i - 1);
            double[] curr = rocPoints.get(i);
            auc += (curr[0] - prev[0]) * (curr[1] + prev[1]) / 2;
        }
        return Math.abs(auc);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RocAucCalculatorTest {

    @Test
    void testCalculateTPR() {
        assertEquals(0.8, RocAucCalculator.calculateTPR(80, 20), 0.001);
        assertEquals(0.0, RocAucCalculator.calculateTPR(0, 0), 0.001);
    }

    @Test
    void testCalculateFPR() {
        assertEquals(0.1, RocAucCalculator.calculateFPR(10, 90), 0.001);
        assertEquals(0.0, RocAucCalculator.calculateFPR(0, 0), 0.001);
    }

    @Test
    void testInterpretAUC() {
        assertEquals("Excellent", RocAucCalculator.interpretAUC(0.95));
        assertEquals("Good", RocAucCalculator.interpretAUC(0.85));
        assertEquals("Fair", RocAucCalculator.interpretAUC(0.75));
        assertEquals("Poor", RocAucCalculator.interpretAUC(0.65));
    }

    @Test
    void testCalculateROCPoints() {
        double[] scores = {0.9, 0.4, 0.35, 0.8};
        int[] labels = {1, 0, 0, 1};
        double[] thresholds = {0.0, 0.5, 1.0};

        var points = RocAucCalculator.calculateROCPoints(scores, labels, thresholds);
        assertEquals(3, points.size());
    }

    @Test
    void testInterpretAUCFail() {
        assertEquals("Fail (worse than random)", RocAucCalculator.interpretAUC(0.4));
    }

    @Test
    void testTPRPerfect() {
        assertEquals(1.0, RocAucCalculator.calculateTPR(100, 0), 0.001);
    }

    @Test
    void testFPRPerfect() {
        assertEquals(0.0, RocAucCalculator.calculateFPR(0, 100), 0.001);
    }

    @Test
    void testTrapezoidalAUC() {
        var points = new java.util.ArrayList<double[]>();
        points.add(new double[]{0.0, 0.0});
        points.add(new double[]{0.5, 0.5});
        points.add(new double[]{1.0, 1.0});
        double auc = RocAucCalculator.trapezoidalAUC(points);
        assertEquals(0.5, auc, 0.01);
    }

    @Test
    void testROCPointsNotEmpty() {
        double[] scores = {0.9, 0.1, 0.8, 0.2};
        int[] labels = {1, 0, 1, 0};
        double[] thresholds = {0.3, 0.5, 0.7};
        var points = RocAucCalculator.calculateROCPoints(scores, labels, thresholds);
        assertFalse(points.isEmpty());
    }

    @Test
    void testTPRZeroCases() {
        assertEquals(0.0, RocAucCalculator.calculateTPR(0, 100), 0.001);
    }
}`,

	hint1: 'TPR = TP / (TP + FN), also called Sensitivity or Recall',
	hint2: 'AUC of 0.5 means random performance, 1.0 is perfect',

	whyItMatters: `ROC-AUC is the standard for binary classification:

- **Threshold independent**: Evaluates across all classification thresholds
- **Imbalance robust**: Works well with imbalanced datasets
- **Interpretable**: Easy to understand probability of correct ranking
- **Comparison**: Standard metric for comparing classifiers

ROC-AUC is essential for model selection in binary classification.`,

	translations: {
		ru: {
			title: 'ROC кривая и AUC',
			description: `# ROC кривая и AUC

Вычисляйте ROC кривые и площадь под кривой для бинарной классификации.

## Задача

Реализуйте анализ ROC-AUC:
- Вычисление точек ROC при разных порогах
- Расчет AUC метрики
- Интерпретация ROC кривой

## Пример

\`\`\`java
ROC roc = new ROC();
roc.eval(labels, predictions);
double auc = roc.calculateAUC();
\`\`\``,
			hint1: 'TPR = TP / (TP + FN), также называется Sensitivity или Recall',
			hint2: 'AUC 0.5 означает случайную производительность, 1.0 идеальную',
			whyItMatters: `ROC-AUC стандарт для бинарной классификации:

- **Независим от порога**: Оценивает по всем порогам классификации
- **Устойчив к дисбалансу**: Хорошо работает с несбалансированными данными
- **Интерпретируемый**: Легко понять вероятность правильного ранжирования
- **Сравнение**: Стандартная метрика для сравнения классификаторов`,
		},
		uz: {
			title: 'ROC egri chizig\'i va AUC',
			description: `# ROC egri chizig'i va AUC

Binar klassifikatsiya uchun ROC egri chiziqlarini va egri chiziq ostidagi maydonni hisoblang.

## Topshiriq

ROC-AUC tahlilini amalga oshiring:
- Turli chegaralarda ROC nuqtalarini hisoblash
- AUC metrikasini hisoblash
- ROC egri chizig'ini talqin qilish

## Misol

\`\`\`java
ROC roc = new ROC();
roc.eval(labels, predictions);
double auc = roc.calculateAUC();
\`\`\``,
			hint1: "TPR = TP / (TP + FN), Sensitivity yoki Recall deb ham ataladi",
			hint2: "AUC 0.5 tasodifiy samaradorlikni, 1.0 mukammallikni bildiradi",
			whyItMatters: `ROC-AUC binar klassifikatsiya uchun standart:

- **Chegaradan mustaqil**: Barcha klassifikatsiya chegaralarida baholaydi
- **Muvozanatsizlikka barqaror**: Muvozanatsiz ma'lumotlar bilan yaxshi ishlaydi
- **Tushunarli**: To'g'ri tartiblash ehtimolini tushunish oson
- **Taqqoslash**: Klassifikatorlarni taqqoslash uchun standart metrika`,
		},
	},
};

export default task;
