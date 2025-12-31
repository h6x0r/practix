import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-cross-validation',
	title: 'Cross-Validation',
	difficulty: 'medium',
	tags: ['validation', 'cv', 'evaluation'],
	estimatedTime: '25m',
	isPremium: false,
	order: 2,
	description: `# Cross-Validation

Implement k-fold cross-validation for robust model evaluation.

## Task

Build cross-validation:
- Split data into k folds
- Train and evaluate on each fold
- Aggregate performance metrics

## Example

\`\`\`java
// 5-fold cross-validation
List<DataSet> folds = createFolds(data, 5);

for (int i = 0; i < 5; i++) {
    DataSet test = folds.get(i);
    DataSet train = mergeFolds(folds, i);
    // Train and evaluate
}
\`\`\``,

	initialCode: `import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.util.List;
import java.util.ArrayList;

public class CrossValidator {

    /**
     * Split dataset into k folds.
     */
    public static List<DataSet> createFolds(DataSet data, int k) {
        return null;
    }

    /**
     * Merge all folds except the test fold.
     */
    public static DataSet mergeTrainFolds(List<DataSet> folds, int testFoldIndex) {
        return null;
    }

    /**
     * Perform k-fold cross-validation and return average accuracy.
     */
    public static double crossValidate(DataSet data, int k,
                                         ModelTrainer trainer) {
        return 0.0;
    }
}

interface ModelTrainer {
    double trainAndEvaluate(DataSet train, DataSet test);
}`,

	solutionCode: `import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

public class CrossValidator {

    /**
     * Split dataset into k folds.
     */
    public static List<DataSet> createFolds(DataSet data, int k) {
        List<DataSet> folds = new ArrayList<>();

        int numExamples = (int) data.numExamples();
        int foldSize = numExamples / k;

        // Shuffle data first
        data.shuffle();

        for (int i = 0; i < k; i++) {
            int start = i * foldSize;
            int end = (i == k - 1) ? numExamples : (i + 1) * foldSize;

            INDArray features = data.getFeatures().get(
                Nd4j.createFromArray(start, end), Nd4j.all()
            );
            INDArray labels = data.getLabels().get(
                Nd4j.createFromArray(start, end), Nd4j.all()
            );

            folds.add(new DataSet(features, labels));
        }

        return folds;
    }

    /**
     * Merge all folds except the test fold.
     */
    public static DataSet mergeTrainFolds(List<DataSet> folds, int testFoldIndex) {
        List<INDArray> featuresList = new ArrayList<>();
        List<INDArray> labelsList = new ArrayList<>();

        for (int i = 0; i < folds.size(); i++) {
            if (i != testFoldIndex) {
                featuresList.add(folds.get(i).getFeatures());
                labelsList.add(folds.get(i).getLabels());
            }
        }

        INDArray features = Nd4j.vstack(featuresList);
        INDArray labels = Nd4j.vstack(labelsList);

        return new DataSet(features, labels);
    }

    /**
     * Perform k-fold cross-validation and return average accuracy.
     */
    public static double crossValidate(DataSet data, int k,
                                         ModelTrainer trainer) {
        List<DataSet> folds = createFolds(data, k);
        double totalScore = 0.0;

        for (int i = 0; i < k; i++) {
            DataSet testFold = folds.get(i);
            DataSet trainFolds = mergeTrainFolds(folds, i);

            double score = trainer.trainAndEvaluate(trainFolds, testFold);
            totalScore += score;
        }

        return totalScore / k;
    }

    /**
     * Stratified k-fold (maintains class distribution).
     */
    public static List<DataSet> createStratifiedFolds(DataSet data, int k) {
        // This is simplified - full implementation would group by class
        return createFolds(data, k);
    }
}

interface ModelTrainer {
    double trainAndEvaluate(DataSet train, DataSet test);
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class CrossValidatorTest {

    @Test
    void testCreateFolds() {
        DataSet data = createTestDataset(100);
        List<DataSet> folds = CrossValidator.createFolds(data, 5);

        assertEquals(5, folds.size());
        // Each fold should have ~20 examples
        assertTrue(folds.get(0).numExamples() >= 18);
    }

    @Test
    void testMergeTrainFolds() {
        DataSet data = createTestDataset(50);
        List<DataSet> folds = CrossValidator.createFolds(data, 5);

        DataSet train = CrossValidator.mergeTrainFolds(folds, 0);

        // Train should have 4 folds worth of data
        assertTrue(train.numExamples() >= 38);
    }

    @Test
    void testCrossValidate() {
        DataSet data = createTestDataset(100);

        double avgScore = CrossValidator.crossValidate(data, 5, (train, test) -> {
            // Mock trainer returns random score
            return 0.8;
        });

        assertEquals(0.8, avgScore, 0.01);
    }

    private DataSet createTestDataset(int n) {
        return new DataSet(
            Nd4j.rand(n, 10),
            Nd4j.rand(n, 2)
        );
    }

    @Test
    void testCreateFoldsReturnsCorrectSize() {
        DataSet data = createTestDataset(60);
        List<DataSet> folds = CrossValidator.createFolds(data, 3);
        assertEquals(3, folds.size());
    }

    @Test
    void testCreateFoldsNotNull() {
        DataSet data = createTestDataset(50);
        List<DataSet> folds = CrossValidator.createFolds(data, 5);
        assertNotNull(folds);
        for (DataSet fold : folds) {
            assertNotNull(fold);
        }
    }

    @Test
    void testMergeTrainFoldsNotNull() {
        DataSet data = createTestDataset(40);
        List<DataSet> folds = CrossValidator.createFolds(data, 4);
        DataSet train = CrossValidator.mergeTrainFolds(folds, 1);
        assertNotNull(train);
    }

    @Test
    void testMergeTrainFoldsExcludesTestFold() {
        DataSet data = createTestDataset(100);
        List<DataSet> folds = CrossValidator.createFolds(data, 5);
        DataSet train = CrossValidator.mergeTrainFolds(folds, 2);
        // Train should have 80% of data (4 out of 5 folds)
        assertTrue(train.numExamples() >= 75);
    }

    @Test
    void testCrossValidateReturnsScore() {
        DataSet data = createTestDataset(50);
        double score = CrossValidator.crossValidate(data, 5, (train, test) -> 0.9);
        assertTrue(score >= 0 && score <= 1);
    }

    @Test
    void testStratifiedFolds() {
        DataSet data = createTestDataset(100);
        List<DataSet> folds = CrossValidator.createStratifiedFolds(data, 5);
        assertEquals(5, folds.size());
    }

    @Test
    void testCrossValidateAveragesScores() {
        DataSet data = createTestDataset(100);
        double[] scores = {0.7, 0.8, 0.9, 0.75, 0.85};
        final int[] i = {0};
        double avgScore = CrossValidator.crossValidate(data, 5, (train, test) -> {
            return scores[i[0]++ % 5];
        });
        assertEquals(0.8, avgScore, 0.01);
    }
}`,

	hint1: 'Shuffle data before splitting into folds',
	hint2: 'Use Nd4j.vstack() to combine multiple folds',

	whyItMatters: `Cross-validation provides robust evaluation:

- **Less variance**: Results less dependent on train/test split
- **Use all data**: Every sample used for both training and testing
- **Detect overfitting**: Compare train vs validation performance
- **Hyperparameter selection**: Compare configurations fairly

CV is essential for reliable model comparison.`,

	translations: {
		ru: {
			title: 'Кросс-валидация',
			description: `# Кросс-валидация

Реализуйте k-fold кросс-валидацию для надежной оценки модели.

## Задача

Создайте кросс-валидацию:
- Разделите данные на k фолдов
- Обучите и оцените на каждом фолде
- Агрегируйте метрики производительности

## Пример

\`\`\`java
// 5-fold cross-validation
List<DataSet> folds = createFolds(data, 5);

for (int i = 0; i < 5; i++) {
    DataSet test = folds.get(i);
    DataSet train = mergeFolds(folds, i);
    // Train and evaluate
}
\`\`\``,
			hint1: 'Перемешайте данные перед разделением на фолды',
			hint2: 'Используйте Nd4j.vstack() для объединения нескольких фолдов',
			whyItMatters: `Кросс-валидация обеспечивает надежную оценку:

- **Меньше дисперсия**: Результаты менее зависят от разбиения
- **Использование всех данных**: Каждый образец для обучения и тестирования
- **Обнаружение переобучения**: Сравнение train vs validation
- **Выбор гиперпараметров**: Честное сравнение конфигураций`,
		},
		uz: {
			title: 'Kross-validatsiya',
			description: `# Kross-validatsiya

Ishonchli model baholash uchun k-fold kross-validatsiyasini amalga oshiring.

## Topshiriq

Kross-validatsiya yarating:
- Ma'lumotlarni k foldga bo'ling
- Har bir foldda o'rgating va baholang
- Samaradorlik metrikalarini yig'ing

## Misol

\`\`\`java
// 5-fold cross-validation
List<DataSet> folds = createFolds(data, 5);

for (int i = 0; i < 5; i++) {
    DataSet test = folds.get(i);
    DataSet train = mergeFolds(folds, i);
    // Train and evaluate
}
\`\`\``,
			hint1: "Foldlarga bo'lishdan oldin ma'lumotlarni aralashtiring",
			hint2: "Bir nechta foldlarni birlashtirish uchun Nd4j.vstack() dan foydalaning",
			whyItMatters: `Kross-validatsiya ishonchli baholashni ta'minlaydi:

- **Kamroq dispersiya**: Natijalar train/test bo'linishiga kamroq bog'liq
- **Barcha ma'lumotlardan foydalanish**: Har bir namuna o'qitish va test uchun ishlatiladi
- **Overfittingni aniqlash**: Train vs validation samaradorligini solishtirish
- **Giperparametr tanlash**: Konfiguratsiyalarni adolatli solishtirish`,
		},
	},
};

export default task;
