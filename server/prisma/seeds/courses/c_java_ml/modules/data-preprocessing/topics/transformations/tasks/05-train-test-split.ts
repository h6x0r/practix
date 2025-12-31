import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-train-test-split',
	title: 'Train-Test Split',
	difficulty: 'easy',
	tags: ['preprocessing', 'split', 'validation'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Train-Test Split

Split datasets into training and testing sets.

## Task

Implement data splitting:
- Random train/test split
- Stratified split for classification
- Handle shuffling

## Example

\`\`\`java
SplitTestAndTrain split = data.splitTestAndTrain(0.8);
DataSet train = split.getTrain();
DataSet test = split.getTest();
\`\`\``,

	initialCode: `import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Random;

public class DataSplitter {

    /**
     * Split dataset with specified ratio.
     * @param ratio Fraction of data for training (0.0-1.0)
     */
    public static SplitTestAndTrain split(DataSet data, double ratio) {
        return null;
    }

    /**
     * Split with shuffling first.
     */
    public static SplitTestAndTrain splitWithShuffle(DataSet data, double ratio) {
        return null;
    }

    /**
     * Create train/validation/test splits.
     */
    public static DataSet[] threeWaySplit(DataSet data,
                                           double trainRatio,
                                           double valRatio) {
        return null;
    }
}`,

	solutionCode: `import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Random;

public class DataSplitter {

    /**
     * Split dataset with specified ratio.
     * @param ratio Fraction of data for training (0.0-1.0)
     */
    public static SplitTestAndTrain split(DataSet data, double ratio) {
        return data.splitTestAndTrain(ratio);
    }

    /**
     * Split with shuffling first.
     */
    public static SplitTestAndTrain splitWithShuffle(DataSet data, double ratio) {
        data.shuffle();
        return data.splitTestAndTrain(ratio);
    }

    /**
     * Create train/validation/test splits.
     */
    public static DataSet[] threeWaySplit(DataSet data,
                                           double trainRatio,
                                           double valRatio) {
        data.shuffle();

        // First split: train vs (val + test)
        SplitTestAndTrain firstSplit = data.splitTestAndTrain(trainRatio);
        DataSet train = firstSplit.getTrain();
        DataSet remaining = firstSplit.getTest();

        // Second split: val vs test from remaining
        double valFromRemaining = valRatio / (1.0 - trainRatio);
        SplitTestAndTrain secondSplit = remaining.splitTestAndTrain(valFromRemaining);

        return new DataSet[] {
            train,
            secondSplit.getTrain(),  // validation
            secondSplit.getTest()     // test
        };
    }

    /**
     * Split with fixed random seed for reproducibility.
     */
    public static SplitTestAndTrain splitReproducible(DataSet data,
                                                        double ratio,
                                                        long seed) {
        data.shuffle(seed);
        return data.splitTestAndTrain(ratio);
    }

    /**
     * Get split sizes.
     */
    public static int[] getSplitSizes(int totalSize, double trainRatio) {
        int trainSize = (int) (totalSize * trainRatio);
        int testSize = totalSize - trainSize;
        return new int[] {trainSize, testSize};
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class DataSplitterTest {

    @Test
    void testSplit() {
        DataSet data = createTestDataset(100);
        SplitTestAndTrain split = DataSplitter.split(data, 0.8);

        assertEquals(80, split.getTrain().numExamples());
        assertEquals(20, split.getTest().numExamples());
    }

    @Test
    void testThreeWaySplit() {
        DataSet data = createTestDataset(100);
        DataSet[] splits = DataSplitter.threeWaySplit(data, 0.6, 0.2);

        assertEquals(60, splits[0].numExamples());  // train
        assertEquals(20, splits[1].numExamples());  // validation
        assertEquals(20, splits[2].numExamples());  // test
    }

    @Test
    void testGetSplitSizes() {
        int[] sizes = DataSplitter.getSplitSizes(100, 0.8);
        assertEquals(80, sizes[0]);
        assertEquals(20, sizes[1]);
    }

    @Test
    void testSplitWithShuffle() {
        DataSet data = createTestDataset(100);
        SplitTestAndTrain split = DataSplitter.splitWithShuffle(data, 0.7);
        assertEquals(70, split.getTrain().numExamples());
        assertEquals(30, split.getTest().numExamples());
    }

    @Test
    void testSplitReproducible() {
        DataSet data1 = createTestDataset(50);
        DataSet data2 = createTestDataset(50);
        SplitTestAndTrain split1 = DataSplitter.splitReproducible(data1, 0.8, 42L);
        SplitTestAndTrain split2 = DataSplitter.splitReproducible(data2, 0.8, 42L);
        assertEquals(split1.getTrain().numExamples(), split2.getTrain().numExamples());
    }

    @Test
    void testSplitHalfHalf() {
        DataSet data = createTestDataset(100);
        SplitTestAndTrain split = DataSplitter.split(data, 0.5);
        assertEquals(50, split.getTrain().numExamples());
        assertEquals(50, split.getTest().numExamples());
    }

    @Test
    void testGetSplitSizesDifferentRatio() {
        int[] sizes = DataSplitter.getSplitSizes(100, 0.7);
        assertEquals(70, sizes[0]);
        assertEquals(30, sizes[1]);
    }

    @Test
    void testSplitSmallDataset() {
        DataSet data = createTestDataset(10);
        SplitTestAndTrain split = DataSplitter.split(data, 0.8);
        assertEquals(8, split.getTrain().numExamples());
        assertEquals(2, split.getTest().numExamples());
    }

    @Test
    void testThreeWaySplitDifferentRatios() {
        DataSet data = createTestDataset(100);
        DataSet[] splits = DataSplitter.threeWaySplit(data, 0.7, 0.15);
        assertEquals(70, splits[0].numExamples());
        assertEquals(15, splits[1].numExamples());
        assertEquals(15, splits[2].numExamples());
    }

    private DataSet createTestDataset(int n) {
        return new DataSet(
            Nd4j.rand(n, 10),
            Nd4j.rand(n, 2)
        );
    }
}`,

	hint1: 'Use splitTestAndTrain() for simple splits',
	hint2: 'Always shuffle before splitting to avoid ordering bias',

	whyItMatters: `Proper data splitting prevents data leakage:

- **Unbiased evaluation**: Test set never seen during training
- **Validation set**: Tune hyperparameters without overfitting to test
- **Shuffling**: Prevent ordering bias
- **Reproducibility**: Use seeds for consistent splits

Correct splitting is fundamental to ML experiments.`,

	translations: {
		ru: {
			title: 'Разбиение на train-test',
			description: `# Разбиение на train-test

Разделяйте датасеты на обучающую и тестовую выборки.

## Задача

Реализуйте разбиение данных:
- Случайное train/test разбиение
- Стратифицированное разбиение для классификации
- Обработка перемешивания

## Пример

\`\`\`java
SplitTestAndTrain split = data.splitTestAndTrain(0.8);
DataSet train = split.getTrain();
DataSet test = split.getTest();
\`\`\``,
			hint1: 'Используйте splitTestAndTrain() для простых разбиений',
			hint2: 'Всегда перемешивайте перед разбиением для избежания смещения',
			whyItMatters: `Правильное разбиение предотвращает утечку данных:

- **Несмещенная оценка**: Тестовый набор никогда не виден при обучении
- **Валидационный набор**: Подбор гиперпараметров без переобучения на тест
- **Перемешивание**: Предотвращение смещения порядка
- **Воспроизводимость**: Используйте seed для консистентных разбиений`,
		},
		uz: {
			title: 'Train-test bo\'lish',
			description: `# Train-test bo'lish

Datasetlarni o'qitish va test to'plamlariga bo'ling.

## Topshiriq

Ma'lumot bo'lishni amalga oshiring:
- Tasodifiy train/test bo'lish
- Klassifikatsiya uchun stratifitsirlangan bo'lish
- Aralashtirish bilan ishlash

## Misol

\`\`\`java
SplitTestAndTrain split = data.splitTestAndTrain(0.8);
DataSet train = split.getTrain();
DataSet test = split.getTest();
\`\`\``,
			hint1: "Oddiy bo'lishlar uchun splitTestAndTrain() dan foydalaning",
			hint2: "Tartib bias dan qochish uchun bo'lishdan oldin doimo aralashtiring",
			whyItMatters: `To'g'ri bo'lish data leakage ni oldini oladi:

- **Xolis baholash**: Test to'plami hech qachon o'qitish paytida ko'rilmaydi
- **Validatsiya to'plami**: Testga overfit bo'lmasdan giperparametrlarni sozlash
- **Aralashtirish**: Tartib bias ni oldini olish
- **Takroriylik**: Izchil bo'lishlar uchun seedlardan foydalaning`,
		},
	},
};

export default task;
