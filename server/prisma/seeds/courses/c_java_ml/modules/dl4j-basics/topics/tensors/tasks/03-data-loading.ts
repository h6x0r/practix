import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-data-loading',
	title: 'Data Loading',
	difficulty: 'medium',
	tags: ['nd4j', 'data', 'datavec'],
	estimatedTime: '20m',
	isPremium: false,
	order: 3,
	description: `# Data Loading with DataVec

Load and preprocess data for DL4J models using DataVec.

## Task

Implement data loading utilities:
- Load CSV data
- Split into train/test sets
- Create DataSet iterators

## Example

\`\`\`java
RecordReader reader = new CSVRecordReader(0, ',');
reader.initialize(new FileSplit(new File("data.csv")));

DataSetIterator iterator = new RecordReaderDataSetIterator(
    reader, batchSize, labelIndex, numClasses
);
\`\`\``,

	initialCode: `import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import java.io.File;

public class DataLoader {

    /**
     */
    public static DataSetIterator loadCSV(String filePath, int batchSize,
                                          int labelIndex, int numClasses) {
        return null;
    }

    /**
     */
    public static DataSet[] trainTestSplit(DataSet dataset, double trainRatio) {
        return null;
    }

    /**
     */
    public static DataSet normalizeMinMax(DataSet dataset) {
        return null;
    }

    /**
     */
    public static DataSet shuffle(DataSet dataset) {
        return null;
    }
}`,

	solutionCode: `import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import java.io.File;
import java.io.IOException;

public class DataLoader {

    /**
     * Create a DataSetIterator from CSV file.
     */
    public static DataSetIterator loadCSV(String filePath, int batchSize,
                                          int labelIndex, int numClasses) {
        try {
            RecordReader reader = new CSVRecordReader(0, ',');
            reader.initialize(new FileSplit(new File(filePath)));

            return new RecordReaderDataSetIterator(
                reader, batchSize, labelIndex, numClasses
            );
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Failed to load CSV: " + e.getMessage(), e);
        }
    }

    /**
     * Split dataset into train and test sets.
     */
    public static DataSet[] trainTestSplit(DataSet dataset, double trainRatio) {
        dataset.shuffle();
        SplitTestAndTrain split = dataset.splitTestAndTrain(trainRatio);

        return new DataSet[] {
            split.getTrain(),
            split.getTest()
        };
    }

    /**
     * Normalize dataset features to [0, 1] range.
     */
    public static DataSet normalizeMinMax(DataSet dataset) {
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fit(dataset);
        normalizer.transform(dataset);
        return dataset;
    }

    /**
     * Shuffle the dataset.
     */
    public static DataSet shuffle(DataSet dataset) {
        dataset.shuffle();
        return dataset;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class DataLoaderTest {

    @Test
    void testTrainTestSplit() {
        INDArray features = Nd4j.rand(100, 5);
        INDArray labels = Nd4j.zeros(100, 2);
        for (int i = 0; i < 100; i++) {
            labels.putScalar(i, i % 2, 1);
        }
        DataSet dataset = new DataSet(features, labels);

        DataSet[] split = DataLoader.trainTestSplit(dataset, 0.8);

        assertEquals(80, split[0].numExamples());
        assertEquals(20, split[1].numExamples());
    }

    @Test
    void testNormalizeMinMax() {
        INDArray features = Nd4j.create(new double[][]{{0, 100}, {50, 200}});
        INDArray labels = Nd4j.ones(2, 1);
        DataSet dataset = new DataSet(features, labels);

        DataSet normalized = DataLoader.normalizeMinMax(dataset);

        double min = normalized.getFeatures().minNumber().doubleValue();
        double max = normalized.getFeatures().maxNumber().doubleValue();
        assertTrue(min >= 0 && max <= 1);
    }

    @Test
    void testShuffle() {
        INDArray features = Nd4j.arange(0, 10).reshape(10, 1);
        INDArray labels = Nd4j.ones(10, 1);
        DataSet dataset = new DataSet(features, labels);

        INDArray original = dataset.getFeatures().dup();
        DataSet shuffled = DataLoader.shuffle(dataset);

        // After shuffle, order should likely change
        assertNotNull(shuffled);
    }

    @Test
    void testTrainTestSplitReturnsArray() {
        INDArray features = Nd4j.rand(50, 3);
        INDArray labels = Nd4j.zeros(50, 2);
        DataSet dataset = new DataSet(features, labels);
        DataSet[] split = DataLoader.trainTestSplit(dataset, 0.7);
        assertEquals(2, split.length);
    }

    @Test
    void testTrainTestSplitSumEquals100() {
        INDArray features = Nd4j.rand(100, 4);
        INDArray labels = Nd4j.ones(100, 1);
        DataSet dataset = new DataSet(features, labels);
        DataSet[] split = DataLoader.trainTestSplit(dataset, 0.6);
        assertEquals(100, split[0].numExamples() + split[1].numExamples());
    }

    @Test
    void testNormalizeMinMaxReturnsDataSet() {
        INDArray features = Nd4j.rand(20, 5);
        INDArray labels = Nd4j.ones(20, 1);
        DataSet dataset = new DataSet(features, labels);
        DataSet result = DataLoader.normalizeMinMax(dataset);
        assertNotNull(result);
        assertNotNull(result.getFeatures());
    }

    @Test
    void testNormalizeMinMaxPreservesShape() {
        INDArray features = Nd4j.rand(30, 6);
        INDArray labels = Nd4j.ones(30, 2);
        DataSet dataset = new DataSet(features, labels);
        DataSet normalized = DataLoader.normalizeMinMax(dataset);
        assertArrayEquals(new long[]{30, 6}, normalized.getFeatures().shape());
    }

    @Test
    void testShuffleReturnsDataSet() {
        INDArray features = Nd4j.rand(25, 4);
        INDArray labels = Nd4j.ones(25, 1);
        DataSet dataset = new DataSet(features, labels);
        DataSet shuffled = DataLoader.shuffle(dataset);
        assertInstanceOf(DataSet.class, shuffled);
    }

    @Test
    void testShufflePreservesSize() {
        INDArray features = Nd4j.rand(40, 3);
        INDArray labels = Nd4j.ones(40, 2);
        DataSet dataset = new DataSet(features, labels);
        DataSet shuffled = DataLoader.shuffle(dataset);
        assertEquals(40, shuffled.numExamples());
    }

    @Test
    void testTrainTestSplitNotNull() {
        INDArray features = Nd4j.rand(60, 4);
        INDArray labels = Nd4j.ones(60, 2);
        DataSet dataset = new DataSet(features, labels);
        DataSet[] split = DataLoader.trainTestSplit(dataset, 0.75);
        assertNotNull(split[0]);
        assertNotNull(split[1]);
    }
}`,

	hint1: 'Use CSVRecordReader for CSV files with header handling',
	hint2: 'NormalizerMinMaxScaler.fit() computes stats, transform() applies them',

	whyItMatters: `Data loading is the first step in any ML pipeline:

- **DataVec**: Powerful data ingestion library
- **Iterators**: Memory-efficient batch processing
- **Preprocessing**: Normalization improves training
- **Splitting**: Proper train/test splits prevent overfitting

Well-prepared data leads to better models.`,

	translations: {
		ru: {
			title: 'Загрузка данных',
			description: `# Загрузка данных с DataVec

Загрузка и предобработка данных для моделей DL4J с помощью DataVec.

## Задача

Реализуйте утилиты загрузки данных:
- Загрузка CSV данных
- Разделение на train/test наборы
- Создание DataSet итераторов

## Пример

\`\`\`java
RecordReader reader = new CSVRecordReader(0, ',');
reader.initialize(new FileSplit(new File("data.csv")));

DataSetIterator iterator = new RecordReaderDataSetIterator(
    reader, batchSize, labelIndex, numClasses
);
\`\`\``,
			hint1: 'Используйте CSVRecordReader для CSV файлов с обработкой заголовков',
			hint2: 'NormalizerMinMaxScaler.fit() вычисляет статистики, transform() применяет их',
			whyItMatters: `Загрузка данных - первый шаг любого ML pipeline:

- **DataVec**: Мощная библиотека для работы с данными
- **Итераторы**: Эффективная по памяти обработка батчей
- **Предобработка**: Нормализация улучшает обучение
- **Разделение**: Правильные train/test splits предотвращают переобучение`,
		},
		uz: {
			title: "Ma'lumotlarni yuklash",
			description: `# DataVec bilan ma'lumotlarni yuklash

DL4J modellari uchun DataVec yordamida ma'lumotlarni yuklash va oldindan qayta ishlash.

## Topshiriq

Ma'lumotlarni yuklash yordamchilarini amalga oshiring:
- CSV ma'lumotlarini yuklash
- Train/test to'plamlariga bo'lish
- DataSet iteratorlarini yaratish

## Misol

\`\`\`java
RecordReader reader = new CSVRecordReader(0, ',');
reader.initialize(new FileSplit(new File("data.csv")));

DataSetIterator iterator = new RecordReaderDataSetIterator(
    reader, batchSize, labelIndex, numClasses
);
\`\`\``,
			hint1: "Sarlavha bilan CSV fayllar uchun CSVRecordReader dan foydalaning",
			hint2: "NormalizerMinMaxScaler.fit() statistikani hisoblaydi, transform() ularni qo'llaydi",
			whyItMatters: `Ma'lumotlarni yuklash har qanday ML pipeline ning birinchi qadami:

- **DataVec**: Kuchli ma'lumotlarni qabul qilish kutubxonasi
- **Iteratorlar**: Xotira samarali batch qayta ishlash
- **Oldindan qayta ishlash**: Normalizatsiya o'qitishni yaxshilaydi
- **Bo'lish**: To'g'ri train/test bo'linishlar overfitting ni oldini oladi`,
		},
	},
};

export default task;
