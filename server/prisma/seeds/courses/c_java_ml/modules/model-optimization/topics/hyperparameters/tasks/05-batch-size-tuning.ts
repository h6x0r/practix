import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-batch-size-tuning',
	title: 'Batch Size Tuning',
	difficulty: 'easy',
	tags: ['batch-size', 'mini-batch', 'hyperparameters'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Batch Size Tuning

Configure batch size for optimal training performance.

## Task

Implement batch size configuration:
- Set mini-batch size for training
- Understand trade-offs of different sizes
- Configure for memory constraints

## Example

\`\`\`java
DataSetIterator iterator = new MnistDataSetIterator(32, true, 12345);
model.fit(iterator);
\`\`\``,

	initialCode: `import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.IOException;

public class BatchSizeTuner {

    /**
     * Create iterator with specified batch size.
     */
    public static DataSetIterator createIterator(int batchSize) throws IOException {
        return null;
    }

    /**
     * Calculate number of batches for dataset.
     */
    public static int calculateNumBatches(int datasetSize, int batchSize) {
        return 0;
    }

    /**
     * Suggest batch size based on available memory.
     */
    public static int suggestBatchSize(long availableMemoryMB, int featureSize) {
        return 32;
    }
}`,

	solutionCode: `import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.IOException;

public class BatchSizeTuner {

    /**
     * Create iterator with specified batch size.
     */
    public static DataSetIterator createIterator(int batchSize) throws IOException {
        return new MnistDataSetIterator(batchSize, true, 12345);
    }

    /**
     * Calculate number of batches for dataset.
     */
    public static int calculateNumBatches(int datasetSize, int batchSize) {
        return (int) Math.ceil((double) datasetSize / batchSize);
    }

    /**
     * Suggest batch size based on available memory.
     */
    public static int suggestBatchSize(long availableMemoryMB, int featureSize) {
        // Rough estimation: each sample needs featureSize * 4 bytes (float)
        // Plus gradients and activations (multiply by 3)
        long bytesPerSample = (long) featureSize * 4 * 3;
        long availableBytes = availableMemoryMB * 1024 * 1024;

        int maxBatchSize = (int) (availableBytes / bytesPerSample);

        // Round down to nearest power of 2
        int batchSize = 1;
        while (batchSize * 2 <= maxBatchSize && batchSize < 256) {
            batchSize *= 2;
        }

        return Math.max(16, batchSize);
    }

    /**
     * Common batch sizes to try.
     */
    public static int[] getBatchSizeGrid() {
        return new int[] {16, 32, 64, 128, 256};
    }

    /**
     * Check if batch size is power of 2 (GPU optimal).
     */
    public static boolean isOptimalForGPU(int batchSize) {
        return batchSize > 0 && (batchSize & (batchSize - 1)) == 0;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import static org.junit.jupiter.api.Assertions.*;

public class BatchSizeTunerTest {

    @Test
    void testCreateIterator() throws Exception {
        DataSetIterator iterator = BatchSizeTuner.createIterator(32);
        assertNotNull(iterator);
    }

    @Test
    void testCalculateNumBatches() {
        assertEquals(4, BatchSizeTuner.calculateNumBatches(100, 32));
        assertEquals(2, BatchSizeTuner.calculateNumBatches(64, 32));
        assertEquals(1, BatchSizeTuner.calculateNumBatches(32, 32));
    }

    @Test
    void testSuggestBatchSize() {
        int suggested = BatchSizeTuner.suggestBatchSize(1024, 784);
        assertTrue(suggested >= 16);
        assertTrue(suggested <= 256);
    }

    @Test
    void testIsOptimalForGPU() {
        assertTrue(BatchSizeTuner.isOptimalForGPU(32));
        assertTrue(BatchSizeTuner.isOptimalForGPU(64));
        assertFalse(BatchSizeTuner.isOptimalForGPU(48));
    }

    @Test
    void testBatchSizeGrid() {
        int[] grid = BatchSizeTuner.getBatchSizeGrid();
        assertEquals(5, grid.length);
        assertEquals(16, grid[0]);
    }

    @Test
    void testBatchSizeGridValues() {
        int[] grid = BatchSizeTuner.getBatchSizeGrid();
        assertEquals(256, grid[4]);
    }

    @Test
    void testCalculateNumBatchesCeiling() {
        assertEquals(3, BatchSizeTuner.calculateNumBatches(65, 32));
    }

    @Test
    void testSuggestBatchSizeMinimum() {
        int suggested = BatchSizeTuner.suggestBatchSize(1, 784);
        assertTrue(suggested >= 16);
    }

    @Test
    void testIsOptimalForGPU128() {
        assertTrue(BatchSizeTuner.isOptimalForGPU(128));
        assertTrue(BatchSizeTuner.isOptimalForGPU(256));
    }

    @Test
    void testIsOptimalForGPUNonPowerOf2() {
        assertFalse(BatchSizeTuner.isOptimalForGPU(100));
        assertFalse(BatchSizeTuner.isOptimalForGPU(50));
    }
}`,

	hint1: 'Powers of 2 (32, 64, 128) are optimal for GPU',
	hint2: 'Larger batches train faster but may need more memory',

	whyItMatters: `Batch size significantly affects training:

- **Memory usage**: Larger batches need more GPU memory
- **Training speed**: Larger batches can be parallelized better
- **Convergence**: Smaller batches add noise that can help generalization
- **GPU efficiency**: Powers of 2 are optimal for hardware

Choosing the right batch size balances speed and performance.`,

	translations: {
		ru: {
			title: 'Настройка размера батча',
			description: `# Настройка размера батча

Настройте размер батча для оптимальной производительности обучения.

## Задача

Реализуйте конфигурацию размера батча:
- Установка размера мини-батча для обучения
- Понимание компромиссов разных размеров
- Настройка под ограничения памяти

## Пример

\`\`\`java
DataSetIterator iterator = new MnistDataSetIterator(32, true, 12345);
model.fit(iterator);
\`\`\``,
			hint1: 'Степени 2 (32, 64, 128) оптимальны для GPU',
			hint2: 'Большие батчи обучаются быстрее, но требуют больше памяти',
			whyItMatters: `Размер батча значительно влияет на обучение:

- **Использование памяти**: Большие батчи требуют больше GPU памяти
- **Скорость обучения**: Большие батчи лучше параллелизуются
- **Сходимость**: Малые батчи добавляют шум помогающий обобщению
- **Эффективность GPU**: Степени 2 оптимальны для железа`,
		},
		uz: {
			title: "Batch o'lchamini sozlash",
			description: `# Batch o'lchamini sozlash

Optimal o'qitish samaradorligi uchun batch o'lchamini sozlang.

## Topshiriq

Batch o'lchamini sozlashni amalga oshiring:
- O'qitish uchun mini-batch o'lchamini o'rnatish
- Turli o'lchamlarning kelishuvlarini tushunish
- Xotira cheklovlari uchun sozlash

## Misol

\`\`\`java
DataSetIterator iterator = new MnistDataSetIterator(32, true, 12345);
model.fit(iterator);
\`\`\``,
			hint1: "2 ning darajalari (32, 64, 128) GPU uchun optimal",
			hint2: "Kattaroq batchlar tezroq o'qitiladi, lekin ko'proq xotira kerak",
			whyItMatters: `Batch o'lchami o'qitishga sezilarli ta'sir qiladi:

- **Xotira ishlatish**: Katta batchlar ko'proq GPU xotirasi kerak
- **O'qitish tezligi**: Katta batchlar yaxshiroq parallellanadi
- **Konvergentsiya**: Kichik batchlar umumlashtirishga yordam beruvchi shovqin qo'shadi
- **GPU samaradorligi**: 2 ning darajalari apparat uchun optimal`,
		},
	},
};

export default task;
