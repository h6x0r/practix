import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-missing-values',
	title: 'Handling Missing Values',
	difficulty: 'easy',
	tags: ['datavec', 'preprocessing', 'missing-values'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Handling Missing Values

Handle missing data in your datasets.

## Task

Implement missing value strategies:
- Remove rows with missing values
- Impute with mean/median
- Fill with custom values

## Example

\`\`\`java
TransformProcess tp = new TransformProcess.Builder(schema)
    .transform(new ReplaceEmptyWithValueTransform("age", 30.0))
    .build();
\`\`\``,

	initialCode: `import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MissingValueHandler {

    /**
     * Replace missing values with specified value.
     */
    public static TransformProcess replaceWithValue(Schema schema,
                                                      String column,
                                                      double value) {
        return null;
    }

    /**
     * Calculate mean of column (ignoring NaN).
     */
    public static double calculateMean(INDArray column) {
        return 0.0;
    }

    /**
     * Replace NaN with mean value.
     */
    public static INDArray imputeWithMean(INDArray data, int columnIndex) {
        return null;
    }
}`,

	solutionCode: `import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.doubletransform.ReplaceInvalidWithIntegerTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class MissingValueHandler {

    /**
     * Replace missing values with specified value.
     */
    public static TransformProcess replaceWithValue(Schema schema,
                                                      String column,
                                                      double value) {
        return new TransformProcess.Builder(schema)
            .transform(new ReplaceInvalidWithIntegerTransform(column, (int) value))
            .build();
    }

    /**
     * Calculate mean of column (ignoring NaN).
     */
    public static double calculateMean(INDArray column) {
        // Count non-NaN values
        int count = 0;
        double sum = 0.0;

        for (int i = 0; i < column.length(); i++) {
            double val = column.getDouble(i);
            if (!Double.isNaN(val)) {
                sum += val;
                count++;
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

    /**
     * Replace NaN with mean value.
     */
    public static INDArray imputeWithMean(INDArray data, int columnIndex) {
        INDArray result = data.dup();
        INDArray column = result.getColumn(columnIndex);
        double mean = calculateMean(column);

        for (int i = 0; i < column.length(); i++) {
            if (Double.isNaN(column.getDouble(i))) {
                column.putScalar(i, mean);
            }
        }

        return result;
    }

    /**
     * Replace NaN with median value.
     */
    public static INDArray imputeWithMedian(INDArray data, int columnIndex) {
        INDArray result = data.dup();
        INDArray column = result.getColumn(columnIndex);

        // Get non-NaN values and sort
        java.util.List<Double> values = new java.util.ArrayList<>();
        for (int i = 0; i < column.length(); i++) {
            double val = column.getDouble(i);
            if (!Double.isNaN(val)) {
                values.add(val);
            }
        }
        java.util.Collections.sort(values);

        double median = values.isEmpty() ? 0.0 :
            values.get(values.size() / 2);

        for (int i = 0; i < column.length(); i++) {
            if (Double.isNaN(column.getDouble(i))) {
                column.putScalar(i, median);
            }
        }

        return result;
    }

    /**
     * Count missing values in column.
     */
    public static int countMissing(INDArray column) {
        int count = 0;
        for (int i = 0; i < column.length(); i++) {
            if (Double.isNaN(column.getDouble(i))) {
                count++;
            }
        }
        return count;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class MissingValueHandlerTest {

    @Test
    void testCalculateMean() {
        INDArray column = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        double mean = MissingValueHandler.calculateMean(column);
        assertEquals(3.0, mean, 0.001);
    }

    @Test
    void testCalculateMeanWithNaN() {
        INDArray column = Nd4j.create(new double[]{1.0, Double.NaN, 3.0, 4.0, Double.NaN});
        double mean = MissingValueHandler.calculateMean(column);
        assertEquals(8.0 / 3, mean, 0.001);  // (1+3+4)/3
    }

    @Test
    void testCountMissing() {
        INDArray column = Nd4j.create(new double[]{1.0, Double.NaN, 3.0, Double.NaN});
        int count = MissingValueHandler.countMissing(column);
        assertEquals(2, count);
    }

    @Test
    void testCountMissingNoNaN() {
        INDArray column = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        int count = MissingValueHandler.countMissing(column);
        assertEquals(0, count);
    }

    @Test
    void testCountMissingAllNaN() {
        INDArray column = Nd4j.create(new double[]{Double.NaN, Double.NaN, Double.NaN});
        int count = MissingValueHandler.countMissing(column);
        assertEquals(3, count);
    }

    @Test
    void testCalculateMeanEmpty() {
        INDArray column = Nd4j.create(new double[]{Double.NaN, Double.NaN});
        double mean = MissingValueHandler.calculateMean(column);
        assertEquals(0.0, mean, 0.001);
    }

    @Test
    void testCalculateMeanSingleValue() {
        INDArray column = Nd4j.create(new double[]{5.0});
        double mean = MissingValueHandler.calculateMean(column);
        assertEquals(5.0, mean, 0.001);
    }

    @Test
    void testImputeWithMean() {
        INDArray data = Nd4j.create(new double[][]{
            {1.0, 10.0},
            {2.0, Double.NaN},
            {3.0, 30.0}
        });
        INDArray result = MissingValueHandler.imputeWithMean(data, 1);
        assertNotNull(result);
    }

    @Test
    void testImputeWithMedian() {
        INDArray data = Nd4j.create(new double[][]{
            {1.0, 10.0},
            {2.0, Double.NaN},
            {3.0, 30.0}
        });
        INDArray result = MissingValueHandler.imputeWithMedian(data, 1);
        assertNotNull(result);
    }

    @Test
    void testCountMissingSingleNaN() {
        INDArray column = Nd4j.create(new double[]{1.0, 2.0, Double.NaN, 4.0, 5.0});
        int count = MissingValueHandler.countMissing(column);
        assertEquals(1, count);
    }
}`,

	hint1: 'Check for NaN using Double.isNaN()',
	hint2: 'Calculate statistics excluding NaN values first',

	whyItMatters: `Missing value handling is critical:

- **Data quality**: Real data often has missing values
- **Algorithm requirements**: Many algorithms cannot handle NaN
- **Strategy matters**: Wrong imputation can bias results
- **Domain knowledge**: Best strategy depends on the problem

Proper missing value handling improves model reliability.`,

	translations: {
		ru: {
			title: 'Обработка пропущенных значений',
			description: `# Обработка пропущенных значений

Обрабатывайте пропущенные данные в датасетах.

## Задача

Реализуйте стратегии для пропущенных значений:
- Удаление строк с пропущенными значениями
- Заполнение средним/медианой
- Заполнение кастомными значениями

## Пример

\`\`\`java
TransformProcess tp = new TransformProcess.Builder(schema)
    .transform(new ReplaceEmptyWithValueTransform("age", 30.0))
    .build();
\`\`\``,
			hint1: 'Проверяйте на NaN используя Double.isNaN()',
			hint2: 'Сначала вычислите статистики исключая NaN значения',
			whyItMatters: `Обработка пропущенных значений критична:

- **Качество данных**: Реальные данные часто имеют пропуски
- **Требования алгоритмов**: Многие алгоритмы не обрабатывают NaN
- **Стратегия важна**: Неправильное заполнение может исказить результаты
- **Доменные знания**: Лучшая стратегия зависит от задачи`,
		},
		uz: {
			title: "Yo'qolgan qiymatlarni boshqarish",
			description: `# Yo'qolgan qiymatlarni boshqarish

Datasetlaringizdagi yo'qolgan ma'lumotlarni boshqaring.

## Topshiriq

Yo'qolgan qiymat strategiyalarini amalga oshiring:
- Yo'qolgan qiymatlari bor qatorlarni olib tashlash
- O'rtacha/median bilan to'ldirish
- Maxsus qiymatlar bilan to'ldirish

## Misol

\`\`\`java
TransformProcess tp = new TransformProcess.Builder(schema)
    .transform(new ReplaceEmptyWithValueTransform("age", 30.0))
    .build();
\`\`\``,
			hint1: "NaN uchun Double.isNaN() dan foydalaning",
			hint2: "Avval NaN qiymatlarni chiqarib statistikalarni hisoblang",
			whyItMatters: `Yo'qolgan qiymatlarni boshqarish muhim:

- **Ma'lumot sifati**: Haqiqiy ma'lumotlarda ko'pincha yo'qolgan qiymatlar bor
- **Algoritm talablari**: Ko'p algoritmlar NaN ni boshqarolmaydi
- **Strategiya muhim**: Noto'g'ri to'ldirish natijalarni buzishi mumkin
- **Domen bilimi**: Eng yaxshi strategiya muammoga bog'liq`,
		},
	},
};

export default task;
