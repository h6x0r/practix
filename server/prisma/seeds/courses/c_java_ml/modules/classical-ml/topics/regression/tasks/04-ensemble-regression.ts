import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-ensemble-regression',
	title: 'Ensemble Regression',
	difficulty: 'medium',
	tags: ['tribuo', 'ensemble', 'bagging'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# Ensemble Regression

Build ensemble models for improved regression predictions.

## Task

Implement ensemble methods:
- Bagging with regression trees
- Combine multiple base learners
- Evaluate ensemble performance

## Example

\`\`\`java
BaggingTrainer<Regressor> ensemble = new BaggingTrainer<>(
    new CARTRegressionTrainer(6),
    new AveragingCombiner(),
    50  // number of trees
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.ensemble.BaggingTrainer;

public class EnsembleRegressor {

    /**
     * Create bagging ensemble with regression trees.
     */
    public static BaggingTrainer<Regressor> createBaggingEnsemble(
            int numTrees, int maxDepth) {
        // Your code here
        return null;
    }

    /**
     * Train ensemble model.
     */
    public static Model<Regressor> trainEnsemble(
            BaggingTrainer<Regressor> trainer, Dataset<Regressor> data) {
        // Your code here
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.evaluation.*;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.ensemble.BaggingTrainer;

public class EnsembleRegressor {

    /**
     * Create bagging ensemble with regression trees.
     */
    public static BaggingTrainer<Regressor> createBaggingEnsemble(
            int numTrees, int maxDepth) {
        CARTRegressionTrainer baseTrainer = new CARTRegressionTrainer(maxDepth);
        AveragingCombiner combiner = new AveragingCombiner();

        return new BaggingTrainer<>(baseTrainer, combiner, numTrees);
    }

    /**
     * Train ensemble model.
     */
    public static Model<Regressor> trainEnsemble(
            BaggingTrainer<Regressor> trainer, Dataset<Regressor> data) {
        return trainer.train(data);
    }

    /**
     * Create ensemble with subsample ratio.
     */
    public static BaggingTrainer<Regressor> createBaggingWithSubsample(
            int numTrees, int maxDepth, double subsampleRatio) {
        CARTRegressionTrainer baseTrainer = new CARTRegressionTrainer(maxDepth);
        AveragingCombiner combiner = new AveragingCombiner();

        return new BaggingTrainer<>(baseTrainer, combiner, numTrees, subsampleRatio);
    }

    /**
     * Evaluate ensemble model.
     */
    public static RegressionEvaluation evaluate(
            Model<Regressor> model, Dataset<Regressor> testData) {
        RegressionEvaluator evaluator = new RegressionEvaluator();
        return evaluator.evaluate(model, testData);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class EnsembleRegressorTest {

    @Test
    void testCreateBaggingEnsemble() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingEnsemble(10, 5);
        assertNotNull(trainer);
    }

    @Test
    void testTrainEnsemble() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingEnsemble(5, 3);
        MutableDataset<Regressor> data = createTestDataset();

        Model<Regressor> model = EnsembleRegressor.trainEnsemble(trainer, data);
        assertNotNull(model);
    }

    private MutableDataset<Regressor> createTestDataset() {
        RegressionFactory factory = new RegressionFactory();
        MutableDataset<Regressor> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 50; i++) {
            double x = Math.random() * 10;
            double y = x * 2 + Math.random();
            dataset.add(new ArrayExample<>(new Regressor("target", y),
                new String[]{"x"}, new double[]{x}));
        }

        return dataset;
    }

    @Test
    void testCreateBaggingEnsembleMoreTrees() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingEnsemble(50, 8);
        assertNotNull(trainer);
    }

    @Test
    void testCreateBaggingEnsembleShallowTrees() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingEnsemble(20, 2);
        assertNotNull(trainer);
    }

    @Test
    void testCreateBaggingWithSubsample() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingWithSubsample(30, 5, 0.7);
        assertNotNull(trainer);
    }

    @Test
    void testDatasetSize() {
        MutableDataset<Regressor> data = createTestDataset();
        assertEquals(50, data.size());
    }

    @Test
    void testTrainedModelNotNull() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingEnsemble(5, 3);
        MutableDataset<Regressor> data = createTestDataset();
        Model<Regressor> model = EnsembleRegressor.trainEnsemble(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testMultipleEnsemblesCanBeCreated() {
        BaggingTrainer<Regressor> t1 = EnsembleRegressor.createBaggingEnsemble(10, 4);
        BaggingTrainer<Regressor> t2 = EnsembleRegressor.createBaggingEnsemble(20, 6);
        assertNotNull(t1);
        assertNotNull(t2);
    }

    @Test
    void testBaggingSingleTree() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingEnsemble(1, 5);
        assertNotNull(trainer);
    }

    @Test
    void testBaggingLargeSubsample() {
        BaggingTrainer<Regressor> trainer = EnsembleRegressor.createBaggingWithSubsample(25, 4, 1.0);
        assertNotNull(trainer);
    }
}`,

	hint1: 'Use AveragingCombiner to combine predictions',
	hint2: 'More trees usually improve performance but increase computation',

	whyItMatters: `Ensembles improve prediction stability:

- **Reduced variance**: Average out individual model errors
- **Better accuracy**: Often beats single models
- **Robustness**: Less sensitive to outliers
- **Foundation**: Basis for Random Forest and boosting

Ensembles are standard practice in production ML.`,

	translations: {
		ru: {
			title: 'Ансамблевая регрессия',
			description: `# Ансамблевая регрессия

Создайте ансамблевые модели для улучшенных регрессионных предсказаний.

## Задача

Реализуйте ансамблевые методы:
- Бэггинг с деревьями регрессии
- Комбинирование нескольких базовых обучающих
- Оценка производительности ансамбля

## Пример

\`\`\`java
BaggingTrainer<Regressor> ensemble = new BaggingTrainer<>(
    new CARTRegressionTrainer(6),
    new AveragingCombiner(),
    50  // number of trees
);
\`\`\``,
			hint1: 'Используйте AveragingCombiner для комбинирования предсказаний',
			hint2: 'Больше деревьев обычно улучшает производительность, но увеличивает вычисления',
			whyItMatters: `Ансамбли улучшают стабильность предсказаний:

- **Уменьшение дисперсии**: Усреднение ошибок отдельных моделей
- **Лучшая точность**: Часто превосходят одиночные модели
- **Устойчивость**: Менее чувствительны к выбросам
- **Основа**: База для Random Forest и бустинга`,
		},
		uz: {
			title: 'Ansambl regressiyasi',
			description: `# Ansambl regressiyasi

Yaxshilangan regressiya bashoratlari uchun ansambl modellarini yarating.

## Topshiriq

Ansambl metodlarini amalga oshiring:
- Regressiya daraxtlari bilan bagging
- Bir nechta bazaviy o'rganuvchilarni birlashtirish
- Ansambl samaradorligini baholash

## Misol

\`\`\`java
BaggingTrainer<Regressor> ensemble = new BaggingTrainer<>(
    new CARTRegressionTrainer(6),
    new AveragingCombiner(),
    50  // number of trees
);
\`\`\``,
			hint1: "Bashoratlarni birlashtirish uchun AveragingCombiner dan foydalaning",
			hint2: "Ko'proq daraxtlar odatda samaradorlikni yaxshilaydi, lekin hisoblashni oshiradi",
			whyItMatters: `Ansambllar bashorat barqarorligini yaxshilaydi:

- **Kamaytirilgan dispersiya**: Individual model xatolarini o'rtacha
- **Yaxshiroq aniqlik**: Ko'pincha yagona modellarni mag'lub etadi
- **Barqarorlik**: Outlierga kam sezgir
- **Asos**: Random Forest va boosting uchun asos`,
		},
	},
};

export default task;
