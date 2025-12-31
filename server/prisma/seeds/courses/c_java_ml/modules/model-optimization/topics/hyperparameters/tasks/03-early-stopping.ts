import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-early-stopping',
	title: 'Early Stopping',
	difficulty: 'easy',
	tags: ['dl4j', 'early-stopping', 'regularization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Early Stopping

Prevent overfitting by stopping training when validation performance degrades.

## Task

Implement early stopping:
- Monitor validation loss
- Stop after patience epochs without improvement
- Save best model

## Example

\`\`\`java
EarlyStoppingConfiguration config = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
    .scoreCalculator(new DataSetLossCalculator(testIterator, true))
    .evaluateEveryNEpochs(1)
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.earlystopping.*;
import org.deeplearning4j.earlystopping.termination.*;
import org.deeplearning4j.earlystopping.scorecalc.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class EarlyStoppingTrainer {

    /**
     * Create early stopping configuration.
     * @param maxEpochs Maximum epochs
     * @param patience Epochs without improvement before stopping
     */
    public static EarlyStoppingConfiguration<MultiLayerNetwork> createConfig(
            int maxEpochs, int patience, DataSetIterator validationData) {
        // Your code here
        return null;
    }

    /**
     * Create trainer with early stopping.
     */
    public static EarlyStoppingTrainer<MultiLayerNetwork> createTrainer(
            EarlyStoppingConfiguration<MultiLayerNetwork> config,
            MultiLayerNetwork network) {
        // Your code here
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.earlystopping.*;
import org.deeplearning4j.earlystopping.termination.*;
import org.deeplearning4j.earlystopping.scorecalc.*;
import org.deeplearning4j.earlystopping.saver.*;
import org.deeplearning4j.earlystopping.trainer.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.File;

public class EarlyStoppingTrainerBuilder {

    /**
     * Create early stopping configuration.
     * @param maxEpochs Maximum epochs
     * @param patience Epochs without improvement before stopping
     */
    public static EarlyStoppingConfiguration<MultiLayerNetwork> createConfig(
            int maxEpochs, int patience, DataSetIterator validationData) {

        return new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
            // Stop after maxEpochs
            .epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs))
            // Stop if no improvement for 'patience' epochs
            .epochTerminationConditions(
                new ScoreImprovementEpochTerminationCondition(patience)
            )
            // Calculate validation score
            .scoreCalculator(new DataSetLossCalculator(validationData, true))
            // Evaluate every epoch
            .evaluateEveryNEpochs(1)
            // Save best model
            .modelSaver(new InMemoryModelSaver<>())
            .build();
    }

    /**
     * Create trainer with early stopping.
     */
    public static EarlyStoppingTrainer<MultiLayerNetwork> createTrainer(
            EarlyStoppingConfiguration<MultiLayerNetwork> config,
            DataSetIterator trainData,
            MultiLayerNetwork network) {

        return new EarlyStoppingTrainer<>(
            config,
            network,
            trainData
        );
    }

    /**
     * Create config that saves to disk.
     */
    public static EarlyStoppingConfiguration<MultiLayerNetwork> createConfigWithSave(
            int maxEpochs, int patience,
            DataSetIterator validationData,
            String savePath) {

        return new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
            .epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs))
            .epochTerminationConditions(
                new ScoreImprovementEpochTerminationCondition(patience)
            )
            .scoreCalculator(new DataSetLossCalculator(validationData, true))
            .evaluateEveryNEpochs(1)
            .modelSaver(new LocalFileModelSaver(new File(savePath)))
            .build();
    }

    /**
     * Train with early stopping and return result.
     */
    public static EarlyStoppingResult<MultiLayerNetwork> trainWithEarlyStopping(
            MultiLayerNetwork network,
            DataSetIterator trainData,
            DataSetIterator validationData,
            int maxEpochs,
            int patience) {

        EarlyStoppingConfiguration<MultiLayerNetwork> config = createConfig(
            maxEpochs, patience, validationData
        );

        EarlyStoppingTrainer<MultiLayerNetwork> trainer = createTrainer(
            config, trainData, network
        );

        return trainer.fit();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.earlystopping.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import java.util.Collections;
import static org.junit.jupiter.api.Assertions.*;

public class EarlyStoppingTrainerBuilderTest {

    @Test
    void testCreateConfig() {
        DataSetIterator valData = createMockIterator();

        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);

        assertNotNull(config);
    }

    @Test
    void testPatience() {
        DataSetIterator valData = createMockIterator();

        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(50, 10, valData);

        assertNotNull(config);
        // Verify termination conditions are set
        assertFalse(config.getEpochTerminationConditions().isEmpty());
    }

    private DataSetIterator createMockIterator() {
        DataSet mock = new DataSet(
            org.nd4j.linalg.factory.Nd4j.rand(10, 5),
            org.nd4j.linalg.factory.Nd4j.rand(10, 2)
        );
        return new ListDataSetIterator<>(Collections.singletonList(mock));
    }

    @Test
    void testCreateConfigNotNull() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);
        assertNotNull(config);
    }

    @Test
    void testConfigHasTerminationConditions() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(50, 10, valData);
        assertNotNull(config.getEpochTerminationConditions());
    }

    @Test
    void testConfigHasScoreCalculator() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);
        assertNotNull(config.getScoreCalculator());
    }

    @Test
    void testConfigHasModelSaver() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);
        assertNotNull(config.getModelSaver());
    }

    @Test
    void testDifferentMaxEpochs() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config1 =
            EarlyStoppingTrainerBuilder.createConfig(50, 5, valData);
        EarlyStoppingConfiguration<MultiLayerNetwork> config2 =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);
        assertNotNull(config1);
        assertNotNull(config2);
    }

    @Test
    void testDifferentPatience() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config1 =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);
        EarlyStoppingConfiguration<MultiLayerNetwork> config2 =
            EarlyStoppingTrainerBuilder.createConfig(100, 10, valData);
        assertNotNull(config1);
        assertNotNull(config2);
    }

    @Test
    void testEvaluateEveryNEpochs() {
        DataSetIterator valData = createMockIterator();
        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfig(100, 5, valData);
        assertTrue(config.getEvaluateEveryNEpochs() > 0);
    }

    @Test
    void testConfigWithSave() throws Exception {
        DataSetIterator valData = createMockIterator();
        java.io.File tempDir = java.io.File.createTempFile("model", "");
        tempDir.delete();
        tempDir.mkdir();

        EarlyStoppingConfiguration<MultiLayerNetwork> config =
            EarlyStoppingTrainerBuilder.createConfigWithSave(100, 5, valData, tempDir.getAbsolutePath());
        assertNotNull(config);
    }
}`,

	hint1: 'ScoreImprovementEpochTerminationCondition checks patience epochs',
	hint2: 'InMemoryModelSaver keeps best model in memory',

	whyItMatters: `Early stopping is a simple but effective technique:

- **Prevent overfitting**: Stop before model memorizes training data
- **Save time**: No need to run full training if converged
- **Automatic**: No manual epoch tuning needed
- **Best model**: Automatically saves the best checkpoint

Early stopping is a must-have for deep learning training.`,

	translations: {
		ru: {
			title: 'Ранняя остановка',
			description: `# Ранняя остановка

Предотвратите переобучение остановкой обучения при ухудшении валидации.

## Задача

Реализуйте раннюю остановку:
- Мониторинг validation loss
- Остановка после patience эпох без улучшения
- Сохранение лучшей модели

## Пример

\`\`\`java
EarlyStoppingConfiguration config = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
    .scoreCalculator(new DataSetLossCalculator(testIterator, true))
    .evaluateEveryNEpochs(1)
    .build();
\`\`\``,
			hint1: 'ScoreImprovementEpochTerminationCondition проверяет patience эпох',
			hint2: 'InMemoryModelSaver хранит лучшую модель в памяти',
			whyItMatters: `Ранняя остановка - простая но эффективная техника:

- **Предотвращение переобучения**: Остановка до запоминания данных
- **Экономия времени**: Не нужно полное обучение при сходимости
- **Автоматизация**: Не нужен ручной подбор эпох
- **Лучшая модель**: Автоматическое сохранение лучшего checkpoint`,
		},
		uz: {
			title: 'Erta to\'xtatish',
			description: `# Erta to'xtatish

Validatsiya samaradorligi yomonlashganda o'qitishni to'xtatib overfittingni oldini oling.

## Topshiriq

Erta to'xtatishni amalga oshiring:
- Validation lossni monitoring qiling
- Yaxshilanishsiz patience epochlardan keyin to'xtating
- Eng yaxshi modelni saqlang

## Misol

\`\`\`java
EarlyStoppingConfiguration config = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
    .scoreCalculator(new DataSetLossCalculator(testIterator, true))
    .evaluateEveryNEpochs(1)
    .build();
\`\`\``,
			hint1: "ScoreImprovementEpochTerminationCondition patience epochlarni tekshiradi",
			hint2: "InMemoryModelSaver eng yaxshi modelni xotirada saqlaydi",
			whyItMatters: `Erta to'xtatish oddiy lekin samarali texnika:

- **Overfittingni oldini olish**: Model o'qitish ma'lumotlarini yodlamasdan oldin to'xtatish
- **Vaqtni tejash**: Agar konvergeniya bo'lsa to'liq o'qitish shart emas
- **Avtomatik**: Qo'lda epoch sozlash kerak emas
- **Eng yaxshi model**: Avtomatik ravishda eng yaxshi checkpointni saqlaydi`,
		},
	},
};

export default task;
