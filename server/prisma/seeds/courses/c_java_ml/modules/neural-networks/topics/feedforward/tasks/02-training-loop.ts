import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-training-loop',
	title: 'Training Loop',
	difficulty: 'medium',
	tags: ['dl4j', 'training', 'neural-network'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Training Loop

Implement a complete training loop for DL4J models with validation and callbacks.

## Task

Create a trainer class that:
- Runs training epochs
- Evaluates on validation data
- Tracks training metrics
- Supports early stopping

## Example

\`\`\`java
for (int epoch = 0; epoch < numEpochs; epoch++) {
    model.fit(trainIterator);

    Evaluation eval = model.evaluate(testIterator);
    System.out.println("Epoch " + epoch + " Accuracy: " + eval.accuracy());
}
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.util.ArrayList;
import java.util.List;

public class ModelTrainer {

    private MultiLayerNetwork model;
    private List<Double> trainLosses;
    private List<Double> valAccuracies;

    public ModelTrainer(MultiLayerNetwork model) {
        this.model = model;
        this.trainLosses = new ArrayList<>();
        this.valAccuracies = new ArrayList<>();
    }

    /**
     * Train for specified number of epochs.
     */
    public void train(DataSetIterator trainData, DataSetIterator valData,
                      int numEpochs) {
    }

    /**
     * Evaluate model on test data.
     */
    public Evaluation evaluate(DataSetIterator testData) {
        return null;
    }

    /**
     * Train with early stopping.
     */
    public void trainWithEarlyStopping(DataSetIterator trainData,
                                        DataSetIterator valData,
                                        int maxEpochs, int patience) {
    }

    public List<Double> getTrainLosses() {
        return trainLosses;
    }

    public List<Double> getValAccuracies() {
        return valAccuracies;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.util.ArrayList;
import java.util.List;

public class ModelTrainer {

    private MultiLayerNetwork model;
    private List<Double> trainLosses;
    private List<Double> valAccuracies;

    public ModelTrainer(MultiLayerNetwork model) {
        this.model = model;
        this.trainLosses = new ArrayList<>();
        this.valAccuracies = new ArrayList<>();
    }

    /**
     * Train for specified number of epochs.
     */
    public void train(DataSetIterator trainData, DataSetIterator valData,
                      int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Reset iterators
            trainData.reset();

            // Train one epoch
            model.fit(trainData);

            // Get training loss
            double loss = model.score();
            trainLosses.add(loss);

            // Evaluate on validation data
            if (valData != null) {
                valData.reset();
                Evaluation eval = model.evaluate(valData);
                double accuracy = eval.accuracy();
                valAccuracies.add(accuracy);

                System.out.printf("Epoch %d - Loss: %.4f, Val Accuracy: %.4f%n",
                    epoch + 1, loss, accuracy);
            } else {
                System.out.printf("Epoch %d - Loss: %.4f%n", epoch + 1, loss);
            }
        }
    }

    /**
     * Evaluate model on test data.
     */
    public Evaluation evaluate(DataSetIterator testData) {
        testData.reset();
        return model.evaluate(testData);
    }

    /**
     * Train with early stopping.
     */
    public void trainWithEarlyStopping(DataSetIterator trainData,
                                        DataSetIterator valData,
                                        int maxEpochs, int patience) {
        double bestAccuracy = 0.0;
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            trainData.reset();
            model.fit(trainData);

            double loss = model.score();
            trainLosses.add(loss);

            valData.reset();
            Evaluation eval = model.evaluate(valData);
            double accuracy = eval.accuracy();
            valAccuracies.add(accuracy);

            System.out.printf("Epoch %d - Loss: %.4f, Val Accuracy: %.4f%n",
                epoch + 1, loss, accuracy);

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
            }

            if (epochsWithoutImprovement >= patience) {
                System.out.println("Early stopping at epoch " + (epoch + 1));
                break;
            }
        }
    }

    public List<Double> getTrainLosses() {
        return trainLosses;
    }

    public List<Double> getValAccuracies() {
        return valAccuracies;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class ModelTrainerTest {

    @Test
    void testTrainerInitialization() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);

        assertNotNull(trainer.getTrainLosses());
        assertNotNull(trainer.getValAccuracies());
        assertTrue(trainer.getTrainLosses().isEmpty());
    }

    @Test
    void testGettersReturnLists() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);

        assertTrue(trainer.getTrainLosses() instanceof java.util.List);
        assertTrue(trainer.getValAccuracies() instanceof java.util.List);
    }

    @Test
    void testTrainLossesIsEmpty() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        assertEquals(0, trainer.getTrainLosses().size());
    }

    @Test
    void testValAccuraciesIsEmpty() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        assertEquals(0, trainer.getValAccuracies().size());
    }

    @Test
    void testTrainerNotNull() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        assertNotNull(trainer);
    }

    @Test
    void testTrainLossesListNotNull() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        assertNotNull(trainer.getTrainLosses());
    }

    @Test
    void testValAccuraciesListNotNull() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        assertNotNull(trainer.getValAccuracies());
    }

    @Test
    void testTrainLossesIsMutable() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        trainer.getTrainLosses().add(0.5);
        assertEquals(1, trainer.getTrainLosses().size());
    }

    @Test
    void testValAccuraciesIsMutable() {
        MultiLayerNetwork mockModel = mock(MultiLayerNetwork.class);
        ModelTrainer trainer = new ModelTrainer(mockModel);
        trainer.getValAccuracies().add(0.9);
        assertEquals(1, trainer.getValAccuracies().size());
    }
}`,

	hint1: 'Reset iterators before each epoch with iterator.reset()',
	hint2: 'model.score() returns the current loss value',

	whyItMatters: `Training loops are where learning happens:

- **Epochs**: Multiple passes over the data
- **Validation**: Monitor generalization during training
- **Early stopping**: Prevent overfitting
- **Metrics tracking**: Understand training dynamics

Well-designed training leads to better models.`,

	translations: {
		ru: {
			title: 'Цикл обучения',
			description: `# Цикл обучения

Реализуйте полный цикл обучения для моделей DL4J с валидацией и callbacks.

## Задача

Создайте класс тренера:
- Запуск эпох обучения
- Оценка на валидационных данных
- Отслеживание метрик
- Поддержка early stopping

## Пример

\`\`\`java
for (int epoch = 0; epoch < numEpochs; epoch++) {
    model.fit(trainIterator);

    Evaluation eval = model.evaluate(testIterator);
    System.out.println("Epoch " + epoch + " Accuracy: " + eval.accuracy());
}
\`\`\``,
			hint1: 'Сбрасывайте итераторы перед каждой эпохой с iterator.reset()',
			hint2: 'model.score() возвращает текущее значение loss',
			whyItMatters: `Циклы обучения - место, где происходит обучение:

- **Эпохи**: Множественные проходы по данным
- **Валидация**: Мониторинг обобщения в процессе обучения
- **Early stopping**: Предотвращение переобучения
- **Отслеживание метрик**: Понимание динамики обучения`,
		},
		uz: {
			title: "O'qitish sikli",
			description: `# O'qitish sikli

DL4J modellari uchun validatsiya va callbacks bilan to'liq o'qitish siklini amalga oshiring.

## Topshiriq

Trainer sinfini yarating:
- O'qitish davrlarini ishga tushirish
- Validatsiya ma'lumotlarida baholash
- Metrikalarni kuzatish
- Early stopping ni qo'llab-quvvatlash

## Misol

\`\`\`java
for (int epoch = 0; epoch < numEpochs; epoch++) {
    model.fit(trainIterator);

    Evaluation eval = model.evaluate(testIterator);
    System.out.println("Epoch " + epoch + " Accuracy: " + eval.accuracy());
}
\`\`\``,
			hint1: "Har bir davrdan oldin iterator.reset() bilan iteratorlarni qayta tiklang",
			hint2: "model.score() joriy loss qiymatini qaytaradi",
			whyItMatters: `O'qitish sikllari o'rganish sodir bo'ladigan joy:

- **Davrlar**: Ma'lumotlar bo'yicha ko'p o'tishlar
- **Validatsiya**: O'qitish davomida umumlashtirishni kuzatish
- **Early stopping**: Overfitting ni oldini olish
- **Metrikalarni kuzatish**: O'qitish dinamikasini tushunish`,
		},
	},
};

export default task;
