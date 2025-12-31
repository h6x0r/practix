import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-learning-rate',
	title: 'Learning Rate Optimization',
	difficulty: 'medium',
	tags: ['dl4j', 'optimization', 'learning-rate'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Learning Rate Optimization

Find optimal learning rate for neural network training.

## Task

Implement learning rate strategies:
- Learning rate schedules
- Warm-up strategies
- Learning rate finder

## Example

\`\`\`java
ISchedule schedule = new ExponentialSchedule(
    ScheduleType.ITERATION,
    0.1,   // initial LR
    0.99   // decay rate
);

IUpdater updater = new Adam(schedule);
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.schedule.*;

public class LearningRateOptimizer {

    /**
     * Create constant learning rate updater.
     */
    public static IUpdater createConstantLR(double learningRate) {
        return null;
    }

    /**
     * Create exponential decay schedule.
     */
    public static IUpdater createExponentialDecay(double initialLR,
                                                    double decayRate) {
        return null;
    }

    /**
     * Create step decay schedule.
     */
    public static IUpdater createStepDecay(double initialLR,
                                             double decayRate,
                                             int stepSize) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.schedule.*;

public class LearningRateOptimizer {

    /**
     * Create constant learning rate updater.
     */
    public static IUpdater createConstantLR(double learningRate) {
        return new Adam(learningRate);
    }

    /**
     * Create exponential decay schedule.
     */
    public static IUpdater createExponentialDecay(double initialLR,
                                                    double decayRate) {
        ISchedule schedule = new ExponentialSchedule(
            ScheduleType.ITERATION,
            initialLR,
            decayRate
        );
        return new Adam(schedule);
    }

    /**
     * Create step decay schedule.
     */
    public static IUpdater createStepDecay(double initialLR,
                                             double decayRate,
                                             int stepSize) {
        ISchedule schedule = new StepSchedule(
            ScheduleType.ITERATION,
            initialLR,
            decayRate,
            stepSize
        );
        return new Adam(schedule);
    }

    /**
     * Create SGD with momentum.
     */
    public static IUpdater createSGDMomentum(double learningRate, double momentum) {
        return new Nesterovs(learningRate, momentum);
    }

    /**
     * Create polynomial decay schedule.
     */
    public static IUpdater createPolynomialDecay(double initialLR,
                                                   double power,
                                                   int maxIterations) {
        ISchedule schedule = new PolySchedule(
            ScheduleType.ITERATION,
            initialLR,
            power,
            maxIterations
        );
        return new Adam(schedule);
    }

    /**
     * Create cyclic learning rate.
     */
    public static IUpdater createCyclicLR(double baseLR, double maxLR, int stepSize) {
        ISchedule schedule = new CycleSchedule(
            ScheduleType.ITERATION,
            baseLR,
            maxLR,
            stepSize
        );
        return new Sgd(schedule);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.learning.config.*;
import static org.junit.jupiter.api.Assertions.*;

public class LearningRateOptimizerTest {

    @Test
    void testCreateConstantLR() {
        IUpdater updater = LearningRateOptimizer.createConstantLR(0.001);
        assertNotNull(updater);
        assertTrue(updater instanceof Adam);
    }

    @Test
    void testCreateExponentialDecay() {
        IUpdater updater = LearningRateOptimizer.createExponentialDecay(0.01, 0.99);
        assertNotNull(updater);
    }

    @Test
    void testCreateStepDecay() {
        IUpdater updater = LearningRateOptimizer.createStepDecay(0.1, 0.1, 1000);
        assertNotNull(updater);
    }

    @Test
    void testCreateSGDMomentum() {
        IUpdater updater = LearningRateOptimizer.createSGDMomentum(0.01, 0.9);
        assertNotNull(updater);
        assertTrue(updater instanceof Nesterovs);
    }

    @Test
    void testConstantLRReturnsAdam() {
        IUpdater updater = LearningRateOptimizer.createConstantLR(0.01);
        assertInstanceOf(Adam.class, updater);
    }

    @Test
    void testExponentialDecayNotNull() {
        IUpdater updater = LearningRateOptimizer.createExponentialDecay(0.1, 0.95);
        assertNotNull(updater);
    }

    @Test
    void testStepDecayWithDifferentParams() {
        IUpdater updater = LearningRateOptimizer.createStepDecay(0.01, 0.5, 500);
        assertNotNull(updater);
    }

    @Test
    void testSGDMomentumReturnsNesterovs() {
        IUpdater updater = LearningRateOptimizer.createSGDMomentum(0.001, 0.95);
        assertInstanceOf(Nesterovs.class, updater);
    }

    @Test
    void testPolynomialDecay() {
        IUpdater updater = LearningRateOptimizer.createPolynomialDecay(0.1, 2.0, 10000);
        assertNotNull(updater);
    }

    @Test
    void testCyclicLR() {
        IUpdater updater = LearningRateOptimizer.createCyclicLR(0.001, 0.01, 2000);
        assertNotNull(updater);
    }
}`,

	hint1: 'Use ExponentialSchedule or StepSchedule with IUpdater',
	hint2: 'ScheduleType.ITERATION applies per batch, EPOCH per epoch',

	whyItMatters: `Learning rate is the most critical hyperparameter:

- **Too high**: Training diverges or oscillates
- **Too low**: Training is slow and may get stuck
- **Schedules**: Adapt LR during training for best results
- **Warmup**: Start low and increase for stability

Proper LR tuning can dramatically improve model performance.`,

	translations: {
		ru: {
			title: 'Оптимизация learning rate',
			description: `# Оптимизация learning rate

Найдите оптимальный learning rate для обучения нейронной сети.

## Задача

Реализуйте стратегии learning rate:
- Расписания learning rate
- Стратегии разогрева
- Поиск learning rate

## Пример

\`\`\`java
ISchedule schedule = new ExponentialSchedule(
    ScheduleType.ITERATION,
    0.1,   // initial LR
    0.99   // decay rate
);

IUpdater updater = new Adam(schedule);
\`\`\``,
			hint1: 'Используйте ExponentialSchedule или StepSchedule с IUpdater',
			hint2: 'ScheduleType.ITERATION применяется к каждому batch, EPOCH к каждой эпохе',
			whyItMatters: `Learning rate - самый критичный гиперпараметр:

- **Слишком высокий**: Обучение расходится или осциллирует
- **Слишком низкий**: Обучение медленное и может застрять
- **Расписания**: Адаптация LR во время обучения для лучших результатов
- **Разогрев**: Начните с низкого и увеличивайте для стабильности`,
		},
		uz: {
			title: 'Learning rate optimallashtirish',
			description: `# Learning rate optimallashtirish

Neyron tarmoq o'qitish uchun optimal learning rate ni toping.

## Topshiriq

Learning rate strategiyalarini amalga oshiring:
- Learning rate jadvallari
- Isitish strategiyalari
- Learning rate topuvchi

## Misol

\`\`\`java
ISchedule schedule = new ExponentialSchedule(
    ScheduleType.ITERATION,
    0.1,   // initial LR
    0.99   // decay rate
);

IUpdater updater = new Adam(schedule);
\`\`\``,
			hint1: "IUpdater bilan ExponentialSchedule yoki StepSchedule dan foydalaning",
			hint2: "ScheduleType.ITERATION har bir batchga, EPOCH har bir epochga qo'llaniladi",
			whyItMatters: `Learning rate eng muhim giperparametr:

- **Juda yuqori**: O'qitish divergentlashadi yoki tebranadi
- **Juda past**: O'qitish sekin va tiqilib qolishi mumkin
- **Jadvallar**: Eng yaxshi natijalar uchun o'qitish paytida LR ni moslashtirish
- **Isitish**: Barqarorlik uchun pastdan boshlang va oshiring`,
		},
	},
};

export default task;
