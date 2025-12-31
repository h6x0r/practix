import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-svm',
	title: 'Support Vector Machine',
	difficulty: 'medium',
	tags: ['tribuo', 'svm', 'classification'],
	estimatedTime: '25m',
	isPremium: true,
	order: 5,
	description: `# Support Vector Machine

Implement SVM classifiers for binary and multiclass problems.

## Task

Build SVM classifiers:
- Configure kernel functions
- Tune regularization (C parameter)
- Handle multiclass classification

## Example

\`\`\`java
LibSVMClassificationTrainer trainer = new LibSVMClassificationTrainer(
    new SVMParameters(
        new RBFKernel(0.5),  // RBF kernel with gamma=0.5
        1.0  // C parameter
    )
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.libsvm.*;

public class SVMClassifier {

    /**
     * @param C Regularization parameter
     */
    public static LibSVMClassificationTrainer createLinearSVM(double C) {
        return null;
    }

    /**
     * @param gamma RBF kernel parameter
     * @param C Regularization parameter
     */
    public static LibSVMClassificationTrainer createRBFSVM(double gamma, double C) {
        return null;
    }

    /**
     */
    public static Model<Label> train(LibSVMClassificationTrainer trainer,
                                       Dataset<Label> data) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.libsvm.*;
import org.tribuo.common.libsvm.*;

public class SVMClassifier {

    /**
     * Create linear SVM trainer.
     * @param C Regularization parameter
     */
    public static LibSVMClassificationTrainer createLinearSVM(double C) {
        SVMParameters<Label> params = new SVMParameters<>(
            new LinearSVMType<>(SVMType.NU_SVC),
            KernelType.LINEAR
        );
        params.setCost(C);
        return new LibSVMClassificationTrainer(params);
    }

    /**
     * Create RBF kernel SVM.
     * @param gamma RBF kernel parameter
     * @param C Regularization parameter
     */
    public static LibSVMClassificationTrainer createRBFSVM(double gamma, double C) {
        SVMParameters<Label> params = new SVMParameters<>(
            new LinearSVMType<>(SVMType.C_SVC),
            KernelType.RBF
        );
        params.setGamma(gamma);
        params.setCost(C);
        return new LibSVMClassificationTrainer(params);
    }

    /**
     * Train and return model.
     */
    public static Model<Label> train(LibSVMClassificationTrainer trainer,
                                       Dataset<Label> data) {
        return trainer.train(data);
    }

    /**
     * Create polynomial kernel SVM.
     */
    public static LibSVMClassificationTrainer createPolySVM(int degree, double C) {
        SVMParameters<Label> params = new SVMParameters<>(
            new LinearSVMType<>(SVMType.C_SVC),
            KernelType.POLY
        );
        params.setDegree(degree);
        params.setCost(C);
        return new LibSVMClassificationTrainer(params);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.classification.libsvm.*;
import static org.junit.jupiter.api.Assertions.*;

public class SVMClassifierTest {

    @Test
    void testCreateLinearSVM() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createLinearSVM(1.0);
        assertNotNull(trainer);
    }

    @Test
    void testCreateRBFSVM() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createRBFSVM(0.5, 1.0);
        assertNotNull(trainer);
    }

    @Test
    void testCreateLinearSVMWithLargeC() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createLinearSVM(10.0);
        assertNotNull(trainer);
    }

    @Test
    void testCreateLinearSVMWithSmallC() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createLinearSVM(0.01);
        assertNotNull(trainer);
    }

    @Test
    void testCreateRBFSVMWithDifferentGamma() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createRBFSVM(0.1, 1.0);
        assertNotNull(trainer);
    }

    @Test
    void testCreateRBFSVMHighGamma() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createRBFSVM(2.0, 5.0);
        assertNotNull(trainer);
    }

    @Test
    void testCreatePolySVM() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createPolySVM(3, 1.0);
        assertNotNull(trainer);
    }

    @Test
    void testCreatePolySVMDegree2() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createPolySVM(2, 0.5);
        assertNotNull(trainer);
    }

    @Test
    void testCreatePolySVMHighDegree() {
        LibSVMClassificationTrainer trainer = SVMClassifier.createPolySVM(5, 1.0);
        assertNotNull(trainer);
    }
}`,

	hint1: 'Use SVMParameters to configure kernel and regularization',
	hint2: 'RBF kernel is good for non-linear boundaries',

	whyItMatters: `SVMs remain powerful classifiers:

- **Strong theory**: Well-founded in statistical learning
- **Kernel trick**: Handle non-linear data
- **Margin maximization**: Good generalization
- **Small data**: Work well with limited samples

SVMs are often best for small-to-medium datasets.`,

	translations: {
		ru: {
			title: 'Метод опорных векторов',
			description: `# Метод опорных векторов

Реализуйте SVM классификаторы для бинарных и многоклассовых задач.

## Задача

Создайте SVM классификаторы:
- Настройте функции ядра
- Подберите регуляризацию (параметр C)
- Обработайте многоклассовую классификацию

## Пример

\`\`\`java
LibSVMClassificationTrainer trainer = new LibSVMClassificationTrainer(
    new SVMParameters(
        new RBFKernel(0.5),  // RBF kernel with gamma=0.5
        1.0  // C parameter
    )
);
\`\`\``,
			hint1: 'Используйте SVMParameters для настройки ядра и регуляризации',
			hint2: 'RBF ядро хорошо для нелинейных границ',
			whyItMatters: `SVM остаются мощными классификаторами:

- **Сильная теория**: Основаны на статистическом обучении
- **Трюк с ядром**: Обработка нелинейных данных
- **Максимизация отступа**: Хорошее обобщение
- **Малые данные**: Хорошо работают с ограниченными выборками`,
		},
		uz: {
			title: 'Support Vector Machine',
			description: `# Support Vector Machine

Binary va ko'p sinfli muammolar uchun SVM klassifikatorlarini amalga oshiring.

## Topshiriq

SVM klassifikatorlarini yarating:
- Yadro funksiyalarini sozlang
- Regularizatsiyani (C parametri) sozlang
- Ko'p sinfli klassifikatsiyani boshqaring

## Misol

\`\`\`java
LibSVMClassificationTrainer trainer = new LibSVMClassificationTrainer(
    new SVMParameters(
        new RBFKernel(0.5),  // RBF kernel with gamma=0.5
        1.0  // C parameter
    )
);
\`\`\``,
			hint1: "Yadro va regularizatsiyani sozlash uchun SVMParameters dan foydalaning",
			hint2: "RBF yadro nolinear chegaralar uchun yaxshi",
			whyItMatters: `SVMlar kuchli klassifikatorlar bo'lib qolmoqda:

- **Kuchli nazariya**: Statistik o'qitishga asoslangan
- **Yadro hiylasi**: Nolinear ma'lumotlarni boshqarish
- **Margin maksimallash**: Yaxshi umumlashtirish
- **Kichik ma'lumotlar**: Cheklangan namunalar bilan yaxshi ishlaydi`,
		},
	},
};

export default task;
