import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-lambda-scope',
    title: 'Lambda Scope and Effectively Final',
    difficulty: 'medium',
    tags: ['java', 'lambda', 'scope', 'closure', 'effectively-final'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Lambda Scope and Effectively Final

Lambda expressions can access variables from their enclosing scope (closure), but with restrictions. Local variables must be final or effectively final. Understanding lambda scope rules is crucial for avoiding common pitfalls and writing correct functional code.

## Requirements:
1. Demonstrate variable access in lambdas:
   1.1. Instance variables (can be modified)
   1.2. Static variables (can be modified)
   1.3. Local variables (must be effectively final)
   1.4. Method parameters (effectively final)

2. Show effectively final concept:
   2.1. Variables that are never reassigned
   2.2. Compile error when trying to modify
   2.3. Why this restriction exists

3. Demonstrate 'this' reference:
   3.1. 'this' in lambda refers to enclosing instance
   3.2. Difference from anonymous classes
   3.3. Accessing instance methods and fields

4. Show closure behavior:
   4.1. Capturing variables from outer scope
   4.2. Variable shadowing
   4.3. Best practices for lambda variable access

## Example Output:
\`\`\`
=== Variable Access in Lambdas ===
Instance variable: count = 10
Static variable: total = 100
Local variable (effectively final): multiplier = 5
Parameter (effectively final): base = 3

=== This Reference ===
Lambda this: MethodScope@hashcode
Anonymous class this: Anonymous@hashcode
Instance method called: Hello from instance

=== Closure Behavior ===
Captured value: prefix = "Result: "
Lambda output: Result: 42
Outer scope modified: newPrefix = "Output: "

=== Effectively Final Restriction ===
Valid: Using final variable = 10
Valid: Using effectively final variable = 20
Error: Cannot modify local variable in lambda
\`\`\``,
    initialCode: `public class LambdaScope {
    private int instanceVar = 10;
    private static int staticVar = 100;

    public void demonstrateScope() {
        // TODO: Demonstrate local variable access (effectively final)

        // TODO: Demonstrate instance and static variable access

        // TODO: Show this reference behavior

        // TODO: Demonstrate closure behavior
    }

    public static void main(String[] args) {
        LambdaScope demo = new LambdaScope();
        demo.demonstrateScope();
    }
}`,
    solutionCode: `import java.util.function.*;

public class LambdaScope {
    private int instanceVar = 10;
    private static int staticVar = 100;

    public void demonstrateScope() {
        System.out.println("=== Variable Access in Lambdas ===");

        // Local variable - must be effectively final
        int multiplier = 5;
        int base = 3;

        // Lambda can access effectively final local variables
        IntFunction<Integer> multiply = x -> x * multiplier;
        System.out.println("Local variable (effectively final): multiplier = " + multiplier);

        // Lambda can access parameters (effectively final)
        Function<Integer, Integer> power = exp -> {
            int result = 1;
            for (int i = 0; i < exp; i++) {
                result *= base; // base is effectively final parameter
            }
            return result;
        };
        System.out.println("Parameter (effectively final): base = " + base);

        // Lambda can access and modify instance variables
        Runnable modifyInstance = () -> {
            instanceVar += 5;
            System.out.println("Instance variable: count = " + instanceVar);
        };
        modifyInstance.run();

        // Lambda can access and modify static variables
        Runnable modifyStatic = () -> {
            staticVar += 10;
            System.out.println("Static variable: total = " + staticVar);
        };
        modifyStatic.run();

        // This would cause compilation error:
        // int mutableVar = 10;
        // Runnable invalid = () -> mutableVar++; // Error: must be final or effectively final
        // mutableVar = 20; // This reassignment makes it not effectively final

        System.out.println("\\n=== This Reference ===");

        // In lambda, 'this' refers to the enclosing instance
        Supplier<String> lambdaThis = () -> {
            System.out.println("Lambda this: " + this.getClass().getName() + "@" +
                Integer.toHexString(this.hashCode()));
            return this.getInstanceMessage();
        };
        lambdaThis.get();

        // In anonymous class, 'this' refers to the anonymous class instance
        Supplier<String> anonymousThis = new Supplier<>() {
            @Override
            public String get() {
                System.out.println("Anonymous class this: " + this.getClass().getName() +
                    "@" + Integer.toHexString(this.hashCode()));
                return "Anonymous class";
            }
        };
        anonymousThis.get();

        System.out.println("Instance method called: " + lambdaThis.get());

        System.out.println("\\n=== Closure Behavior ===");

        // Lambda captures variables from enclosing scope (closure)
        String prefix = "Result: ";
        Function<Integer, String> formatter = num -> prefix + num;
        System.out.println("Captured value: prefix = \\"" + prefix + "\\"");
        System.out.println("Lambda output: " + formatter.apply(42));

        // Can create new variable with same name in outer scope
        String newPrefix = "Output: ";
        System.out.println("Outer scope modified: newPrefix = \\"" + newPrefix + "\\"");
        // Original lambda still uses captured 'prefix' value

        System.out.println("\\n=== Effectively Final Restriction ===");

        // Final variable - explicitly declared
        final int finalVar = 10;
        Supplier<Integer> useFinal = () -> finalVar * 2;
        System.out.println("Valid: Using final variable = " + finalVar);

        // Effectively final - never reassigned
        int effectivelyFinalVar = 20;
        Supplier<Integer> useEffectivelyFinal = () -> effectivelyFinalVar * 2;
        System.out.println("Valid: Using effectively final variable = " + effectivelyFinalVar);

        // This demonstrates the error (commented to avoid compilation error):
        // int notFinal = 30;
        // Supplier<Integer> invalid = () -> notFinal * 2;
        // notFinal = 40; // This makes it NOT effectively final
        System.out.println("Error: Cannot modify local variable in lambda");
    }

    private String getInstanceMessage() {
        return "Hello from instance";
    }

    public static void main(String[] args) {
        LambdaScope demo = new LambdaScope();
        demo.demonstrateScope();
    }
}`,
    hint1: `Local variables used in lambdas must be final or effectively final (never reassigned). This is because lambdas can outlive the method scope and need stable variable values.`,
    hint2: `In lambdas, 'this' refers to the enclosing class instance, not the lambda itself. This is different from anonymous classes where 'this' refers to the anonymous class instance.`,
    whyItMatters: `Understanding lambda scope rules prevents common bugs and helps write correct functional code. The effectively final requirement ensures thread safety and prevents confusing behavior when lambdas are passed around. Knowing how 'this' works in lambdas versus anonymous classes is essential for accessing instance members correctly and avoiding subtle bugs.`,
    order: 5,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.function.*;

// Test1: Verify lambda accessing effectively final local variable
class Test1 {
    @Test
    public void test() {
        int multiplier = 5;
        IntFunction<Integer> multiply = x -> x * multiplier;
        assertEquals(Integer.valueOf(25), multiply.apply(5));
    }
}

// Test2: Verify lambda can access instance variables
class Test2 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            LambdaScope scope = new LambdaScope();
            scope.demonstrateScope();
            String output = out.toString();
            assertTrue(output.contains("Instance variable") || output.contains("instanceVar"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test3: Verify lambda with method parameter
class Test3 {
    @Test
    public void test() {
        int base = 3;
        Function<Integer, Integer> power = exp -> {
            int result = 1;
            for (int i = 0; i < exp; i++) {
                result *= base;
            }
            return result;
        };
        assertEquals(Integer.valueOf(27), power.apply(3));
    }
}

// Test4: Verify 'this' reference in lambda
class Test4 {
    @Test
    public void test() {
        LambdaScope scope = new LambdaScope();
        Supplier<String> getMsg = () -> scope.getInstanceMessage();
        assertEquals("Hello from instance", getMsg.get());
    }
}

// Test5: Verify closure captures variable value
class Test5 {
    @Test
    public void test() {
        String prefix = "Result: ";
        Function<Integer, String> formatter = num -> prefix + num;
        assertEquals("Result: 42", formatter.apply(42));
    }
}

// Test6: Verify lambda cannot modify local variable
class Test6 {
    @Test
    public void test() {
        final int finalVar = 10;
        Supplier<Integer> useFinal = () -> finalVar * 2;
        assertEquals(Integer.valueOf(20), useFinal.get());
    }
}

// Test7: Verify effectively final variable
class Test7 {
    @Test
    public void test() {
        int effectivelyFinal = 20;
        Supplier<Integer> useVar = () -> effectivelyFinal * 2;
        assertEquals(Integer.valueOf(40), useVar.get());
    }
}

// Test8: Verify lambda can modify instance variable
class Test8 {
    @Test
    public void test() {
        int[] counter = {0};  // Effectively final array allows modification
        Runnable incrementer = () -> counter[0]++;
        incrementer.run();
        incrementer.run();
        incrementer.run();
        assertEquals(3, counter[0]);
    }
}

// Test9: Verify lambda captures variable at definition time
class Test9 {
    @Test
    public void test() {
        String captured = "Original";
        Function<String, String> concat = s -> captured + s;
        assertEquals("OriginalTest", concat.apply("Test"));
    }
}

// Test10: Verify multiple lambdas can access same variable
class Test10 {
    @Test
    public void test() {
        int value = 100;
        Supplier<Integer> s1 = () -> value;
        Supplier<Integer> s2 = () -> value * 2;
        assertEquals(Integer.valueOf(100), s1.get());
        assertEquals(Integer.valueOf(200), s2.get());
    }
}`,
    translations: {
        ru: {
            title: 'Область видимости лямбд и эффективно финальные переменные',
            solutionCode: `import java.util.function.*;

public class LambdaScope {
    private int instanceVar = 10;
    private static int staticVar = 100;

    public void demonstrateScope() {
        System.out.println("=== Доступ к переменным в лямбдах ===");

        // Локальная переменная - должна быть эффективно финальной
        int multiplier = 5;
        int base = 3;

        // Лямбда может обращаться к эффективно финальным локальным переменным
        IntFunction<Integer> multiply = x -> x * multiplier;
        System.out.println("Local variable (effectively final): multiplier = " + multiplier);

        // Лямбда может обращаться к параметрам (эффективно финальным)
        Function<Integer, Integer> power = exp -> {
            int result = 1;
            for (int i = 0; i < exp; i++) {
                result *= base; // base - эффективно финальный параметр
            }
            return result;
        };
        System.out.println("Parameter (effectively final): base = " + base);

        // Лямбда может обращаться и изменять переменные экземпляра
        Runnable modifyInstance = () -> {
            instanceVar += 5;
            System.out.println("Instance variable: count = " + instanceVar);
        };
        modifyInstance.run();

        // Лямбда может обращаться и изменять статические переменные
        Runnable modifyStatic = () -> {
            staticVar += 10;
            System.out.println("Static variable: total = " + staticVar);
        };
        modifyStatic.run();

        // Это вызовет ошибку компиляции:
        // int mutableVar = 10;
        // Runnable invalid = () -> mutableVar++; // Ошибка: должна быть final или эффективно final
        // mutableVar = 20; // Это переназначение делает её не эффективно final

        System.out.println("\\n=== Ссылка This ===");

        // В лямбде 'this' ссылается на объемлющий экземпляр
        Supplier<String> lambdaThis = () -> {
            System.out.println("Lambda this: " + this.getClass().getName() + "@" +
                Integer.toHexString(this.hashCode()));
            return this.getInstanceMessage();
        };
        lambdaThis.get();

        // В анонимном классе 'this' ссылается на экземпляр анонимного класса
        Supplier<String> anonymousThis = new Supplier<>() {
            @Override
            public String get() {
                System.out.println("Anonymous class this: " + this.getClass().getName() +
                    "@" + Integer.toHexString(this.hashCode()));
                return "Anonymous class";
            }
        };
        anonymousThis.get();

        System.out.println("Instance method called: " + lambdaThis.get());

        System.out.println("\\n=== Поведение замыкания ===");

        // Лямбда захватывает переменные из объемлющей области (замыкание)
        String prefix = "Result: ";
        Function<Integer, String> formatter = num -> prefix + num;
        System.out.println("Captured value: prefix = \\"" + prefix + "\\"");
        System.out.println("Lambda output: " + formatter.apply(42));

        // Можно создать новую переменную с тем же именем во внешней области
        String newPrefix = "Output: ";
        System.out.println("Outer scope modified: newPrefix = \\"" + newPrefix + "\\"");
        // Исходная лямбда по-прежнему использует захваченное значение 'prefix'

        System.out.println("\\n=== Ограничение эффективно финальных ===");

        // Final переменная - явно объявлена
        final int finalVar = 10;
        Supplier<Integer> useFinal = () -> finalVar * 2;
        System.out.println("Valid: Using final variable = " + finalVar);

        // Эффективно финальная - никогда не переназначается
        int effectivelyFinalVar = 20;
        Supplier<Integer> useEffectivelyFinal = () -> effectivelyFinalVar * 2;
        System.out.println("Valid: Using effectively final variable = " + effectivelyFinalVar);

        // Это демонстрирует ошибку (закомментировано, чтобы избежать ошибки компиляции):
        // int notFinal = 30;
        // Supplier<Integer> invalid = () -> notFinal * 2;
        // notFinal = 40; // Это делает её НЕ эффективно final
        System.out.println("Error: Cannot modify local variable in lambda");
    }

    private String getInstanceMessage() {
        return "Hello from instance";
    }

    public static void main(String[] args) {
        LambdaScope demo = new LambdaScope();
        demo.demonstrateScope();
    }
}`,
            description: `# Область видимости лямбд и эффективно финальные переменные

Лямбда-выражения могут обращаться к переменным из их объемлющей области (замыкание), но с ограничениями. Локальные переменные должны быть final или эффективно финальными. Понимание правил области видимости лямбд имеет решающее значение для избежания распространенных ошибок и написания корректного функционального кода.

## Требования:
1. Продемонстрируйте доступ к переменным в лямбдах:
   1.1. Переменные экземпляра (могут быть изменены)
   1.2. Статические переменные (могут быть изменены)
   1.3. Локальные переменные (должны быть эффективно финальными)
   1.4. Параметры методов (эффективно финальные)

2. Покажите концепцию эффективно финальных:
   2.1. Переменные, которые никогда не переназначаются
   2.2. Ошибка компиляции при попытке изменения
   2.3. Почему существует это ограничение

3. Продемонстрируйте ссылку 'this':
   3.1. 'this' в лямбде ссылается на объемлющий экземпляр
   3.2. Разница с анонимными классами
   3.3. Доступ к методам и полям экземпляра

4. Покажите поведение замыкания:
   4.1. Захват переменных из внешней области
   4.2. Затенение переменных
   4.3. Лучшие практики доступа к переменным в лямбдах

## Пример вывода:
\`\`\`
=== Variable Access in Lambdas ===
Instance variable: count = 10
Static variable: total = 100
Local variable (effectively final): multiplier = 5
Parameter (effectively final): base = 3

=== This Reference ===
Lambda this: MethodScope@hashcode
Anonymous class this: Anonymous@hashcode
Instance method called: Hello from instance

=== Closure Behavior ===
Captured value: prefix = "Result: "
Lambda output: Result: 42
Outer scope modified: newPrefix = "Output: "

=== Effectively Final Restriction ===
Valid: Using final variable = 10
Valid: Using effectively final variable = 20
Error: Cannot modify local variable in lambda
\`\`\``,
            hint1: `Локальные переменные, используемые в лямбдах, должны быть final или эффективно финальными (никогда не переназначаться). Это потому, что лямбды могут пережить область видимости метода и нуждаются в стабильных значениях переменных.`,
            hint2: `В лямбдах 'this' ссылается на экземпляр объемлющего класса, а не на саму лямбду. Это отличается от анонимных классов, где 'this' ссылается на экземпляр анонимного класса.`,
            whyItMatters: `Понимание правил области видимости лямбд предотвращает распространенные ошибки и помогает писать корректный функциональный код. Требование эффективно финальных переменных обеспечивает потокобезопасность и предотвращает запутывающее поведение, когда лямбды передаются. Знание того, как работает 'this' в лямбдах по сравнению с анонимными классами, необходимо для правильного доступа к членам экземпляра и избежания тонких ошибок.`
        },
        uz: {
            title: `Lambda ko'rish doirasi va samarali yakuniy`,
            solutionCode: `import java.util.function.*;

public class LambdaScope {
    private int instanceVar = 10;
    private static int staticVar = 100;

    public void demonstrateScope() {
        System.out.println("=== Lambdalarda o'zgaruvchilarga kirish ===");

        // Lokal o'zgaruvchi - samarali yakuniy bo'lishi kerak
        int multiplier = 5;
        int base = 3;

        // Lambda samarali yakuniy lokal o'zgaruvchilarga kirishi mumkin
        IntFunction<Integer> multiply = x -> x * multiplier;
        System.out.println("Local variable (effectively final): multiplier = " + multiplier);

        // Lambda parametrlarga kirishi mumkin (samarali yakuniy)
        Function<Integer, Integer> power = exp -> {
            int result = 1;
            for (int i = 0; i < exp; i++) {
                result *= base; // base samarali yakuniy parametr
            }
            return result;
        };
        System.out.println("Parameter (effectively final): base = " + base);

        // Lambda instansiya o'zgaruvchilariga kirishi va o'zgartirishi mumkin
        Runnable modifyInstance = () -> {
            instanceVar += 5;
            System.out.println("Instance variable: count = " + instanceVar);
        };
        modifyInstance.run();

        // Lambda statik o'zgaruvchilarga kirishi va o'zgartirishi mumkin
        Runnable modifyStatic = () -> {
            staticVar += 10;
            System.out.println("Static variable: total = " + staticVar);
        };
        modifyStatic.run();

        // Bu kompilyatsiya xatosiga olib keladi:
        // int mutableVar = 10;
        // Runnable invalid = () -> mutableVar++; // Xato: final yoki samarali final bo'lishi kerak
        // mutableVar = 20; // Bu qayta tayinlash uni samarali final emas qiladi

        System.out.println("\\n=== This havolasi ===");

        // Lambdada 'this' o'rab turgan instansiyaga ishora qiladi
        Supplier<String> lambdaThis = () -> {
            System.out.println("Lambda this: " + this.getClass().getName() + "@" +
                Integer.toHexString(this.hashCode()));
            return this.getInstanceMessage();
        };
        lambdaThis.get();

        // Anonim klassda 'this' anonim klass instansiyasiga ishora qiladi
        Supplier<String> anonymousThis = new Supplier<>() {
            @Override
            public String get() {
                System.out.println("Anonymous class this: " + this.getClass().getName() +
                    "@" + Integer.toHexString(this.hashCode()));
                return "Anonymous class";
            }
        };
        anonymousThis.get();

        System.out.println("Instance method called: " + lambdaThis.get());

        System.out.println("\\n=== Yopilish xatti-harakati ===");

        // Lambda o'rab turgan doiradan o'zgaruvchilarni ushlaydi (yopilish)
        String prefix = "Result: ";
        Function<Integer, String> formatter = num -> prefix + num;
        System.out.println("Captured value: prefix = \\"" + prefix + "\\"");
        System.out.println("Lambda output: " + formatter.apply(42));

        // Tashqi doirada xuddi shu nom bilan yangi o'zgaruvchi yaratish mumkin
        String newPrefix = "Output: ";
        System.out.println("Outer scope modified: newPrefix = \\"" + newPrefix + "\\"");
        // Asl lambda hali ham ushlangan 'prefix' qiymatidan foydalanadi

        System.out.println("\\n=== Samarali yakuniy cheklov ===");

        // Final o'zgaruvchi - aniq e'lon qilingan
        final int finalVar = 10;
        Supplier<Integer> useFinal = () -> finalVar * 2;
        System.out.println("Valid: Using final variable = " + finalVar);

        // Samarali yakuniy - hech qachon qayta tayinlanmaydi
        int effectivelyFinalVar = 20;
        Supplier<Integer> useEffectivelyFinal = () -> effectivelyFinalVar * 2;
        System.out.println("Valid: Using effectively final variable = " + effectivelyFinalVar);

        // Bu xatoni namoyish qiladi (kompilyatsiya xatosidan qochish uchun sharh qilingan):
        // int notFinal = 30;
        // Supplier<Integer> invalid = () -> notFinal * 2;
        // notFinal = 40; // Bu uni samarali final EMAS qiladi
        System.out.println("Error: Cannot modify local variable in lambda");
    }

    private String getInstanceMessage() {
        return "Hello from instance";
    }

    public static void main(String[] args) {
        LambdaScope demo = new LambdaScope();
        demo.demonstrateScope();
    }
}`,
            description: `# Lambda ko'rish doirasi va samarali yakuniy

Lambda ifodalari o'rab turgan doiradan (yopilish) o'zgaruvchilarga kirishi mumkin, lekin cheklovlar bilan. Lokal o'zgaruvchilar final yoki samarali yakuniy bo'lishi kerak. Lambda ko'rish doirasi qoidalarini tushunish umumiy xatolardan qochish va to'g'ri funksional kod yozish uchun juda muhimdir.

## Talablar:
1. Lambdalarda o'zgaruvchilarga kirishni namoyish eting:
   1.1. Instansiya o'zgaruvchilari (o'zgartirilishi mumkin)
   1.2. Statik o'zgaruvchilar (o'zgartirilishi mumkin)
   1.3. Lokal o'zgaruvchilar (samarali yakuniy bo'lishi kerak)
   1.4. Metod parametrlari (samarali yakuniy)

2. Samarali yakuniy kontseptsiyasini ko'rsating:
   2.1. Hech qachon qayta tayinlanmaydigan o'zgaruvchilar
   2.2. O'zgartirishga urinishda kompilyatsiya xatosi
   2.3. Nima uchun bu cheklov mavjud

3. 'this' havolasini namoyish eting:
   3.1. Lambdada 'this' o'rab turgan instansiyaga ishora qiladi
   3.2. Anonim klasslardan farq
   3.3. Instansiya metodlari va maydonlariga kirish

4. Yopilish xatti-harakatini ko'rsating:
   4.1. Tashqi doiradan o'zgaruvchilarni ushlash
   4.2. O'zgaruvchi soyalash
   4.3. Lambda o'zgaruvchilarga kirish uchun eng yaxshi amaliyotlar

## Chiqish namunasi:
\`\`\`
=== Variable Access in Lambdas ===
Instance variable: count = 10
Static variable: total = 100
Local variable (effectively final): multiplier = 5
Parameter (effectively final): base = 3

=== This Reference ===
Lambda this: MethodScope@hashcode
Anonymous class this: Anonymous@hashcode
Instance method called: Hello from instance

=== Closure Behavior ===
Captured value: prefix = "Result: "
Lambda output: Result: 42
Outer scope modified: newPrefix = "Output: "

=== Effectively Final Restriction ===
Valid: Using final variable = 10
Valid: Using effectively final variable = 20
Error: Cannot modify local variable in lambda
\`\`\``,
            hint1: `Lambdalarda ishlatilgan lokal o'zgaruvchilar final yoki samarali yakuniy bo'lishi kerak (hech qachon qayta tayinlanmaslik). Bu lambdalar metod ko'rish doirasidan oshib ketishi va barqaror o'zgaruvchi qiymatlariga muhtoj bo'lganligi uchun.`,
            hint2: `Lambdalarda 'this' o'rab turgan klass instansiyasiga ishora qiladi, lambdaning o'ziga emas. Bu anonim klasslardan farq qiladi, bunda 'this' anonim klass instansiyasiga ishora qiladi.`,
            whyItMatters: `Lambda ko'rish doirasi qoidalarini tushunish umumiy xatolarni oldini oladi va to'g'ri funksional kod yozishga yordam beradi. Samarali yakuniy talabi thread xavfsizligini ta'minlaydi va lambdalar o'tkazilganda chalkash xatti-harakatning oldini oladi. Lambdalarda va anonim klasslarda 'this' qanday ishlashini bilish instansiya a'zolariga to'g'ri kirish va nozik xatolardan qochish uchun zarurdir.`
        }
    }
};

export default task;
