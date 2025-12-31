import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-state',
	title: 'State Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'behavioral', 'state'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `## State Pattern

The **State** pattern allows an object to alter its behavior when its internal state changes. The object will appear to change its class.

---

### Key Components

| Component | Description |
|-----------|-------------|
| **State** | Interface declaring state-specific behavior methods |
| **ConcreteState** | Implements behavior for a particular state |
| **Context** | Maintains current state and delegates behavior to it |

---

### Your Task

Implement a media player with state-dependent behavior:

1. **State interface** - \`play()\`, \`pause()\`, \`stop()\` methods
2. **MediaPlayer (Context)** - holds current state, delegates to it
3. **StoppedState** - play starts, pause/stop are invalid
4. **PlayingState** - play is no-op, pause transitions, stop transitions
5. **PausedState** - play resumes, pause is no-op, stop transitions

---

### Example Usage

\`\`\`java
MediaPlayer player = new MediaPlayer();	// starts in StoppedState

String r1 = player.play();	// "Starting playback" → transitions to PlayingState
String r2 = player.play();	// "Already playing" → stays in PlayingState

String r3 = player.pause();	// "Pausing playback" → transitions to PausedState
String r4 = player.pause();	// "Already paused" → stays in PausedState

String r5 = player.play();	// "Resuming playback" → transitions to PlayingState
String r6 = player.stop();	// "Stopping playback" → transitions to StoppedState

String r7 = player.pause();	// "Cannot pause, already stopped" → invalid operation
String r8 = player.stop();	// "Already stopped" → stays in StoppedState
\`\`\`

---

### Key Insight

State replaces complex conditional logic with polymorphism. Each state class encapsulates the behavior specific to that state, making it easy to add new states without modifying existing code.`,
	initialCode: `interface State {
    String play(MediaPlayer player);
    String pause(MediaPlayer player);
    String stop(MediaPlayer player);
}

class MediaPlayer {
    private State state;

    public MediaPlayer() {
    }

    public void setState(State state) {
    }

    public String play() { return state.play(this); }
    public String pause() { return state.pause(this); }
    public String stop() { return state.stop(this); }
}

class StoppedState implements State {
    @Override
    public String play(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
    @Override
    public String pause(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
    @Override
    public String stop(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
}

class PlayingState implements State {
    @Override
    public String play(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
    @Override
    public String pause(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
    @Override
    public String stop(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
}

class PausedState implements State {
    @Override
    public String play(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
    @Override
    public String pause(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
    @Override
    public String stop(MediaPlayer player) {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `interface State {	// State interface - defines state-specific behavior
    String play(MediaPlayer player);	// handle play action
    String pause(MediaPlayer player);	// handle pause action
    String stop(MediaPlayer player);	// handle stop action
}

class MediaPlayer {	// Context - maintains current state
    private State state;	// reference to current state

    public MediaPlayer() {	// constructor - start in stopped state
        this.state = new StoppedState();	// initial state is stopped
    }

    public void setState(State state) {	// change current state
        this.state = state;	// set new state
    }

    public String play() { return state.play(this); }	// delegate to current state
    public String pause() { return state.pause(this); }	// delegate to current state
    public String stop() { return state.stop(this); }	// delegate to current state
}

class StoppedState implements State {	// ConcreteState - stopped state behavior
    @Override
    public String play(MediaPlayer player) {	// play from stopped
        player.setState(new PlayingState());	// transition to playing
        return "Starting playback";	// action performed
    }
    @Override
    public String pause(MediaPlayer player) {	// pause from stopped
        return "Cannot pause, already stopped";	// invalid action
    }
    @Override
    public String stop(MediaPlayer player) {	// stop from stopped
        return "Already stopped";	// no-op, already in this state
    }
}

class PlayingState implements State {	// ConcreteState - playing state behavior
    @Override
    public String play(MediaPlayer player) {	// play from playing
        return "Already playing";	// no-op, already in this state
    }
    @Override
    public String pause(MediaPlayer player) {	// pause from playing
        player.setState(new PausedState());	// transition to paused
        return "Pausing playback";	// action performed
    }
    @Override
    public String stop(MediaPlayer player) {	// stop from playing
        player.setState(new StoppedState());	// transition to stopped
        return "Stopping playback";	// action performed
    }
}

class PausedState implements State {	// ConcreteState - paused state behavior
    @Override
    public String play(MediaPlayer player) {	// play from paused
        player.setState(new PlayingState());	// transition to playing
        return "Resuming playback";	// action performed
    }
    @Override
    public String pause(MediaPlayer player) {	// pause from paused
        return "Already paused";	// no-op, already in this state
    }
    @Override
    public String stop(MediaPlayer player) {	// stop from paused
        player.setState(new StoppedState());	// transition to stopped
        return "Stopping playback";	// action performed
    }
}`,
	hint1: `### Understanding State Transitions

Each state handles actions differently and may trigger transitions:

\`\`\`java
// StoppedState transitions:
play()  → PlayingState  ("Starting playback")
pause() → stays Stopped ("Cannot pause, already stopped")
stop()  → stays Stopped ("Already stopped")

// PlayingState transitions:
play()  → stays Playing ("Already playing")
pause() → PausedState   ("Pausing playback")
stop()  → StoppedState  ("Stopping playback")

// PausedState transitions:
play()  → PlayingState  ("Resuming playback")
pause() → stays Paused  ("Already paused")
stop()  → StoppedState  ("Stopping playback")
\`\`\`

Use \`player.setState(new NewState())\` to transition.`,
	hint2: `### Implementing State Classes

Each state class follows this pattern:

\`\`\`java
class StoppedState implements State {
    @Override
    public String play(MediaPlayer player) {
        // 1. Perform transition
        player.setState(new PlayingState());
        // 2. Return result
        return "Starting playback";
    }

    @Override
    public String pause(MediaPlayer player) {
        // Invalid action - return error message
        return "Cannot pause, already stopped";
    }

    @Override
    public String stop(MediaPlayer player) {
        // No-op - already in this state
        return "Already stopped";
    }
}
\`\`\`

Remember: "Already" for same state, transition message for valid actions, error for invalid actions.`,
	whyItMatters: `## Why State Pattern Matters

### The Problem and Solution

**Without State:**
\`\`\`java
// Complex conditionals spread throughout code
class MediaPlayer {
    private String state = "stopped";	// state as string

    public String play() {
        if (state.equals("stopped")) {	// check current state
            state = "playing";
            return "Starting playback";
        } else if (state.equals("playing")) {	// another condition
            return "Already playing";
        } else if (state.equals("paused")) {	// yet another condition
            state = "playing";
            return "Resuming playback";
        }
        return "Unknown state";
    }
    // pause() and stop() have similar conditionals...
    // Adding new state means modifying ALL methods!
}
\`\`\`

**With State:**
\`\`\`java
// Clean, encapsulated state behavior
class MediaPlayer {
    private State state;	// state as object

    public String play() {
        return state.play(this);	// delegate to state
    }
}

class PlayingState implements State {
    public String play(MediaPlayer p) {
        return "Already playing";	// state-specific behavior
    }
}
// Adding new state = add new class, no changes to MediaPlayer!
\`\`\`

---

### Real-World Applications

| Application | Context | States |
|-------------|---------|--------|
| **TCP Connection** | Socket | Closed, Listen, Established, Closing |
| **Order Processing** | Order | Pending, Paid, Shipped, Delivered, Cancelled |
| **Document** | Document | Draft, Moderation, Published, Archived |
| **Thread** | Thread | New, Runnable, Blocked, Waiting, Terminated |
| **Game Character** | Character | Idle, Walking, Running, Jumping, Falling |

---

### Production Pattern: Order State Machine

\`\`\`java
// State interface for order processing
interface OrderState {	// order state contract
    OrderState pay(Order order);	// process payment
    OrderState ship(Order order);	// ship the order
    OrderState deliver(Order order);	// mark as delivered
    OrderState cancel(Order order);	// cancel the order
    String getStatus();	// get current status name
}

class PendingState implements OrderState {	// initial state - awaiting payment
    @Override
    public OrderState pay(Order order) {	// payment received
        order.processPayment();	// process the payment
        return new PaidState();	// transition to paid
    }

    @Override
    public OrderState ship(Order order) {	// can't ship unpaid
        throw new IllegalStateException("Cannot ship unpaid order");
    }

    @Override
    public OrderState deliver(Order order) {	// can't deliver unshipped
        throw new IllegalStateException("Cannot deliver unshipped order");
    }

    @Override
    public OrderState cancel(Order order) {	// cancel is allowed
        order.notifyCustomer("Order cancelled");	// notify customer
        return new CancelledState();	// transition to cancelled
    }

    @Override
    public String getStatus() { return "PENDING"; }	// status name
}

class PaidState implements OrderState {	// paid - ready to ship
    @Override
    public OrderState pay(Order order) {	// already paid
        throw new IllegalStateException("Already paid");
    }

    @Override
    public OrderState ship(Order order) {	// ready to ship
        order.createShipment();	// create shipment
        order.notifyCustomer("Your order has been shipped");	// notify
        return new ShippedState();	// transition to shipped
    }

    @Override
    public OrderState deliver(Order order) {	// can't deliver yet
        throw new IllegalStateException("Must ship before delivery");
    }

    @Override
    public OrderState cancel(Order order) {	// cancel with refund
        order.refundPayment();	// refund the payment
        order.notifyCustomer("Order cancelled, refund processed");
        return new CancelledState();	// transition to cancelled
    }

    @Override
    public String getStatus() { return "PAID"; }	// status name
}

class ShippedState implements OrderState {	// shipped - in transit
    @Override
    public OrderState pay(Order order) {	// already paid
        throw new IllegalStateException("Already paid");
    }

    @Override
    public OrderState ship(Order order) {	// already shipped
        throw new IllegalStateException("Already shipped");
    }

    @Override
    public OrderState deliver(Order order) {	// delivery confirmed
        order.notifyCustomer("Your order has been delivered");
        return new DeliveredState();	// transition to delivered
    }

    @Override
    public OrderState cancel(Order order) {	// too late to cancel
        throw new IllegalStateException("Cannot cancel shipped order");
    }

    @Override
    public String getStatus() { return "SHIPPED"; }	// status name
}

// Context class
class Order {	// context holding state
    private OrderState state = new PendingState();	// start in pending

    public void pay() { state = state.pay(this); }	// delegate to state
    public void ship() { state = state.ship(this); }	// delegate to state
    public void deliver() { state = state.deliver(this); }	// delegate to state
    public void cancel() { state = state.cancel(this); }	// delegate to state
    public String getStatus() { return state.getStatus(); }	// get current status
}

// Usage:
Order order = new Order();	// create new order
order.pay();	// PENDING → PAID
order.ship();	// PAID → SHIPPED
order.deliver();	// SHIPPED → DELIVERED
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Creating new states each transition** | Memory overhead | Use singleton states for stateless states |
| **State knowing too much** | Tight coupling | Pass only necessary context to state |
| **Missing transitions** | Undefined behavior | Handle all possible actions in each state |
| **No initial state** | NullPointerException | Always set initial state in constructor |
| **Circular transitions** | Infinite loops | Design clear, acyclic state machine |`,
	order: 3,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;

// Test1: MediaPlayer starts in StoppedState
class Test1 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        String result = player.play();
        assertEquals("Starting playback", result);
    }
}

// Test2: Playing play returns already playing
class Test2 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        player.play();
        String result = player.play();
        assertEquals("Already playing", result);
    }
}

// Test3: Playing pause transitions to paused
class Test3 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        player.play();
        String result = player.pause();
        assertEquals("Pausing playback", result);
    }
}

// Test4: Paused play resumes
class Test4 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        player.play();
        player.pause();
        String result = player.play();
        assertEquals("Resuming playback", result);
    }
}

// Test5: Stopped pause is invalid
class Test5 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        String result = player.pause();
        assertEquals("Cannot pause, already stopped", result);
    }
}

// Test6: Stopped stop returns already stopped
class Test6 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        String result = player.stop();
        assertEquals("Already stopped", result);
    }
}

// Test7: Playing stop transitions to stopped
class Test7 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        player.play();
        String result = player.stop();
        assertEquals("Stopping playback", result);
    }
}

// Test8: Paused pause returns already paused
class Test8 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        player.play();
        player.pause();
        String result = player.pause();
        assertEquals("Already paused", result);
    }
}

// Test9: Paused stop transitions to stopped
class Test9 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        player.play();
        player.pause();
        String result = player.stop();
        assertEquals("Stopping playback", result);
    }
}

// Test10: Full state cycle
class Test10 {
    @Test
    public void test() {
        MediaPlayer player = new MediaPlayer();
        assertEquals("Starting playback", player.play());
        assertEquals("Pausing playback", player.pause());
        assertEquals("Resuming playback", player.play());
        assertEquals("Stopping playback", player.stop());
        assertEquals("Starting playback", player.play());
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн State (Состояние)',
			description: `## Паттерн State (Состояние)

Паттерн **State** позволяет объекту изменять своё поведение при изменении внутреннего состояния. Объект будет выглядеть так, будто изменился его класс.

---

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **State** | Интерфейс, объявляющий методы поведения для состояний |
| **ConcreteState** | Реализует поведение для конкретного состояния |
| **Context** | Хранит текущее состояние и делегирует ему поведение |

---

### Ваша задача

Реализуйте медиаплеер с зависящим от состояния поведением:

1. **Интерфейс State** - методы \`play()\`, \`pause()\`, \`stop()\`
2. **MediaPlayer (Context)** - хранит текущее состояние, делегирует ему
3. **StoppedState** - play начинает, pause/stop невалидны
4. **PlayingState** - play без действия, pause переходит, stop переходит
5. **PausedState** - play возобновляет, pause без действия, stop переходит

---

### Пример использования

\`\`\`java
MediaPlayer player = new MediaPlayer();	// начинает в StoppedState

String r1 = player.play();	// "Starting playback" → переход в PlayingState
String r2 = player.play();	// "Already playing" → остаётся в PlayingState

String r3 = player.pause();	// "Pausing playback" → переход в PausedState
String r4 = player.pause();	// "Already paused" → остаётся в PausedState

String r5 = player.play();	// "Resuming playback" → переход в PlayingState
String r6 = player.stop();	// "Stopping playback" → переход в StoppedState

String r7 = player.pause();	// "Cannot pause, already stopped" → недопустимая операция
String r8 = player.stop();	// "Already stopped" → остаётся в StoppedState
\`\`\`

---

### Ключевая идея

State заменяет сложную условную логику полиморфизмом. Каждый класс состояния инкапсулирует поведение, специфичное для этого состояния, что упрощает добавление новых состояний без изменения существующего кода.`,
			hint1: `### Понимание переходов состояний

Каждое состояние обрабатывает действия по-разному и может вызывать переходы:

\`\`\`java
// Переходы StoppedState:
play()  → PlayingState  ("Starting playback")
pause() → остаётся Stopped ("Cannot pause, already stopped")
stop()  → остаётся Stopped ("Already stopped")

// Переходы PlayingState:
play()  → остаётся Playing ("Already playing")
pause() → PausedState   ("Pausing playback")
stop()  → StoppedState  ("Stopping playback")

// Переходы PausedState:
play()  → PlayingState  ("Resuming playback")
pause() → остаётся Paused  ("Already paused")
stop()  → StoppedState  ("Stopping playback")
\`\`\`

Используйте \`player.setState(new NewState())\` для перехода.`,
			hint2: `### Реализация классов состояний

Каждый класс состояния следует этому шаблону:

\`\`\`java
class StoppedState implements State {
    @Override
    public String play(MediaPlayer player) {
        // 1. Выполняем переход
        player.setState(new PlayingState());
        // 2. Возвращаем результат
        return "Starting playback";
    }

    @Override
    public String pause(MediaPlayer player) {
        // Недопустимое действие - возвращаем сообщение об ошибке
        return "Cannot pause, already stopped";
    }

    @Override
    public String stop(MediaPlayer player) {
        // Без действия - уже в этом состоянии
        return "Already stopped";
    }
}
\`\`\`

Помните: "Already" для того же состояния, сообщение о переходе для допустимых действий, ошибка для недопустимых.`,
			whyItMatters: `## Почему паттерн State важен

### Проблема и решение

**Без State:**
\`\`\`java
// Сложные условия разбросаны по коду
class MediaPlayer {
    private String state = "stopped";	// состояние как строка

    public String play() {
        if (state.equals("stopped")) {	// проверка текущего состояния
            state = "playing";
            return "Starting playback";
        } else if (state.equals("playing")) {	// ещё одно условие
            return "Already playing";
        } else if (state.equals("paused")) {	// и ещё одно условие
            state = "playing";
            return "Resuming playback";
        }
        return "Unknown state";
    }
    // pause() и stop() имеют аналогичные условия...
    // Добавление нового состояния требует изменения ВСЕХ методов!
}
\`\`\`

**С State:**
\`\`\`java
// Чистое, инкапсулированное поведение состояний
class MediaPlayer {
    private State state;	// состояние как объект

    public String play() {
        return state.play(this);	// делегируем состоянию
    }
}

class PlayingState implements State {
    public String play(MediaPlayer p) {
        return "Already playing";	// поведение конкретного состояния
    }
}
// Добавление нового состояния = добавить новый класс, без изменений в MediaPlayer!
\`\`\`

---

### Применение в реальном мире

| Применение | Context | Состояния |
|------------|---------|-----------|
| **TCP соединение** | Socket | Closed, Listen, Established, Closing |
| **Обработка заказов** | Order | Pending, Paid, Shipped, Delivered, Cancelled |
| **Документ** | Document | Draft, Moderation, Published, Archived |
| **Поток** | Thread | New, Runnable, Blocked, Waiting, Terminated |
| **Игровой персонаж** | Character | Idle, Walking, Running, Jumping, Falling |

---

### Продакшен паттерн: Машина состояний заказа

\`\`\`java
// Интерфейс состояния для обработки заказов
interface OrderState {	// контракт состояния заказа
    OrderState pay(Order order);	// обработать оплату
    OrderState ship(Order order);	// отправить заказ
    OrderState deliver(Order order);	// отметить как доставленный
    OrderState cancel(Order order);	// отменить заказ
    String getStatus();	// получить название текущего статуса
}

class PendingState implements OrderState {	// начальное состояние - ожидание оплаты
    @Override
    public OrderState pay(Order order) {	// оплата получена
        order.processPayment();	// обработать оплату
        return new PaidState();	// переход в оплачено
    }

    @Override
    public OrderState ship(Order order) {	// нельзя отправить неоплаченный
        throw new IllegalStateException("Cannot ship unpaid order");
    }

    @Override
    public OrderState deliver(Order order) {	// нельзя доставить неотправленный
        throw new IllegalStateException("Cannot deliver unshipped order");
    }

    @Override
    public OrderState cancel(Order order) {	// отмена разрешена
        order.notifyCustomer("Order cancelled");	// уведомить клиента
        return new CancelledState();	// переход в отменён
    }

    @Override
    public String getStatus() { return "PENDING"; }	// название статуса
}

class PaidState implements OrderState {	// оплачен - готов к отправке
    @Override
    public OrderState pay(Order order) {	// уже оплачен
        throw new IllegalStateException("Already paid");
    }

    @Override
    public OrderState ship(Order order) {	// готов к отправке
        order.createShipment();	// создать отправление
        order.notifyCustomer("Your order has been shipped");	// уведомить
        return new ShippedState();	// переход в отправлен
    }

    @Override
    public OrderState deliver(Order order) {	// ещё нельзя доставить
        throw new IllegalStateException("Must ship before delivery");
    }

    @Override
    public OrderState cancel(Order order) {	// отмена с возвратом
        order.refundPayment();	// вернуть оплату
        order.notifyCustomer("Order cancelled, refund processed");
        return new CancelledState();	// переход в отменён
    }

    @Override
    public String getStatus() { return "PAID"; }	// название статуса
}

class ShippedState implements OrderState {	// отправлен - в пути
    @Override
    public OrderState pay(Order order) {	// уже оплачен
        throw new IllegalStateException("Already paid");
    }

    @Override
    public OrderState ship(Order order) {	// уже отправлен
        throw new IllegalStateException("Already shipped");
    }

    @Override
    public OrderState deliver(Order order) {	// доставка подтверждена
        order.notifyCustomer("Your order has been delivered");
        return new DeliveredState();	// переход в доставлен
    }

    @Override
    public OrderState cancel(Order order) {	// слишком поздно отменять
        throw new IllegalStateException("Cannot cancel shipped order");
    }

    @Override
    public String getStatus() { return "SHIPPED"; }	// название статуса
}

// Класс контекста
class Order {	// контекст, хранящий состояние
    private OrderState state = new PendingState();	// начало в pending

    public void pay() { state = state.pay(this); }	// делегируем состоянию
    public void ship() { state = state.ship(this); }	// делегируем состоянию
    public void deliver() { state = state.deliver(this); }	// делегируем состоянию
    public void cancel() { state = state.cancel(this); }	// делегируем состоянию
    public String getStatus() { return state.getStatus(); }	// получить текущий статус
}

// Использование:
Order order = new Order();	// создать новый заказ
order.pay();	// PENDING → PAID
order.ship();	// PAID → SHIPPED
order.deliver();	// SHIPPED → DELIVERED
\`\`\`

---

### Частые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Создание новых состояний при каждом переходе** | Накладные расходы памяти | Используйте синглтон-состояния для stateless состояний |
| **Состояние знает слишком много** | Тесная связанность | Передавайте только необходимый контекст |
| **Отсутствующие переходы** | Неопределённое поведение | Обрабатывайте все действия в каждом состоянии |
| **Нет начального состояния** | NullPointerException | Всегда устанавливайте начальное состояние в конструкторе |
| **Циклические переходы** | Бесконечные циклы | Проектируйте чёткую ациклическую машину состояний |`
		},
		uz: {
			title: 'State (Holat) Pattern',
			description: `## State (Holat) Pattern

**State** pattern obyektga ichki holati o'zgarganda o'z xatti-harakatini o'zgartirishga imkon beradi. Obyekt o'z klassi o'zgargandek ko'rinadi.

---

### Asosiy Komponentlar

| Komponent | Tavsif |
|-----------|--------|
| **State** | Holatga xos xatti-harakat metodlarini e'lon qiluvchi interfeys |
| **ConcreteState** | Ma'lum holat uchun xatti-harakatni amalga oshiradi |
| **Context** | Joriy holatni saqlaydi va unga xatti-harakatni delegatsiya qiladi |

---

### Vazifangiz

Holatga bog'liq xatti-harakatli media pleyer amalga oshiring:

1. **State interfeysi** - \`play()\`, \`pause()\`, \`stop()\` metodlari
2. **MediaPlayer (Context)** - joriy holatni saqlaydi, unga delegatsiya qiladi
3. **StoppedState** - play boshlaydi, pause/stop yaroqsiz
4. **PlayingState** - play hech nima qilmaydi, pause o'tadi, stop o'tadi
5. **PausedState** - play davom ettiradi, pause hech nima qilmaydi, stop o'tadi

---

### Foydalanish Namunasi

\`\`\`java
MediaPlayer player = new MediaPlayer();	// StoppedState da boshlanadi

String r1 = player.play();	// "Starting playback" → PlayingState ga o'tadi
String r2 = player.play();	// "Already playing" → PlayingState da qoladi

String r3 = player.pause();	// "Pausing playback" → PausedState ga o'tadi
String r4 = player.pause();	// "Already paused" → PausedState da qoladi

String r5 = player.play();	// "Resuming playback" → PlayingState ga o'tadi
String r6 = player.stop();	// "Stopping playback" → StoppedState ga o'tadi

String r7 = player.pause();	// "Cannot pause, already stopped" → yaroqsiz operatsiya
String r8 = player.stop();	// "Already stopped" → StoppedState da qoladi
\`\`\`

---

### Asosiy Fikr

State murakkab shartli mantiqni polimorfizm bilan almashtiradi. Har bir holat klassi o'sha holatga xos xatti-harakatni inkapsulyatsiya qiladi, bu mavjud kodni o'zgartirmasdan yangi holatlarni qo'shishni osonlashtiradi.`,
			hint1: `### Holat O'tishlarini Tushunish

Har bir holat harakatlarni boshqacha boshqaradi va o'tishlarni boshlashi mumkin:

\`\`\`java
// StoppedState o'tishlari:
play()  → PlayingState  ("Starting playback")
pause() → Stopped da qoladi ("Cannot pause, already stopped")
stop()  → Stopped da qoladi ("Already stopped")

// PlayingState o'tishlari:
play()  → Playing da qoladi ("Already playing")
pause() → PausedState   ("Pausing playback")
stop()  → StoppedState  ("Stopping playback")

// PausedState o'tishlari:
play()  → PlayingState  ("Resuming playback")
pause() → Paused da qoladi  ("Already paused")
stop()  → StoppedState  ("Stopping playback")
\`\`\`

O'tish uchun \`player.setState(new NewState())\` ishlating.`,
			hint2: `### Holat Klasslarini Amalga Oshirish

Har bir holat klassi ushbu shablonga amal qiladi:

\`\`\`java
class StoppedState implements State {
    @Override
    public String play(MediaPlayer player) {
        // 1. O'tishni amalga oshirish
        player.setState(new PlayingState());
        // 2. Natijani qaytarish
        return "Starting playback";
    }

    @Override
    public String pause(MediaPlayer player) {
        // Yaroqsiz harakat - xato xabarini qaytarish
        return "Cannot pause, already stopped";
    }

    @Override
    public String stop(MediaPlayer player) {
        // Hech nima qilmaslik - allaqachon bu holatda
        return "Already stopped";
    }
}
\`\`\`

Esda tuting: bir xil holat uchun "Already", to'g'ri harakatlar uchun o'tish xabari, yaroqsiz harakatlar uchun xato.`,
			whyItMatters: `## Nima Uchun State Pattern Muhim

### Muammo va Yechim

**State siz:**
\`\`\`java
// Murakkab shartlar kod bo'ylab tarqalgan
class MediaPlayer {
    private String state = "stopped";	// holat satr sifatida

    public String play() {
        if (state.equals("stopped")) {	// joriy holatni tekshirish
            state = "playing";
            return "Starting playback";
        } else if (state.equals("playing")) {	// yana bir shart
            return "Already playing";
        } else if (state.equals("paused")) {	// yana bir shart
            state = "playing";
            return "Resuming playback";
        }
        return "Unknown state";
    }
    // pause() va stop() da ham shunga o'xshash shartlar bor...
    // Yangi holat qo'shish BARCHA metodlarni o'zgartirishni talab qiladi!
}
\`\`\`

**State bilan:**
\`\`\`java
// Toza, inkapsulyatsiya qilingan holat xatti-harakati
class MediaPlayer {
    private State state;	// holat obyekt sifatida

    public String play() {
        return state.play(this);	// holatga delegatsiya
    }
}

class PlayingState implements State {
    public String play(MediaPlayer p) {
        return "Already playing";	// holatga xos xatti-harakat
    }
}
// Yangi holat qo'shish = yangi klass qo'shish, MediaPlayer da o'zgarishlar yo'q!
\`\`\`

---

### Haqiqiy Dunyo Qo'llanilishi

| Qo'llanish | Context | Holatlar |
|------------|---------|----------|
| **TCP Ulanish** | Socket | Closed, Listen, Established, Closing |
| **Buyurtma Qayta Ishlash** | Order | Pending, Paid, Shipped, Delivered, Cancelled |
| **Hujjat** | Document | Draft, Moderation, Published, Archived |
| **Thread** | Thread | New, Runnable, Blocked, Waiting, Terminated |
| **O'yin Personaji** | Character | Idle, Walking, Running, Jumping, Falling |

---

### Prodakshen Pattern: Buyurtma Holat Mashinasi

\`\`\`java
// Buyurtmalarni qayta ishlash uchun holat interfeysi
interface OrderState {	// buyurtma holati shartnomasi
    OrderState pay(Order order);	// to'lovni qayta ishlash
    OrderState ship(Order order);	// buyurtmani jo'natish
    OrderState deliver(Order order);	// yetkazib berilgan deb belgilash
    OrderState cancel(Order order);	// buyurtmani bekor qilish
    String getStatus();	// joriy status nomini olish
}

class PendingState implements OrderState {	// boshlang'ich holat - to'lov kutilmoqda
    @Override
    public OrderState pay(Order order) {	// to'lov qabul qilindi
        order.processPayment();	// to'lovni qayta ishlash
        return new PaidState();	// to'langan ga o'tish
    }

    @Override
    public OrderState ship(Order order) {	// to'lanmaganini jo'natib bo'lmaydi
        throw new IllegalStateException("Cannot ship unpaid order");
    }

    @Override
    public OrderState deliver(Order order) {	// jo'natilmaganini yetkazib bo'lmaydi
        throw new IllegalStateException("Cannot deliver unshipped order");
    }

    @Override
    public OrderState cancel(Order order) {	// bekor qilish ruxsat etiladi
        order.notifyCustomer("Order cancelled");	// mijozni xabardor qilish
        return new CancelledState();	// bekor qilingan ga o'tish
    }

    @Override
    public String getStatus() { return "PENDING"; }	// status nomi
}

class PaidState implements OrderState {	// to'langan - jo'natishga tayyor
    @Override
    public OrderState pay(Order order) {	// allaqachon to'langan
        throw new IllegalStateException("Already paid");
    }

    @Override
    public OrderState ship(Order order) {	// jo'natishga tayyor
        order.createShipment();	// jo'natmani yaratish
        order.notifyCustomer("Your order has been shipped");	// xabardor qilish
        return new ShippedState();	// jo'natilgan ga o'tish
    }

    @Override
    public OrderState deliver(Order order) {	// hali yetkazib bo'lmaydi
        throw new IllegalStateException("Must ship before delivery");
    }

    @Override
    public OrderState cancel(Order order) {	// qaytarish bilan bekor qilish
        order.refundPayment();	// to'lovni qaytarish
        order.notifyCustomer("Order cancelled, refund processed");
        return new CancelledState();	// bekor qilingan ga o'tish
    }

    @Override
    public String getStatus() { return "PAID"; }	// status nomi
}

class ShippedState implements OrderState {	// jo'natilgan - yo'lda
    @Override
    public OrderState pay(Order order) {	// allaqachon to'langan
        throw new IllegalStateException("Already paid");
    }

    @Override
    public OrderState ship(Order order) {	// allaqachon jo'natilgan
        throw new IllegalStateException("Already shipped");
    }

    @Override
    public OrderState deliver(Order order) {	// yetkazib berish tasdiqlandi
        order.notifyCustomer("Your order has been delivered");
        return new DeliveredState();	// yetkazilgan ga o'tish
    }

    @Override
    public OrderState cancel(Order order) {	// bekor qilish uchun juda kech
        throw new IllegalStateException("Cannot cancel shipped order");
    }

    @Override
    public String getStatus() { return "SHIPPED"; }	// status nomi
}

// Context klassi
class Order {	// holatni saqlovchi kontekst
    private OrderState state = new PendingState();	// pending da boshlash

    public void pay() { state = state.pay(this); }	// holatga delegatsiya
    public void ship() { state = state.ship(this); }	// holatga delegatsiya
    public void deliver() { state = state.deliver(this); }	// holatga delegatsiya
    public void cancel() { state = state.cancel(this); }	// holatga delegatsiya
    public String getStatus() { return state.getStatus(); }	// joriy statusni olish
}

// Foydalanish:
Order order = new Order();	// yangi buyurtma yaratish
order.pay();	// PENDING → PAID
order.ship();	// PAID → SHIPPED
order.deliver();	// SHIPPED → DELIVERED
\`\`\`

---

### Oldini Olish Kerak Bo'lgan Xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Har bir o'tishda yangi holatlar yaratish** | Xotira sarfi | Holatsiz holatlar uchun singleton holatlardan foydalaning |
| **Holat juda ko'p biladi** | Qattiq bog'lanish | Holatga faqat kerakli kontekstni uzating |
| **Yo'qolgan o'tishlar** | Aniqlanmagan xatti-harakat | Har bir holatda barcha harakatlarni boshqaring |
| **Boshlang'ich holat yo'q** | NullPointerException | Konstruktorda har doim boshlang'ich holatni o'rnating |
| **Siklik o'tishlar** | Cheksiz sikllar | Aniq, atsiklik holat mashinasini loyihalang |`
		}
	}
};

export default task;
