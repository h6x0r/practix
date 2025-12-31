import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-mediator',
	title: 'Mediator Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'behavioral', 'mediator'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Mediator Pattern

The **Mediator Pattern** defines an object that encapsulates how a set of objects interact. It promotes loose coupling by keeping objects from referring to each other explicitly, letting you vary their interaction independently.

---

### Key Components

| Component | Role |
|-----------|------|
| **Mediator** | Interface defining communication methods |
| **ConcreteMediator** | Coordinates communication between colleagues |
| **Colleague** | Object that communicates through the mediator |
| **ConcreteColleague** | Specific colleague implementation |

---

### Your Task

Implement a **Chat Room** using the Mediator pattern:

1. **ChatRoom** (ConcreteMediator): Manages users and broadcasts messages
2. **User** (Colleague): Sends messages through mediator, receives from others

---

### Example Usage

\`\`\`java
ChatRoom chatRoom = new ChatRoom();	// create mediator (chat room)

User alice = new User("Alice", chatRoom);	// create user with mediator reference
User bob = new User("Bob", chatRoom);	// create another user
User charlie = new User("Charlie", chatRoom);	// create third user

chatRoom.addUser(alice);	// register Alice with mediator
chatRoom.addUser(bob);	// register Bob with mediator
chatRoom.addUser(charlie);	// register Charlie with mediator

alice.send("Hello everyone!");	// Alice sends via mediator

// Bob and Charlie receive: "[Alice]: Hello everyone!"
// Alice does NOT receive her own message
System.out.println(bob.getMessages());	// [[Alice]: Hello everyone!]
System.out.println(charlie.getMessages());	// [[Alice]: Hello everyone!]
System.out.println(alice.getMessages());	// [] (sender doesn't receive)
\`\`\`

---

### Key Insight

> Objects never communicate directly with each other. All communication goes through the mediator, which decides who receives what. This centralizes the communication logic in one place.`,
	initialCode: `import java.util.*;

interface ChatMediator {
    void sendMessage(String msg, User user);
    void addUser(User user);
}

class ChatRoom implements ChatMediator {
    private List<User> users = new ArrayList<>();

    @Override
    public void addUser(User user) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void sendMessage(String msg, User sender) {
        throw new UnsupportedOperationException("TODO");
    }
}

class User {
    private String name;
    private ChatMediator mediator;
    private List<String> messages = new ArrayList<>();

    public User(String name, ChatMediator mediator) {
    }

    public String getName() { return name; }
    public List<String> getMessages() { return messages; }

    public void send(String msg) {
        throw new UnsupportedOperationException("TODO");
    }

    public void receive(String msg, String from) {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `import java.util.*;	// import ArrayList, List for user management

interface ChatMediator {	// Mediator interface - defines communication contract
    void sendMessage(String msg, User user);	// method to broadcast message from a user
    void addUser(User user);	// method to register new user with mediator
}

class ChatRoom implements ChatMediator {	// ConcreteMediator - coordinates all user communication
    private List<User> users = new ArrayList<>();	// stores all registered users

    @Override
    public void addUser(User user) {	// register new user with chat room
        users.add(user);	// add user to the list of participants
    }

    @Override
    public void sendMessage(String msg, User sender) {	// broadcast message to all except sender
        for (User user : users) {	// iterate through all registered users
            if (user != sender) {	// skip the sender (don't echo back)
                user.receive(msg, sender.getName());	// deliver message to recipient
            }
        }
    }
}

class User {	// Colleague - communicates through mediator only
    private String name;	// user identifier
    private ChatMediator mediator;	// reference to mediator for communication
    private List<String> messages = new ArrayList<>();	// stores received messages

    public User(String name, ChatMediator mediator) {	// constructor with mediator injection
        this.name = name;	// set user name
        this.mediator = mediator;	// store mediator reference
    }

    public String getName() { return name; }	// getter for user name
    public List<String> getMessages() { return messages; }	// getter for received messages

    public void send(String msg) {	// send message via mediator
        mediator.sendMessage(msg, this);	// delegate to mediator, pass self as sender
    }

    public void receive(String msg, String from) {	// called by mediator when message arrives
        messages.add("[" + from + "]: " + msg);	// format and store received message
    }
}`,
	hint1: `## Hint 1: User Communication

Users should never communicate directly with each other. All communication goes through the mediator:

\`\`\`java
public void send(String msg) {	// user wants to send message
    mediator.sendMessage(msg, this);	// delegate to mediator with message and sender reference
}

public void receive(String msg, String from) {	// called by mediator
    messages.add("[" + from + "]: " + msg);	// format: [SenderName]: message
}
\`\`\`

The **send()** method delegates to the mediator. The **receive()** method is called BY the mediator when another user sends a message.`,
	hint2: `## Hint 2: Mediator Broadcasting

The mediator is responsible for routing messages to the correct recipients:

\`\`\`java
@Override
public void sendMessage(String msg, User sender) {	// broadcast to all except sender
    for (User user : users) {	// iterate all registered users
        if (user != sender) {	// skip the original sender
            user.receive(msg, sender.getName());	// call receive on each recipient
        }
    }
}

@Override
public void addUser(User user) {	// registration method
    users.add(user);	// simply add to the list
}
\`\`\`

The mediator knows about ALL users and decides who receives each message. Users only know about the mediator, not about each other.`,
	whyItMatters: `## Why Mediator Pattern Matters

### The Problem: Direct Communication Creates Tight Coupling

Without Mediator, objects must know about and communicate with every other object:

\`\`\`java
// ❌ Without Mediator - users communicate directly
class User {	// user class with direct references
    private String name;	// user identifier
    private List<User> contacts = new ArrayList<>();	// must know ALL other users

    public void addContact(User user) {	// must manually manage contacts
        contacts.add(user);	// track each contact individually
    }

    public void sendMessage(String msg) {	// send to all contacts
        for (User contact : contacts) {	// iterate through all known contacts
            contact.receiveMessage(msg, this.name);	// call directly on each
        }
    }
}

// Problem: Each user must know about every other user
// Adding/removing users requires updating ALL other users
User alice = new User("Alice");	// create user
User bob = new User("Bob");	// create another
User charlie = new User("Charlie");	// create third
// Must manually connect everyone to everyone!
alice.addContact(bob);	// alice knows bob
alice.addContact(charlie);	// alice knows charlie
bob.addContact(alice);	// bob knows alice
bob.addContact(charlie);	// bob knows charlie
charlie.addContact(alice);	// charlie knows alice
charlie.addContact(bob);	// charlie knows bob
// N users = N*(N-1) connections! O(N²) complexity
\`\`\`

\`\`\`java
// ✅ With Mediator - communication is centralized
ChatMediator chatRoom = new ChatRoom();	// single coordination point

User alice = new User("Alice", chatRoom);	// users only know mediator
User bob = new User("Bob", chatRoom);	// no direct references between users
User charlie = new User("Charlie", chatRoom);	// same mediator for all

chatRoom.addUser(alice);	// register with mediator
chatRoom.addUser(bob);	// mediator manages all connections
chatRoom.addUser(charlie);	// N users = N connections! O(N) complexity

alice.send("Hello!");	// alice doesn't know who receives
// Mediator routes to bob and charlie automatically
\`\`\`

---

### Real-World Applications

| Application | Mediator | Colleagues | Benefit |
|-------------|----------|------------|---------|
| **Chat Server** | ChatRoom | Users | Users don't know about each other |
| **Air Traffic Control** | Control Tower | Aircraft | Planes communicate through tower only |
| **UI Dialog** | DialogBox | Buttons, TextFields | Components coordinate via dialog |
| **Event Bus** | EventBus | Subscribers | Publishers don't know subscribers |
| **Stock Exchange** | Exchange | Traders | Buyers/sellers don't interact directly |

---

### Production Pattern: Air Traffic Control System

\`\`\`java
interface ATCMediator {	// Air Traffic Control mediator interface
    void registerFlight(Aircraft aircraft);	// register aircraft with ATC
    void requestLanding(Aircraft aircraft);	// aircraft requests landing clearance
    void requestTakeoff(Aircraft aircraft);	// aircraft requests takeoff clearance
    void notifyAll(String message, Aircraft sender);	// broadcast to all aircraft
}

class ControlTower implements ATCMediator {	// ConcreteMediator - coordinates all aircraft
    private List<Aircraft> flights = new ArrayList<>();	// all registered aircraft
    private String runwayStatus = "CLEAR";	// runway state (CLEAR or OCCUPIED)
    private Queue<Aircraft> landingQueue = new LinkedList<>();	// waiting for landing
    private Queue<Aircraft> takeoffQueue = new LinkedList<>();	// waiting for takeoff

    @Override
    public void registerFlight(Aircraft aircraft) {	// register new aircraft
        flights.add(aircraft);	// add to managed flights
        aircraft.receive("Welcome to airspace. Runway is " + runwayStatus);	// initial status
    }

    @Override
    public void requestLanding(Aircraft aircraft) {	// handle landing request
        if (runwayStatus.equals("CLEAR")) {	// runway available
            runwayStatus = "OCCUPIED";	// mark runway busy
            aircraft.receive("CLEARED for landing");	// grant clearance
            notifyAll("HOLD: " + aircraft.getCallSign() + " landing", aircraft);	// alert others
        } else {	// runway busy
            landingQueue.add(aircraft);	// add to queue
            aircraft.receive("HOLD position. You are #" + landingQueue.size());	// queue position
        }
    }

    @Override
    public void requestTakeoff(Aircraft aircraft) {	// handle takeoff request
        if (runwayStatus.equals("CLEAR")) {	// runway available
            runwayStatus = "OCCUPIED";	// mark runway busy
            aircraft.receive("CLEARED for takeoff");	// grant clearance
            notifyAll("HOLD: " + aircraft.getCallSign() + " taking off", aircraft);	// alert others
        } else {	// runway busy
            takeoffQueue.add(aircraft);	// add to queue
            aircraft.receive("HOLD position. Runway occupied");	// inform waiting
        }
    }

    public void runwayCleared() {	// called when aircraft completes landing/takeoff
        runwayStatus = "CLEAR";	// mark runway free
        if (!landingQueue.isEmpty()) {	// landings have priority
            Aircraft next = landingQueue.poll();	// get next in landing queue
            requestLanding(next);	// process landing request
        } else if (!takeoffQueue.isEmpty()) {	// then process takeoffs
            Aircraft next = takeoffQueue.poll();	// get next in takeoff queue
            requestTakeoff(next);	// process takeoff request
        } else {	// no one waiting
            notifyAll("Runway is now CLEAR", null);	// broadcast clear status
        }
    }

    @Override
    public void notifyAll(String message, Aircraft sender) {	// broadcast to all
        for (Aircraft flight : flights) {	// iterate all aircraft
            if (flight != sender) {	// skip sender
                flight.receive("ATC: " + message);	// deliver message
            }
        }
    }
}

abstract class Aircraft {	// Colleague base class
    protected String callSign;	// unique identifier (e.g., "UA123")
    protected ATCMediator atc;	// reference to control tower
    protected List<String> communications = new ArrayList<>();	// message log

    public Aircraft(String callSign, ATCMediator atc) {	// constructor
        this.callSign = callSign;	// set identifier
        this.atc = atc;	// store mediator reference
    }

    public String getCallSign() { return callSign; }	// getter for ID
    public List<String> getCommunications() { return communications; }	// getter for log

    public void receive(String message) {	// called by mediator
        communications.add(message);	// log all communications
    }

    public abstract void requestLanding();	// request landing clearance
    public abstract void requestTakeoff();	// request takeoff clearance
}

class CommercialFlight extends Aircraft {	// ConcreteColleague - commercial aircraft
    public CommercialFlight(String callSign, ATCMediator atc) {	// constructor
        super(callSign, atc);	// call parent constructor
    }

    @Override
    public void requestLanding() {	// request landing via mediator
        communications.add("Requesting landing clearance");	// log request
        atc.requestLanding(this);	// delegate to ATC
    }

    @Override
    public void requestTakeoff() {	// request takeoff via mediator
        communications.add("Requesting takeoff clearance");	// log request
        atc.requestTakeoff(this);	// delegate to ATC
    }
}

// Usage:
ControlTower tower = new ControlTower();	// create mediator

Aircraft ua123 = new CommercialFlight("UA123", tower);	// create aircraft
Aircraft dl456 = new CommercialFlight("DL456", tower);	// create another
Aircraft aa789 = new CommercialFlight("AA789", tower);	// create third

tower.registerFlight(ua123);	// register all with mediator
tower.registerFlight(dl456);	// mediator manages all communication
tower.registerFlight(aa789);	// aircraft don't know about each other

ua123.requestLanding();	// UA123: CLEARED for landing
dl456.requestLanding();	// DL456: HOLD position. You are #1
aa789.requestTakeoff();	// AA789: HOLD position. Runway occupied

tower.runwayCleared();	// UA123 landed, process next
// DL456 now gets: CLEARED for landing
// AA789 and UA123 get: HOLD: DL456 landing
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Colleagues referencing each other | Defeats the purpose of mediator | All communication through mediator |
| God Mediator | Mediator becomes too complex | Split into multiple specialized mediators |
| Bidirectional dependency | Memory leaks, tight coupling | Use weak references or clear on removal |
| Not using interface | Hard to swap mediators | Define mediator interface |
| Synchronous only | Blocks on slow operations | Add async message handling |`,
	order: 6,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;

// Test1: ChatRoom addUser adds user
class Test1 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        room.addUser(alice);
        alice.send("Hi");
    }
}

// Test2: User send delivers to others
class Test2 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        User bob = new User("Bob", room);
        room.addUser(alice);
        room.addUser(bob);
        alice.send("Hello");
        assertEquals(1, bob.getMessages().size());
    }
}

// Test3: Sender does not receive own message
class Test3 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        room.addUser(alice);
        alice.send("Test");
        assertTrue(alice.getMessages().isEmpty());
    }
}

// Test4: Message format is correct
class Test4 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        User bob = new User("Bob", room);
        room.addUser(alice);
        room.addUser(bob);
        alice.send("Hi Bob");
        assertEquals("[Alice]: Hi Bob", bob.getMessages().get(0));
    }
}

// Test5: Multiple users receive message
class Test5 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        User bob = new User("Bob", room);
        User charlie = new User("Charlie", room);
        room.addUser(alice);
        room.addUser(bob);
        room.addUser(charlie);
        alice.send("Hello all");
        assertEquals(1, bob.getMessages().size());
        assertEquals(1, charlie.getMessages().size());
    }
}

// Test6: User getName returns name
class Test6 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        assertEquals("Alice", alice.getName());
    }
}

// Test7: Multiple messages accumulate
class Test7 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        User bob = new User("Bob", room);
        room.addUser(alice);
        room.addUser(bob);
        alice.send("First");
        alice.send("Second");
        assertEquals(2, bob.getMessages().size());
    }
}

// Test8: Empty messages list initially
class Test8 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        assertTrue(alice.getMessages().isEmpty());
    }
}

// Test9: receive formats message correctly
class Test9 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User bob = new User("Bob", room);
        bob.receive("Test message", "Alice");
        assertEquals("[Alice]: Test message", bob.getMessages().get(0));
    }
}

// Test10: Bidirectional communication works
class Test10 {
    @Test
    public void test() {
        ChatRoom room = new ChatRoom();
        User alice = new User("Alice", room);
        User bob = new User("Bob", room);
        room.addUser(alice);
        room.addUser(bob);
        alice.send("Hi Bob");
        bob.send("Hi Alice");
        assertEquals("[Bob]: Hi Alice", alice.getMessages().get(0));
        assertEquals("[Alice]: Hi Bob", bob.getMessages().get(0));
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Mediator (Посредник)',
			description: `## Паттерн Mediator (Посредник)

Паттерн **Mediator** определяет объект, который инкапсулирует способ взаимодействия набора объектов. Он способствует слабой связанности, не позволяя объектам явно ссылаться друг на друга, что позволяет независимо изменять их взаимодействие.

---

### Ключевые компоненты

| Компонент | Роль |
|-----------|------|
| **Mediator** | Интерфейс, определяющий методы коммуникации |
| **ConcreteMediator** | Координирует коммуникацию между коллегами |
| **Colleague** | Объект, который общается через посредника |
| **ConcreteColleague** | Конкретная реализация коллеги |

---

### Ваша задача

Реализуйте **Чат-комнату** используя паттерн Mediator:

1. **ChatRoom** (ConcreteMediator): Управляет пользователями и рассылает сообщения
2. **User** (Colleague): Отправляет сообщения через посредника, получает от других

---

### Пример использования

\`\`\`java
ChatRoom chatRoom = new ChatRoom();	// создаём посредника (чат-комнату)

User alice = new User("Alice", chatRoom);	// создаём пользователя со ссылкой на посредника
User bob = new User("Bob", chatRoom);	// создаём другого пользователя
User charlie = new User("Charlie", chatRoom);	// создаём третьего пользователя

chatRoom.addUser(alice);	// регистрируем Alice в посреднике
chatRoom.addUser(bob);	// регистрируем Bob в посреднике
chatRoom.addUser(charlie);	// регистрируем Charlie в посреднике

alice.send("Hello everyone!");	// Alice отправляет через посредника

// Bob и Charlie получают: "[Alice]: Hello everyone!"
// Alice НЕ получает своё собственное сообщение
System.out.println(bob.getMessages());	// [[Alice]: Hello everyone!]
System.out.println(charlie.getMessages());	// [[Alice]: Hello everyone!]
System.out.println(alice.getMessages());	// [] (отправитель не получает)
\`\`\`

---

### Ключевая идея

> Объекты никогда не общаются друг с другом напрямую. Вся коммуникация проходит через посредника, который решает, кто что получает. Это централизует логику коммуникации в одном месте.`,
			hint1: `## Подсказка 1: Коммуникация пользователей

Пользователи никогда не должны общаться друг с другом напрямую. Вся коммуникация проходит через посредника:

\`\`\`java
public void send(String msg) {	// пользователь хочет отправить сообщение
    mediator.sendMessage(msg, this);	// делегируем посреднику с сообщением и ссылкой на отправителя
}

public void receive(String msg, String from) {	// вызывается посредником
    messages.add("[" + from + "]: " + msg);	// формат: [ИмяОтправителя]: сообщение
}
\`\`\`

Метод **send()** делегирует посреднику. Метод **receive()** вызывается ПОСРЕДНИКОМ, когда другой пользователь отправляет сообщение.`,
			hint2: `## Подсказка 2: Рассылка посредником

Посредник отвечает за маршрутизацию сообщений правильным получателям:

\`\`\`java
@Override
public void sendMessage(String msg, User sender) {	// рассылка всем кроме отправителя
    for (User user : users) {	// перебираем всех зарегистрированных пользователей
        if (user != sender) {	// пропускаем исходного отправителя
            user.receive(msg, sender.getName());	// вызываем receive на каждом получателе
        }
    }
}

@Override
public void addUser(User user) {	// метод регистрации
    users.add(user);	// просто добавляем в список
}
\`\`\`

Посредник знает о ВСЕХ пользователях и решает, кто получит каждое сообщение. Пользователи знают только о посреднике, но не друг о друге.`,
			whyItMatters: `## Почему паттерн Mediator важен

### Проблема: Прямая коммуникация создаёт сильную связанность

Без Mediator объекты должны знать о каждом другом объекте и общаться с ним:

\`\`\`java
// ❌ Без Mediator - пользователи общаются напрямую
class User {	// класс пользователя с прямыми ссылками
    private String name;	// идентификатор пользователя
    private List<User> contacts = new ArrayList<>();	// должен знать ВСЕХ других пользователей

    public void addContact(User user) {	// должен вручную управлять контактами
        contacts.add(user);	// отслеживать каждый контакт отдельно
    }

    public void sendMessage(String msg) {	// отправить всем контактам
        for (User contact : contacts) {	// перебрать все известные контакты
            contact.receiveMessage(msg, this.name);	// вызвать напрямую на каждом
        }
    }
}

// Проблема: Каждый пользователь должен знать о каждом другом
// Добавление/удаление пользователей требует обновления ВСЕХ других
User alice = new User("Alice");	// создаём пользователя
User bob = new User("Bob");	// создаём другого
User charlie = new User("Charlie");	// создаём третьего
// Нужно вручную связать всех со всеми!
alice.addContact(bob);	// alice знает bob
alice.addContact(charlie);	// alice знает charlie
bob.addContact(alice);	// bob знает alice
bob.addContact(charlie);	// bob знает charlie
charlie.addContact(alice);	// charlie знает alice
charlie.addContact(bob);	// charlie знает bob
// N пользователей = N*(N-1) связей! O(N²) сложность
\`\`\`

\`\`\`java
// ✅ С Mediator - коммуникация централизована
ChatMediator chatRoom = new ChatRoom();	// единая точка координации

User alice = new User("Alice", chatRoom);	// пользователи знают только посредника
User bob = new User("Bob", chatRoom);	// нет прямых ссылок между пользователями
User charlie = new User("Charlie", chatRoom);	// один посредник для всех

chatRoom.addUser(alice);	// регистрация у посредника
chatRoom.addUser(bob);	// посредник управляет всеми связями
chatRoom.addUser(charlie);	// N пользователей = N связей! O(N) сложность

alice.send("Hello!");	// alice не знает, кто получит
// Посредник маршрутизирует bob и charlie автоматически
\`\`\`

---

### Применение в реальном мире

| Применение | Посредник | Коллеги | Преимущество |
|------------|-----------|---------|--------------|
| **Чат-сервер** | ChatRoom | Пользователи | Пользователи не знают друг о друге |
| **Управление воздушным движением** | Диспетчерская вышка | Самолёты | Самолёты общаются только через вышку |
| **UI диалог** | DialogBox | Кнопки, Поля ввода | Компоненты координируются через диалог |
| **Шина событий** | EventBus | Подписчики | Издатели не знают подписчиков |
| **Фондовая биржа** | Биржа | Трейдеры | Покупатели/продавцы не взаимодействуют напрямую |

---

### Продакшн паттерн: Система управления воздушным движением

\`\`\`java
interface ATCMediator {	// интерфейс посредника управления воздушным движением
    void registerFlight(Aircraft aircraft);	// регистрация самолёта в УВД
    void requestLanding(Aircraft aircraft);	// самолёт запрашивает разрешение на посадку
    void requestTakeoff(Aircraft aircraft);	// самолёт запрашивает разрешение на взлёт
    void notifyAll(String message, Aircraft sender);	// рассылка всем самолётам
}

class ControlTower implements ATCMediator {	// ConcreteMediator - координирует все самолёты
    private List<Aircraft> flights = new ArrayList<>();	// все зарегистрированные самолёты
    private String runwayStatus = "CLEAR";	// состояние полосы (CLEAR или OCCUPIED)
    private Queue<Aircraft> landingQueue = new LinkedList<>();	// ожидают посадки
    private Queue<Aircraft> takeoffQueue = new LinkedList<>();	// ожидают взлёта

    @Override
    public void registerFlight(Aircraft aircraft) {	// регистрация нового самолёта
        flights.add(aircraft);	// добавить к управляемым рейсам
        aircraft.receive("Welcome to airspace. Runway is " + runwayStatus);	// начальный статус
    }

    @Override
    public void requestLanding(Aircraft aircraft) {	// обработка запроса на посадку
        if (runwayStatus.equals("CLEAR")) {	// полоса свободна
            runwayStatus = "OCCUPIED";	// пометить полосу занятой
            aircraft.receive("CLEARED for landing");	// выдать разрешение
            notifyAll("HOLD: " + aircraft.getCallSign() + " landing", aircraft);	// предупредить остальных
        } else {	// полоса занята
            landingQueue.add(aircraft);	// добавить в очередь
            aircraft.receive("HOLD position. You are #" + landingQueue.size());	// позиция в очереди
        }
    }

    @Override
    public void requestTakeoff(Aircraft aircraft) {	// обработка запроса на взлёт
        if (runwayStatus.equals("CLEAR")) {	// полоса свободна
            runwayStatus = "OCCUPIED";	// пометить полосу занятой
            aircraft.receive("CLEARED for takeoff");	// выдать разрешение
            notifyAll("HOLD: " + aircraft.getCallSign() + " taking off", aircraft);	// предупредить остальных
        } else {	// полоса занята
            takeoffQueue.add(aircraft);	// добавить в очередь
            aircraft.receive("HOLD position. Runway occupied");	// сообщить об ожидании
        }
    }

    public void runwayCleared() {	// вызывается когда самолёт завершил посадку/взлёт
        runwayStatus = "CLEAR";	// пометить полосу свободной
        if (!landingQueue.isEmpty()) {	// посадки имеют приоритет
            Aircraft next = landingQueue.poll();	// взять следующего из очереди на посадку
            requestLanding(next);	// обработать запрос на посадку
        } else if (!takeoffQueue.isEmpty()) {	// затем обработать взлёты
            Aircraft next = takeoffQueue.poll();	// взять следующего из очереди на взлёт
            requestTakeoff(next);	// обработать запрос на взлёт
        } else {	// никто не ждёт
            notifyAll("Runway is now CLEAR", null);	// объявить о свободной полосе
        }
    }

    @Override
    public void notifyAll(String message, Aircraft sender) {	// рассылка всем
        for (Aircraft flight : flights) {	// перебрать все самолёты
            if (flight != sender) {	// пропустить отправителя
                flight.receive("ATC: " + message);	// доставить сообщение
            }
        }
    }
}

abstract class Aircraft {	// базовый класс Colleague
    protected String callSign;	// уникальный идентификатор (напр., "UA123")
    protected ATCMediator atc;	// ссылка на диспетчерскую вышку
    protected List<String> communications = new ArrayList<>();	// журнал сообщений

    public Aircraft(String callSign, ATCMediator atc) {	// конструктор
        this.callSign = callSign;	// установить идентификатор
        this.atc = atc;	// сохранить ссылку на посредника
    }

    public String getCallSign() { return callSign; }	// геттер для ID
    public List<String> getCommunications() { return communications; }	// геттер для журнала

    public void receive(String message) {	// вызывается посредником
        communications.add(message);	// логировать всю коммуникацию
    }

    public abstract void requestLanding();	// запросить разрешение на посадку
    public abstract void requestTakeoff();	// запросить разрешение на взлёт
}

class CommercialFlight extends Aircraft {	// ConcreteColleague - коммерческий самолёт
    public CommercialFlight(String callSign, ATCMediator atc) {	// конструктор
        super(callSign, atc);	// вызвать родительский конструктор
    }

    @Override
    public void requestLanding() {	// запросить посадку через посредника
        communications.add("Requesting landing clearance");	// логировать запрос
        atc.requestLanding(this);	// делегировать УВД
    }

    @Override
    public void requestTakeoff() {	// запросить взлёт через посредника
        communications.add("Requesting takeoff clearance");	// логировать запрос
        atc.requestTakeoff(this);	// делегировать УВД
    }
}

// Использование:
ControlTower tower = new ControlTower();	// создать посредника

Aircraft ua123 = new CommercialFlight("UA123", tower);	// создать самолёт
Aircraft dl456 = new CommercialFlight("DL456", tower);	// создать другой
Aircraft aa789 = new CommercialFlight("AA789", tower);	// создать третий

tower.registerFlight(ua123);	// зарегистрировать всех у посредника
tower.registerFlight(dl456);	// посредник управляет всей коммуникацией
tower.registerFlight(aa789);	// самолёты не знают друг о друге

ua123.requestLanding();	// UA123: CLEARED for landing
dl456.requestLanding();	// DL456: HOLD position. You are #1
aa789.requestTakeoff();	// AA789: HOLD position. Runway occupied

tower.runwayCleared();	// UA123 приземлился, обработать следующего
// DL456 теперь получает: CLEARED for landing
// AA789 и UA123 получают: HOLD: DL456 landing
\`\`\`

---

### Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| Коллеги ссылаются друг на друга | Нарушает смысл посредника | Вся коммуникация через посредника |
| Бог-посредник | Посредник становится слишком сложным | Разделить на несколько специализированных посредников |
| Двунаправленная зависимость | Утечки памяти, сильная связанность | Использовать слабые ссылки или очищать при удалении |
| Не использовать интерфейс | Сложно заменить посредника | Определить интерфейс посредника |
| Только синхронный | Блокируется на медленных операциях | Добавить асинхронную обработку сообщений |`
		},
		uz: {
			title: 'Mediator Pattern',
			description: `## Mediator Pattern

**Mediator Pattern** ob'ektlar to'plamining o'zaro ta'sirini inkapsulyatsiya qiluvchi ob'ektni belgilaydi. U ob'ektlarni bir-biriga aniq murojaat qilishdan saqlash orqali yumshoq bog'lanishni ta'minlaydi, bu ularning o'zaro ta'sirini mustaqil ravishda o'zgartirish imkonini beradi.

---

### Asosiy komponentlar

| Komponent | Vazifa |
|-----------|--------|
| **Mediator** | Kommunikatsiya metodlarini belgilovchi interfeys |
| **ConcreteMediator** | Hamkasblar orasidagi kommunikatsiyani muvofiqlashtiradi |
| **Colleague** | Mediator orqali muloqot qiluvchi ob'ekt |
| **ConcreteColleague** | Aniq hamkasb realizatsiyasi |

---

### Vazifangiz

Mediator patternidan foydalanib **Chat xonasini** amalga oshiring:

1. **ChatRoom** (ConcreteMediator): Foydalanuvchilarni boshqaradi va xabarlarni tarqatadi
2. **User** (Colleague): Mediator orqali xabar yuboradi, boshqalardan qabul qiladi

---

### Foydalanish namunasi

\`\`\`java
ChatRoom chatRoom = new ChatRoom();	// mediator (chat xona) yaratish

User alice = new User("Alice", chatRoom);	// mediator referensiga ega foydalanuvchi yaratish
User bob = new User("Bob", chatRoom);	// boshqa foydalanuvchi yaratish
User charlie = new User("Charlie", chatRoom);	// uchinchi foydalanuvchi yaratish

chatRoom.addUser(alice);	// Alice ni mediatorga ro'yxatdan o'tkazish
chatRoom.addUser(bob);	// Bob ni mediatorga ro'yxatdan o'tkazish
chatRoom.addUser(charlie);	// Charlie ni mediatorga ro'yxatdan o'tkazish

alice.send("Hello everyone!");	// Alice mediator orqali yuboradi

// Bob va Charlie qabul qiladi: "[Alice]: Hello everyone!"
// Alice o'z xabarini QABUL QILMAYDI
System.out.println(bob.getMessages());	// [[Alice]: Hello everyone!]
System.out.println(charlie.getMessages());	// [[Alice]: Hello everyone!]
System.out.println(alice.getMessages());	// [] (jo'natuvchi qabul qilmaydi)
\`\`\`

---

### Asosiy tushuncha

> Ob'ektlar hech qachon bir-biri bilan to'g'ridan-to'g'ri muloqot qilmaydi. Barcha kommunikatsiya mediator orqali o'tadi, u kim nimani olishini hal qiladi. Bu kommunikatsiya mantiqini bir joyga jamlaydi.`,
			hint1: `## Maslahat 1: Foydalanuvchi kommunikatsiyasi

Foydalanuvchilar hech qachon bir-biri bilan to'g'ridan-to'g'ri muloqot qilmasligi kerak. Barcha kommunikatsiya mediator orqali:

\`\`\`java
public void send(String msg) {	// foydalanuvchi xabar yubormoqchi
    mediator.sendMessage(msg, this);	// xabar va jo'natuvchi referensi bilan mediatorga delegatsiya
}

public void receive(String msg, String from) {	// mediator tomonidan chaqiriladi
    messages.add("[" + from + "]: " + msg);	// format: [Jo'natuvchiIsmi]: xabar
}
\`\`\`

**send()** metodi mediatorga delegatsiya qiladi. **receive()** metodi boshqa foydalanuvchi xabar yuborganda MEDIATOR TOMONIDAN chaqiriladi.`,
			hint2: `## Maslahat 2: Mediator tarqatishi

Mediator xabarlarni to'g'ri qabul qiluvchilarga yo'naltirish uchun javobgar:

\`\`\`java
@Override
public void sendMessage(String msg, User sender) {	// jo'natuvchidan tashqari hammaga tarqatish
    for (User user : users) {	// barcha ro'yxatdan o'tgan foydalanuvchilarni takrorlash
        if (user != sender) {	// asl jo'natuvchini o'tkazib yuborish
            user.receive(msg, sender.getName());	// har bir qabul qiluvchida receive chaqirish
        }
    }
}

@Override
public void addUser(User user) {	// ro'yxatdan o'tkazish metodi
    users.add(user);	// shunchaki ro'yxatga qo'shish
}
\`\`\`

Mediator BARCHA foydalanuvchilar haqida biladi va har bir xabarni kim olishini hal qiladi. Foydalanuvchilar faqat mediator haqida biladi, bir-biri haqida emas.`,
			whyItMatters: `## Nima uchun Mediator Pattern muhim

### Muammo: To'g'ridan-to'g'ri kommunikatsiya kuchli bog'lanish yaratadi

Mediatorsiz ob'ektlar har bir boshqa ob'ekt haqida bilishi va u bilan muloqot qilishi kerak:

\`\`\`java
// ❌ Mediatorsiz - foydalanuvchilar to'g'ridan-to'g'ri muloqot qiladi
class User {	// to'g'ridan-to'g'ri referenslarga ega foydalanuvchi klassi
    private String name;	// foydalanuvchi identifikatori
    private List<User> contacts = new ArrayList<>();	// BARCHA boshqa foydalanuvchilarni bilishi kerak

    public void addContact(User user) {	// kontaktlarni qo'lda boshqarishi kerak
        contacts.add(user);	// har bir kontaktni alohida kuzatish
    }

    public void sendMessage(String msg) {	// barcha kontaktlarga yuborish
        for (User contact : contacts) {	// barcha ma'lum kontaktlarni takrorlash
            contact.receiveMessage(msg, this.name);	// har birida to'g'ridan-to'g'ri chaqirish
        }
    }
}

// Muammo: Har bir foydalanuvchi har bir boshqa foydalanuvchi haqida bilishi kerak
// Foydalanuvchi qo'shish/o'chirish BARCHA boshqalarni yangilashni talab qiladi
User alice = new User("Alice");	// foydalanuvchi yaratish
User bob = new User("Bob");	// boshqasini yaratish
User charlie = new User("Charlie");	// uchinchisini yaratish
// Hammani hamma bilan qo'lda bog'lash kerak!
alice.addContact(bob);	// alice bob ni biladi
alice.addContact(charlie);	// alice charlie ni biladi
bob.addContact(alice);	// bob alice ni biladi
bob.addContact(charlie);	// bob charlie ni biladi
charlie.addContact(alice);	// charlie alice ni biladi
charlie.addContact(bob);	// charlie bob ni biladi
// N foydalanuvchi = N*(N-1) bog'lanish! O(N²) murakkablik
\`\`\`

\`\`\`java
// ✅ Mediator bilan - kommunikatsiya markazlashgan
ChatMediator chatRoom = new ChatRoom();	// yagona muvofiqlashtirish nuqtasi

User alice = new User("Alice", chatRoom);	// foydalanuvchilar faqat mediatorni biladi
User bob = new User("Bob", chatRoom);	// foydalanuvchilar orasida to'g'ridan-to'g'ri referens yo'q
User charlie = new User("Charlie", chatRoom);	// hammasi uchun bitta mediator

chatRoom.addUser(alice);	// mediatorga ro'yxatdan o'tkazish
chatRoom.addUser(bob);	// mediator barcha bog'lanishlarni boshqaradi
chatRoom.addUser(charlie);	// N foydalanuvchi = N bog'lanish! O(N) murakkablik

alice.send("Hello!");	// alice kim olishini bilmaydi
// Mediator bob va charlie ga avtomatik yo'naltiradi
\`\`\`

---

### Haqiqiy dunyo qo'llanilishi

| Qo'llanilish | Mediator | Hamkasblar | Foyda |
|--------------|----------|------------|-------|
| **Chat server** | ChatRoom | Foydalanuvchilar | Foydalanuvchilar bir-birini bilmaydi |
| **Havo harakatini boshqarish** | Dispetcherlik minorasi | Samolyotlar | Samolyotlar faqat minora orqali muloqot qiladi |
| **UI dialog** | DialogBox | Tugmalar, Matn maydonlari | Komponentlar dialog orqali muvofiqlashtiradi |
| **Voqealar shinasi** | EventBus | Obunachi lar | Nashriyotlar obunachillarni bilmaydi |
| **Fond birjasi** | Birja | Treyderlar | Xaridorlar/sotuvchilar to'g'ridan-to'g'ri ta'sirlashmaydi |

---

### Production Pattern: Havo harakatini boshqarish tizimi

\`\`\`java
interface ATCMediator {	// Havo harakatini boshqarish mediator interfeysi
    void registerFlight(Aircraft aircraft);	// samolyotni ATC ga ro'yxatdan o'tkazish
    void requestLanding(Aircraft aircraft);	// samolyot qo'nish ruxsatini so'raydi
    void requestTakeoff(Aircraft aircraft);	// samolyot uchish ruxsatini so'raydi
    void notifyAll(String message, Aircraft sender);	// barcha samolyotlarga tarqatish
}

class ControlTower implements ATCMediator {	// ConcreteMediator - barcha samolyotlarni muvofiqlashtiradi
    private List<Aircraft> flights = new ArrayList<>();	// barcha ro'yxatdan o'tgan samolyotlar
    private String runwayStatus = "CLEAR";	// uchish-qo'nish yo'lagi holati (CLEAR yoki OCCUPIED)
    private Queue<Aircraft> landingQueue = new LinkedList<>();	// qo'nishni kutayotganlar
    private Queue<Aircraft> takeoffQueue = new LinkedList<>();	// uchishni kutayotganlar

    @Override
    public void registerFlight(Aircraft aircraft) {	// yangi samolyotni ro'yxatdan o'tkazish
        flights.add(aircraft);	// boshqarilayotgan reysarga qo'shish
        aircraft.receive("Welcome to airspace. Runway is " + runwayStatus);	// boshlang'ich holat
    }

    @Override
    public void requestLanding(Aircraft aircraft) {	// qo'nish so'rovini qayta ishlash
        if (runwayStatus.equals("CLEAR")) {	// yo'lak bo'sh
            runwayStatus = "OCCUPIED";	// yo'lakni band deb belgilash
            aircraft.receive("CLEARED for landing");	// ruxsat berish
            notifyAll("HOLD: " + aircraft.getCallSign() + " landing", aircraft);	// boshqalarni ogohlantirish
        } else {	// yo'lak band
            landingQueue.add(aircraft);	// navbatga qo'shish
            aircraft.receive("HOLD position. You are #" + landingQueue.size());	// navbat o'rni
        }
    }

    @Override
    public void requestTakeoff(Aircraft aircraft) {	// uchish so'rovini qayta ishlash
        if (runwayStatus.equals("CLEAR")) {	// yo'lak bo'sh
            runwayStatus = "OCCUPIED";	// yo'lakni band deb belgilash
            aircraft.receive("CLEARED for takeoff");	// ruxsat berish
            notifyAll("HOLD: " + aircraft.getCallSign() + " taking off", aircraft);	// boshqalarni ogohlantirish
        } else {	// yo'lak band
            takeoffQueue.add(aircraft);	// navbatga qo'shish
            aircraft.receive("HOLD position. Runway occupied");	// kutish haqida xabar
        }
    }

    public void runwayCleared() {	// samolyot qo'nish/uchishni tugatganda chaqiriladi
        runwayStatus = "CLEAR";	// yo'lakni bo'sh deb belgilash
        if (!landingQueue.isEmpty()) {	// qo'nishlar ustuvor
            Aircraft next = landingQueue.poll();	// qo'nish navbatidan keyingisini olish
            requestLanding(next);	// qo'nish so'rovini qayta ishlash
        } else if (!takeoffQueue.isEmpty()) {	// keyin uchishlarni qayta ishlash
            Aircraft next = takeoffQueue.poll();	// uchish navbatidan keyingisini olish
            requestTakeoff(next);	// uchish so'rovini qayta ishlash
        } else {	// hech kim kutmayapti
            notifyAll("Runway is now CLEAR", null);	// bo'sh holat haqida e'lon
        }
    }

    @Override
    public void notifyAll(String message, Aircraft sender) {	// hammaga tarqatish
        for (Aircraft flight : flights) {	// barcha samolyotlarni takrorlash
            if (flight != sender) {	// jo'natuvchini o'tkazib yuborish
                flight.receive("ATC: " + message);	// xabarni yetkazish
            }
        }
    }
}

abstract class Aircraft {	// Colleague bazaviy klassi
    protected String callSign;	// noyob identifikator (masalan, "UA123")
    protected ATCMediator atc;	// dispetcherlik minorasiga referens
    protected List<String> communications = new ArrayList<>();	// xabarlar jurnali

    public Aircraft(String callSign, ATCMediator atc) {	// konstruktor
        this.callSign = callSign;	// identifikatorni o'rnatish
        this.atc = atc;	// mediator referensini saqlash
    }

    public String getCallSign() { return callSign; }	// ID uchun getter
    public List<String> getCommunications() { return communications; }	// jurnal uchun getter

    public void receive(String message) {	// mediator tomonidan chaqiriladi
        communications.add(message);	// barcha kommunikatsiyani jurnalga yozish
    }

    public abstract void requestLanding();	// qo'nish ruxsatini so'rash
    public abstract void requestTakeoff();	// uchish ruxsatini so'rash
}

class CommercialFlight extends Aircraft {	// ConcreteColleague - tijorat samolyoti
    public CommercialFlight(String callSign, ATCMediator atc) {	// konstruktor
        super(callSign, atc);	// ota konstruktorini chaqirish
    }

    @Override
    public void requestLanding() {	// mediator orqali qo'nishni so'rash
        communications.add("Requesting landing clearance");	// so'rovni jurnalga yozish
        atc.requestLanding(this);	// ATC ga delegatsiya
    }

    @Override
    public void requestTakeoff() {	// mediator orqali uchishni so'rash
        communications.add("Requesting takeoff clearance");	// so'rovni jurnalga yozish
        atc.requestTakeoff(this);	// ATC ga delegatsiya
    }
}

// Foydalanish:
ControlTower tower = new ControlTower();	// mediator yaratish

Aircraft ua123 = new CommercialFlight("UA123", tower);	// samolyot yaratish
Aircraft dl456 = new CommercialFlight("DL456", tower);	// boshqasini yaratish
Aircraft aa789 = new CommercialFlight("AA789", tower);	// uchinchisini yaratish

tower.registerFlight(ua123);	// hammasini mediatorga ro'yxatdan o'tkazish
tower.registerFlight(dl456);	// mediator barcha kommunikatsiyani boshqaradi
tower.registerFlight(aa789);	// samolyotlar bir-birini bilmaydi

ua123.requestLanding();	// UA123: CLEARED for landing
dl456.requestLanding();	// DL456: HOLD position. You are #1
aa789.requestTakeoff();	// AA789: HOLD position. Runway occupied

tower.runwayCleared();	// UA123 qo'ndi, keyingisini qayta ishlash
// DL456 endi oladi: CLEARED for landing
// AA789 va UA123 oladi: HOLD: DL456 landing
\`\`\`

---

### Oldini olish kerak bo'lgan keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| Hamkasblar bir-biriga murojaat qiladi | Mediator maqsadini buzadi | Barcha kommunikatsiya mediator orqali |
| Xudo-mediator | Mediator juda murakkab bo'lib ketadi | Bir nechta ixtisoslashgan mediatorlarga bo'lish |
| Ikki tomonlama bog'liqlik | Xotira oqishi, kuchli bog'lanish | Zaif referenslardan foydalanish yoki o'chirishda tozalash |
| Interfeys ishlatmaslik | Mediatorni almashtirish qiyin | Mediator interfeysini belgilash |
| Faqat sinxron | Sekin operatsiyalarda bloklanadi | Asinxron xabar ishlov berishni qo'shish |`
		}
	}
};

export default task;
