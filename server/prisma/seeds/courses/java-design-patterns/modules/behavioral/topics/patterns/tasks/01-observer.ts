import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-observer',
	title: 'Observer Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'behavioral', 'observer'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Observer Pattern

The **Observer** pattern defines a one-to-many dependency between objects so that when one object (Subject) changes state, all its dependents (Observers) are notified and updated automatically.

---

### Key Components

| Component | Description |
|-----------|-------------|
| **Observer** | Interface with \`update()\` method for receiving notifications |
| **Subject** | Maintains list of observers and notifies them of state changes |
| **ConcreteObserver** | Implements Observer interface with specific reaction logic |

---

### Your Task

Implement a news notification system:

1. **Observer interface** - \`update(message)\` method returning notification string
2. **NewsAgency (Subject)** - \`attach()\`, \`detach()\`, \`setNews()\` methods
3. **EmailSubscriber** - returns "Email to {email}: {message}"
4. **SMSSubscriber** - returns "SMS to {phone}: {message}"

---

### Example Usage

\`\`\`java
NewsAgency agency = new NewsAgency();	// create the subject (publisher)

Observer email = new EmailSubscriber("user@example.com");	// create email observer
Observer sms = new SMSSubscriber("+1234567890");	// create SMS observer

agency.attach(email);	// register email subscriber
agency.attach(sms);	// register SMS subscriber

List<String> responses = agency.setNews("Breaking News!");	// publish news, notify all observers
// responses contains:
// ["Email to user@example.com: Breaking News!", "SMS to +1234567890: Breaking News!"]

agency.detach(sms);	// unsubscribe SMS observer
List<String> newResponses = agency.setNews("Update");	// only email observer notified
// newResponses contains: ["Email to user@example.com: Update"]
\`\`\`

---

### Key Insight

Observer enables **loose coupling** between the subject and its observers - the subject doesn't need to know the concrete types of its observers, only that they implement the Observer interface.`,
	initialCode: `import java.util.ArrayList;
import java.util.List;

interface Observer {
    String update(String message);
}

class NewsAgency {
    private List<Observer> observers = new ArrayList<>();
    private String news;

    public void attach(Observer observer) {
        throw new UnsupportedOperationException("TODO");
    }

    public void detach(Observer observer) {
        throw new UnsupportedOperationException("TODO");
    }

    public List<String> setNews(String news) {
        throw new UnsupportedOperationException("TODO");
    }
}

class EmailSubscriber implements Observer {
    private String email;

    public EmailSubscriber(String email) {
    }

    @Override
    public String update(String message) {
        throw new UnsupportedOperationException("TODO");
    }
}

class SMSSubscriber implements Observer {
    private String phone;

    public SMSSubscriber(String phone) {
    }

    @Override
    public String update(String message) {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `import java.util.ArrayList;	// import ArrayList for observer storage
import java.util.List;	// import List interface

interface Observer {	// Observer interface - defines update contract
    String update(String message);	// receives notification, returns formatted response
}

class NewsAgency {	// Subject (Publisher) - maintains observers and notifies them
    private List<Observer> observers = new ArrayList<>();	// list of registered observers
    private String news;	// current news state

    public void attach(Observer observer) {	// register new observer
        observers.add(observer);	// add observer to list
    }

    public void detach(Observer observer) {	// unregister observer
        observers.remove(observer);	// remove observer from list
    }

    public List<String> setNews(String news) {	// update state and notify all observers
        this.news = news;	// store new news
        List<String> responses = new ArrayList<>();	// collect all responses
        for (Observer observer : observers) {	// iterate through all observers
            responses.add(observer.update(news));	// notify each observer, collect response
        }
        return responses;	// return all notification responses
    }
}

class EmailSubscriber implements Observer {	// ConcreteObserver - email notification
    private String email;	// subscriber's email address

    public EmailSubscriber(String email) {	// constructor with email
        this.email = email;	// store email address
    }

    @Override
    public String update(String message) {	// handle notification
        return "Email to " + email + ": " + message;	// format email notification
    }
}

class SMSSubscriber implements Observer {	// ConcreteObserver - SMS notification
    private String phone;	// subscriber's phone number

    public SMSSubscriber(String phone) {	// constructor with phone
        this.phone = phone;	// store phone number
    }

    @Override
    public String update(String message) {	// handle notification
        return "SMS to " + phone + ": " + message;	// format SMS notification
    }
}`,
	hint1: `### Understanding Observer Structure

The Observer pattern has three main parts:

\`\`\`java
// 1. Observer interface - contract for receiving updates
interface Observer {
    String update(String message);	// Returns notification string
}

// 2. Subject (NewsAgency) - manages observers
class NewsAgency {
    private List<Observer> observers;	// Store observers here

    public void attach(Observer o) {
        observers.add(o);	// Add to list
    }

    public void detach(Observer o) {
        observers.remove(o);	// Remove from list
    }
}

// 3. ConcreteObservers - implement specific behavior
class EmailSubscriber implements Observer {
    public String update(String message) {
        return "Email to " + email + ": " + message;
    }
}
\`\`\``,
	hint2: `### Implementing setNews - The Notification Loop

The \`setNews\` method is the heart of the pattern - it updates state and notifies all observers:

\`\`\`java
public List<String> setNews(String news) {
    this.news = news;	// 1. Update internal state

    List<String> responses = new ArrayList<>();	// 2. Prepare response collection

    for (Observer observer : observers) {	// 3. Iterate all observers
        String response = observer.update(news);	// 4. Notify each one
        responses.add(response);	// 5. Collect response
    }

    return responses;	// 6. Return all responses
}
\`\`\`

Each observer formats their notification differently:
- EmailSubscriber: "Email to {email}: {message}"
- SMSSubscriber: "SMS to {phone}: {message}"`,
	whyItMatters: `## Why Observer Pattern Matters

### The Problem and Solution

**Without Observer:**
\`\`\`java
// Tight coupling - NewsAgency must know all subscriber types
class NewsAgency {
    private EmailSubscriber email;	// direct reference
    private SMSSubscriber sms;	// direct reference
    private PushSubscriber push;	// must add for each new type!

    public void setNews(String news) {
        email.sendEmail(news);	// different method
        sms.sendSMS(news);	// different method
        push.sendPush(news);	// must modify for each new type
    }
}
\`\`\`

**With Observer:**
\`\`\`java
// Loose coupling - NewsAgency only knows Observer interface
class NewsAgency {
    private List<Observer> observers;	// uniform collection

    public void setNews(String news) {
        for (Observer o : observers) {	// iterate all
            o.update(news);	// same interface for all
        }
    }
}
// Adding new subscriber types requires NO changes to NewsAgency!
\`\`\`

---

### Real-World Applications

| Application | Subject | Observers |
|-------------|---------|-----------|
| **GUI Events** | Button, TextField | ActionListener, KeyListener |
| **Stock Market** | Stock, Index | Trader, Dashboard, Alert |
| **Social Media** | User, Channel | Followers, Notification Service |
| **MVC Pattern** | Model | View components |
| **Message Queue** | Topic | Message consumers |

---

### Production Pattern: Event System

\`\`\`java
// Generic event system with type safety
interface EventListener<T> {	// generic observer interface
    void onEvent(T event);	// receive typed event
}

class EventBus {	// subject managing multiple event types
    private Map<Class<?>, List<EventListener<?>>> listeners = new HashMap<>();	// event type -> listeners

    public <T> void subscribe(Class<T> eventType, EventListener<T> listener) {	// register for specific event type
        listeners.computeIfAbsent(eventType, k -> new ArrayList<>()).add(listener);	// add to type's list
    }

    public <T> void unsubscribe(Class<T> eventType, EventListener<T> listener) {	// unregister
        List<EventListener<?>> list = listeners.get(eventType);	// get listeners for type
        if (list != null) list.remove(listener);	// remove if exists
    }

    @SuppressWarnings("unchecked")
    public <T> void publish(T event) {	// publish event to all subscribers
        List<EventListener<?>> list = listeners.get(event.getClass());	// get listeners for event type
        if (list != null) {	// if any listeners registered
            for (EventListener<?> listener : list) {	// iterate all
                ((EventListener<T>) listener).onEvent(event);	// notify with type cast
            }
        }
    }
}

// Usage:
EventBus bus = new EventBus();	// create event bus
bus.subscribe(OrderCreated.class, e -> sendConfirmation(e));	// subscribe to order events
bus.subscribe(OrderCreated.class, e -> updateInventory(e));	// multiple subscribers per event
bus.subscribe(PaymentReceived.class, e -> processPayment(e));	// different event type
bus.publish(new OrderCreated(orderId, items));	// publish notifies all OrderCreated listeners
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Memory leaks** | Observers not detached | Always detach in cleanup/dispose |
| **Notification order** | Assuming specific order | Design observers to be order-independent |
| **Cascading updates** | Observer triggers more updates | Use flag to prevent recursive notifications |
| **Sync vs Async** | Blocking slow observers | Consider async notification for heavy operations |
| **Null observers** | NullPointerException | Check for null before adding/notifying |`,
	order: 0,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;

// Test1: NewsAgency attach adds observer
class Test1 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        Observer email = new EmailSubscriber("test@test.com");
        agency.attach(email);
        List<String> responses = agency.setNews("Test");
        assertEquals(1, responses.size());
    }
}

// Test2: EmailSubscriber update returns correct format
class Test2 {
    @Test
    public void test() {
        Observer email = new EmailSubscriber("user@example.com");
        String result = email.update("Breaking News");
        assertEquals("Email to user@example.com: Breaking News", result);
    }
}

// Test3: SMSSubscriber update returns correct format
class Test3 {
    @Test
    public void test() {
        Observer sms = new SMSSubscriber("+1234567890");
        String result = sms.update("Alert");
        assertEquals("SMS to +1234567890: Alert", result);
    }
}

// Test4: NewsAgency detach removes observer
class Test4 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        Observer email = new EmailSubscriber("test@test.com");
        agency.attach(email);
        agency.detach(email);
        List<String> responses = agency.setNews("Test");
        assertEquals(0, responses.size());
    }
}

// Test5: Multiple observers notified
class Test5 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        agency.attach(new EmailSubscriber("a@a.com"));
        agency.attach(new SMSSubscriber("+111"));
        List<String> responses = agency.setNews("News");
        assertEquals(2, responses.size());
    }
}

// Test6: setNews returns all responses
class Test6 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        agency.attach(new EmailSubscriber("x@x.com"));
        List<String> responses = agency.setNews("Update");
        assertTrue(responses.get(0).contains("Email to x@x.com"));
    }
}

// Test7: Empty agency returns empty list
class Test7 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        List<String> responses = agency.setNews("Test");
        assertTrue(responses.isEmpty());
    }
}

// Test8: Observer message propagated correctly
class Test8 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        agency.attach(new EmailSubscriber("test@mail.com"));
        List<String> responses = agency.setNews("Important");
        assertTrue(responses.get(0).endsWith("Important"));
    }
}

// Test9: Multiple notifications work
class Test9 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        agency.attach(new EmailSubscriber("a@b.com"));
        agency.setNews("First");
        List<String> responses = agency.setNews("Second");
        assertTrue(responses.get(0).contains("Second"));
    }
}

// Test10: Detach specific observer only
class Test10 {
    @Test
    public void test() {
        NewsAgency agency = new NewsAgency();
        Observer email = new EmailSubscriber("keep@test.com");
        Observer sms = new SMSSubscriber("+999");
        agency.attach(email);
        agency.attach(sms);
        agency.detach(sms);
        List<String> responses = agency.setNews("Test");
        assertEquals(1, responses.size());
        assertTrue(responses.get(0).startsWith("Email"));
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Observer (Наблюдатель)',
			description: `## Паттерн Observer (Наблюдатель)

Паттерн **Observer** определяет зависимость один-ко-многим между объектами таким образом, что когда один объект (Subject) меняет состояние, все его зависимые (Observers) автоматически уведомляются и обновляются.

---

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **Observer** | Интерфейс с методом \`update()\` для получения уведомлений |
| **Subject** | Хранит список наблюдателей и уведомляет их об изменениях |
| **ConcreteObserver** | Реализует интерфейс Observer с конкретной логикой реакции |

---

### Ваша задача

Реализуйте систему уведомлений о новостях:

1. **Интерфейс Observer** - метод \`update(message)\`, возвращающий строку уведомления
2. **NewsAgency (Subject)** - методы \`attach()\`, \`detach()\`, \`setNews()\`
3. **EmailSubscriber** - возвращает "Email to {email}: {message}"
4. **SMSSubscriber** - возвращает "SMS to {phone}: {message}"

---

### Пример использования

\`\`\`java
NewsAgency agency = new NewsAgency();	// создаём субъект (издатель)

Observer email = new EmailSubscriber("user@example.com");	// создаём email-наблюдателя
Observer sms = new SMSSubscriber("+1234567890");	// создаём SMS-наблюдателя

agency.attach(email);	// регистрируем email-подписчика
agency.attach(sms);	// регистрируем SMS-подписчика

List<String> responses = agency.setNews("Breaking News!");	// публикуем новость, уведомляем всех
// responses содержит:
// ["Email to user@example.com: Breaking News!", "SMS to +1234567890: Breaking News!"]

agency.detach(sms);	// отписываем SMS-наблюдателя
List<String> newResponses = agency.setNews("Update");	// уведомляется только email
// newResponses содержит: ["Email to user@example.com: Update"]
\`\`\`

---

### Ключевая идея

Observer обеспечивает **слабую связанность** между субъектом и его наблюдателями — субъект не знает конкретные типы своих наблюдателей, только то, что они реализуют интерфейс Observer.`,
			hint1: `### Понимание структуры Observer

Паттерн Observer состоит из трёх основных частей:

\`\`\`java
// 1. Интерфейс Observer - контракт для получения обновлений
interface Observer {
    String update(String message);	// Возвращает строку уведомления
}

// 2. Subject (NewsAgency) - управляет наблюдателями
class NewsAgency {
    private List<Observer> observers;	// Храним наблюдателей здесь

    public void attach(Observer o) {
        observers.add(o);	// Добавляем в список
    }

    public void detach(Observer o) {
        observers.remove(o);	// Удаляем из списка
    }
}

// 3. ConcreteObservers - реализуют конкретное поведение
class EmailSubscriber implements Observer {
    public String update(String message) {
        return "Email to " + email + ": " + message;
    }
}
\`\`\``,
			hint2: `### Реализация setNews - Цикл уведомлений

Метод \`setNews\` - сердце паттерна - он обновляет состояние и уведомляет всех наблюдателей:

\`\`\`java
public List<String> setNews(String news) {
    this.news = news;	// 1. Обновляем внутреннее состояние

    List<String> responses = new ArrayList<>();	// 2. Готовим коллекцию ответов

    for (Observer observer : observers) {	// 3. Перебираем всех наблюдателей
        String response = observer.update(news);	// 4. Уведомляем каждого
        responses.add(response);	// 5. Собираем ответ
    }

    return responses;	// 6. Возвращаем все ответы
}
\`\`\`

Каждый наблюдатель форматирует уведомление по-своему:
- EmailSubscriber: "Email to {email}: {message}"
- SMSSubscriber: "SMS to {phone}: {message}"`,
			whyItMatters: `## Почему паттерн Observer важен

### Проблема и решение

**Без Observer:**
\`\`\`java
// Тесная связанность - NewsAgency должен знать все типы подписчиков
class NewsAgency {
    private EmailSubscriber email;	// прямая ссылка
    private SMSSubscriber sms;	// прямая ссылка
    private PushSubscriber push;	// нужно добавлять для каждого нового типа!

    public void setNews(String news) {
        email.sendEmail(news);	// разные методы
        sms.sendSMS(news);	// разные методы
        push.sendPush(news);	// нужно изменять для каждого нового типа
    }
}
\`\`\`

**С Observer:**
\`\`\`java
// Слабая связанность - NewsAgency знает только интерфейс Observer
class NewsAgency {
    private List<Observer> observers;	// единообразная коллекция

    public void setNews(String news) {
        for (Observer o : observers) {	// перебираем всех
            o.update(news);	// единый интерфейс для всех
        }
    }
}
// Добавление новых типов подписчиков НЕ требует изменений в NewsAgency!
\`\`\`

---

### Применение в реальном мире

| Применение | Subject | Observers |
|------------|---------|-----------|
| **GUI события** | Button, TextField | ActionListener, KeyListener |
| **Биржа** | Stock, Index | Trader, Dashboard, Alert |
| **Соцсети** | User, Channel | Followers, Notification Service |
| **Паттерн MVC** | Model | View компоненты |
| **Очередь сообщений** | Topic | Потребители сообщений |

---

### Продакшен паттерн: Система событий

\`\`\`java
// Обобщённая система событий с типобезопасностью
interface EventListener<T> {	// обобщённый интерфейс наблюдателя
    void onEvent(T event);	// получение типизированного события
}

class EventBus {	// субъект, управляющий множеством типов событий
    private Map<Class<?>, List<EventListener<?>>> listeners = new HashMap<>();	// тип события -> слушатели

    public <T> void subscribe(Class<T> eventType, EventListener<T> listener) {	// регистрация на конкретный тип
        listeners.computeIfAbsent(eventType, k -> new ArrayList<>()).add(listener);	// добавляем в список типа
    }

    public <T> void unsubscribe(Class<T> eventType, EventListener<T> listener) {	// отмена регистрации
        List<EventListener<?>> list = listeners.get(eventType);	// получаем слушателей типа
        if (list != null) list.remove(listener);	// удаляем если есть
    }

    @SuppressWarnings("unchecked")
    public <T> void publish(T event) {	// публикация события всем подписчикам
        List<EventListener<?>> list = listeners.get(event.getClass());	// получаем слушателей типа события
        if (list != null) {	// если есть зарегистрированные
            for (EventListener<?> listener : list) {	// перебираем всех
                ((EventListener<T>) listener).onEvent(event);	// уведомляем с приведением типа
            }
        }
    }
}

// Использование:
EventBus bus = new EventBus();	// создаём шину событий
bus.subscribe(OrderCreated.class, e -> sendConfirmation(e));	// подписка на события заказа
bus.subscribe(OrderCreated.class, e -> updateInventory(e));	// несколько подписчиков на событие
bus.subscribe(PaymentReceived.class, e -> processPayment(e));	// другой тип события
bus.publish(new OrderCreated(orderId, items));	// публикация уведомляет всех слушателей OrderCreated
\`\`\`

---

### Частые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Утечки памяти** | Наблюдатели не отписаны | Всегда отписывайте в cleanup/dispose |
| **Порядок уведомлений** | Предполагается определённый порядок | Проектируйте наблюдателей независимыми от порядка |
| **Каскадные обновления** | Наблюдатель вызывает новые обновления | Используйте флаг для предотвращения рекурсии |
| **Синхр. vs Асинхр.** | Блокировка медленными наблюдателями | Рассмотрите асинхронные уведомления |
| **Null наблюдатели** | NullPointerException | Проверяйте на null перед добавлением/уведомлением |`
		},
		uz: {
			title: 'Observer (Kuzatuvchi) Pattern',
			description: `## Observer (Kuzatuvchi) Pattern

**Observer** pattern bir-ko'p bog'lanishni aniqlaydi, shuning uchun bir obyekt (Subject) holati o'zgarganda, uning barcha bog'liqlari (Observers) avtomatik xabardor qilinadi va yangilanadi.

---

### Asosiy Komponentlar

| Komponent | Tavsif |
|-----------|--------|
| **Observer** | Bildirishnomalarni qabul qilish uchun \`update()\` metodi bilan interfeys |
| **Subject** | Kuzatuvchilar ro'yxatini saqlaydi va holat o'zgarishlarini xabar qiladi |
| **ConcreteObserver** | Observer interfeysini maxsus reaktsiya mantiqiy bilan amalga oshiradi |

---

### Vazifangiz

Yangiliklar bildirishnoma tizimini amalga oshiring:

1. **Observer interfeysi** - bildirishnoma satrini qaytaruvchi \`update(message)\` metodi
2. **NewsAgency (Subject)** - \`attach()\`, \`detach()\`, \`setNews()\` metodlari
3. **EmailSubscriber** - "Email to {email}: {message}" qaytaradi
4. **SMSSubscriber** - "SMS to {phone}: {message}" qaytaradi

---

### Foydalanish Namunasi

\`\`\`java
NewsAgency agency = new NewsAgency();	// subyektni (nashriyot) yaratamiz

Observer email = new EmailSubscriber("user@example.com");	// email kuzatuvchisini yaratamiz
Observer sms = new SMSSubscriber("+1234567890");	// SMS kuzatuvchisini yaratamiz

agency.attach(email);	// email obunachilikni ro'yxatdan o'tkazamiz
agency.attach(sms);	// SMS obunachilikni ro'yxatdan o'tkazamiz

List<String> responses = agency.setNews("Breaking News!");	// yangilik nashr qilamiz, barchasini xabardor qilamiz
// responses o'z ichiga oladi:
// ["Email to user@example.com: Breaking News!", "SMS to +1234567890: Breaking News!"]

agency.detach(sms);	// SMS kuzatuvchisini obunachilikdan chiqaramiz
List<String> newResponses = agency.setNews("Update");	// faqat email kuzatuvchisi xabardor qilinadi
// newResponses o'z ichiga oladi: ["Email to user@example.com: Update"]
\`\`\`

---

### Asosiy Fikr

Observer subyekt va uning kuzatuvchilari o'rtasida **zaif bog'lanish**ni ta'minlaydi — subyekt o'z kuzatuvchilarining aniq turlarini bilishi shart emas, faqat ular Observer interfeysini amalga oshirishini biladi.`,
			hint1: `### Observer Strukturasini Tushunish

Observer pattern uch asosiy qismdan iborat:

\`\`\`java
// 1. Observer interfeysi - yangilanishlarni qabul qilish shartnomasi
interface Observer {
    String update(String message);	// Bildirishnoma satrini qaytaradi
}

// 2. Subject (NewsAgency) - kuzatuvchilarni boshqaradi
class NewsAgency {
    private List<Observer> observers;	// Kuzatuvchilarni bu yerda saqlaymiz

    public void attach(Observer o) {
        observers.add(o);	// Ro'yxatga qo'shamiz
    }

    public void detach(Observer o) {
        observers.remove(o);	// Ro'yxatdan olib tashlaymiz
    }
}

// 3. ConcreteObservers - maxsus xatti-harakatni amalga oshiradi
class EmailSubscriber implements Observer {
    public String update(String message) {
        return "Email to " + email + ": " + message;
    }
}
\`\`\``,
			hint2: `### setNews Amalga Oshirish - Xabarnoma Sikli

\`setNews\` metodi patternning yuragi - u holatni yangilaydi va barcha kuzatuvchilarni xabardor qiladi:

\`\`\`java
public List<String> setNews(String news) {
    this.news = news;	// 1. Ichki holatni yangilaymiz

    List<String> responses = new ArrayList<>();	// 2. Javoblar to'plamini tayyorlaymiz

    for (Observer observer : observers) {	// 3. Barcha kuzatuvchilarni aylanamiz
        String response = observer.update(news);	// 4. Har birini xabardor qilamiz
        responses.add(response);	// 5. Javobni yig'amiz
    }

    return responses;	// 6. Barcha javoblarni qaytaramiz
}
\`\`\`

Har bir kuzatuvchi o'z bildirishnomasini boshqacha formatlaydi:
- EmailSubscriber: "Email to {email}: {message}"
- SMSSubscriber: "SMS to {phone}: {message}"`,
			whyItMatters: `## Nima Uchun Observer Pattern Muhim

### Muammo va Yechim

**Observer siz:**
\`\`\`java
// Qattiq bog'lanish - NewsAgency barcha obunachi turlarini bilishi kerak
class NewsAgency {
    private EmailSubscriber email;	// to'g'ridan-to'g'ri havola
    private SMSSubscriber sms;	// to'g'ridan-to'g'ri havola
    private PushSubscriber push;	// har bir yangi tur uchun qo'shish kerak!

    public void setNews(String news) {
        email.sendEmail(news);	// turli metodlar
        sms.sendSMS(news);	// turli metodlar
        push.sendPush(news);	// har bir yangi tur uchun o'zgartirish kerak
    }
}
\`\`\`

**Observer bilan:**
\`\`\`java
// Zaif bog'lanish - NewsAgency faqat Observer interfeysini biladi
class NewsAgency {
    private List<Observer> observers;	// bir xil to'plam

    public void setNews(String news) {
        for (Observer o : observers) {	// barchasini aylanamiz
            o.update(news);	// hammasi uchun bir xil interfeys
        }
    }
}
// Yangi obunachi turlarini qo'shish NewsAgency da O'ZGARISHLAR talab qilmaydi!
\`\`\`

---

### Haqiqiy Dunyo Qo'llanilishi

| Qo'llanish | Subject | Observers |
|------------|---------|-----------|
| **GUI Hodisalari** | Button, TextField | ActionListener, KeyListener |
| **Birja** | Stock, Index | Trader, Dashboard, Alert |
| **Ijtimoiy Tarmoqlar** | User, Channel | Followers, Notification Service |
| **MVC Pattern** | Model | View komponentlari |
| **Xabar Navbati** | Topic | Xabar iste'molchilari |

---

### Prodakshen Pattern: Hodisalar Tizimi

\`\`\`java
// Tur xavfsizligi bilan umumiy hodisalar tizimi
interface EventListener<T> {	// umumiy kuzatuvchi interfeysi
    void onEvent(T event);	// turlangan hodisani qabul qilish
}

class EventBus {	// bir nechta hodisa turlarini boshqaruvchi subyekt
    private Map<Class<?>, List<EventListener<?>>> listeners = new HashMap<>();	// hodisa turi -> tinglovchilar

    public <T> void subscribe(Class<T> eventType, EventListener<T> listener) {	// maxsus hodisa turi uchun ro'yxatdan o'tish
        listeners.computeIfAbsent(eventType, k -> new ArrayList<>()).add(listener);	// tur ro'yxatiga qo'shish
    }

    public <T> void unsubscribe(Class<T> eventType, EventListener<T> listener) {	// ro'yxatdan chiqarish
        List<EventListener<?>> list = listeners.get(eventType);	// tur uchun tinglovchilarni olish
        if (list != null) list.remove(listener);	// mavjud bo'lsa olib tashlash
    }

    @SuppressWarnings("unchecked")
    public <T> void publish(T event) {	// hodisani barcha obunachilarla nashr qilish
        List<EventListener<?>> list = listeners.get(event.getClass());	// hodisa turi uchun tinglovchilarni olish
        if (list != null) {	// agar ro'yxatdan o'tgan tinglovchilar bo'lsa
            for (EventListener<?> listener : list) {	// barchasini aylantirish
                ((EventListener<T>) listener).onEvent(event);	// tur o'zgartirish bilan xabardor qilish
            }
        }
    }
}

// Foydalanish:
EventBus bus = new EventBus();	// hodisalar shinasini yaratish
bus.subscribe(OrderCreated.class, e -> sendConfirmation(e));	// buyurtma hodisalariga obuna
bus.subscribe(OrderCreated.class, e -> updateInventory(e));	// bir hodisaga bir nechta obunachi
bus.subscribe(PaymentReceived.class, e -> processPayment(e));	// boshqa hodisa turi
bus.publish(new OrderCreated(orderId, items));	// nashr qilish barcha OrderCreated tinglovchilarini xabardor qiladi
\`\`\`

---

### Oldini Olish Kerak Bo'lgan Xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Xotira oqishi** | Kuzatuvchilar ajratilmagan | Har doim cleanup/dispose da ajrating |
| **Xabarnoma tartibi** | Ma'lum tartib kutilmoqda | Kuzatuvchilarni tartibdan mustaqil loyihalang |
| **Kaskad yangilanishlar** | Kuzatuvchi ko'proq yangilanishlarni boshlaydi | Rekursiv xabarnomalarni oldini olish uchun bayroq ishlating |
| **Sinxron vs Asinxron** | Sekin kuzatuvchilar bloklaydi | Og'ir operatsiyalar uchun asinxron xabarnomalarni ko'rib chiqing |
| **Null kuzatuvchilar** | NullPointerException | Qo'shish/xabardor qilishdan oldin null tekshiring |`
		}
	}
};

export default task;
