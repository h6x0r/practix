import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-adapter',
	title: 'Adapter Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'structural', 'adapter'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Adapter pattern in Java - convert interface of a class into another interface clients expect.

**You will implement:**

1. **MediaPlayer interface** - Target interface
2. **AdvancedMediaPlayer** - Adaptee interface
3. **MediaAdapter** - Adapts advanced players to MediaPlayer

**Example Usage:**

\`\`\`java
MediaPlayer player = new AudioPlayer();	// client uses target interface
String mp3 = player.play("mp3", "song.mp3");	// native support: "Playing mp3 file: song.mp3"
String vlc = player.play("vlc", "movie.vlc");	// uses adapter: "Playing vlc file: movie.vlc"
String mp4 = player.play("mp4", "video.mp4");	// uses adapter: "Playing mp4 file: video.mp4"
String invalid = player.play("avi", "clip.avi");	// "Invalid media type: avi"
\`\`\``,
	initialCode: `interface MediaPlayer {
    String play(String audioType, String fileName);
}

interface AdvancedMediaPlayer {
    String playVlc(String fileName);
    String playMp4(String fileName);
}

class VlcPlayer implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public String playMp4(String fileName) {
        return null; // VLC player doesn't support mp4
    }
}

class Mp4Player implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return null;
    }

    @Override
    public String playMp4(String fileName) {
        throw new UnsupportedOperationException("TODO");
    }
}

class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedPlayer;

    public MediaAdapter(String audioType) {
        throw new UnsupportedOperationException("TODO: create appropriate player");
    }

    @Override
    public String play(String audioType, String fileName) {
        throw new UnsupportedOperationException("TODO");
    }
}

class AudioPlayer implements MediaPlayer {
    @Override
    public String play(String audioType, String fileName) {
        }
        throw new UnsupportedOperationException("TODO: use MediaAdapter");
    }
}`,
	solutionCode: `interface MediaPlayer {	// Target interface - what client expects
    String play(String audioType, String fileName);	// unified play method
}

interface AdvancedMediaPlayer {	// Adaptee interface - incompatible interface
    String playVlc(String fileName);	// VLC-specific method
    String playMp4(String fileName);	// MP4-specific method
}

class VlcPlayer implements AdvancedMediaPlayer {	// Concrete Adaptee - VLC implementation
    @Override
    public String playVlc(String fileName) {	// VLC playback
        return "Playing vlc file: " + fileName;	// VLC-specific behavior
    }

    @Override
    public String playMp4(String fileName) {	// not supported
        return null;	// VLC player doesn't support mp4
    }
}

class Mp4Player implements AdvancedMediaPlayer {	// Concrete Adaptee - MP4 implementation
    @Override
    public String playVlc(String fileName) {	// not supported
        return null;	// MP4 player doesn't support vlc
    }

    @Override
    public String playMp4(String fileName) {	// MP4 playback
        return "Playing mp4 file: " + fileName;	// MP4-specific behavior
    }
}

class MediaAdapter implements MediaPlayer {	// Adapter - bridges the gap
    private AdvancedMediaPlayer advancedPlayer;	// holds adaptee reference

    public MediaAdapter(String audioType) {	// constructor creates appropriate player
        if (audioType.equalsIgnoreCase("vlc")) {	// check for VLC
            advancedPlayer = new VlcPlayer();	// create VLC player
        } else if (audioType.equalsIgnoreCase("mp4")) {	// check for MP4
            advancedPlayer = new Mp4Player();	// create MP4 player
        }
    }

    @Override
    public String play(String audioType, String fileName) {	// adapt to target interface
        if (audioType.equalsIgnoreCase("vlc")) {	// delegate VLC calls
            return advancedPlayer.playVlc(fileName);	// call adaptee method
        } else if (audioType.equalsIgnoreCase("mp4")) {	// delegate MP4 calls
            return advancedPlayer.playMp4(fileName);	// call adaptee method
        }
        return null;	// unsupported type
    }
}

class AudioPlayer implements MediaPlayer {	// Client - uses adapter transparently
    @Override
    public String play(String audioType, String fileName) {	// unified interface
        if (audioType.equalsIgnoreCase("mp3")) {	// native support
            return "Playing mp3 file: " + fileName;	// built-in functionality
        } else if (audioType.equalsIgnoreCase("vlc") || audioType.equalsIgnoreCase("mp4")) {
            MediaAdapter adapter = new MediaAdapter(audioType);	// create adapter
            return adapter.play(audioType, fileName);	// delegate to adapter
        }
        return "Invalid media type: " + audioType;	// handle unknown types
    }
}`,
	hint1: `**Adaptee Implementations:**

VlcPlayer and Mp4Player implement AdvancedMediaPlayer with their specific methods:

\`\`\`java
class VlcPlayer implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return "Playing vlc file: " + fileName;	// VLC implementation
    }

    @Override
    public String playMp4(String fileName) {
        return null;	// not supported by this player
    }
}

class Mp4Player implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return null;	// not supported by this player
    }

    @Override
    public String playMp4(String fileName) {
        return "Playing mp4 file: " + fileName;	// MP4 implementation
    }
}
\`\`\``,
	hint2: `**MediaAdapter Implementation:**

The adapter converts AdvancedMediaPlayer to MediaPlayer interface:

\`\`\`java
class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedPlayer;	// holds the adaptee

    public MediaAdapter(String audioType) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer = new VlcPlayer();	// create VLC adaptee
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer = new Mp4Player();	// create MP4 adaptee
        }
    }

    @Override
    public String play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("vlc")) {
            return advancedPlayer.playVlc(fileName);	// delegate to VLC
        } else if (audioType.equalsIgnoreCase("mp4")) {
            return advancedPlayer.playMp4(fileName);	// delegate to MP4
        }
        return null;
    }
}
\`\`\`

AudioPlayer uses adapter for unsupported formats:

\`\`\`java
if (audioType.equalsIgnoreCase("vlc") || audioType.equalsIgnoreCase("mp4")) {
    MediaAdapter adapter = new MediaAdapter(audioType);	// create adapter
    return adapter.play(audioType, fileName);	// delegate
}
\`\`\``,
	whyItMatters: `## Why Adapter Exists

Adapter allows classes with incompatible interfaces to work together. It wraps an existing class with a new interface without modifying the original code.

**Problem - Incompatible Interfaces:**

\`\`\`java
// ❌ Bad: Client code coupled to specific implementations
class MediaApplication {
    public void playMedia(String type, String file) {
        if (type.equals("vlc")) {
            VlcPlayer vlc = new VlcPlayer();	// direct dependency
            vlc.playVlc(file);	// VLC-specific method
        } else if (type.equals("mp4")) {
            Mp4Player mp4 = new Mp4Player();	// another dependency
            mp4.playMp4(file);	// MP4-specific method
        }
        // Adding new format requires modifying this class
    }
}
\`\`\`

**Solution - Adapter Provides Uniform Interface:**

\`\`\`java
// ✅ Good: Client uses unified interface
class MediaApplication {
    private final MediaPlayer player;	// depends on abstraction

    public MediaApplication() {
        this.player = new AudioPlayer();	// uses adapter internally
    }

    public void playMedia(String type, String file) {
        player.play(type, file);	// same interface for all types
    }
}
\`\`\`

---

## Real-World Examples

1. **Arrays.asList()** - Adapts array to List interface
2. **InputStreamReader** - Adapts byte stream to character stream
3. **Collections.enumeration()** - Adapts Iterator to Enumeration
4. **OutputStreamWriter** - Adapts character stream to byte stream
5. **JDBC Drivers** - Adapt vendor-specific APIs to standard JDBC interface

---

## Production Pattern: Payment Gateway Adapter

\`\`\`java
// Target Interface - what our application expects
interface PaymentProcessor {	// unified payment interface
    PaymentResult charge(String customerId, BigDecimal amount, String currency);	// charge method
    RefundResult refund(String transactionId, BigDecimal amount);	// refund method
    PaymentStatus getStatus(String transactionId);	// status check
}

// Adaptee 1: Stripe API (third-party SDK)
class StripeAPI {	// Stripe's actual interface
    public StripeCharge createCharge(String customer, long amountCents, String currency) {
        // Stripe SDK implementation
        return new StripeCharge("ch_" + UUID.randomUUID(), "succeeded");
    }

    public StripeRefund createRefund(String chargeId, long amountCents) {
        return new StripeRefund("re_" + UUID.randomUUID(), "succeeded");
    }

    public StripeCharge retrieveCharge(String chargeId) {
        return new StripeCharge(chargeId, "succeeded");
    }
}

// Adaptee 2: PayPal API (different third-party SDK)
class PayPalSDK {	// PayPal's actual interface
    public PayPalPayment capturePayment(String payerId, double amount, String currencyCode) {
        return new PayPalPayment("PAY-" + UUID.randomUUID(), "COMPLETED");
    }

    public PayPalRefund refundPayment(String paymentId, double amount) {
        return new PayPalRefund("REF-" + UUID.randomUUID(), "COMPLETED");
    }

    public PayPalPayment getPaymentDetails(String paymentId) {
        return new PayPalPayment(paymentId, "COMPLETED");
    }
}

// Adapter for Stripe
class StripeAdapter implements PaymentProcessor {	// adapts Stripe to our interface
    private final StripeAPI stripe;	// holds Stripe SDK

    public StripeAdapter(String apiKey) {	// constructor with config
        this.stripe = new StripeAPI();	// initialize Stripe
    }

    @Override
    public PaymentResult charge(String customerId, BigDecimal amount, String currency) {
        long cents = amount.multiply(BigDecimal.valueOf(100)).longValue();	// convert to cents
        StripeCharge charge = stripe.createCharge(customerId, cents, currency);	// call Stripe API
        return new PaymentResult(charge.getId(), mapStatus(charge.getStatus()));	// adapt response
    }

    @Override
    public RefundResult refund(String transactionId, BigDecimal amount) {
        long cents = amount.multiply(BigDecimal.valueOf(100)).longValue();	// convert to cents
        StripeRefund refund = stripe.createRefund(transactionId, cents);	// call Stripe API
        return new RefundResult(refund.getId(), mapRefundStatus(refund.getStatus()));	// adapt
    }

    @Override
    public PaymentStatus getStatus(String transactionId) {
        StripeCharge charge = stripe.retrieveCharge(transactionId);	// call Stripe API
        return mapStatus(charge.getStatus());	// adapt status
    }

    private PaymentStatus mapStatus(String stripeStatus) {	// map Stripe status to our enum
        return "succeeded".equals(stripeStatus) ? PaymentStatus.SUCCESS : PaymentStatus.FAILED;
    }

    private RefundStatus mapRefundStatus(String status) {	// map refund status
        return "succeeded".equals(status) ? RefundStatus.COMPLETED : RefundStatus.FAILED;
    }
}

// Adapter for PayPal
class PayPalAdapter implements PaymentProcessor {	// adapts PayPal to our interface
    private final PayPalSDK paypal;	// holds PayPal SDK

    public PayPalAdapter(String clientId, String clientSecret) {	// constructor with credentials
        this.paypal = new PayPalSDK();	// initialize PayPal
    }

    @Override
    public PaymentResult charge(String customerId, BigDecimal amount, String currency) {
        double amountDouble = amount.doubleValue();	// PayPal uses double
        PayPalPayment payment = paypal.capturePayment(customerId, amountDouble, currency);
        return new PaymentResult(payment.getId(), mapStatus(payment.getState()));	// adapt response
    }

    @Override
    public RefundResult refund(String transactionId, BigDecimal amount) {
        double amountDouble = amount.doubleValue();	// PayPal uses double
        PayPalRefund refund = paypal.refundPayment(transactionId, amountDouble);	// call PayPal
        return new RefundResult(refund.getId(), mapRefundStatus(refund.getState()));	// adapt
    }

    @Override
    public PaymentStatus getStatus(String transactionId) {
        PayPalPayment payment = paypal.getPaymentDetails(transactionId);	// call PayPal API
        return mapStatus(payment.getState());	// adapt status
    }

    private PaymentStatus mapStatus(String paypalStatus) {	// map PayPal status to our enum
        return "COMPLETED".equals(paypalStatus) ? PaymentStatus.SUCCESS : PaymentStatus.FAILED;
    }

    private RefundStatus mapRefundStatus(String status) {	// map refund status
        return "COMPLETED".equals(status) ? RefundStatus.COMPLETED : RefundStatus.FAILED;
    }
}

// Client code - works with any payment processor
class CheckoutService {	// uses the unified interface
    private final PaymentProcessor processor;	// depends on abstraction

    public CheckoutService(PaymentProcessor processor) {	// inject any adapter
        this.processor = processor;	// store reference
    }

    public OrderResult checkout(Order order, String customerId) {	// checkout logic
        PaymentResult result = processor.charge(	// use unified interface
            customerId, order.getTotal(), order.getCurrency());
        if (result.getStatus() == PaymentStatus.SUCCESS) {	// check status
            return new OrderResult(order.getId(), result.getTransactionId(), "COMPLETED");
        }
        return new OrderResult(order.getId(), null, "FAILED");
    }
}

// Usage - easy to switch payment providers
PaymentProcessor stripe = new StripeAdapter("sk_test_xxx");	// Stripe adapter
PaymentProcessor paypal = new PayPalAdapter("client_id", "secret");	// PayPal adapter
CheckoutService service = new CheckoutService(stripe);	// inject adapter
\`\`\`

---

## Object Adapter vs Class Adapter

\`\`\`java
// Object Adapter (composition) - preferred in Java
class ObjectAdapter implements Target {
    private final Adaptee adaptee;	// composition

    public ObjectAdapter(Adaptee adaptee) {
        this.adaptee = adaptee;	// inject adaptee
    }

    public void request() {
        adaptee.specificRequest();	// delegate
    }
}

// Class Adapter (inheritance) - less flexible
class ClassAdapter extends Adaptee implements Target {	// multiple inheritance
    public void request() {
        specificRequest();	// call inherited method
    }
}
\`\`\`

---

## Common Mistakes to Avoid

1. **Modifying the adaptee** - Adapter wraps, doesn't change original class
2. **Too much logic in adapter** - Keep it thin, just translate interfaces
3. **Not handling edge cases** - Map error states and exceptions properly
4. **Creating adapter per call** - Consider caching/reusing adapters`,
	order: 0,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void vlcPlayerPlaysVlcFile() {
        VlcPlayer player = new VlcPlayer();
        String result = player.playVlc("movie.vlc");
        assertEquals("Playing vlc file: movie.vlc", result, "VlcPlayer should play vlc files");
    }
}

class Test2 {
    @Test
    void vlcPlayerReturnsNullForMp4() {
        VlcPlayer player = new VlcPlayer();
        String result = player.playMp4("video.mp4");
        assertNull(result, "VlcPlayer should return null for mp4 files");
    }
}

class Test3 {
    @Test
    void mp4PlayerPlaysMp4File() {
        Mp4Player player = new Mp4Player();
        String result = player.playMp4("video.mp4");
        assertEquals("Playing mp4 file: video.mp4", result, "Mp4Player should play mp4 files");
    }
}

class Test4 {
    @Test
    void mp4PlayerReturnsNullForVlc() {
        Mp4Player player = new Mp4Player();
        String result = player.playVlc("movie.vlc");
        assertNull(result, "Mp4Player should return null for vlc files");
    }
}

class Test5 {
    @Test
    void mediaAdapterPlaysVlcFiles() {
        MediaAdapter adapter = new MediaAdapter("vlc");
        String result = adapter.play("vlc", "movie.vlc");
        assertEquals("Playing vlc file: movie.vlc", result, "MediaAdapter should play vlc files");
    }
}

class Test6 {
    @Test
    void mediaAdapterPlaysMp4Files() {
        MediaAdapter adapter = new MediaAdapter("mp4");
        String result = adapter.play("mp4", "video.mp4");
        assertEquals("Playing mp4 file: video.mp4", result, "MediaAdapter should play mp4 files");
    }
}

class Test7 {
    @Test
    void audioPlayerPlaysMp3Natively() {
        AudioPlayer player = new AudioPlayer();
        String result = player.play("mp3", "song.mp3");
        assertEquals("Playing mp3 file: song.mp3", result, "AudioPlayer should play mp3 natively");
    }
}

class Test8 {
    @Test
    void audioPlayerPlaysVlcViaAdapter() {
        AudioPlayer player = new AudioPlayer();
        String result = player.play("vlc", "movie.vlc");
        assertEquals("Playing vlc file: movie.vlc", result, "AudioPlayer should play vlc via adapter");
    }
}

class Test9 {
    @Test
    void audioPlayerPlaysMp4ViaAdapter() {
        AudioPlayer player = new AudioPlayer();
        String result = player.play("mp4", "video.mp4");
        assertEquals("Playing mp4 file: video.mp4", result, "AudioPlayer should play mp4 via adapter");
    }
}

class Test10 {
    @Test
    void audioPlayerReturnsInvalidForUnsupportedType() {
        AudioPlayer player = new AudioPlayer();
        String result = player.play("avi", "clip.avi");
        assertEquals("Invalid media type: avi", result, "AudioPlayer should return invalid for unsupported types");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Adapter (Адаптер)',
			description: `Реализуйте паттерн Adapter на Java — преобразуйте интерфейс класса в другой интерфейс, ожидаемый клиентами.

**Вы реализуете:**

1. **Интерфейс MediaPlayer** - Целевой интерфейс
2. **AdvancedMediaPlayer** - Интерфейс адаптируемого
3. **MediaAdapter** - Адаптирует продвинутые плееры к MediaPlayer

**Пример использования:**

\`\`\`java
MediaPlayer player = new AudioPlayer();	// клиент использует целевой интерфейс
String mp3 = player.play("mp3", "song.mp3");	// нативная поддержка: "Playing mp3 file: song.mp3"
String vlc = player.play("vlc", "movie.vlc");	// через адаптер: "Playing vlc file: movie.vlc"
String mp4 = player.play("mp4", "video.mp4");	// через адаптер: "Playing mp4 file: video.mp4"
String invalid = player.play("avi", "clip.avi");	// "Invalid media type: avi"
\`\`\``,
			hint1: `**Реализации адаптируемых:**

VlcPlayer и Mp4Player реализуют AdvancedMediaPlayer со своими специфичными методами:

\`\`\`java
class VlcPlayer implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return "Playing vlc file: " + fileName;	// VLC реализация
    }

    @Override
    public String playMp4(String fileName) {
        return null;	// не поддерживается этим плеером
    }
}

class Mp4Player implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return null;	// не поддерживается этим плеером
    }

    @Override
    public String playMp4(String fileName) {
        return "Playing mp4 file: " + fileName;	// MP4 реализация
    }
}
\`\`\``,
			hint2: `**Реализация MediaAdapter:**

Адаптер преобразует AdvancedMediaPlayer в интерфейс MediaPlayer:

\`\`\`java
class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedPlayer;	// хранит адаптируемого

    public MediaAdapter(String audioType) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer = new VlcPlayer();	// создаём VLC адаптируемого
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer = new Mp4Player();	// создаём MP4 адаптируемого
        }
    }

    @Override
    public String play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("vlc")) {
            return advancedPlayer.playVlc(fileName);	// делегируем VLC
        } else if (audioType.equalsIgnoreCase("mp4")) {
            return advancedPlayer.playMp4(fileName);	// делегируем MP4
        }
        return null;
    }
}
\`\`\`

AudioPlayer использует адаптер для неподдерживаемых форматов:

\`\`\`java
if (audioType.equalsIgnoreCase("vlc") || audioType.equalsIgnoreCase("mp4")) {
    MediaAdapter adapter = new MediaAdapter(audioType);	// создаём адаптер
    return adapter.play(audioType, fileName);	// делегируем
}
\`\`\``,
			whyItMatters: `## Зачем нужен Adapter

Adapter позволяет классам с несовместимыми интерфейсами работать вместе. Он оборачивает существующий класс новым интерфейсом без изменения оригинального кода.

**Проблема - Несовместимые интерфейсы:**

\`\`\`java
// ❌ Плохо: Клиентский код связан с конкретными реализациями
class MediaApplication {
    public void playMedia(String type, String file) {
        if (type.equals("vlc")) {
            VlcPlayer vlc = new VlcPlayer();	// прямая зависимость
            vlc.playVlc(file);	// VLC-специфичный метод
        } else if (type.equals("mp4")) {
            Mp4Player mp4 = new Mp4Player();	// ещё одна зависимость
            mp4.playMp4(file);	// MP4-специфичный метод
        }
        // Добавление нового формата требует изменения этого класса
    }
}
\`\`\`

**Решение - Adapter предоставляет унифицированный интерфейс:**

\`\`\`java
// ✅ Хорошо: Клиент использует унифицированный интерфейс
class MediaApplication {
    private final MediaPlayer player;	// зависит от абстракции

    public MediaApplication() {
        this.player = new AudioPlayer();	// использует адаптер внутри
    }

    public void playMedia(String type, String file) {
        player.play(type, file);	// один интерфейс для всех типов
    }
}
\`\`\`

---

## Примеры из реального мира

1. **Arrays.asList()** - Адаптирует массив к интерфейсу List
2. **InputStreamReader** - Адаптирует байтовый поток к символьному
3. **Collections.enumeration()** - Адаптирует Iterator к Enumeration
4. **OutputStreamWriter** - Адаптирует символьный поток к байтовому
5. **JDBC Drivers** - Адаптируют vendor-специфичные API к стандартному JDBC интерфейсу

---

## Production паттерн: Адаптер платёжного шлюза

\`\`\`java
// Целевой интерфейс - что ожидает наше приложение
interface PaymentProcessor {	// унифицированный платёжный интерфейс
    PaymentResult charge(String customerId, BigDecimal amount, String currency);	// метод оплаты
    RefundResult refund(String transactionId, BigDecimal amount);	// метод возврата
    PaymentStatus getStatus(String transactionId);	// проверка статуса
}

// Адаптируемый 1: Stripe API (сторонний SDK)
class StripeAPI {	// фактический интерфейс Stripe
    public StripeCharge createCharge(String customer, long amountCents, String currency) {
        // Реализация Stripe SDK
        return new StripeCharge("ch_" + UUID.randomUUID(), "succeeded");
    }

    public StripeRefund createRefund(String chargeId, long amountCents) {
        return new StripeRefund("re_" + UUID.randomUUID(), "succeeded");
    }

    public StripeCharge retrieveCharge(String chargeId) {
        return new StripeCharge(chargeId, "succeeded");
    }
}

// Адаптируемый 2: PayPal API (другой сторонний SDK)
class PayPalSDK {	// фактический интерфейс PayPal
    public PayPalPayment capturePayment(String payerId, double amount, String currencyCode) {
        return new PayPalPayment("PAY-" + UUID.randomUUID(), "COMPLETED");
    }

    public PayPalRefund refundPayment(String paymentId, double amount) {
        return new PayPalRefund("REF-" + UUID.randomUUID(), "COMPLETED");
    }

    public PayPalPayment getPaymentDetails(String paymentId) {
        return new PayPalPayment(paymentId, "COMPLETED");
    }
}

// Адаптер для Stripe
class StripeAdapter implements PaymentProcessor {	// адаптирует Stripe к нашему интерфейсу
    private final StripeAPI stripe;	// хранит Stripe SDK

    public StripeAdapter(String apiKey) {	// конструктор с конфигурацией
        this.stripe = new StripeAPI();	// инициализируем Stripe
    }

    @Override
    public PaymentResult charge(String customerId, BigDecimal amount, String currency) {
        long cents = amount.multiply(BigDecimal.valueOf(100)).longValue();	// конвертируем в центы
        StripeCharge charge = stripe.createCharge(customerId, cents, currency);	// вызываем Stripe API
        return new PaymentResult(charge.getId(), mapStatus(charge.getStatus()));	// адаптируем ответ
    }

    @Override
    public RefundResult refund(String transactionId, BigDecimal amount) {
        long cents = amount.multiply(BigDecimal.valueOf(100)).longValue();	// конвертируем в центы
        StripeRefund refund = stripe.createRefund(transactionId, cents);	// вызываем Stripe API
        return new RefundResult(refund.getId(), mapRefundStatus(refund.getStatus()));	// адаптируем
    }

    @Override
    public PaymentStatus getStatus(String transactionId) {
        StripeCharge charge = stripe.retrieveCharge(transactionId);	// вызываем Stripe API
        return mapStatus(charge.getStatus());	// адаптируем статус
    }

    private PaymentStatus mapStatus(String stripeStatus) {	// маппим статус Stripe в наш enum
        return "succeeded".equals(stripeStatus) ? PaymentStatus.SUCCESS : PaymentStatus.FAILED;
    }

    private RefundStatus mapRefundStatus(String status) {	// маппим статус возврата
        return "succeeded".equals(status) ? RefundStatus.COMPLETED : RefundStatus.FAILED;
    }
}

// Адаптер для PayPal
class PayPalAdapter implements PaymentProcessor {	// адаптирует PayPal к нашему интерфейсу
    private final PayPalSDK paypal;	// хранит PayPal SDK

    public PayPalAdapter(String clientId, String clientSecret) {	// конструктор с учётными данными
        this.paypal = new PayPalSDK();	// инициализируем PayPal
    }

    @Override
    public PaymentResult charge(String customerId, BigDecimal amount, String currency) {
        double amountDouble = amount.doubleValue();	// PayPal использует double
        PayPalPayment payment = paypal.capturePayment(customerId, amountDouble, currency);
        return new PaymentResult(payment.getId(), mapStatus(payment.getState()));	// адаптируем ответ
    }

    @Override
    public RefundResult refund(String transactionId, BigDecimal amount) {
        double amountDouble = amount.doubleValue();	// PayPal использует double
        PayPalRefund refund = paypal.refundPayment(transactionId, amountDouble);	// вызываем PayPal
        return new RefundResult(refund.getId(), mapRefundStatus(refund.getState()));	// адаптируем
    }

    @Override
    public PaymentStatus getStatus(String transactionId) {
        PayPalPayment payment = paypal.getPaymentDetails(transactionId);	// вызываем PayPal API
        return mapStatus(payment.getState());	// адаптируем статус
    }

    private PaymentStatus mapStatus(String paypalStatus) {	// маппим статус PayPal в наш enum
        return "COMPLETED".equals(paypalStatus) ? PaymentStatus.SUCCESS : PaymentStatus.FAILED;
    }

    private RefundStatus mapRefundStatus(String status) {	// маппим статус возврата
        return "COMPLETED".equals(status) ? RefundStatus.COMPLETED : RefundStatus.FAILED;
    }
}

// Клиентский код - работает с любым процессором платежей
class CheckoutService {	// использует унифицированный интерфейс
    private final PaymentProcessor processor;	// зависит от абстракции

    public CheckoutService(PaymentProcessor processor) {	// внедряем любой адаптер
        this.processor = processor;	// сохраняем ссылку
    }

    public OrderResult checkout(Order order, String customerId) {	// логика оформления
        PaymentResult result = processor.charge(	// используем унифицированный интерфейс
            customerId, order.getTotal(), order.getCurrency());
        if (result.getStatus() == PaymentStatus.SUCCESS) {	// проверяем статус
            return new OrderResult(order.getId(), result.getTransactionId(), "COMPLETED");
        }
        return new OrderResult(order.getId(), null, "FAILED");
    }
}

// Использование - легко переключать платёжных провайдеров
PaymentProcessor stripe = new StripeAdapter("sk_test_xxx");	// Stripe адаптер
PaymentProcessor paypal = new PayPalAdapter("client_id", "secret");	// PayPal адаптер
CheckoutService service = new CheckoutService(stripe);	// внедряем адаптер
\`\`\`

---

## Object Adapter vs Class Adapter

\`\`\`java
// Object Adapter (композиция) - предпочтителен в Java
class ObjectAdapter implements Target {
    private final Adaptee adaptee;	// композиция

    public ObjectAdapter(Adaptee adaptee) {
        this.adaptee = adaptee;	// внедряем адаптируемого
    }

    public void request() {
        adaptee.specificRequest();	// делегируем
    }
}

// Class Adapter (наследование) - менее гибкий
class ClassAdapter extends Adaptee implements Target {	// множественное наследование
    public void request() {
        specificRequest();	// вызываем унаследованный метод
    }
}
\`\`\`

---

## Частые ошибки, которых следует избегать

1. **Модификация адаптируемого** - Adapter оборачивает, не изменяет оригинальный класс
2. **Слишком много логики в адаптере** - Держите его тонким, только переводите интерфейсы
3. **Не обрабатывать крайние случаи** - Правильно маппьте состояния ошибок и исключения
4. **Создание адаптера на каждый вызов** - Рассмотрите кэширование/переиспользование адаптеров`
		},
		uz: {
			title: 'Adapter Pattern',
			description: `Java da Adapter patternini amalga oshiring — klass interfeysini mijozlar kutgan boshqa interfeysga aylantiring.

**Siz amalga oshirasiz:**

1. **MediaPlayer interfeysi** - Maqsad interfeys
2. **AdvancedMediaPlayer** - Adaptilanuvchi interfeys
3. **MediaAdapter** - Kengaytirilgan pleyerlarni MediaPlayer ga moslaydi

**Foydalanish misoli:**

\`\`\`java
MediaPlayer player = new AudioPlayer();	// mijoz maqsad interfeysidan foydalanadi
String mp3 = player.play("mp3", "song.mp3");	// mahalliy qo'llab-quvvatlash: "Playing mp3 file: song.mp3"
String vlc = player.play("vlc", "movie.vlc");	// adapter orqali: "Playing vlc file: movie.vlc"
String mp4 = player.play("mp4", "video.mp4");	// adapter orqali: "Playing mp4 file: video.mp4"
String invalid = player.play("avi", "clip.avi");	// "Invalid media type: avi"
\`\`\``,
			hint1: `**Adaptilanuvchi amalga oshirishlar:**

VlcPlayer va Mp4Player AdvancedMediaPlayer ni o'ziga xos metodlari bilan amalga oshiradi:

\`\`\`java
class VlcPlayer implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return "Playing vlc file: " + fileName;	// VLC amalga oshirish
    }

    @Override
    public String playMp4(String fileName) {
        return null;	// bu pleyer tomonidan qo'llab-quvvatlanmaydi
    }
}

class Mp4Player implements AdvancedMediaPlayer {
    @Override
    public String playVlc(String fileName) {
        return null;	// bu pleyer tomonidan qo'llab-quvvatlanmaydi
    }

    @Override
    public String playMp4(String fileName) {
        return "Playing mp4 file: " + fileName;	// MP4 amalga oshirish
    }
}
\`\`\``,
			hint2: `**MediaAdapter amalga oshirish:**

Adapter AdvancedMediaPlayer ni MediaPlayer interfeysiga aylantiradi:

\`\`\`java
class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedPlayer;	// adaptilanuvchini saqlaydi

    public MediaAdapter(String audioType) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer = new VlcPlayer();	// VLC adaptilanuvchisini yaratamiz
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer = new Mp4Player();	// MP4 adaptilanuvchisini yaratamiz
        }
    }

    @Override
    public String play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("vlc")) {
            return advancedPlayer.playVlc(fileName);	// VLC ga delegatsiya
        } else if (audioType.equalsIgnoreCase("mp4")) {
            return advancedPlayer.playMp4(fileName);	// MP4 ga delegatsiya
        }
        return null;
    }
}
\`\`\`

AudioPlayer qo'llab-quvvatlanmaydigan formatlar uchun adapterdan foydalanadi:

\`\`\`java
if (audioType.equalsIgnoreCase("vlc") || audioType.equalsIgnoreCase("mp4")) {
    MediaAdapter adapter = new MediaAdapter(audioType);	// adapter yaratamiz
    return adapter.play(audioType, fileName);	// delegatsiya
}
\`\`\``,
			whyItMatters: `## Adapter nima uchun kerak

Adapter mos kelmaydigan interfeyslarga ega klasslarning birga ishlashiga imkon beradi. U mavjud klassni asl kodni o'zgartirmasdan yangi interfeys bilan o'raydi.

**Muammo - Mos kelmaydigan interfeyslar:**

\`\`\`java
// ❌ Yomon: Mijoz kodi aniq amalga oshirishlarga bog'langan
class MediaApplication {
    public void playMedia(String type, String file) {
        if (type.equals("vlc")) {
            VlcPlayer vlc = new VlcPlayer();	// to'g'ridan-to'g'ri bog'liqlik
            vlc.playVlc(file);	// VLC ga xos metod
        } else if (type.equals("mp4")) {
            Mp4Player mp4 = new Mp4Player();	// yana bir bog'liqlik
            mp4.playMp4(file);	// MP4 ga xos metod
        }
        // Yangi format qo'shish bu klassni o'zgartirishni talab qiladi
    }
}
\`\`\`

**Yechim - Adapter yagona interfeys taqdim etadi:**

\`\`\`java
// ✅ Yaxshi: Mijoz yagona interfeysdan foydalanadi
class MediaApplication {
    private final MediaPlayer player;	// abstraksiyaga bog'liq

    public MediaApplication() {
        this.player = new AudioPlayer();	// ichida adapterdan foydalanadi
    }

    public void playMedia(String type, String file) {
        player.play(type, file);	// barcha turlar uchun bir xil interfeys
    }
}
\`\`\`

---

## Haqiqiy dunyo misollari

1. **Arrays.asList()** - Massivni List interfeysiga moslaydi
2. **InputStreamReader** - Bayt oqimini belgi oqimiga moslaydi
3. **Collections.enumeration()** - Iterator ni Enumeration ga moslaydi
4. **OutputStreamWriter** - Belgi oqimini bayt oqimiga moslaydi
5. **JDBC Drivers** - Vendor-ga xos API larni standart JDBC interfeysiga moslaydi

---

## Production pattern: To'lov shlyuzi adapteri

\`\`\`java
// Maqsad interfeys - bizning ilovamiz nimani kutadi
interface PaymentProcessor {	// yagona to'lov interfeysi
    PaymentResult charge(String customerId, BigDecimal amount, String currency);	// to'lov metodi
    RefundResult refund(String transactionId, BigDecimal amount);	// qaytarim metodi
    PaymentStatus getStatus(String transactionId);	// holat tekshiruvi
}

// Adaptilanuvchi 1: Stripe API (uchinchi tomon SDK)
class StripeAPI {	// Stripe ning haqiqiy interfeysi
    public StripeCharge createCharge(String customer, long amountCents, String currency) {
        // Stripe SDK amalga oshirish
        return new StripeCharge("ch_" + UUID.randomUUID(), "succeeded");
    }

    public StripeRefund createRefund(String chargeId, long amountCents) {
        return new StripeRefund("re_" + UUID.randomUUID(), "succeeded");
    }

    public StripeCharge retrieveCharge(String chargeId) {
        return new StripeCharge(chargeId, "succeeded");
    }
}

// Adaptilanuvchi 2: PayPal API (boshqa uchinchi tomon SDK)
class PayPalSDK {	// PayPal ning haqiqiy interfeysi
    public PayPalPayment capturePayment(String payerId, double amount, String currencyCode) {
        return new PayPalPayment("PAY-" + UUID.randomUUID(), "COMPLETED");
    }

    public PayPalRefund refundPayment(String paymentId, double amount) {
        return new PayPalRefund("REF-" + UUID.randomUUID(), "COMPLETED");
    }

    public PayPalPayment getPaymentDetails(String paymentId) {
        return new PayPalPayment(paymentId, "COMPLETED");
    }
}

// Stripe uchun adapter
class StripeAdapter implements PaymentProcessor {	// Stripe ni bizning interfeysga moslaydi
    private final StripeAPI stripe;	// Stripe SDK ni saqlaydi

    public StripeAdapter(String apiKey) {	// konfiguratsiya bilan konstruktor
        this.stripe = new StripeAPI();	// Stripe ni initsializatsiya qilamiz
    }

    @Override
    public PaymentResult charge(String customerId, BigDecimal amount, String currency) {
        long cents = amount.multiply(BigDecimal.valueOf(100)).longValue();	// sentlarga aylantiramiz
        StripeCharge charge = stripe.createCharge(customerId, cents, currency);	// Stripe API chaqiramiz
        return new PaymentResult(charge.getId(), mapStatus(charge.getStatus()));	// javobni moslaymiz
    }

    @Override
    public RefundResult refund(String transactionId, BigDecimal amount) {
        long cents = amount.multiply(BigDecimal.valueOf(100)).longValue();	// sentlarga aylantiramiz
        StripeRefund refund = stripe.createRefund(transactionId, cents);	// Stripe API chaqiramiz
        return new RefundResult(refund.getId(), mapRefundStatus(refund.getStatus()));	// moslaymiz
    }

    @Override
    public PaymentStatus getStatus(String transactionId) {
        StripeCharge charge = stripe.retrieveCharge(transactionId);	// Stripe API chaqiramiz
        return mapStatus(charge.getStatus());	// holatni moslaymiz
    }

    private PaymentStatus mapStatus(String stripeStatus) {	// Stripe holatini bizning enum ga moslaymiz
        return "succeeded".equals(stripeStatus) ? PaymentStatus.SUCCESS : PaymentStatus.FAILED;
    }

    private RefundStatus mapRefundStatus(String status) {	// qaytarim holatini moslaymiz
        return "succeeded".equals(status) ? RefundStatus.COMPLETED : RefundStatus.FAILED;
    }
}

// PayPal uchun adapter
class PayPalAdapter implements PaymentProcessor {	// PayPal ni bizning interfeysga moslaydi
    private final PayPalSDK paypal;	// PayPal SDK ni saqlaydi

    public PayPalAdapter(String clientId, String clientSecret) {	// hisob ma'lumotlari bilan konstruktor
        this.paypal = new PayPalSDK();	// PayPal ni initsializatsiya qilamiz
    }

    @Override
    public PaymentResult charge(String customerId, BigDecimal amount, String currency) {
        double amountDouble = amount.doubleValue();	// PayPal double ishlatadi
        PayPalPayment payment = paypal.capturePayment(customerId, amountDouble, currency);
        return new PaymentResult(payment.getId(), mapStatus(payment.getState()));	// javobni moslaymiz
    }

    @Override
    public RefundResult refund(String transactionId, BigDecimal amount) {
        double amountDouble = amount.doubleValue();	// PayPal double ishlatadi
        PayPalRefund refund = paypal.refundPayment(transactionId, amountDouble);	// PayPal chaqiramiz
        return new RefundResult(refund.getId(), mapRefundStatus(refund.getState()));	// moslaymiz
    }

    @Override
    public PaymentStatus getStatus(String transactionId) {
        PayPalPayment payment = paypal.getPaymentDetails(transactionId);	// PayPal API chaqiramiz
        return mapStatus(payment.getState());	// holatni moslaymiz
    }

    private PaymentStatus mapStatus(String paypalStatus) {	// PayPal holatini bizning enum ga moslaymiz
        return "COMPLETED".equals(paypalStatus) ? PaymentStatus.SUCCESS : PaymentStatus.FAILED;
    }

    private RefundStatus mapRefundStatus(String status) {	// qaytarim holatini moslaymiz
        return "COMPLETED".equals(status) ? RefundStatus.COMPLETED : RefundStatus.FAILED;
    }
}

// Mijoz kodi - har qanday to'lov protsessori bilan ishlaydi
class CheckoutService {	// yagona interfeysdan foydalanadi
    private final PaymentProcessor processor;	// abstraksiyaga bog'liq

    public CheckoutService(PaymentProcessor processor) {	// har qanday adapterni kiritamiz
        this.processor = processor;	// havolani saqlaymiz
    }

    public OrderResult checkout(Order order, String customerId) {	// buyurtma berish mantiqi
        PaymentResult result = processor.charge(	// yagona interfeysdan foydalanamiz
            customerId, order.getTotal(), order.getCurrency());
        if (result.getStatus() == PaymentStatus.SUCCESS) {	// holatni tekshiramiz
            return new OrderResult(order.getId(), result.getTransactionId(), "COMPLETED");
        }
        return new OrderResult(order.getId(), null, "FAILED");
    }
}

// Foydalanish - to'lov provayderlarini osonlik bilan almashtirish mumkin
PaymentProcessor stripe = new StripeAdapter("sk_test_xxx");	// Stripe adapter
PaymentProcessor paypal = new PayPalAdapter("client_id", "secret");	// PayPal adapter
CheckoutService service = new CheckoutService(stripe);	// adapterni kiritamiz
\`\`\`

---

## Object Adapter vs Class Adapter

\`\`\`java
// Object Adapter (kompozitsiya) - Java da afzal
class ObjectAdapter implements Target {
    private final Adaptee adaptee;	// kompozitsiya

    public ObjectAdapter(Adaptee adaptee) {
        this.adaptee = adaptee;	// adaptilanuvchini kiritamiz
    }

    public void request() {
        adaptee.specificRequest();	// delegatsiya
    }
}

// Class Adapter (meros) - kamroq moslashuvchan
class ClassAdapter extends Adaptee implements Target {	// ko'p meroslik
    public void request() {
        specificRequest();	// meros qilib olingan metodini chaqiramiz
    }
}
\`\`\`

---

## Qochish kerak bo'lgan keng tarqalgan xatolar

1. **Adaptilanuvchini o'zgartirish** - Adapter o'raydi, asl klassni o'zgartirmaydi
2. **Adapterda juda ko'p mantiq** - Uni yupqa saqlang, faqat interfeyslarni tarjima qiling
3. **Chegaraviy holatlarni boshqarmaslik** - Xato holatlarini va istisnolarni to'g'ri moslang
4. **Har bir chaqiruvda adapter yaratish** - Adapterlarni keshlash/qayta ishlatishni ko'rib chiqing`
		}
	}
};

export default task;
