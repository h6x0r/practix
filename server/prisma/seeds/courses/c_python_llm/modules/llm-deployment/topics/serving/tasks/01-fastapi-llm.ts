import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-fastapi-llm-api',
	title: 'FastAPI LLM API',
	difficulty: 'medium',
	tags: ['fastapi', 'api', 'deployment'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# FastAPI LLM API

Build a production-ready API for serving LLM predictions.

## Task

Create a FastAPI application that:
- Loads an LLM on startup
- Handles text generation requests
- Supports streaming responses
- Includes proper error handling

## Example

\`\`\`python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
async def generate(request: GenerateRequest):
    response = model.generate(request.prompt)
    return {"text": response}
\`\`\``,

	initialCode: `from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="LLM API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    tokens_used: int

# Your code here - implement these endpoints

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    # Your code here
    pass

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    # Your code here
    pass

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Stream generated text."""
    # Your code here
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Your code here
    pass
`,

	solutionCode: `from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="LLM API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    tokens_used: int

# Global model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model, tokenizer

    model_name = "gpt2"  # Use small model for demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_used = len(outputs[0])

        return GenerateResponse(
            text=generated_text,
            tokens_used=tokens_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_tokens(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """Generate tokens and yield them one by one."""
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs.input_ids

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.multinomial(
                torch.softmax(next_token_logits, dim=-1),
                num_samples=1
            )

        generated = torch.cat([generated, next_token], dim=1)
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

        yield f"data: {token_text}\\n\\n"
        await asyncio.sleep(0.01)

        if next_token.item() == tokenizer.eos_token_id:
            break

    yield "data: [DONE]\\n\\n"

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Stream generated text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StreamingResponse(
        stream_tokens(request.prompt, request.max_tokens, request.temperature),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
`,

	testCode: `import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

class TestLLMAPI(unittest.TestCase):
    def test_generate_request_model(self):
        request = GenerateRequest(prompt="Hello", max_tokens=50)
        self.assertEqual(request.prompt, "Hello")
        self.assertEqual(request.max_tokens, 50)
        self.assertEqual(request.temperature, 0.7)

    def test_generate_request_defaults(self):
        request = GenerateRequest(prompt="Test")
        self.assertEqual(request.max_tokens, 100)
        self.assertEqual(request.stream, False)

    def test_generate_response_model(self):
        response = GenerateResponse(text="Hello world", tokens_used=10)
        self.assertEqual(response.text, "Hello world")
        self.assertEqual(response.tokens_used, 10)

    def test_request_has_prompt(self):
        request = GenerateRequest(prompt="Test prompt")
        self.assertEqual(request.prompt, "Test prompt")

    def test_request_temperature_default(self):
        request = GenerateRequest(prompt="Test")
        self.assertEqual(request.temperature, 0.7)

    def test_response_has_text(self):
        response = GenerateResponse(text="Output", tokens_used=5)
        self.assertIsInstance(response.text, str)

    def test_response_has_tokens_used(self):
        response = GenerateResponse(text="Out", tokens_used=3)
        self.assertIsInstance(response.tokens_used, int)

    def test_request_stream_default_false(self):
        request = GenerateRequest(prompt="P")
        self.assertFalse(request.stream)

    def test_request_custom_temperature(self):
        request = GenerateRequest(prompt="P", temperature=0.5)
        self.assertEqual(request.temperature, 0.5)

    def test_app_exists(self):
        self.assertIsNotNone(app)
`,

	hint1: 'Use @app.on_event("startup") to load heavy models once',
	hint2: 'Use StreamingResponse with async generator for streaming',

	whyItMatters: `FastAPI is ideal for LLM serving:

- **Async**: Handle concurrent requests efficiently
- **Type hints**: Auto-generated API documentation
- **Streaming**: SSE for real-time token streaming
- **Production-ready**: Easy to deploy with Uvicorn

This is how many LLM APIs are built.`,

	translations: {
		ru: {
			title: 'FastAPI LLM API',
			description: `# FastAPI LLM API

Создайте production-ready API для сервинга предсказаний LLM.

## Задача

Создайте FastAPI приложение, которое:
- Загружает LLM при старте
- Обрабатывает запросы на генерацию текста
- Поддерживает стриминг ответов
- Включает правильную обработку ошибок

## Пример

\`\`\`python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
async def generate(request: GenerateRequest):
    response = model.generate(request.prompt)
    return {"text": response}
\`\`\``,
			hint1: 'Используйте @app.on_event("startup") для загрузки тяжелых моделей один раз',
			hint2: 'Используйте StreamingResponse с async генератором для стриминга',
			whyItMatters: `FastAPI идеален для сервинга LLM:

- **Async**: Эффективная обработка конкурентных запросов
- **Type hints**: Автогенерация документации API
- **Streaming**: SSE для потоковой передачи токенов
- **Production-ready**: Легкий деплой с Uvicorn`,
		},
		uz: {
			title: 'FastAPI LLM API',
			description: `# FastAPI LLM API

LLM bashoratlarini serving qilish uchun production-ready API yarating.

## Topshiriq

FastAPI ilovasini yarating:
- Boshlanganda LLM ni yuklaydi
- Matn generatsiya so'rovlarini qayta ishlaydi
- Streaming javoblarni qo'llab-quvvatlaydi
- To'g'ri xato boshqaruvini o'z ichiga oladi

## Misol

\`\`\`python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
async def generate(request: GenerateRequest):
    response = model.generate(request.prompt)
    return {"text": response}
\`\`\``,
			hint1: 'Og\'ir modellarni bir marta yuklash uchun @app.on_event("startup") dan foydalaning',
			hint2: "Streaming uchun async generator bilan StreamingResponse dan foydalaning",
			whyItMatters: `FastAPI LLM serving uchun ideal:

- **Async**: Bir vaqtdagi so'rovlarni samarali qayta ishlash
- **Type hints**: API hujjatlarini avtogeneratsiya
- **Streaming**: Real-time token streaming uchun SSE
- **Production-ready**: Uvicorn bilan oson deploy`,
		},
	},
};

export default task;
