import React, { useState, useRef, useEffect } from "react";
import { useUITranslation } from "@/contexts/LanguageContext";

export interface Snippet {
  id: string;
  name: string;
  description: string;
  language: string;
  code: string;
}

const SNIPPETS: Snippet[] = [
  // Go Snippets
  {
    id: "go-http-server",
    name: "HTTP Server",
    description: "Basic HTTP server with handler",
    language: "go",
    code: `package main

import (
    "fmt"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
}

func main() {
    http.HandleFunc("/", handler)
    fmt.Println("Server starting on :8080...")
    http.ListenAndServe(":8080", nil)
}
`,
  },
  {
    id: "go-goroutines",
    name: "Goroutines & Channels",
    description: "Concurrent execution with channels",
    language: "go",
    code: `package main

import "fmt"

func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\\n", id, j)
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 5)
    results := make(chan int, 5)

    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }

    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)

    for a := 1; a <= 5; a++ {
        fmt.Println("Result:", <-results)
    }
}
`,
  },
  // Java Snippets
  {
    id: "java-stream",
    name: "Stream API",
    description: "Filter, map, reduce with streams",
    language: "java",
    code: `import java.util.*;
import java.util.stream.*;

public class Main {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        int sum = numbers.stream()
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .reduce(0, Integer::sum);

        System.out.println("Sum of squares of even numbers: " + sum);

        List<String> words = numbers.stream()
            .map(n -> "Item " + n)
            .collect(Collectors.toList());

        words.forEach(System.out::println);
    }
}
`,
  },
  {
    id: "java-optional",
    name: "Optional Pattern",
    description: "Safe null handling with Optional",
    language: "java",
    code: `import java.util.Optional;

public class Main {
    public static void main(String[] args) {
        Optional<String> name = findUserName(1);

        String greeting = name
            .map(n -> "Hello, " + n + "!")
            .orElse("Hello, Guest!");

        System.out.println(greeting);

        name.ifPresent(n -> System.out.println("Found: " + n));

        String defaultName = name.orElseGet(() -> "DefaultUser");
        System.out.println("Name: " + defaultName);
    }

    static Optional<String> findUserName(int id) {
        if (id == 1) return Optional.of("Alice");
        return Optional.empty();
    }
}
`,
  },
  // Python Snippets
  {
    id: "python-comprehension",
    name: "List Comprehensions",
    description: "Pythonic list transformations",
    language: "python",
    code: `# List comprehension basics
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter and transform
squares_of_even = [n**2 for n in numbers if n % 2 == 0]
print(f"Squares of even: {squares_of_even}")

# Nested comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(f"Matrix: {matrix}")

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["apple", "banana", "cherry"]}
print(f"Word lengths: {word_lengths}")

# Set comprehension
unique_chars = {char.lower() for char in "Hello World" if char.isalpha()}
print(f"Unique chars: {unique_chars}")
`,
  },
  {
    id: "python-decorators",
    name: "Decorators",
    description: "Function decorators pattern",
    language: "python",
    code: `import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
            raise Exception(f"Failed after {max_attempts} attempts")
        return wrapper
    return decorator

@timer
@retry(max_attempts=2)
def fetch_data(url):
    print(f"Fetching {url}...")
    return "data"

result = fetch_data("https://api.example.com")
print(f"Result: {result}")
`,
  },
  // TypeScript Snippets
  {
    id: "ts-generics",
    name: "Generics",
    description: "Type-safe generic functions",
    language: "typescript",
    code: `// Generic function
function identity<T>(arg: T): T {
    return arg;
}

// Generic interface
interface Result<T, E = Error> {
    data?: T;
    error?: E;
}

// Generic class
class Queue<T> {
    private items: T[] = [];

    enqueue(item: T): void {
        this.items.push(item);
    }

    dequeue(): T | undefined {
        return this.items.shift();
    }
}

// Usage
const stringResult: Result<string> = { data: "Hello" };
const numberResult: Result<number> = { data: 42 };

const queue = new Queue<number>();
queue.enqueue(1);
queue.enqueue(2);
console.log(queue.dequeue()); // 1

console.log(identity<string>("TypeScript"));
console.log(identity(123)); // Type inference
`,
  },
  {
    id: "ts-async",
    name: "Async/Await Patterns",
    description: "Promise handling and async patterns",
    language: "typescript",
    code: `// Async function
async function fetchUser(id: number): Promise<{ name: string }> {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({ name: \`User \${id}\` });
        }, 100);
    });
}

// Parallel execution
async function fetchAllUsers(ids: number[]) {
    const users = await Promise.all(ids.map(fetchUser));
    return users;
}

// Sequential execution
async function fetchUsersSequentially(ids: number[]) {
    const users = [];
    for (const id of ids) {
        users.push(await fetchUser(id));
    }
    return users;
}

// Main
(async () => {
    console.log("Parallel fetch:");
    const parallel = await fetchAllUsers([1, 2, 3]);
    parallel.forEach(u => console.log(u.name));

    console.log("\\nSequential fetch:");
    const sequential = await fetchUsersSequentially([4, 5]);
    sequential.forEach(u => console.log(u.name));
})();
`,
  },
];

interface SnippetsLibraryProps {
  currentLanguage: string;
  onInsertSnippet: (code: string, language: string) => void;
}

export const SnippetsLibrary: React.FC<SnippetsLibraryProps> = ({
  currentLanguage,
  onInsertSnippet,
}) => {
  const { tUI } = useUITranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState<string>("all");
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const filteredSnippets =
    filter === "all"
      ? SNIPPETS
      : SNIPPETS.filter((s) => s.language === filter);

  const languages = Array.from(new Set(SNIPPETS.map((s) => s.language)));

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#21262d] rounded transition-colors"
        title={tUI("playground.snippets")}
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
          />
        </svg>
        <span className="hidden sm:inline">{tUI("playground.snippets")}</span>
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-1 bg-white dark:bg-[#161b22] rounded-lg shadow-lg z-50 w-[320px] border border-gray-200 dark:border-[#21262d]">
          {/* Header */}
          <div className="p-3 border-b border-gray-200 dark:border-[#21262d]">
            <div className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
              {tUI("playground.snippetsTitle")}
            </div>
            {/* Filter */}
            <div className="flex gap-1 flex-wrap">
              <FilterButton
                active={filter === "all"}
                onClick={() => setFilter("all")}
              >
                All
              </FilterButton>
              {languages.map((lang) => (
                <FilterButton
                  key={lang}
                  active={filter === lang}
                  onClick={() => setFilter(lang)}
                >
                  {lang.charAt(0).toUpperCase() + lang.slice(1)}
                </FilterButton>
              ))}
            </div>
          </div>

          {/* Snippets List */}
          <div className="max-h-[300px] overflow-y-auto">
            {filteredSnippets.map((snippet) => (
              <SnippetItem
                key={snippet.id}
                snippet={snippet}
                isCurrentLanguage={snippet.language === currentLanguage}
                onInsert={() => {
                  onInsertSnippet(snippet.code, snippet.language);
                  setIsOpen(false);
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

interface FilterButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

const FilterButton: React.FC<FilterButtonProps> = ({
  active,
  onClick,
  children,
}) => (
  <button
    onClick={onClick}
    className={`px-2 py-1 text-xs rounded transition-colors ${
      active
        ? "bg-brand-500 text-white"
        : "bg-gray-100 dark:bg-[#21262d] text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-[#30363d]"
    }`}
  >
    {children}
  </button>
);

interface SnippetItemProps {
  snippet: Snippet;
  isCurrentLanguage: boolean;
  onInsert: () => void;
}

const SnippetItem: React.FC<SnippetItemProps> = ({
  snippet,
  isCurrentLanguage,
  onInsert,
}) => {
  const languageColors: Record<string, string> = {
    go: "text-cyan-500",
    java: "text-orange-500",
    python: "text-green-500",
    typescript: "text-blue-500",
  };

  return (
    <button
      onClick={onInsert}
      className="w-full text-left p-3 hover:bg-gray-50 dark:hover:bg-[#21262d] transition-colors border-b border-gray-100 dark:border-[#21262d] last:border-b-0"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={`text-xs font-medium ${languageColors[snippet.language] || "text-gray-500"}`}
            >
              {snippet.language.toUpperCase()}
            </span>
            <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
              {snippet.name}
            </span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate">
            {snippet.description}
          </p>
        </div>
        {isCurrentLanguage && (
          <span className="flex-shrink-0 px-1.5 py-0.5 bg-brand-100 dark:bg-brand-900/30 text-brand-600 dark:text-brand-400 text-[10px] font-medium rounded">
            Current
          </span>
        )}
      </div>
    </button>
  );
};

export default SnippetsLibrary;
