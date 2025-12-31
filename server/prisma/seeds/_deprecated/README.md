# Deprecated Seed Files

These files contain old seed data that has been replaced by the new modular structure in `/courses` and `/shared/modules`.

## Old Files
- `go-advanced.ts` - Go advanced topics (now in shared/modules/go)
- `go-concurrency.ts` - Go concurrency modules (now in shared/modules/go)
- `go-error-handling.ts` - Go error handling (now in shared/modules/go)
- `go-http-grpc.ts` - Go HTTP/gRPC topics (now in shared/modules/go)
- `go-patterns.ts` - Go design patterns (now in shared/modules/go)

## New Structure
```
seeds/
├── courses/           # Course definitions
│   ├── go-basics/
│   ├── go-concurrency/
│   ├── go-web-apis/
│   ├── go-production/
│   ├── java-core/
│   ├── java-modern/
│   └── java-advanced/
├── shared/
│   └── modules/       # Reusable modules
│       ├── go/        # 25 Go modules
│       └── java/      # 35 Java modules
└── types.ts           # TypeScript interfaces
```

## Migration Date
2025-12-12
