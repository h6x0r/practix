# User Courses Module

This module manages user course enrollment and progress tracking.

## Features

- Track which courses a user has started
- Monitor course progress (0-100%)
- Track last accessed time for each course
- Auto-complete courses when progress reaches 100%

## API Endpoints

### GET /users/me/courses
Get all courses started by the authenticated user.

**Response:**
```json
[
  {
    "id": "uuid",
    "slug": "go-basics",
    "title": "Go Basics",
    "description": "Learn Go fundamentals",
    "category": "language",
    "icon": "🚀",
    "estimatedTime": "10h",
    "translations": {},
    "progress": 45,
    "startedAt": "2025-01-01T00:00:00Z",
    "lastAccessedAt": "2025-01-15T10:30:00Z",
    "completedAt": null
  }
]
```

### POST /users/me/courses/:courseSlug/start
Start a new course or resume an existing one.

**Response:**
```json
{
  "id": "uuid",
  "slug": "go-basics",
  "title": "Go Basics",
  "description": "Learn Go fundamentals",
  "category": "language",
  "icon": "🚀",
  "estimatedTime": "10h",
  "translations": {},
  "progress": 0,
  "startedAt": "2025-01-01T00:00:00Z",
  "lastAccessedAt": "2025-01-01T00:00:00Z",
  "completedAt": null
}
```

### PATCH /users/me/courses/:courseSlug/progress
Update progress for a course.

**Request Body:**
```json
{
  "progress": 75
}
```

**Response:**
```json
{
  "courseSlug": "go-basics",
  "progress": 75,
  "lastAccessedAt": "2025-01-15T10:30:00Z",
  "completedAt": null
}
```

## Database Schema

The module uses the `UserCourse` model from Prisma:

```prisma
model UserCourse {
  id             String   @id @default(uuid())
  userId         String
  courseSlug     String
  progress       Int      @default(0)
  startedAt      DateTime @default(now())
  lastAccessedAt DateTime @default(now())
  completedAt    DateTime?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([userId, courseSlug])
  @@index([userId])
}
```

## Service Methods

### getUserCourses(userId: string)
Retrieves all courses started by a user with their progress.

### startCourse(userId: string, courseSlug: string)
Creates or updates a UserCourse record when a user starts a course.

### updateProgress(userId: string, courseSlug: string, progress: number)
Updates the progress percentage (0-100) for a user's course. Auto-completes the course when progress reaches 100%.

### updateLastAccessed(userId: string, courseSlug: string)
Updates the last accessed timestamp. Creates a UserCourse record if it doesn't exist.

## Authentication

All endpoints require JWT authentication using `JwtAuthGuard`.
The user ID is extracted from the JWT token in the request.

## Validation

- Progress must be between 0 and 100 (validated using class-validator)
- Course must exist before starting
- User must have started a course before updating progress
