# KODLA Backend Setup Guide

This guide describes how to initialize and run the NestJS backend with PostgreSQL.

## Prerequisites

- Docker & Docker Compose
- Node.js (v18+)

## Step 1: Prepare Configuration Files

The system generated the configuration files with `.txt` extensions to ensure delivery. You need to rename them.

1.  **Rename Env File:**
    ```bash
    mv server/env.txt server/.env
    ```

2.  **Rename Prisma Schema:**
    ```bash
    mv server/prisma/schema.prisma.txt server/prisma/schema.prisma
    ```

## Step 2: Start Database

Run PostgreSQL using Docker Compose from the project root:

```bash
docker-compose up -d
```

Check if the container `kodla_postgres` is running.

## Step 3: Install Backend Dependencies

Navigate to the server directory and install packages:

```bash
cd server
npm install
```

## Step 4: Initialize Database Schema

Push the Prisma schema to the running PostgreSQL database:

```bash
npx prisma db push
```

*This command creates the tables (User, Task, Submission, etc.) in the database.*

## Step 5: Start the Server

Run the backend in development mode:

```bash
npm run start:dev
```

The server should start on `http://localhost:8080`.

## Step 6: Verify

Open `http://localhost:8080` in your browser. You should see "Hello World!" (or a 404 if the root route isn't defined, which is normal for an API).

## Next Steps for Development

1.  **Generate Prisma Client:** If you see type errors in `prisma.service.ts`, run `npx prisma generate` inside the `server` folder.
2.  **Seed Data:** You will need to create a seed script to populate `Task` table with the mock data from frontend if you want to use the backend immediately.
