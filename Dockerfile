# syntax=docker/dockerfile:1

FROM node:20-bullseye AS builder
WORKDIR /app

# Build arguments for Vite
ARG VITE_API_URL=http://localhost:8080
ARG VITE_APP_ENV=production

ENV VITE_API_URL=${VITE_API_URL}
ENV VITE_APP_ENV=${VITE_APP_ENV}

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source files
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public

# Build
RUN npm run build

# Production stage
FROM nginx:1.27-alpine AS runner

# Copy nginx config
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
