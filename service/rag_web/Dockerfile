# frontend/Dockerfile
FROM node:20-alpine

# non-root 유저 생성
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts || npm install --ignore-scripts

COPY rag_web/src ./src
COPY rag_web/index.html ./index.html
COPY rag_web/tsconfig.json ./tsconfig.json
COPY rag_web/vite.config.ts ./vite.config.ts

# non-root user로 실행
USER appuser

EXPOSE 5173
CMD ["npm", "run", "dev"]