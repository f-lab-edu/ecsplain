# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts || npm install --ignore-scripts

RUN addgroup -S appgroup && adduser -S appuser -G appgroup \
    && chown -R appuser:appgroup /app


COPY src ./src
COPY index.html ./index.html
COPY tsconfig.json ./tsconfig.json
COPY vite.config.ts ./vite.config.ts

# non-root user로 실행
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 5173
CMD ["npm", "run", "dev"]
