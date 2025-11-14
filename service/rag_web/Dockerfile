# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci || npm install

COPY rag_web/src ./src
COPY rag_web/index.html ./index.html
COPY rag_web/tsconfig.json ./tsconfig.json
COPY rag_web/vite.config.ts ./vite.config.ts

EXPOSE 5173
CMD ["npm", "run", "dev"]