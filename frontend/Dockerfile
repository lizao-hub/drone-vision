# Build the Vue app, then serve it with nginx.
FROM node:22-alpine AS build

ARG NPM_REGISTRY=https://registry.npmmirror.com
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --registry=$NPM_REGISTRY

COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
