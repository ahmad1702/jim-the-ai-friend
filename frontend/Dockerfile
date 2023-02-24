# Build step #1: build the React front end
FROM node:lts-alpine as build-step
WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH

COPY .env ./
COPY index.html ./
COPY vite.config.ts ./
COPY tsconfig.node.json  ./
COPY tsconfig.json  ./
COPY package.json  ./
COPY ./src ./src
COPY ./public ./public

RUN npm install
RUN npm run build

# Build step #2: build an Caddy container
FROM caddy:alpine
EXPOSE 80
EXPOSE 443
COPY --from=build-step /app/dist /usr/share/caddy