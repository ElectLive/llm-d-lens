# Node is a frontend build dependency, not an application runtime.
FROM node:26-alpine AS build

WORKDIR /app

COPY package*.json ./
COPY docs/site/package.json ./docs/site/package.json
COPY apps/frontend/package.json ./apps/frontend/package.json
COPY packages/shared/package.json ./packages/shared/package.json
RUN npm ci --ignore-scripts

COPY . .
RUN npm run build

# Serve stage
FROM python:3.12-slim

WORKDIR /app

ARG TARGETARCH
ARG KUBECTL_VERSION=v1.36.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git jq openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
    && curl -fsSLo /tmp/kubectl.sha256 "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl.sha256" \
    && printf '%s  /tmp/kubectl\n' "$(cat /tmp/kubectl.sha256)" | sha256sum --check \
    && install -m 0755 /tmp/kubectl /usr/local/bin/kubectl \
    && rm /tmp/kubectl /tmp/kubectl.sha256

COPY apps/backend/pyproject.toml ./apps/backend/pyproject.toml
COPY apps/backend/src ./apps/backend/src
RUN python -m pip install --no-cache-dir -e "apps/backend[embedded-db,ldap]"
COPY packages/shared ./packages/shared
COPY --from=build /app/apps/frontend/dist ./apps/frontend/dist
COPY hack/run-backend.sh hack/storage-env.sh ./hack/

ENV LENS_DATA_DIR=/var/lib/lens
ENV LENS_CACHE_DIR=/var/cache/lens
ENV LENS_LOG_DIR=/var/log/lens
VOLUME ["/var/lib/lens", "/var/cache/lens", "/var/log/lens"]
ENV PORT=8080
ENV NODE_ENV=production
EXPOSE 8080

CMD ["bash", "hack/run-backend.sh"]
