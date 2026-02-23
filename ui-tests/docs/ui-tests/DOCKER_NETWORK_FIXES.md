# Docker Proxy Configuration for Corporate Networks

## Method 1: Configure Docker Desktop Proxy Settings
1. Open Docker Desktop
2. Go to Settings → Resources → Proxies
3. Enable "Manual proxy configuration"
4. Set HTTP/HTTPS proxy to your corporate proxy
5. Add localhost,127.0.0.1 to the "Bypass proxy settings" list

## Method 2: Configure Docker Daemon Proxy (Linux/macOS)
Create or edit ~/.docker/config.json:

```json
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://your-proxy-server:port",
     "httpsProxy": "http://your-proxy-server:port",
     "noProxy": "localhost,127.0.0.1"
   }
 }
}
```

## Method 3: Use Alternative Docker Images
The Dockerfiles have been updated to use alternative base images:

### Original (causing issues):
FROM node:18-alpine

### Alternative options:
FROM mcr.microsoft.com/node:18-alpine  # Microsoft Container Registry
FROM ghcr.io/nodejs/node:18-alpine     # GitHub Container Registry
FROM quay.io/nodejs/node:18-alpine     # Red Hat Quay

## Method 4: Download Image Manually
If you can access Docker Hub through a browser:
1. Download the node:18-alpine image manually
2. Load it into Docker:
   docker load -i node-18-alpine.tar

## Method 5: Use Multi-stage Build with Local Dependencies
Update Dockerfiles to use a local Node.js installation or multi-stage builds.
