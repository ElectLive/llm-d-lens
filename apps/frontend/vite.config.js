import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'fs'

// When TLS_CERT_FILE/TLS_KEY_FILE are set (see hack/lens-local.sh and
// hack/generate-self-signed-cert.sh), serve the Vite dev server itself
// over HTTPS too -- browsers only expose features like microphone access
// (getUserMedia) in a "secure context" (HTTPS or http://localhost), so a
// bare-metal box reached over http://<lan-ip> needs both the frontend dev
// server and the Python API server on HTTPS.
const tlsCertFile = process.env.TLS_CERT_FILE
const tlsKeyFile = process.env.TLS_KEY_FILE
const httpsOptions =
  tlsCertFile && tlsKeyFile
    ? { cert: fs.readFileSync(tlsCertFile), key: fs.readFileSync(tlsKeyFile) }
    : undefined
// The Python API server is started with the same cert, so
// proxied requests need https:// targets too when TLS is enabled.
const proxyScheme = httpsOptions ? 'https' : 'http'
const backendPort = process.env.BACKEND_PORT || process.env.SERVER_PORT || process.env.PORT || 3000

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  envPrefix: ['VITE_', 'REACT_APP_'],
  server: {
    host: process.env.VITE_HOST || '0.0.0.0',
    port: Number(process.env.VITE_PORT) || 5173,
    strictPort: true,
    https: httpsOptions,
    watch: {
      ignored: ['**/.venv/**', '**/node_modules/**', '**/dist/**', '**/.git/**'],
    },
    proxy: {
      '/api/cluster-overview': {
        target: process.env.VITE_AUTH_PROXY_TARGET || `${proxyScheme}://localhost:${backendPort}`,
        changeOrigin: true,
        secure: false,
      },
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || `${proxyScheme}://localhost:${backendPort}`,
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
