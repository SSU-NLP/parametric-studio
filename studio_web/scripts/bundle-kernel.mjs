// Copy the Python kernel package into src-tauri/resources so Tauri bundles it into the app.
// Runs at build time (tauri.conf.json beforeBuildCommand). Cross-platform (Node fs), skips __pycache__.
import { cpSync, rmSync, existsSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))       // studio_web/scripts
const src = resolve(here, '..', '..', 'parametic_studio')  // repo/parametic_studio
const dst = resolve(here, '..', 'src-tauri', 'resources', 'parametic_studio')
const reqSrc = resolve(here, '..', '..', 'requirements-studio.txt')
const reqDst = resolve(here, '..', 'src-tauri', 'resources', 'requirements-studio.txt')

if (!existsSync(src)) { console.error('[bundle-kernel] kernel source not found:', src); process.exit(1) }
if (existsSync(dst)) rmSync(dst, { recursive: true, force: true })
mkdirSync(dirname(dst), { recursive: true })
cpSync(src, dst, { recursive: true, filter: (p) => !p.includes('__pycache__') && !p.endsWith('.pyc') })
console.log('[bundle-kernel] bundled kernel →', dst)

if (!existsSync(reqSrc)) { console.error('[bundle-kernel] requirements source not found:', reqSrc); process.exit(1) }
cpSync(reqSrc, reqDst)
console.log('[bundle-kernel] bundled requirements →', reqDst)
