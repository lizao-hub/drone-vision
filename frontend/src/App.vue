<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { connectWhep, mediaUrl } from './whep'

const EMPTY_STATUS = {
  algorithm: { running: false, online: false },
  paths: { drone1: false, drone1_yolo: false },
}

const status = ref(EMPTY_STATUS)
const mode = ref('raw')
const busy = ref({ algorithm: false })
const apiError = ref('')
const whepError = ref('')
const connection = ref('')
const videoEl = ref(null)

let attachedPath = null
let connectingPath = null
let sessionClose = null
let attachToken = 0
let pollTimer = null
let retryTimer = null

const rawOnline = computed(() => status.value.paths.drone1)
const yoloOnline = computed(() => status.value.paths.drone1_yolo)
const canStartAlgorithm = computed(
  () => rawOnline.value && !(status.value.algorithm.running && mode.value === 'yolo'),
)

const activePath = computed(() =>
  mode.value === 'yolo' ? 'drone1_yolo' : 'drone1',
)

const overlayText = computed(() => {
  if (whepError.value) return `播放连接失败：${whepError.value}`
  if (mode.value === 'raw') {
    if (rawOnline.value) return ''
    return '等待信号源：在 Windows 运行 scripts\\push-windows.ps1'
  }
  if (yoloOnline.value) return ''
  return status.value.algorithm.running
    ? '算法运行中，等待检测流…'
    : '算法未运行，点击「启动算法」'
})

const isWaiting = computed(
  () =>
    !whepError.value &&
    ((mode.value === 'raw' && !rawOnline.value) ||
      (mode.value === 'yolo' && !yoloOnline.value)),
)

async function api(path, method = 'GET') {
  const response = await fetch(path, { method })
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = null
  }
  if (!response.ok) {
    throw new Error(data?.detail || `${response.status} ${response.statusText}`)
  }
  return data
}

async function refresh() {
  try {
    status.value = await api('/api/status')
    apiError.value = ''
  } catch (error) {
    apiError.value = `控制服务不可达：${error.message}`
  }
}

function pathOnline(path) {
  return path === 'drone1_yolo' ? yoloOnline.value : rawOnline.value
}

function closeSession() {
  if (sessionClose) {
    sessionClose()
    sessionClose = null
  }
  if (videoEl.value) videoEl.value.srcObject = null
  connection.value = ''
}

function detach() {
  attachToken += 1
  closeSession()
  attachedPath = null
  connectingPath = null
  whepError.value = ''
}

async function attach(path) {
  const token = ++attachToken
  connectingPath = path
  closeSession()

  try {
    const close = await connectWhep(videoEl.value, path, (state) => {
      if (token === attachToken) connection.value = state
    })
    if (token !== attachToken) {
      close()
      return
    }
    sessionClose = close
    attachedPath = path
    whepError.value = ''
  } catch (error) {
    if (token !== attachToken) return
    closeSession()
    attachedPath = null
    whepError.value = error.message
    clearTimeout(retryTimer)
    retryTimer = setTimeout(() => {
      if (activePath.value === path && pathOnline(path)) attach(path)
    }, 3000)
  } finally {
    if (token === attachToken) connectingPath = null
  }
}

function syncPlayback() {
  const path = activePath.value
  if (!pathOnline(path)) {
    if (attachedPath || connectingPath || whepError.value) detach()
    return
  }
  if (attachedPath === path && sessionClose) return
  if (connectingPath === path) return
  attach(path)
}

async function startAlgorithm() {
  busy.value.algorithm = true
  apiError.value = ''
  try {
    // If the worker is already running (e.g. after a restart), just switch
    // the view instead of calling the API again.
    if (!status.value.algorithm.running) {
      status.value = await api('/api/algorithm/start', 'POST')
    }
    mode.value = 'yolo'
    await refresh()
  } catch (error) {
    apiError.value = error.message
  } finally {
    busy.value.algorithm = false
  }
}

async function stopAlgorithm() {
  busy.value.algorithm = true
  apiError.value = ''
  try {
    status.value = await api('/api/algorithm/stop', 'POST')
    mode.value = 'raw'
    await refresh()
  } catch (error) {
    apiError.value = error.message
  } finally {
    busy.value.algorithm = false
  }
}

watch([mode, rawOnline, yoloOnline], syncPlayback)

onMounted(async () => {
  await refresh()
  syncPlayback()
  pollTimer = setInterval(refresh, 2000)
})

onUnmounted(() => {
  clearInterval(pollTimer)
  clearTimeout(retryTimer)
  detach()
})
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="logo">◉</span>
        <div>
          <h1>无人机直播演示</h1>
          <p>Windows 推流 · MediaMTX · mock YOLO</p>
        </div>
      </div>
      <div class="chips">
        <span class="chip" :class="{ on: rawOnline }"><i />原画</span>
        <span class="chip" :class="{ on: status.algorithm.running }"><i />算法</span>
        <span class="chip" :class="{ on: yoloOnline }"><i />检测</span>
      </div>
    </header>

    <main class="layout">
      <section class="stage-card">
        <div class="stage">
          <video ref="videoEl" muted autoplay playsinline />
          <div class="stage-label">
            {{ mode === 'yolo' ? 'YOLO 检测画面' : '原始画面' }}
          </div>
          <div v-if="connection && !overlayText" class="conn">
            {{ connection }}
          </div>
          <div v-if="overlayText" class="overlay">
            <div v-if="isWaiting" class="spinner" />
            <p>{{ overlayText }}</p>
          </div>
        </div>
        <div class="stage-foot">
          <span>{{ mediaUrl(activePath) }}</span>
          <a :href="mediaUrl(activePath)" target="_blank" rel="noreferrer">
            打不开？用 MediaMTX 播放页
          </a>
        </div>
      </section>

      <aside class="panel">
        <h2>控制台</h2>

        <div class="group">
          <div class="group-head">
            <span>信号源</span>
            <em :class="{ live: rawOnline }">
              {{ rawOnline ? '在线' : '等待推流' }}
            </em>
          </div>
          <p class="hint">
            推流模拟真实设备，由 Windows 侧独立启动：<br />
            <code>scripts\push-windows.ps1</code>
          </p>
        </div>

        <div class="group">
          <div class="group-head">
            <span>检测算法</span>
            <em :class="{ live: status.algorithm.online }">
              {{
                status.algorithm.online
                  ? '在线'
                  : status.algorithm.running
                    ? '启动中'
                    : '离线'
              }}
            </em>
          </div>
          <div class="row">
            <button
              class="btn accent"
              :disabled="!canStartAlgorithm || busy.algorithm"
              @click="startAlgorithm"
            >
              {{ busy.algorithm && !status.algorithm.running ? '启动中…' : '启动算法' }}
            </button>
            <button
              class="btn ghost"
              :disabled="!status.algorithm.running || busy.algorithm"
              @click="stopAlgorithm"
            >
              {{ busy.algorithm && status.algorithm.running ? '停止中…' : '停止算法' }}
            </button>
          </div>
          <p class="hint">
            启动 worker 容器，输出 drone1_yolo；停止后自动回到原画。
          </p>
        </div>

        <div v-if="apiError" class="error">{{ apiError }}</div>

        <div class="meta">
          <div><span>原画</span><code>drone1</code></div>
          <div><span>检测</span><code>drone1_yolo</code></div>
        </div>
      </aside>
    </main>

    <footer>本地演示 · 视频不经过控制服务</footer>
  </div>
</template>
