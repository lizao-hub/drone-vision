// Minimal WHEP client for MediaMTX.
// MediaMTX exposes http://<host>:8889/<path>/whep and expects an SDP offer.

const MEDIA_BASE = `${window.location.protocol}//${window.location.hostname}:8889`

function waitForIce(pc, timeoutMs = 1200) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise((resolve) => {
    const timer = setTimeout(done, timeoutMs)
    function done() {
      clearTimeout(timer)
      pc.removeEventListener('icegatheringstatechange', check)
      resolve()
    }
    function check() {
      if (pc.iceGatheringState === 'complete') done()
    }
    pc.addEventListener('icegatheringstatechange', check)
  })
}

export function mediaUrl(path) {
  return `${MEDIA_BASE}/${path}`
}

export async function connectWhep(video, path, onState = () => {}) {
  const pc = new RTCPeerConnection({ iceServers: [], bundlePolicy: 'max-bundle' })
  let closed = false

  const teardown = () => {
    if (closed) return
    closed = true
    try {
      pc.close()
    } catch {
      // ignore
    }
  }

  try {
    pc.ontrack = (event) => {
      if (event.streams && event.streams[0]) {
        video.srcObject = event.streams[0]
      } else if (event.track) {
        const stream = new MediaStream([event.track])
        video.srcObject = stream
      }
      video.play().catch(() => {})
    }
    pc.onconnectionstatechange = () => onState(pc.connectionState)

    pc.addTransceiver('video', { direction: 'recvonly' })
    await pc.setLocalDescription(await pc.createOffer())
    await waitForIce(pc)

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 10000)
    let response
    try {
      response = await fetch(`${MEDIA_BASE}/${path}/whep`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp', Accept: 'application/sdp' },
        body: pc.localDescription.sdp,
        signal: controller.signal,
      })
    } finally {
      clearTimeout(timer)
    }

    if (!response.ok) {
      throw new Error(`WHEP ${response.status} ${response.statusText}`)
    }

    const answer = await response.text()
    await pc.setRemoteDescription({ type: 'answer', sdp: answer })
    return teardown
  } catch (error) {
    teardown()
    throw error
  }
}
