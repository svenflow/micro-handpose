# micro-handpose Development Notes

## Real Device Testing (LambdaTest)

We use [LambdaTest](https://app.lambdatest.com) Real Device Plus for testing on real iPhones remotely. Account: `nicklaudethorat@gmail.com` (password in keychain under `lambdatest.com`).

### Direct Session URL Pattern

Start a real device Safari session by navigating to a URL with parameters:

```
https://applive.lambdatest.com/live?os=ios&brand=iphone&device=<DEVICE>&osVersion=<VERSION>&browser=safari&url=<ENCODED_URL>
```

**Example — iPhone 17 Pro Max, iOS 26, opening diag.html:**
```
https://applive.lambdatest.com/live?os=ios&brand=iphone&device=iPhone%2017%20Pro%20Max&osVersion=26&browser=safari&url=https%3A%2F%2Fsvenflow.github.io%2Fmicro-handpose%2Fdiag.html
```

This auto-provisions a real device and opens Safari to the specified URL. No manual device selection needed.

### Key Findings

- **WebGPU requires iOS 26+** — iPhone 16 Pro Max on iOS 18 reports "No WebGPU". Only iOS 26 devices have WebGPU in Safari.
- **shader-f16 works on iOS 26** — Apple GPU arch reports `shader-f16: true`, f16 shaders compile successfully.
- **Camera not available** — LambdaTest remote devices don't provide camera access, so camera-based benchmarks won't run. Use `diag.html` (synthetic test data) for automated testing.

### Available Devices (iOS 26 with WebGPU)

| Device | OS | WebGPU |
|--------|-----|--------|
| iPhone 17 Pro Max | 26 | ✅ |
| iPhone 17 Pro | 26 | ✅ |
| iPhone Air | 26 | ✅ |
| iPhone 17 | 26 | ✅ |
| iPhone 16 Pro Max | 18 | ❌ |
| iPhone 15 Pro Max | 17 | ❌ |

### Automation from CLI

To test on a real iPhone from the command line (via Chrome extension):

```bash
# Open LambdaTest session with specific device + URL
chrome -p 0 open "https://applive.lambdatest.com/live?os=ios&brand=iphone&device=iPhone%2017%20Pro%20Max&osVersion=26&browser=safari&url=https%3A%2F%2Fsvenflow.github.io%2Fmicro-handpose%2Fdiag.html"

# Wait ~15s for device to boot + page to load, then screenshot
sleep 15
chrome -p 0 screenshot <tab_id>

# End session when done (via sidebar aria-label)
chrome -p 0 js <tab_id> "Array.from(document.querySelectorAll('[aria-label]')).find(d => d.getAttribute('aria-label').includes('End Session')).click()"
# Then confirm:
chrome -p 0 js <tab_id> "Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Yes, End Session')).click()"

# Close tab
chrome -p 0 close <tab_id>
```

### Limitations

- **No programmatic touch input** — LambdaTest's device interaction layer uses WebRTC video streaming with proprietary WebSocket touch forwarding. Standard DOM click/touch events do NOT get forwarded to the device. The only reliable way to interact is through their web UI manually.
- **Device selection UI** — LambdaTest's React UI blocks programmatic clicks on device selection. Use the direct URL pattern above to bypass device selection entirely.
- **Session timeout** — Sessions auto-close after ~3 min of inactivity. Keep alive by periodically taking screenshots.
- **1 parallel session** on the current plan ($47/mo monthly).

## Test Pages

| Page | Purpose | Camera Required |
|------|---------|----------------|
| `diag.html` | Full diagnostic — WebGPU, f16, model load, inference, benchmark | No |
| `?testProfile` | Live camera benchmark with MediaPipe comparison | Yes |
| `test.html` | Unit tests with synthetic image data | No |

## Build & Deploy

```bash
npm run build          # Build library
npm run build:docs     # Build docs/demo site
# GH Pages auto-deploys from docs/ on push to main
```
