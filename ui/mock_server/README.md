# mock_server — Node.js WebSocket replay server

Broadcasts fixture events at realistic cadence for browser-side development.

## Start

```bash
# From ui/ directory:
npm run mock-server

# Custom options:
node --experimental-strip-types mock_server/server.ts --port 8765 --speed 1.5 --loop
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8765` | WebSocket port |
| `--speed` | `1.0` | Playback speed multiplier (2.0 = 2× faster) |
| `--case` | `chest_pain_01` | Fixture case_id (loads `fixtures/<case>_demo.json`) |
| `--loop` | off | Replay fixture continuously |

## Wire endpoint

```
ws://localhost:8765/session/chest_pain_01/events
```

## Fixture format

See `ui/fixtures/chest_pain_demo.json`. Fixture schema matches the dispatch brief §8:

```json
{
  "case_id": "chest_pain_01",
  "events": [
    {"t_ms": 0,    "type": "turn",        "turn": {...}},
    {"t_ms": 3500, "type": "claim",       "claim": {...}},
    {"t_ms": 11800,"type": "supersede",   "edge": {...}},
    {"t_ms": 20000,"type": "differential","ranking": {...}},
    {"t_ms": 66000,"type": "verifier",    "verifier": {...}},
    {"t_ms": 68000,"type": "soap_delta",  "delta": {...}}
  ]
}
```

Each fixture event is translated into the SubstrateEvent wire frames expected by
`ui/src/lib/ws.ts` — the same shapes as `src/substrate/event_bus.py`.

## Note

Phase 1 uses the in-process `MockServer` class (`ui/src/mock/MockServer.ts`) so
the browser can run without this server running. This Node server is for:
- Integration testing with the real substrate API shape
- Demo recording where you want a separate process driving the UI
- Future wiring once `src/api/` ships a real WebSocket
