# Virtual Human — Conversational Mode Handover

Turns the Virtual Human pipeline from push-to-talk into a real back-and-forth
conversation: press **T** once to start, and the LLM decides when the
conversation ends. No other input is required in between.

## How to apply

Copy each file below into the matching path in the project, overwriting the
existing file. The folder structure here mirrors the project root exactly,
so you can drag the whole `Handover` folder's contents on top of the project
root and let it overwrite in place.

| File in this folder | Destination |
|---|---|
| `Config/DefaultGame.ini` | `Config/DefaultGame.ini` |
| `Source/KairosSample/Public/VirtualHumanConversationComponent.h` | `Source/KairosSample/Public/VirtualHumanConversationComponent.h` |
| `Source/KairosSample/Private/VirtualHumanConversationComponent.cpp` | `Source/KairosSample/Private/VirtualHumanConversationComponent.cpp` |
| `LLM/aisegment.py` | `LLM/aisegment.py` |
| `VirtualHuman_Setup.md` | `VirtualHuman_Setup.md` |

After copying:

1. **Recompile** — reopen the `.uproject` (or rebuild from the `.sln`) so the
   changed C++ picks up.
2. **Blueprint rewiring (one-time, in-editor):** open `BP_FirstPersonCharacter`,
   find the `IA_Talk` Enhanced Input node, and **remove the wire from its
   `Completed` pin to `Stop Listening`** (leave `Completed` unwired). Only the
   `Started` pin → `Start Listening` binding is needed now — this is no
   longer push-to-talk, and leaving the old release-binding in place is
   harmless but pointless.
3. Restart the FastAPI/Ollama backend (`aisegment.py` changed) — `uvicorn
   server:app --host 127.0.0.1 --port 8008`, plus `ollama serve` as before.
4. In the editor, double check *Project Settings → Plugins → NVIDIA ACE ASR →
   Capture → Max Capture Inactivity Seconds* reads `2.5` (from the
   `DefaultGame.ini` change) — this is what auto-finalizes each utterance on
   a pause now that there's no key-release to do it.

Full behavior details, troubleshooting, and the safety-gate notes are in
`VirtualHuman_Setup.md` — the "What `UVirtualHumanConversationComponent` does"
and "If something's off" sections were rewritten for this change.

## What changed, in one paragraph

`StartListening()` is now the only thing bound to input. Once a conversation
starts, ACE ASR's own inactivity watchdog auto-finalizes each utterance after
a pause, and the component automatically loops listen → LLM → speak → listen
again on its own after every reply. `aisegment.py` keeps the running
conversation history (the ACE bridge only ever sends one utterance at a
time, no history) and, after generating each reply, runs one extra small
model call asking whether the conversation has reached a natural close. If
so, it appends a `[[END_SESSION]]` marker to the reply; the C++ component
strips that marker before it's ever spoken or shown, and uses it to decide
whether to keep looping or return to Idle and wait for the next T press. The
self-harm override and the guard-blocked fallback line never trigger that
end check, so a crisis moment or a blocked exchange can never auto-end the
session.

**Not yet compiled/tested in-editor** — no UBT/Visual Studio toolchain was
available in the environment this was written in. Please build and run
through the "Press Play and test" steps in `VirtualHuman_Setup.md` before
relying on it.
