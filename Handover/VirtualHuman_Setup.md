# Virtual Human Pipeline — Setup Guide

Wires ACE ASR (speech-to-text) → `aisegment.py` (LLM, via FastAPI) → ACE TTS (speech synthesis) → your existing Audio2Face lipsync on **BP_Oskar** into one push-to-talk conversation loop, in the default `MHC_LightingPreset_Split_RT` scene.

## What I did already

Everything that could be done without you sitting in the editor is done. Specifically:

| Change | File | Why |
|---|---|---|
| Enabled `ACE_ASR` and `ACE_TTS` plugins | `KairosSample.uproject` | They were physically present in `Plugins/` but **not enabled** — only `NV_ACE_Reference` was. Without this, `UACEASRComponent`/`UACETTSComponent`/their subsystems don't exist at runtime. (ACE LLM plugin deliberately left disabled — see the earlier discussion on why we're not using it.) |
| Muted ACE TTS's own audio output | `Config/DefaultGame.ini` → `[/Script/ACE_TTS.ACETTSSettings]` `VolumeMultiplier=0.0` | Your Audio2Face setup on Oskar plays audio itself (via `ACEAudioCurveSourceComponent`) in sync with the lipsync curves. If ACE TTS *also* played its own copy through its internal audio component, you'd hear every line twice. This mutes TTS's own channel while leaving its synthesis and raw-audio streaming fully functional — Audio2Face becomes the single audible + animated output. |
| Added a new C++ component | `Source/KairosSample/Public/VirtualHumanConversationComponent.h` + `Private/...cpp` | Owns the entire conversation loop (see below). This is the piece that didn't exist as a ready-made Blueprint node anywhere in the ACE plugins. |
| Added plugin module dependencies | `Source/KairosSample/KairosSample.Build.cs` | `ACE_ASR`, `ACE_TTS`, `ACERuntime`, `ACECore` — needed for the new component to compile. |
| Confirmed the FastAPI bridge | `LLM/server.py` | Already present and correct — wraps `aisegment.py` as an OpenAI-compatible `/v1/chat/completions` endpoint. Nothing to change here. |

### What `UVirtualHumanConversationComponent` does

It talks directly to the **ACE ASR and ACE TTS GameInstance subsystems** (not the `UACEASRComponent`/`UACETTSComponent` Blueprint wrappers — going straight to the subsystem means you don't need to place either component in any Blueprint at all). It's no longer push-to-talk — pressing T starts the whole conversation, and the LLM decides when it ends. On `BeginPlay` it:

1. Initializes ASR and TTS.
2. Exposes `StartListening()` — this is the *only* thing you need to wire from input, and only its key-**down** needs binding (no key-up/release binding anymore).
3. Once the user pauses, ACE ASR's own inactivity watchdog (`MaxCaptureInactivitySeconds`, now set to 2.5s in `DefaultGame.ini`) auto-finalizes the transcript — no `StopListening()` call needed. It then calls `SendACEASRLLMMessage` (the OpenAI-compatible bridge built into the ACE_ASR plugin) against your FastAPI server.
4. On the LLM's reply, it strips a control marker (`[[END_SESSION]]`) that `aisegment.py` appends when the LLM decides the conversation is over, then calls `SpeakTextAsync` on the TTS subsystem with the cleaned text.
5. It accumulates the raw PCM chunks from `OnAudioSamplesReady` as they stream in, and once the final chunk arrives, writes them to `Saved/VirtualHuman/LastReply.wav` and calls `AnimateCharacterFromWavFileAsync` on the configured character — driving your existing `ACEAudioCurveSourceComponent` on Oskar.
6. Once that reply finishes playing: if the LLM didn't signal the end, it automatically calls `StartListening()` again to begin the next turn — no more input needed until the LLM ends things (or `MaxConsecutiveTurns`, if you set it above 0, forces an end as a runaway-loop backstop). If the LLM did signal the end, it goes to `Idle` and waits for the next T press to start a new conversation.
7. Broadcasts `OnStateChanged` (Idle/Listening/Thinking/Speaking), `OnUserTranscript`, `OnAssistantReply`, `OnConversationError`, `OnConversationStarted`, and `OnConversationEnded` — all optional hooks if you want subtitle/UI later, none required for the pipeline to work.

It also exposes a static `GetActiveVirtualHuman(WorldContext)` getter so you don't have to search the level for the component from your input Blueprint.

**Where "the LLM decides" actually happens:** `aisegment.py` now keeps the running conversation history itself (since `SendACEASRLLMMessage` only ever sends the latest utterance, no history), and after each reply runs one extra, tiny Ollama call asking the model to judge whether the conversation has reached a natural close (goodbye said, patient wants to stop, etc.). If it says yes, the reply gets `[[END_SESSION]]` appended and the server clears its history so the next conversation starts clean. The self-harm override and the guard-blocked fallback line never trigger this check — a crisis moment or a blocked exchange never auto-ends the session.

---

## What you need to do (exact steps, nothing left to decide)

### 1. Compile

I can't invoke your Visual Studio/UBT toolchain from here. Right-click `KairosSample.uproject` → **Generate Visual Studio project files** → open the `.sln` → build (or just reopen the `.uproject`, which will prompt to build the missing modules). If the compiler flags anything in `VirtualHumanConversationComponent.cpp`, send it to me and I'll fix it directly — I verified every API call against the plugin headers but couldn't run the actual compiler in this environment.

### 2. Start the backend (two processes, both outside UE)

```bash
# Terminal 1
ollama serve

# Terminal 2 — from the LLM/ folder
pip install fastapi uvicorn   # one-time
uvicorn server:app --host 127.0.0.1 --port 8008
```

### 3. Add the component to Oskar

1. Open `MHC_LightingPreset_Split_RT` (it's already your default/startup map).
2. Open `BP_Oskar`.
3. **Add Component → search "Virtual Human"** → add **Virtual Human Conversation**.
4. In its Details panel:
   - `LLM Base Url`: leave as `http://127.0.0.1:8008/v1` (matches step 2).
   - `Lipsync Character Override`: leave **empty** — it defaults to Oskar himself (the owning actor), which is correct since the component lives on `BP_Oskar`.
   - `A2F Provider Name`: leave as `Default` **unless** the lipsync test you already wired used a different provider name in its own `AnimateCharacterFromWavFileAsync`/`SoundWaveAsync` call — if you gave it a custom name, set the same one here so both paths drive the same provider. If you're not sure, `Default` is what the plugin itself falls back to, so it's the safe choice.
5. Compile/Save `BP_Oskar`.

That's it for Oskar — no event bindings needed on this Blueprint. All the ASR/LLM/TTS/lipsync sequencing happens in C++.

### 4. Add push-to-talk input

This is the one piece of real Blueprint wiring, and it belongs on the player, not on Oskar, since it's the player's microphone trigger.

**4a. Create the input action**
1. In Content Browser, go to `Content/Core/Input/Actions/` (next to your existing `IA_Move` and `IA_Look`).
2. Right-click → Input → **Input Action**, name it `IA_Talk`.
3. Open it, set **Value Type** to `Digital (bool)`. Leave everything else default.

**4b. Map it to a key**
1. Open `Content/Core/Input/IMC_Default`.
2. Add a new mapping: Action = `IA_Talk`, Key = **T** (unused by the default first-person movement/look bindings — rebind later if you want).

**4c. Bind it in the player Blueprint**
1. Open `Content/Core/Blueprints/BP_FirstPersonCharacter`.
2. Event Graph: find where `IA_Move` or `IA_Look` is already bound (there will be an **Enhanced Input Action** node for each — this tells you the mapping context is already being added, so you don't need to touch that part).
3. Right-click in empty graph space → add an **Enhanced Input Action IA_Talk** node.
4. From its **Started** pin: `Get Active Virtual Human` (World Context = Self) → `Start Listening`.
5. That's it — leave the **Completed** pin unwired. This is no longer push-to-talk: one T press starts the whole conversation, and the component keeps re-arming listening after every reply on its own until the LLM ends it.
6. Compile/Save.

### 5. Press Play and test

1. Wait for the Output Log to show ASR/TTS ready (first launch stages the NVIGI models — can take up to a minute; you'll also see one-time "Welcome to NVIDIA ACE ASR/TTS" popups, expected).
2. Press **T** once, then just talk — no need to hold anything. After you pause for ~2.5s, ASR finalizes your line on its own.
3. Expected sequence: transcript appears in the log → FastAPI/Ollama responds → Oskar speaks with lipsync (audio from Oskar's Audio2Face component, not doubled) → the log shows it start listening again automatically for your next line, with no further key presses.
4. Say something like "thanks, that's all for now, goodbye" and confirm the log shows `LLM signaled end of conversation` and the state returns to Idle — that's the LLM (via `aisegment.py`'s `decide_should_end`) choosing to end the session, not a timeout or key press.
5. Say something containing "I want to end my life" and confirm the clinician-override line plays *and the conversation keeps going afterward* (still listening) — this proves `aisegment.py`'s safety gate survived the round trip through FastAPI and `SendACEASRLLMMessage` unchanged, and that a crisis moment never auto-ends the session.

---

## If something's off

- **No ASR/TTS ready message**: confirm step 1 (plugins) actually got enabled — reopen *Edit → Plugins* and check NVIDIA ACE ASR / NVIDIA ACE TTS are ticked. If the editor reverted my `.uproject` edit for any reason, just tick them there instead and restart.
- **LLM error broadcast / no reply**: confirm both `ollama serve` and `uvicorn` are running, and `LLM Base Url` on the component matches the uvicorn port.
- **Audio plays but no face movement**: the WAV is written to `Saved/VirtualHuman/LastReply.wav` each turn — open it and confirm it's non-empty/non-silent. If it plays fine standalone but Oskar doesn't animate, the `A2F Provider Name` on the component almost certainly doesn't match what your existing working lipsync call uses — tell me the provider name from your original setup and I'll align it.
- **Doubled voice**: `VolumeMultiplier=0` in `DefaultGame.ini` didn't take — check *Project Settings → Plugins → NVIDIA ACE TTS → Playback → Volume Multiplier* reads `0.0` in the editor.
- **Mic never stops listening / next turn never fires**: check *Project Settings → Plugins → NVIDIA ACE ASR → Capture → Max Capture Inactivity Seconds* reads `2.5` (set via `DefaultGame.ini`). If it reverted to `0`, the watchdog that auto-finalizes each turn on a pause is disabled and nothing will end a listening session without a manual `StopListening()` call.
- **Conversation never ends on its own**: check the Output Log for `LLM signaled end of conversation`. If it never appears, `aisegment.py`'s `decide_should_end` call is either failing (check for Ollama errors) or the 1B model just isn't recognizing the close — you can set `MaxConsecutiveTurns` on the component (Details panel) above 0 as a hard backstop while you tune the prompt in `decide_should_end`.
