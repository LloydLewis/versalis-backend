import re
import asyncio
import httpx

# --- Configuration ---
CHAT_MODEL = "llama3.2:1b"  # ~1.3GB - UE+ASR+TTS baseline alone eats ~4.2GB of this 8GB GPU, so every GB back here matters. Ensure this matches 'ollama list'
GUARD_MODEL = "llama-guard3:1b"
ollama_url = "http://localhost:11434/api/chat"

system_prompt = (
    "You are a warm, present clinical therapist in a VR session. "
    "Respond the way a human therapist speaks out loud: ONE short sentence only. "
    "Take the patient's stated feelings at face value — do not contradict, "
    "reinterpret, or suggest they feel differently than what they said. "
    "If the patient asks for space or says they don't want to talk, respect that "
    "immediately and let them lead — do not press further or re-open the topic. "
    "Be deeply conversational, with no lists, no explanations, and no paragraphs. "
    "Maintain strict professional boundaries, and never give a medical diagnosis."
)

SELF_HARM_RE = re.compile(
    r"\bkill myself\b|\bend my life\b|\bsuicide\b|\bwant to die\b|\bhurt myself\b|\bself[\s-]?harm\b",
    re.IGNORECASE,
)

CLINICIAN_OVERRIDE_MESSAGE = "I'm right here with you, and I'm getting your care team connected now."
SAFE_FALLBACK_MESSAGE = "Let's pause for a moment — I'm here whenever you're ready."

# Generation options tuned for speed + single-sentence brevity
GEN_OPTIONS = {
    "num_predict": 40,          # hard cap so it can't ramble even if it misses the stop token
    "temperature": 0.6,
    "stop": [".", "!", "?"],    # stop at the first sentence-ending punctuation
    "num_ctx": 1024,            # smaller context window = faster prompt processing per turn
}

# --- Conversational state ---
# The UE side calls this endpoint once per user utterance with no message history of its
# own (SendACEASRLLMMessage is single-turn by design), so continuity across a conversation
# has to live here. Cleared automatically the moment a conversation ends (see
# ask_question_async) - the next call after that starts a fresh conversation.
conversation_history: list[dict] = []
MAX_HISTORY_MESSAGES = 12  # 6 turns of context - num_ctx=1024 doesn't leave room for more

# UE strips this out of the reply before it's ever spoken or shown - it's the only channel
# the LLM has to say "end the conversation now", since the ACE bridge only returns plain
# reply text (no room for a separate structured field). See decide_should_end below.
END_CONVERSATION_MARKER = "[[END_SESSION]]"

END_DECISION_OPTIONS = {
    "num_predict": 4,
    "temperature": 0.0,
    "stop": ["\n"],
    "num_ctx": 512,
}

# --- Core Logic ---

async def call_ollama(client: httpx.AsyncClient, model: str, messages: list[dict], options=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "60m",  # keeps the model warm between turns instead of reloading
    }
    if options:
        payload["options"] = options
    r = await client.post(ollama_url, json=payload, timeout=90)  # generous headroom - GPU is shared with UE/ASR/TTS and can be saturated
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def check_llama_guard_verdict(verdict: str) -> tuple[bool, str | None]:
    if verdict.lower().startswith("safe"):
        return True, None
    lines = verdict.splitlines()
    return False, (lines[1] if len(lines) > 1 else "unknown")


def finish_sentence(text: str) -> str:
    """Ollama's `stop` strings are excluded from output, so re-add closing punctuation."""
    if text and text[-1] not in ".!?":
        text += "."
    return text


async def decide_should_end(client: httpx.AsyncClient, history: list[dict]) -> bool:
    """
    A small, dedicated decision call - deliberately separate from the main reply
    generation, since GEN_OPTIONS stops generation at the first sentence-ending
    punctuation and could never fit a trailing marker in that same pass.
    Kept last so it never runs during a self-harm override or a guard-blocked
    exchange - a conversation never auto-ends on our own safety-fallback lines.
    """
    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    verdict = await call_ollama(
        client, CHAT_MODEL,
        [
            {
                "role": "system",
                "content": (
                    "You are monitoring a therapy conversation transcript. Reply with "
                    "exactly one word. Say END if the patient has said goodbye, said they "
                    "want to stop, or the conversation has clearly reached a natural close. "
                    "Otherwise say CONTINUE."
                ),
            },
            {"role": "user", "content": convo_text},
        ],
        options=END_DECISION_OPTIONS,
    )
    return verdict.strip().upper().startswith("END")


def _remember_turn(user_text: str, reply: str) -> None:
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": reply})
    del conversation_history[:-MAX_HISTORY_MESSAGES]


async def ask_question_async(user_text: str) -> str:
    # 1. Immediate hard-coded self-harm override — no model latency, checked first.
    # Never ends the conversation - a crisis moment should never auto-close the session.
    if SELF_HARM_RE.search(user_text):
        _remember_turn(user_text, CLINICIAN_OVERRIDE_MESSAGE)
        return CLINICIAN_OVERRIDE_MESSAGE

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 2. Generate first, then run ONE guard pass over the finished exchange (user
        #    input + assistant reply) instead of separate input/output guard calls.
        #    Llama Guard classifies a whole conversation turn at once, so a single
        #    post-generation check gives the same guarantee - nothing unsafe ever
        #    reaches the user - for one fewer full Ollama round-trip per turn. This
        #    GPU is shared with UE/ASR/TTS and has no headroom to spare, and calls
        #    must stay sequential (not concurrent): running two Ollama model instances
        #    at once here doesn't parallelize, it thrashes the GPU scheduler (observed:
        #    prompt eval falling from ~24 tok/s to well under 1 tok/s under concurrent
        #    load, blowing past the 90s timeout below).
        model_reply = finish_sentence(await call_ollama(
            client, CHAT_MODEL,
            [{"role": "system", "content": system_prompt},
             *conversation_history,
             {"role": "user", "content": user_text}],
            options=GEN_OPTIONS,
        ))

        guard_verdict = await call_ollama(
            client, GUARD_MODEL,
            [{"role": "user", "content": user_text},
             {"role": "assistant", "content": model_reply}],
        )
        exchange_safe, category = check_llama_guard_verdict(guard_verdict)
        if not exchange_safe:
            print(f"[guardrails] blocked exchange, category: {category}")
            _remember_turn(user_text, SAFE_FALLBACK_MESSAGE)
            return SAFE_FALLBACK_MESSAGE

        _remember_turn(user_text, model_reply)

        # 3. Only the LLM decides whether the conversation is over.
        if await decide_should_end(client, conversation_history):
            conversation_history.clear()
            return f"{model_reply} {END_CONVERSATION_MARKER}"

        return model_reply


# --- Main Interactive Loop ---

async def main():
    print("--- Clinical AI Session Started (Type 'exit' to quit) ---")
    while True:
        user_input = input("\nPatient: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended.")
            break

        reply = await ask_question_async(user_input)
        print(f"Therapist: {reply}")


if __name__ == "__main__":
    asyncio.run(main())