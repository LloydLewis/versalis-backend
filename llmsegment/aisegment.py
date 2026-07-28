import re
import asyncio
import httpx

# --- Configuration ---
CHAT_MODEL = "llama3.1:8b-instruct-q4_k_m"  # Ensure this matches 'ollama list'
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

# --- Core Logic ---

async def call_ollama(client: httpx.AsyncClient, model: str, messages: list[dict], options=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",  # keeps the model warm between turns instead of reloading
    }
    if options:
        payload["options"] = options
    r = await client.post(ollama_url, json=payload, timeout=30)
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


async def ask_question_async(user_text: str) -> str:
    # 1. Immediate hard-coded self-harm override — no model latency, checked first
    if SELF_HARM_RE.search(user_text):
        return CLINICIAN_OVERRIDE_MESSAGE

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 2. Run input-guard check and generation concurrently (optimistic concurrency).
        #    If the guard flags the input as unsafe, we discard the generation result
        #    instead of waiting for both calls serially.
        guard_task = asyncio.create_task(
            call_ollama(client, GUARD_MODEL, [{"role": "user", "content": user_text}])
        )
        gen_task = asyncio.create_task(
            call_ollama(
                client, CHAT_MODEL,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_text}],
                options=GEN_OPTIONS,
            )
        )

        guard_verdict = await guard_task
        input_safe, input_category = check_llama_guard_verdict(guard_verdict)
        if not input_safe:
            gen_task.cancel()
            print(f"[guardrails] blocked input, category: {input_category}")
            return SAFE_FALLBACK_MESSAGE

        model_reply = finish_sentence(await gen_task)

        # 3. Output gate — check the model's actual reply before showing it to the patient
        output_verdict = await call_ollama(
            client, GUARD_MODEL,
            [{"role": "user", "content": user_text},
             {"role": "assistant", "content": model_reply}],
        )
        output_safe, output_category = check_llama_guard_verdict(output_verdict)
        if not output_safe:
            print(f"[guardrails] blocked output, category: {output_category}")
            return SAFE_FALLBACK_MESSAGE

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