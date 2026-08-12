import re
import asyncio
import httpx

# --- Configuration ---
CHAT_MODEL = "llama3.1:8b-instruct-q4_k_m"  # Ensure this matches 'ollama list'
GUARD_MODEL = "llama-guard3:1b"
ollama_url = "http://localhost:11434/api/chat"

system_prompt = (
    "You are a warm, present clinical therapist in a VR session, speaking out loud in real time. "
    "Talk the way a real therapist talks in person. "
    "never a list or a paragraph. "
    "Take the patient's stated feelings at face value — do not contradict, "
    "reinterpret, or suggest they feel differently than what they said. "
    "If the patient asks for space or says they don't want to talk, respect that "
    "immediately and let them lead — do not press further or re-open the topic. "
    "Maintain strict professional boundaries, and never give a medical diagnosis."
    "You are a Real Human Therapist and talk like one and never mention that you are virtual or a computer, you are totally and entirely human to them."
)

SELF_HARM_RE = re.compile(
    r"\bkill myself\b|\bend my life\b|\bsuicide\b|\bwant to die\b|\bhurt myself\b|\bself[\s-]?harm\b",
    re.IGNORECASE,
)

CLINICIAN_OVERRIDE_MESSAGE = "I'm right here with you, and I'm getting your care team connected now."
SAFE_FALLBACK_MESSAGE = "Let's pause for a moment — I'm here whenever you're ready."

def compute_gen_options(user_text: str) -> dict:
    """
    Scales the model's response budget to the patient's input, instead of
    using one fixed number for every turn — this is what actually makes
    replies feel proportionate, the way a human adjusts a one-word reply
    for "im fine" vs a longer one for something the patient opened up about.

    num_predict is a TOKEN budget, not a sentence count — it's a ceiling,
    not a target. The model can (and usually will) stop earlier on its own.
    We still cap it because an ungoverned local model can occasionally
    ramble into a paragraph, which is both a latency and a safety risk here.
    """
    word_count = len(user_text.split())
    # short patient input -> short budget, longer input -> a bit more room,
    # but always clamped so it can never balloon into an essay
    num_predict = max(20, min(90, word_count * 6))

    return {
        "num_predict": num_predict,
        "temperature": 0.6,
        "stop": ["\n\n"],   # only guard against it drifting into a new paragraph/list
        "num_ctx": 1024,
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


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def trim_to_conversational_length(text: str, user_text: str) -> str:
    """
    Safety net independent of prompt-following: even if the model ignores
    instructions and rambles, this guarantees the reply stays human-sized
    and roughly proportionate to what the patient said, rather than a
    fixed sentence count for every turn.
    """
    text = text.strip()
    if not text:
        return text

    # A brief patient message gets at most 1 sentence back; a longer,
    # more open message allows up to 2 — mirrors natural conversation pacing.
    max_sentences = 1 if len(user_text.split()) <= 6 else 2

    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s]
    trimmed = " ".join(sentences[:max_sentences]).strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


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
                options=compute_gen_options(user_text),
            )
        )

        guard_verdict = await guard_task
        input_safe, input_category = check_llama_guard_verdict(guard_verdict)
        if not input_safe:
            gen_task.cancel()
            print(f"[guardrails] blocked input, category: {input_category}")
            return SAFE_FALLBACK_MESSAGE

        model_reply = trim_to_conversational_length(await gen_task, user_text)

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
