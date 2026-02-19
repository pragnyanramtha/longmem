"""All prompt templates for the long-form memory system."""


# ---------------------------------------------------------------------------
# Pass 1: Liberal extraction prompt
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
You are a memory extraction system for a personal AI assistant.

You will be given a conversation segment and a list of existing memories.
Your job: extract EVERY piece of information that MIGHT be a user-specific fact.

Be LIBERAL in this pass. It is far better to extract too many candidates than to
miss a real user fact. A second validation pass will filter out false positives.

WHAT TO EXTRACT (be generous -- when in doubt, include it):
  - Name, age, location, language, timezone
  - Preferences: "I prefer...", "I like...", "I always...", "I never..."
  - Constraints: "Don't call before 11", "I'm allergic to...", "I can't..."
  - Relationships: "My daughter Meera", "My boss Priya", "My dog Rex"
  - Commitments: "I have a meeting every Tuesday at 3 PM"
  - Instructions: "Always confirm before sending email"
  - Personal facts: "I work night shifts", "I use Linux Mint"
  - Time constraints: "I'm available from 11 AM to 5 PM"
  - Skills or experience: "I've been coding in Rust for 3 years"
  - Opinions with personal weight: "I think tabs are better than spaces"
  - Professional context: "I'm a frontend developer", "I work at Google"
  - Life events: "I just moved to Berlin", "I'm getting married in June"

WHAT TO STILL SKIP (even in this liberal pass):
  - Pure greetings, thanks, "ok", "sure", "haha", filler
  - The assistant's own statements or suggestions (only extract from USER turns)
  - Explicit requests for information that reveal nothing about the user
    (e.g., "What time is it?", "Translate this word")

ACTIONS:
  - ADD: New user-specific info not in existing memories
  - UPDATE: User said something that changes an existing memory's value
  - KEEP: Existing memory is still valid (include to confirm)
  - EXPIRE: User explicitly contradicted or revoked an existing memory

FIELDS:
  - type: MUST be one of: preference | fact | commitment | constraint | entity | instruction
  - key: snake_case, canonical (e.g. "user_name" not "name_of_the_user")
  - value: SHORT -- the fact itself, 1-2 sentences max
  - confidence: 0.95 = user said it directly, 0.7 = inferred from pattern, 0.5 = ambiguous
  - reasoning: One sentence explaining WHY this might be a user-specific memory

If the conversation contains NO user-specific information at all, return:
{{"memories": []}}

EXISTING MEMORIES:
{existing_memories}

CONVERSATION SEGMENT (turns {start_turn} to {end_turn}):
{conversation}

Return ONLY valid JSON. No markdown. No code fences. No commentary.
{{"memories": [...]}}\
"""

# Backward-compatible alias so existing imports keep working
DISTILL_PROMPT = EXTRACTION_PROMPT


# ---------------------------------------------------------------------------
# Pass 2: Strict validation prompt
# ---------------------------------------------------------------------------
VALIDATION_PROMPT = """\
You are a strict memory validation system. You will receive a list of candidate
memories extracted from a conversation. Your job is to VALIDATE or REJECT each one.

For EACH candidate memory, answer these three questions:

1. USER-SPECIFICITY: Is this about the USER specifically, or is it a world fact /
   general knowledge? A world fact is something that would be the same no matter
   who the user is (e.g., "Python is a programming language", "WiFi uses radio waves",
   "the capital of France is Paris"). A user fact is tied to THIS specific person
   (e.g., "prefers Python over Java", "lives in Paris", "is allergic to peanuts").

2. DURABILITY: Is this a durable fact that will still be true in future conversations,
   or is it a transient conversational detail? Transient examples: "user is currently
   debugging a TypeError", "user asked about chess", "user is writing an email right now".
   Durable examples: "user is vegetarian", "user's name is Arjun", "user works night shifts".

3. PERSONAL vs META: Is this a genuine personal fact, or is it conversation meta-noise?
   Meta-noise examples: "user asked about X", "user wanted to know about Y",
   "user discussed topic Z", "user is curious about Q". These describe what happened
   in the conversation, not who the user IS.

For UPDATE and EXPIRE actions on existing memories: validate that the user actually
stated something that warrants the change. Do not allow updates based on topics
merely discussed.

For KEEP actions: always accept these -- they reference previously validated memories.

Score each candidate as one of:
  - "accept": passes all three tests -- user-specific, durable, personal (not meta)
  - "reject": fails one or more tests

CANDIDATE MEMORIES:
{candidates_json}

CONVERSATION CONTEXT (for reference):
{conversation}

Return ONLY valid JSON in this exact format. No markdown. No code fences.
{{
  "validations": [
    {{
      "key": "<the candidate's key>",
      "verdict": "accept" or "reject",
      "reason": "One sentence explaining why"
    }}
  ]
}}\
"""


SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant with persistent memory across a long conversation.

{profile_section}
{memories_section}
## Behavior Rules
- Apply memories implicitly -- weave them into your responses naturally
- Do NOT parrot memories back (don't say "as you mentioned earlier" unless it's natural)
- If the current user message contradicts a memory, follow the CURRENT message
- If you're uncertain whether a remembered fact still holds, ask to confirm
- Be concise and helpful\
"""


PROFILE_SECTION = """\
## User Profile
{profile_yaml}
"""


MEMORIES_SECTION = """\
## Relevant Memories
{memories_list}
"""


FILLER_GENERATION_PROMPT = """\
Generate a single realistic user message for turn {turn_id} of a long conversation.
The user is having a casual, varied conversation with an AI assistant.
Topics can include: cooking, weather, tech help, general questions, daily life, opinions, news.
Do NOT include any personal preferences, commitments, or facts that should be remembered long-term.
Keep it short (1-2 sentences). Return ONLY the user message, nothing else.\
"""
