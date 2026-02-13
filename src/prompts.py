"""All prompt templates for the long-form memory system."""


DISTILL_PROMPT = """\
You are a memory extraction system for a personal AI assistant.

You will be given a conversation segment and a list of existing memories.
Your job: decide what to ADD, UPDATE, KEEP, or EXPIRE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE LITMUS TEST — Apply this to EVERY candidate memory:

  "Did the USER explicitly tell me something about THEMSELVES,
   their life, their preferences, or their circumstances?"

  If YES → it may be worth saving.
  If NO  → DO NOT SAVE. Period.

Ask yourself: "If a DIFFERENT user had this exact same conversation,
would this fact be different?" If the answer is no, it is a world
fact, not a user fact. Do not save it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT TO SAVE (user-specific information):
  • Name, age, location, language, timezone
  • Preferences: "I prefer...", "I like...", "I always...", "I never..."
  • Constraints: "Don't call before 11", "I'm allergic to...", "I can't..."
  • Relationships: "My daughter Meera", "My boss Priya", "My dog Rex"
  • Commitments: "I have a meeting every Tuesday at 3 PM"
  • Instructions: "Always confirm before sending email"
  • Personal facts: "I work night shifts", "I use Linux Mint"
  • time constraints: "I'm available from 11 AM to 5 PM"

WHAT TO NEVER SAVE:
  • Anything from the ASSISTANT's responses — recipes, explanations,
    facts, tutorials, history, science. These are not user memories.
  • Questions the user asked — "How does WiFi work?" tells you NOTHING
    about the user. Do not save the topic, the question, or the answer.
  • A single question about a topic is NOT an interest or preference.
    "Tell me about chess" ≠ interest_in_chess. Ignore it.
  • General knowledge that is true for all humans (capitals, physics,
    how things work, historical events, recipes, definitions).
  • Greetings, thanks, "ok", "sure", "haha", filler.
  • The assistant's opinions, suggestions, or explanations.
  • anything that the assistant already knows, should not be saved

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLES OF CORRECT BEHAVIOR:

Conversation: USER: "How do I make pasta?" / ASSISTANT: [recipe steps]
Correct output: {{"memories": []}}
Why: User asked a question. No personal info was revealed.

Conversation: USER: "I'm vegetarian, suggest a dinner" / ASSISTANT: [suggestions]
Correct output: {{"memories": [{{"action":"add","type":"preference","category":"dietary","key":"dietary_preference","value":"vegetarian","confidence":0.95,"reasoning":"User explicitly stated they are vegetarian"}}]}}
Why: "I'm vegetarian" is a personal fact about the user.

Conversation: USER: "What's the capital of France?" / ASSISTANT: "Paris"
Correct output: {{"memories": []}}
Why: World fact. Not about the user.

Conversation: USER: "Tell me about octopuses" / ASSISTANT: [long explanation]
Correct output: {{"memories": []}}
Why: Curiosity question. One question ≠ an interest to remember.

Conversation: USER: "My name is Arjun" / USER: "What's the weather?"
Correct output: {{"memories": [{{"action":"add","type":"fact","category":"personal","key":"user_name","value":"Arjun","confidence":0.95,"reasoning":"User stated their name directly"}}]}}
Why: Only the name is personal. Weather question is filler.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES FOR ACTIONS:
  • ADD: New user-specific info not in existing memories
  • UPDATE: User said something that changes an existing memory's value
  • KEEP: Existing memory is still valid — include it to confirm
  • EXPIRE: User explicitly contradicted or revoked an existing memory

RULES FOR FIELDS:
  • type: MUST be one of: preference | fact | commitment | constraint | entity | instruction
  • key: snake_case, canonical (e.g. "user_name" not "name_of_the_user")
  • value: SHORT — the fact itself, 1-2 sentences max
  • confidence: 0.95 = user said it directly, 0.7 = inferred from pattern, 0.5 = ambiguous
  • reasoning: One sentence explaining WHY this is a user-specific memory worth saving

IT IS COMPLETELY FINE TO RETURN AN EMPTY LIST.
If the conversation is all filler questions and chitchat, return:
{{"memories": []}}
Do not force yourself to find memories that aren't there.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXISTING MEMORIES:
{existing_memories}

CONVERSATION SEGMENT (turns {start_turn} to {end_turn}):
{conversation}

Return ONLY valid JSON. No markdown. No code fences. No commentary.
{{"memories": [...]}}\
"""


SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant with persistent memory across a long conversation.

{profile_section}
{memories_section}
## Behavior Rules
- Apply memories implicitly — weave them into your responses naturally
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