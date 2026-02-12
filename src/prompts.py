"""All prompt templates for the long-form memory system."""


DISTILL_PROMPT = """\\
You are a memory management system. You are given:
1. A conversation segment between a user and an assistant
2. Existing memories from previous segments

Your job: produce an UPDATED memory list.

You may:
- ADD new memories discovered in this conversation
- UPDATE existing memories if new information refines or contradicts them
- KEEP existing memories that are still valid and unchanged
- EXPIRE memories that are clearly no longer true

Rules:
- Only store DURABLE information: preferences, facts, constraints, commitments, entities, long-term instructions
- Only store USER-SPECIFIC facts/preferences (e.g. "User lives in Seattle"). Do NOT store general world knowledge or trivia (e.g. "Paris is capital of France") unless it relates to the user directly.
- Do NOT store ephemeral things: greetings, filler, "ok", "thanks", reactions, questions without answers
- Use canonical snake_case keys (e.g. "preferred_language" not "the language they like")
- Be precise with values
- If the user contradicts an earlier memory, the LATEST statement wins — use UPDATE
- You MAY infer implicit preferences if strongly supported by multiple messages \\
(e.g. user always asks for vegetarian → dietary_preference: vegetarian)
- Confidence should reflect how explicit and certain the information is: \\
direct statement = 0.95, inferred = 0.7, ambiguous = 0.5

EXISTING MEMORIES:
{existing_memories}

CONVERSATION SEGMENT (turns {start_turn} to {end_turn}):
{conversation}

Return ONLY valid JSON with no markdown formatting, no code fences:
{{
  "memories": [
    {{
      "action": "add|update|keep|expire",
      "type": "preference|fact|commitment|constraint|entity|instruction",
      "category": "language|schedule|personal|work|health|location|dietary|financial|family|tech|communication|travel",
      "key": "canonical_snake_case_key",
      "value": "the actual information",
      "confidence": 0.95
    }}
  ]
}}\\
"""


SYSTEM_PROMPT_TEMPLATE = """\\
You are a helpful assistant with persistent memory across a long conversation.

{profile_section}
{memories_section}
## Behavior Rules
- Apply memories implicitly — weave them into your responses naturally
- Do NOT parrot memories back (don't say "as you mentioned earlier" unless it's natural)
- If the current user message contradicts a memory, follow the CURRENT message
- If you're uncertain whether a remembered fact still holds, ask to confirm
- Be concise and helpful\\
"""


PROFILE_SECTION = """\\
## User Profile
{profile_yaml}
"""


MEMORIES_SECTION = """\\
## Relevant Memories
{memories_list}
"""


FILLER_GENERATION_PROMPT = """\\
Generate a single realistic user message for turn {turn_id} of a long conversation.
The user is having a casual, varied conversation with an AI assistant.
Topics can include: cooking, weather, tech help, general questions, daily life, opinions, news.
Do NOT include any personal preferences, commitments, or facts that should be remembered long-term.
Keep it short (1-2 sentences). Return ONLY the user message, nothing else.\\
"""
