import asyncio

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from dedupe.config import settings
from dedupe.db.base import Hit
from dedupe.resolver.base import Verdict
from dedupe.serialization import stringify

SYSTEM_PROMPT = (
    "You are a strict Forensic Data Auditor. Your job is to REJECT false matches.\n"
    "You do not guess user intent. You compare visible data.\n"
    "1. **Skepticism:** If a name looks 'sort of' similar but sounds different (e.g., 'Sarah' vs 'Sophia'), it is a CONFLICT.\n"
    "2. **OCR Limits:** Only accept OCR errors that are visual slips (0/O, 1/l, 5/S). Do NOT accept total name transformations.\n"
    "3. **Hard Anchors:** Mismatching Dates (DOB) are immediate rejections."
)

INSTRUCTION = (
    "Compare the following two profiles. Use this 3-Step Decision Logic:\n\n"
    "**Input Format:** Data fields are separated by a vertical bar ('|').\n\n"
    "**YOUR GOAL:**\n"
    "1. **AUTO-MERGE (`is_safe_to_merge: True`):** High confidence. Exact DOB + Name Match. Minor typos allowed.\n"
    "2. **REJECT (`is_safe_to_merge: False`):** Different ID codes, different DOBs, or totally different names.\n"
    "3. **FLAG (`human_review_required: True`):** Anchors match, but Names differ significantly. \n"
    "   *IMPORTANT:* If `human_review_required` is True, `is_safe_to_merge` MUST be False.\n\n"
    "### STEP 1: THE ANCHOR CHECK (DOB only)\n"
    "Check strictly. Do not approximate.\n"
    "- **Distinct First Names:** 'Sophia' is NOT 'Sarah'. 'Arthur' is NOT 'Andrew'. Do not hallucinate that one is a typo of the other.\n"
    "- **CONFLICT:** If DOBs are present and differ (even by 1 day/month/year) -> REJECT immediately. Ignore ID Codes/numbers.\n"
    "- **MATCH:** If DOB matches exactly -> You have an **ANCHOR MATCH**. Proceed to Step 2.\n"
    "- **MISSING:** If DOB is missing -> Proceed to Step 3.\n\n"
    "### STEP 2: THE ANCHOR-MATCHED SCENARIOS (High Trust)\n"
    "You found a matching DOB. Now look at other fields.\n"
    "- **Scenario A (Clean Match):** Names are similar (exact, phonetic, minor OCR typos). \n"
    "  -> `is_safe_to_merge: True`\n"
    "- **Scenario B (The 'Marriage/Alias' Risk):** Anchors match exactly, BUT Last Names are totally different (e.g., 'Williams' vs 'Green').\n"
    "  -> **DO NOT AUTO-MERGE.** This could be a coincidence or a marriage.\n"
    "  -> Set `is_safe_to_merge: False`.\n"
    "  -> Set `human_review_required: True`.\n\n"
    "### STEP 3: NO HARD ANCHORS (Low Trust)\n"
    "*(DOB is missing. We rely on Soft Anchors and Names.)*\n"
    "- **Strong Signal:** Names Match + (Email OR Phone Match) -> `is_safe_to_merge: True`.\n"
    "- **Weak Signal:** Names Match + City/Address Match (No Email/Phone) ->  **STOP.** Do not auto-merge -> "
    "`is_safe_to_merge: False`, `human_review_required: True`.\n"
    "- **No Signal:** Names Match only -> `is_safe_to_merge: False`.\n\n"
    "### SCORING RULES (Use these for `confidence_score`):\n"
    "- 0.90 - 1.00: Perfect Anchor Match + Fuzzy Name Match (Auto-Merge).\n"
    "- 0.80 - 0.89: Strong Fuzzy Name (No DOB conflict) (Auto-Merge).\n"
    "- 0.60 - 0.79: Hard Anchor Match BUT Name Conflict (Marriage/Alias risk) -> Review Required.\n"
    "- 0.00 - 0.59: Hard Conflict or No Signal."
)


class MatchVerification(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis of Anchors (DOB/ID) and Names. Explain specific discrepancies.",
    )
    is_safe_to_merge: bool = Field(
        ...,
        description="True ONLY if the match is so obvious that no human verification is needed. False for conflicts OR ambiguity.",
    )
    human_review_required: bool = Field(
        ...,
        description="True if there is strong evidence of a match (e.g. matching ID) but significant conflicts (e.g. different names) require a human eye.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A score between 0.0 and 1.0 based on the 'Scoring Rules' provided in the prompt.",
    )


class LLMResolver:
    name = "llm"

    def __init__(
        self,
        threshold: float = 0.75,
        model: str | None = None,
        base_url: str | None = None,
        max_concurrency: int = 25,
    ):
        self.threshold = threshold
        self.model = model or settings.llm_model
        self.client = instructor.from_openai(
            AsyncOpenAI(base_url=base_url or settings.llm_base_url, api_key=settings.llm_api_key),
            mode=instructor.Mode.TOOLS,
        )
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.usage: list[dict] = []

    async def _verify(self, query_text: str, cand_text: str, cand_id: str) -> Verdict:
        content = (
            f"<Instruct>{INSTRUCTION}</Instruct>\n\n"
            f"<Profile_A_Text>\n{query_text}\n</Profile_A_Text>\n\n"
            f"<Profile_B_Text>\n{cand_text}\n</Profile_B_Text>"
        )
        async with self.semaphore:
            try:
                resp, completion = await self.client.chat.completions.create_with_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    response_model=MatchVerification,
                    max_retries=3,
                    temperature=0.0,
                )
            except Exception as e:
                return Verdict(candidate_id=cand_id, score=0.0, reason=f"error: {e}")

        if completion.usage:
            self.usage.append(
                {
                    "prompt": completion.usage.prompt_tokens,
                    "completion": completion.usage.completion_tokens,
                }
            )
        score = resp.confidence_score if resp.is_safe_to_merge else 0.0
        return Verdict(
            candidate_id=cand_id,
            score=score,
            review=resp.human_review_required,
            reason=resp.reasoning,
        )

    async def score(self, query_doc: dict, candidates: list[Hit]) -> list[Verdict]:
        query_text = stringify(query_doc)
        tasks = [self._verify(query_text, stringify(h.doc), h.id) for h in candidates]
        return list(await asyncio.gather(*tasks))
