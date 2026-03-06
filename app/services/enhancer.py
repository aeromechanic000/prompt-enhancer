"""Core prompt enhancement service."""

from typing import Optional

from app.models.schemas import EnhancedPromptResult
from app.services.ollama_client import get_ollama_client
from app.services.rag_service import get_rag_service


# Prompt templates
INTENT_EXTRACTION_PROMPT = """Analyze the following user prompt and extract the core intent.

User Prompt: {prompt}

Identify:
1. The main task or question (what the user wants)
2. The domain or context (programming, writing, analysis, etc.)
3. Any specific requirements or constraints mentioned

Respond in a concise format:
Task: [main task]
Domain: [domain]
Requirements: [specific requirements or "none specified"]
Context Needed: [what additional context might help]
"""

ENHANCEMENT_PROMPT = """You are a prompt enhancement assistant. Your task is to improve the given prompt by making it clearer, more specific, and better structured.

Original Prompt: {prompt}

Retrieved Context:
{contexts}

Enhancement Guidelines:
1. Make the intent crystal clear
2. Add relevant context from the retrieved information
3. Structure the prompt with clear sections if complex
4. Specify the expected output format
5. Add relevant constraints or requirements
6. Keep the enhanced prompt concise but comprehensive

Generate an enhanced version of the original prompt. Output ONLY the enhanced prompt, nothing else.

Enhanced Prompt:"""

ENHANCEMENT_PROMPT_NO_RAG = """You are a prompt enhancement assistant. Your task is to improve the given prompt by making it clearer, more specific, and better structured.

Original Prompt: {prompt}

Enhancement Guidelines:
1. Make the intent crystal clear
2. Add structure with clear sections if complex
3. Specify the expected output format
4. Add relevant constraints or requirements
5. Keep the enhanced prompt concise but comprehensive

Generate an enhanced version of the original prompt. Output ONLY the enhanced prompt, nothing else.

Enhanced Prompt:"""


class EnhancerService:
    """Service for enhancing prompts using SLM and RAG."""

    def __init__(self):
        self.ollama = get_ollama_client()
        self.rag = get_rag_service()

    async def extract_intent(self, prompt: str) -> str:
        """Extract intent from a prompt using the local SLM.

        Args:
            prompt: Raw user prompt.

        Returns:
            Extracted intent description.
        """
        extraction_prompt = INTENT_EXTRACTION_PROMPT.format(prompt=prompt)
        intent = await self.ollama.generate(
            extraction_prompt,
            temperature=0.3,  # Lower temperature for more consistent extraction
            max_tokens=512,
        )
        return intent.strip()

    async def enhance(
        self,
        prompt: str,
        use_rag: bool = True,
        context_k: int = 3,
    ) -> EnhancedPromptResult:
        """Enhance a prompt using SLM and optional RAG context.

        Args:
            prompt: Raw user prompt to enhance.
            use_rag: Whether to use RAG for context retrieval.
            context_k: Number of context documents to retrieve.

        Returns:
            EnhancedPromptResult with original, enhanced prompt, and metadata.
        """
        # Step 1: Extract intent
        intent = await self.extract_intent(prompt)

        # Step 2: Retrieve context if RAG is enabled
        contexts_used = []
        if use_rag:
            try:
                # Use intent for more targeted search
                search_results = await self.rag.search(intent, k=context_k)
                contexts_used = [r["content"] for r in search_results]
            except Exception:
                # If RAG fails, continue without context
                contexts_used = []

        # Step 3: Generate enhanced prompt
        if contexts_used:
            context_text = "\n\n---\n\n".join(
                f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts_used)
            )
            enhancement_prompt = ENHANCEMENT_PROMPT.format(
                prompt=prompt,
                contexts=context_text,
            )
        else:
            enhancement_prompt = ENHANCEMENT_PROMPT_NO_RAG.format(prompt=prompt)

        enhanced = await self.ollama.generate(
            enhancement_prompt,
            temperature=0.5,
            max_tokens=1024,
        )

        return EnhancedPromptResult(
            original=prompt,
            enhanced=enhanced.strip(),
            intent=intent,
            contexts_used=contexts_used,
        )


# Singleton instance
_enhancer_service: Optional[EnhancerService] = None


def get_enhancer_service() -> EnhancerService:
    """Get singleton enhancer service instance."""
    global _enhancer_service
    if _enhancer_service is None:
        _enhancer_service = EnhancerService()
    return _enhancer_service
