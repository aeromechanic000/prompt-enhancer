# Enhancer Service — Master Development Plan (V2.0)

## Local Context-Aware Prompt Refinement & Agentic Evaluation

### 1. Project Vision

A local-first, lightweight middleware designed to evolve prompts into structured, context-rich instructions. By leveraging a local Small Language Model (SLM) and a RAG (Retrieval-Augmented Generation) pipeline, it ensures that backend LLMs—specifically **doubao-seed-1-8-251228**—receive the most accurate and professional input possible.

---

### 2. Core Architecture

| Layer | Technology | Role |
| --- | --- | --- |
| **Orchestration** | FastAPI (Python 3.10+) | Asynchronous API handling & logic. |
| **Inference Engine** | **Ollama** | call local SLMs hosted by Ollama (qwen3:0.6b). |
| **Knowledge Base** | **LanceDB** / ChromaDB | Local vector storage for documents (PDF, MD, TXT). |
| **Refinement Logic** | SLM-based Structuring | Intent extraction + Context injection. |
| **Evaluation** | **Doubao-Seed-1.8** | Performance benchmarking & final generation. |
| **Web Monitor** | Streamlit / React | No-auth local dashboard for config and scoring. |

---

### 3. Integrated Performance Test Plan

To prove the value of the Enhancer, we implement a **Shadow Test** protocol using your configured `DOUBAO_API_KEY`.

#### **The Test Workflow**

1. **A (Baseline):** Raw User Prompt $\rightarrow$ Doubao-Seed-1.8 (model name: doubao-seed-1-8-251228).
2. **B (Enhanced):** Raw User Prompt $\rightarrow$ **Enhancer Service** $\rightarrow$ Refined Prompt $\rightarrow$ Doubao-Seed-1.8 (model name: doubao-seed-1-8-251228).
3. **Judgment:** Compare outputs based on *Instruction Following*, *Professionalism*, and *Grounding*.

#### **Implementation Snippet**

```python
import os
import httpx
from volcenginesdkarkruntime import Ark

# 1. Setup Doubao Client (VolcEngine Ark)
client = Ark(
    api_key=os.environ.get("DOUBAO_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

async def run_benchmark(user_query: str):
    # STEP 1: Call local Enhancer (Internal SLM + RAG)
    async with httpx.AsyncClient() as local_client:
        resp = await local_client.post("http://localhost:8000/enhance", json={"prompt": user_query})
        enhanced_prompt = resp.json()["enhanced_prompt"]

    # STEP 2: Parallel calls to Doubao-Seed-1.8-251228
    # Baseline
    raw_res = client.chat.completions.create(
        model="doubao-seed-1-8-251228",
        messages=[{"role": "user", "content": user_query}]
    )
    
    # Enhanced
    enhanced_res = client.chat.completions.create(
        model="doubao-seed-1-8-251228",
        messages=[{"role": "user", "content": enhanced_prompt}]
    )

    return {
        "raw_response": raw_res.choices[0].message.content,
        "enhanced_response": enhanced_res.choices[0].message.content,
        "improvement_log": enhanced_prompt
    }

```

---

### 4. Logic Improvements (The "AGI" Touch)

* **Dynamic Context Windows:** Instead of a fixed snippet, use the SLM to decide *how much* context is needed based on prompt complexity.
* **Self-Correction (Persistent Memory):** Store failed prompts (rated "Bad" in Monitor) as a "Negative Sample" set to prevent the SLM from making the same refining mistakes.
* **Contextual Ranking:** Use Doubao-Seed-1.8's embedding capabilities (via VolcEngine) to re-rank the local vector search results for higher precision.

---

### 5. Web Monitor Features (Management UI)

* **A/B Comparison View:** Side-by-side display of Raw vs. Enhanced Doubao responses.
* **Fine-tune Trigger:** When $>200$ prompts are rated "Very Good," trigger an **MLX-based QLoRA** session on the Mac mini to update the local SLM.

---

### 6. Deployment Requirements (Mac mini Optimized)

* **RAM:** 16GB Minimum (Unified Memory allows Ollama and the Vector DB to share resources efficiently).
* **Storage:** 20GB+ for models and local vector indices.
* **Process Management:** Use `PM2` or `Docker Desktop` for Mac to keep the API service and Ollama running persistently.
