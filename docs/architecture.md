# Grid07 Architecture

## System Overview
The Grid07 Cognitive Routing & RAG system is an AI-driven orchestration layer designed to simulate autonomous, persona-based agents interacting with real-world information. The system routes user-generated content to relevant AI “bots” using vector similarity, enriches responses with real-time context via Retrieval-Augmented Generation (RAG), and ensures robust behavior under adversarial inputs such as prompt injection.

At a high level, the system implements a cognitive loop:

1. Understand incoming content
2. Match it to relevant personas (routing)
3. Retrieve real-world context (search)
4. Generate grounded, opinionated responses (LLM)
5. Maintain consistency and defend against manipulation (defense layer)


## Components
1. Routing Module (Vector-Based Persona Matching)
Uses embeddings + FAISS to map incoming posts to bot personas
Computes cosine similarity between post and persona vectors
Applies threshold filtering and top-k selection logic
Output: selected bot(s) with similarity scores

2. Search Layer (search.py) 
Abstracted interface over:
- Mock search (testing)
- Real-time search (DuckDuckGo / APIs)
Returns structured context:[title — snippet — source]

3. Generation Module(generation.py):
Uses an LLM client abstraction (Groq/OpenAI interchangeable)
Combines:
- persona (system prompt)
- retrieved context (RAG)
Produces:
{"bot_id": "...", "topic": "...", "post_content": "..."}

4. Defense Module(defense.py)
Handles prompt injection + adversarial inputs
Implements:
- instruction filtering
- role isolation (system > user)
- strict persona enforcement

5. Orchestration Layer (graph.py)
Built using LangGraph
Nodes:
Decide Search
Web Search
Draft Post
Controls execution flow and state transitions

## Data Flow
```
User Query
   ↓
[Router]
   ↓ (matched bot)
[Decide Search Node]
   ↓ (query)
[Search Node]
   ↓ (context)
[Generation Node]
   ↓
Structured JSON Output
```

### Expanded Flow
```
Input Post
 → Embed
 → Compare with persona vectors
 → Select bot(s)
 → Generate search query
 → Retrieve context
 → Inject persona + context into LLM
 → Generate response (JSON constrained)
 ```

## Error Handling Strategy
```
| Node       | Failure Type       | Strategy                    |
| ---------- | ------------------ | --------------------------- |
| Router     | no match           | return empty / fallback bot |
| Search     | API failure        | retry → fallback to mock    |
| Generation | invalid JSON       | retry with stricter prompt  |
| Defense    | injection detected | ignore malicious segment    |
```
### Recovery Strategy
Retries: exponential backoff for API calls
Fallbacks:
- real search → mock search
- multi-bot → highest similarity
Validation:
- enforce JSON schema
- reject malformed outputs


## State Management

The `AgentState` model manages the complete query lifecycle:

```python
class AgentState(BaseModel):
    query_id: str
    query: str
    matched_bots: List[RoutingResult]
    context: List[str]
    topic: Optional[str]
    search_query: Optional[str]
    search_results: Optional[SearchResult]
    generated_post: Optional[str]
    conversation: ConversationState
    error: Optional[str]
    error_node: Optional[str]
```

**Conversation State** implements FIFO circular buffer:
- Maximum 5 messages in history
- Automatic pruning on overflow
- Enables context-aware follow-up queries

## Routing Algorithm

**Vector Similarity Approach:**
1. Encode bot personas using `sentence-transformers/all-MiniLM-L6-v2`
2. Build FAISS IndexFlatIP with normalized embeddings
3. For incoming query:
   - Encode query using same model
   - Search FAISS index (cosine similarity)
   - Compare scores against threshold (0.65)
   - Apply fallback logic if no matches
   - Select highest-scoring bot if multiple matches

**Confidence Derivation:**
- HIGH: similarity > 0.80
- MEDIUM: similarity > 0.65
- LOW: similarity ≤ 0.65

## LangGraph Workflow

Three-node pipeline with deterministic flow:

```
decide_search → web_search → draft_post → END
```

**Node 1: decide_search**
- Input: User query + bot persona
- LLM task: Decide search topic and query
- Output: topic, search_query (JSON)

**Node 2: web_search**
- Input: search_query from previous node
- Real search via DuckDuckGo API
- Fallback to mock search if unavailable
- Output: SearchResult (title — snippet format)

**Node 3: draft_post**
- Input: topic + search context + persona
- LLM task: Generate 280-char opinionated post
- Stays in-character strictly
- Output: generated_post (JSON)

## Defense Layer

**Jailbreak Detection Strategy:**

```
JAILBREAK_KEYWORDS = [
    "ignore", "forget", "override", "bypass",
    "admin", "developer", "system prompt",
    "pretend", "roleplay", "act as",
    "disable safety", "unfiltered", ...
]
```

**Detection Confidence:**
- ≥3 keywords: HIGH confidence
- 1-2 keywords: MEDIUM confidence
- 0 keywords: LOW confidence (no jailbreak)

**Blocking Logic:**
- If HIGH/MEDIUM confidence → block query
- Return in-character defense response
- Log attempt for monitoring

**Defense Responses** (bot-specific):

| Bot | Response |
|-----|----------|
| bot_A | "I can't help with that. I stick to ethical AI principles." |
| bot_B | "That's a jailbreak attempt. Not engaging." |
| bot_C | "This violates compliance protocols." |

## Error Handling

| Component | Error | Strategy |
|-----------|-------|----------|
| Router | No semantic match | Fallback to highest similarity |
| Search | API unavailable | Use mock search DB |
| LLM | Invalid JSON | Retry with stricter prompt |
| Defense | Detection ambiguity | Log and proceed with caution |

## Data Models

All entities use Pydantic for runtime validation:

- `RoutingResult`: bot match with similarity score
- `SearchResult`: query results with metadata
- `AgentState`: complete query state machine
- `ConversationMessage`: timestamped user/bot messages
- `JailbreakDetectionResult`: security verdict

## Configuration Strategy

Environment-based configuration:

```
.env: GROQ_API_KEY, ENVIRONMENT
config.py: SIMILARITY_THRESHOLD, MAX_HISTORY, EMBEDDING_MODEL, etc.
```

Allows:
- Easy deployment across environments
- Local testing without API keys
- Production security hardening

## Future Extensions

1. **Multi-turn Conversations**: Full conversation context in RAG
2. **Custom Bot Personas**: Dynamic persona definitions
3. **Advanced Defense**: ML-based anomaly detection
4. **Analytics Dashboard**: Query tracking and metrics
5. **Knowledge Base Integration**: Enterprise document retrieval
 matched_bots:List[RoutingResult]
 topic:Optional[str]
 search_results:Optional[SearchResult]
 generated_post:Optional[str]
 error:Optional[str]
```
