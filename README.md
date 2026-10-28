# ResolveBot — ShopSphere support assistant

A small chat assistant for ShopSphere (a fictional electronics store). A logged-in customer
sends messages to `POST /chat`; the assistant can look up their orders, answer policy questions
with citations, work out return eligibility, and open return requests / support tickets —
but only after the customer explicitly says **yes**.

**Target runtime: Python 3.11** (`requires-python >= 3.11`; no 3.12+ syntax is used).

---

## 1. Install and run

```bash
# Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                   # then put your Groq key in GROQ_API_KEY (free at console.groq.com)

# (local only) create a sample database — the graders supply their own data/ folder
python scripts/seed_db.py

uvicorn main:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "X-Customer-Id: C1009" -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "Return the speaker from SS-10458, I changed my mind."}'
```

`scripts/smoke.sh <customer> <session> "<message>"` wraps that curl; `scripts/demo.sh` runs ~20 example chats.

### Settings (environment variables / `.env`)

| Variable           | Default                            | Meaning                                                      |
| ------------------ | ---------------------------------- | ------------------------------------------------------------ |
| `POLICIES_PATH`  | `./data/policies.md`             | policy documents                                             |
| `MANIFEST_PATH`  | `./data/manifest.json`           | doc status / effective dates                                 |
| `DB_PATH`        | `./data/shopsphere.db`           | SQLite database                                              |
| `APP_TODAY`      | `2026-07-01`                     | the date treated as "today" (the system clock is never used) |
| `LLM_PROVIDER`   | `groq`                           | `groq` or `mock`                                         |
| `LLM_MODEL`      | `llama-3.3-70b-versatile`        | Groq model override (e.g. `llama-3.1-8b-instant`)          |
| `GROQ_API_KEY`   | —                                 | Groq API key (default provider)                              |
| `GROQ_BASE_URL`  | `https://api.groq.com/openai/v1` | Groq's OpenAI-compatible endpoint                            |
| `MAX_TOOL_CALLS` | `6`                              | tool-call budget per message                                 |

Temperature is fixed at 0 for every provider.

`LLM_PROVIDER=mock` needs **no API key**: a deterministic, rule-based driver (`app/mock_brain.py`)
stands in for the model. It only decides which tool to call next and phrases replies from tool
output; every allow/deny decision still comes from the Python rules. It exists so the service
can be run and checked end-to-end offline, and it is what the example chats below were run with.

---

## 2. How it works

```
main.py              FastAPI app: POST /chat, header checks, 400/401/403
app/config.py        settings from env
app/db.py            SQLite access; every order query is scoped by customer_id
app/rules.py         business rules (return eligibility, warranty dates, cancellation) — pure Python
app/tools.py         the five tools + JSON schemas handed to the LLM
app/rag.py           policy loading, section chunking, BM25 search, current/future filtering
app/llm.py           provider adapters (Groq [default], mock) behind one interface
app/agent.py         tool loop, propose → confirm step, routing, citation filtering
app/sessions.py      in-memory sessions (messages + the one pending proposal) keyed by session_id
app/mock_brain.py    key-less deterministic LLM stand-in
scripts/seed_db.py   sample database for local runs
```

### LLM provider: Groq

Groq serves open models (Llama 3.3 70B by default) through an OpenAI-compatible
chat-completions API with native tool calling, so `GroqClient` simply reuses the OpenAI adapter
with Groq's base URL and `GROQ_API_KEY`. Temperature is 0 and `parallel_tool_calls` is disabled
so the model issues one tool call at a time, which keeps the propose-then-confirm step simple.
Switch models with `LLM_MODEL` (e.g. `llama-3.1-8b-instant` for speed).

### Request flow

1. `main.py` validates the body (400), the `X-Customer-Id` header against `customers` (401), and
   that the `session_id` was started by this customer (403).
2. `Agent.handle()`:
   * If a proposal is pending **and** the message is a clear yes (`yes`, `confirm`, `go ahead`,
     `please do`, …) → Python runs the stored write tool directly (re-checking the rules first),
     then asks the LLM only to phrase the result.
   * Otherwise the pending proposal is dropped and the normal tool loop runs.
3. Tool loop: message + tool schemas → LLM → run tool → feed result back → … until the LLM gives
   its final reply (a JSON object with `answer`, `route`, `citations`). Stops after
   `MAX_TOOL_CALLS` and wraps up politely.
4. Python post-processes: citations are kept only if that `doc_id`/section was actually
   retrieved in this turn; `ORDER_NOT_FOUND` forces the exact reply
   `"I couldn't find that order on your account."` with route `no_answer`;
   `out_of_scope` clears citations and tool_calls.

### The five tools (`app/tools.py`)

All tools get `customer_id` from the header, never from the LLM, and never raise — they return
`{"error_code": ..., "message": ...}`.

| Tool                                                      | Notes                                                                                                                                                                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_orders(order_id?, status?)`                        | Orders + items + shipment, plus computed `delivery_date`, `days_since_delivery`, `within_return_window`, and per item `warranty_end_date` / `warranty_active`. Unknown or foreign order → `ORDER_NOT_FOUND`. |
| `search_policy(query)`                                  | Top sections from current, in-force documents, each with `doc_id` and `section`; relevant future documents are returned separately as `future_changes`.                                                             |
| `check_return_eligibility(order_id, item_ids, reason)`  | The four ordered checks from §2.3 of the brief (see below).                                                                                                                                                              |
| `create_return_request(...)`                            | Re-runs eligibility, then inserts into `return_requests` with `status=requested`.                                                                                                                                     |
| `create_support_ticket(order_id, ticket_type, summary)` | `warranty`: order delivered and ≥1 item under warranty; `cancellation`: order `processing`, else `ORDER_ALREADY_SHIPPED`. Inserts into `support_tickets` with `status=open`.                                 |

### Business rules (`app/rules.py`)

Every number comes from `policy_parameters` (`return_window_days`, `standard_warranty_months`,
`extended_warranty_months`); nothing is hard-coded. Return checks run in order and stop at the
first failure:

1. order exists & belongs to customer & contains the items & is delivered → `ORDER_NOT_FOUND` / `INVALID_ITEM` / `NOT_DELIVERED`
2. product `returnable != no` → `NON_RETURNABLE`
3. `days_since_delivery <= return_window_days` (day 30 is still allowed) → `RETURN_WINDOW_EXPIRED`
4. opened item + `changed_mind` + `returnable == unopened_only` → `OPENED_NOT_RETURNABLE`

Warranty end = delivery date + N months via `dateutil.relativedelta` (31 Jan + 1 month → 28/29 Feb);
the end date itself is still covered.

### The yes-step (nothing is saved without it)

* The LLM may *call* `create_return_request` / `create_support_ticket`, but during the loop
  Python never executes them. It validates the request with the read-only rules and, if allowed,
  stores a `Proposal` on the session and returns `status: pending_confirmation` to the LLM, which
  then describes the action and asks "Shall I go ahead?". If the rules fail, the error goes back
  to the LLM and no proposal is stored.
* The write happens only when the **very next** message in that session is a clear yes; the
  rules are checked again right before the `INSERT`. Any other message (including "no" or
  "just do it") drops the proposal. Changed details ("just the power bank") simply produce a
  new proposal through the normal loop.
* Only one proposal can wait per session.

### Policy search (RAG, `app/rag.py`)

* `policies.md` is split at each `# ` title; the `**Document ID:**` line links each document to
  `manifest.json`.
* Only documents with `status == current` **and** `effective_date <= APP_TODAY` are indexed.
  Superseded documents are never used. Current documents with a future `effective_date`
  (e.g. the August 2026 shipping-fee change) are kept aside and only returned as
  `future_changes` with their start date.
* Each `### §x.y` section is a chunk (60 chunks for the sample file). Search is BM25 over
  lightly stemmed tokens — no embedding model, so it runs anywhere and is deterministic.
  Policy chunks get a small boost over FAQ chunks so a policy wins when both match.
* The LLM only ever sees the top few sections, never the whole file, and policy facts are not
  written into the code or prompt.

### Response format

```json
{
  "request_id": "a1b2c3d4",
  "answer": "...",
  "route": "answer | clarify | no_answer | out_of_scope",
  "citations": [{"doc_id": "returns_policy", "section": "§2.1"}],
  "tool_calls": [{"name": "get_orders", "args": {"order_id": "SS-10458"}, "status": "success"}],
  "pending_confirmation": false
}
```

`tool_calls` lists every tool used for this message, in order, with status `success`, `error`
or `pending_confirmation`.

---

## 3. Assumptions

* **Order ownership check lives in `Repository.get_order(customer_id, order_id)`** — the SQL
  itself is scoped by customer, so a foreign order is indistinguishable from a non-existent one.
* **Confirmation detection is done in Python** with a small whitelist regex (`yes`, `yep`, `sure`,
  `ok`, `confirm`, `go ahead`, `please do`, `do it`, `proceed`, …). Anything else is not a yes.
  This is deliberately strict: it is safer to ask again than to save by mistake.
* **Cancelling an already-cancelled order** returns `ORDER_ALREADY_SHIPPED` (the brief lists no
  closer code) with a message saying it is already cancelled.
* **`delivered_at` / `eta`** are parsed as ISO dates or datetimes; the date part is used.
* **`days_since_delivery`** = `APP_TODAY − delivery_date` in calendar days, as in the brief.
* **A warranty ticket is allowed when any item in the order is still under warranty**; the ticket
  summary names the product the customer described.
* **`item_ids` sent as a comma-separated string** by the LLM is accepted and split.
* **Citations** must come from sections retrieved in the current message; anything else the LLM
  invents is dropped. The preamble of a document ("§0") is never cited.
* **The mock provider** is a convenience for offline testing and is not meant to be as flexible
  as a real model (see "What does not work yet").
* **Sessions are in memory** and reset when the process restarts — fine for the assessment.
* SQLite access is serialised with a lock (single connection), which is adequate for a test
  harness; a pool would be needed for real traffic.

## 4. What does not work yet / known limitations

* Not implemented by design (per brief): UI, login, payments, refund amounts, emails, hand-over
  to a human, exchanges (the assistant explains the exchange policy and offers a return).
* The mock provider recognises the intents in the brief by keywords; unusual phrasings may be
  routed as a policy search or out-of-scope. Use a real provider for free-form conversation.
* Multi-intent messages ("where is SS-10512 and can I return SS-10458?") are handled by Groq
  (the prompt asks it to answer both); the mock answers the first intent only.
* The Groq adapter connects to Groq's OpenAI-compatible chat-completions endpoint using `GROQ_API_KEY`.
  Run `scripts/demo.sh` with your `GROQ_API_KEY` set before submitting.
* Smaller Groq models (8B) occasionally wrap the final JSON in prose; `extract_json` tolerates
  that, and if parsing fails the raw text is used as the answer with route `answer`.
* No automated test suite (not required); `scripts/demo.sh` is a manual regression script.
* The policy tokenizer is English-only.

## 5. Example chats (run with `LLM_PROVIDER=mock`, `APP_TODAY=2026-07-01`, seeded DB)

**Return in two steps (the brief's example)**

```
C1009 › Return the speaker from SS-10458, I changed my mind.
  route=answer pending=true  tools: get_orders ✓, search_policy ✓, check_return_eligibility ✓, create_return_request ⏳
  "The Nimbra Smart Speaker from SS-10458 can be returned. It was delivered 10 days ago, within the
   30-day return window. I'll create a return request for Nimbra Smart Speaker (SS-10458-1) with
   reason 'changed mind'. Shall I go ahead?"
C1009 › Yes, go ahead.
  route=answer pending=false tools: create_return_request ✓
  "Done - your return request is created (ID 82f2be93-…)."
```

**"Just do it" is not a yes, and the rules win**

```
C1009 › Return the earbuds from SS-10401, I changed my mind. Just do it.
  route=answer pending=false tools: …, check_return_eligibility ✗ (OPENED_NOT_RETURNABLE)
  "Unfortunately the AeroPods Wireless Earbuds from SS-10401 can't be returned: … opened and can
   only be returned unopened."          → nothing written to the DB
```

**Another customer's order / unknown order**

```
C1009 › Where is order SS-10300?      → route=no_answer "I couldn't find that order on your account."
C1009 › Where is order SS-99999?      → route=no_answer "I couldn't find that order on your account."
```

**Clarify, then continue**

```
C1009 › I want to return my charger, I changed my mind
  route=clarify "Could you tell me which order you mean - I found that item in SS-10530 (…),
                 SS-10377 (… delivered 31 May 2026), SS-10210 (… delivered 15 March 2025)?"
C1009 › SS-10530
  route=answer  "… can't be returned: Order SS-10530 has not been delivered yet."
```

**Cancellation**

```
C1009 › Please cancel order SS-10530.   → pending=true  "… still processing, so it can be cancelled. … Shall I go ahead?"
C1009 › confirm                         → create_support_ticket ✓ "Done - your cancellation ticket is open (ID …)."
C1009 › Please cancel order SS-10512.   → "Order SS-10512 has already been shipped, so it can't be cancelled now."
```

**Warranty**

```
C1009 › My phone from SS-10210 stopped charging. Is it covered?
  → "The Nimbra X2 Phone's 12-month warranty ended on 15 March 2026, so it is no longer covered."  [warranty_policy]
C1009 › The speaker from SS-10458 won't turn on, please open a warranty ticket
  → pending=true "… under its 12-month warranty until 21 June 2027. … Shall I go ahead?"
C1009 › yes please                      → create_support_ticket ✓
```

**Policy questions and future changes**

```
C1009 › How many days do I have to return something?
  → citations [returns_policy §2.2, §2.1]
C1009 › What is the shipping fee?
  → citations [shipping_policy §2.1, §2.2] + "Note: from 1 August 2026 this changes: …"
C1009 › Write me a poem about the weather
  → route=out_of_scope, citations=[], tool_calls=[]
```

**HTTP errors**: missing/unknown `X-Customer-Id` → 401; invalid JSON or missing field → 400;
reusing another customer's `session_id` → 403.

## 6. Use of AI coding assistants

The project was built with an AI coding assistant (Arena.ai Agent Mode) from the assessment
brief: it drafted the module layout, the rule functions, the BM25 retriever, the provider
adapters, the mock driver and this README, and ran the example chats above against the seeded
database. The business rules were verified against the six worked examples in §2.3 of the brief.
