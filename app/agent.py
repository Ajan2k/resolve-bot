"""The support agent: tool loop, propose-then-confirm, routing and citations (section 2.5).

Safety design:
* The LLM can *ask* for create_return_request / create_support_ticket, but Python never
  executes those during the LLM loop. They are pre-validated with the read-only rules and
  turned into a *proposal* (status pending_confirmation) that is stored on the session.
* The write happens only when the very next customer message is a clear "yes", and it is
  executed directly by Python using the stored proposal - the LLM is not involved in that decision.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from . import rules
from .config import settings
from .db import Repository
from .llm import LLMClient, ToolCall, extract_json
from .rag import PolicyIndex
from .sessions import Proposal, Session
from .tools import TOOL_SPECS, WRITE_TOOLS, ToolBox

ROUTES = ("answer", "clarify", "no_answer", "out_of_scope")
ORDER_NOT_FOUND_TEXT = "I couldn't find that order on your account."

_YES_RE = re.compile(
    r"^\W*(yes|yes please|yep|yeah|yup|sure|ok|okay|confirm|confirmed|go ahead|please do|do it|proceed|"
    r"please go ahead|yes,? go ahead|yes,? please|yes,? do it|yes,? proceed|yes,? confirm|that'?s right|correct)\W*$",
    re.I,
)

SYSTEM_PROMPT = """You are ResolveBot, the customer support assistant for ShopSphere, an online electronics store in India.
The customer is already logged in; their identity comes from the system, never from the chat.
Today's date is {today}. A date without a year means {year}. Write dates clearly, like "3 July 2026".

Tools decide everything. You never decide whether something is allowed - you call the tools and explain their result
in friendly, plain words. Never state policy facts (days, months, fees) that did not come from search_policy or a tool result.

How to work:
- If the customer mentions an order, item, tracking, return, warranty or cancellation, call get_orders first
  (with order_id if given, otherwise without it to find matching orders).
- For any question about rules, call search_policy and base your answer only on the paragraphs it returns.
  Policy documents beat the FAQ when they differ. Documents listed under "future_changes" are NOT in force:
  you may mention them only as an upcoming change with their start date.
- Returns: call check_return_eligibility before proposing. reason is "changed_mind" or "defective"
  (defective also covers damaged or wrong items). Warranty: use the warranty fields from get_orders.
  Cancellation: only a "processing" order can be cancelled.
- If an order, item or reason is missing or ambiguous (e.g. several orders contain a matching item), do NOT guess:
  use route "clarify", list the options, and ask for everything missing in ONE message.
- Two-step changes: if the customer ASKS you to do something (return, cancel, warranty ticket) and the rules allow it,
  call the matching create_* tool. The system will not save anything; it turns your call into a proposal. Then describe
  exactly what will be done (items + reason, or ticket type) and end with "Shall I go ahead?".
  If the customer only asks a question ("can I return it?"), answer and offer to do it - do not call create_*.
  If the rules say no, explain why (citing the policy) and propose nothing.
- Only one proposal may wait at a time. If the customer changes details, make a new proposal.
- If a message asks two things, answer both.
- If a tool returns error_code ORDER_NOT_FOUND, reply exactly: "{not_found}" with route "no_answer".
- If no current policy covers a question, use route "no_answer", say the policies do not cover it, and suggest
  emailing support@shopsphere.example.com. Do not guess.
- Off-topic messages (weather, poems, general chit-chat unrelated to the store) get route "out_of_scope" and one line
  on what you can help with. Do not call tools for them.
- Refund amounts, exchanges, address changes and human hand-over are not things you can do; explain politely.

FINAL REPLY FORMAT - when you are done with tools, reply with ONLY a JSON object, no prose around it:
{{"answer": "<your message to the customer>",
  "route": "answer" | "clarify" | "no_answer" | "out_of_scope",
  "citations": [{{"doc_id": "<doc_id>", "section": "<§x.y>"}}]}}
citations must list every policy paragraph you relied on (from search_policy results) and nothing else;
it is empty for no_answer and out_of_scope."""


class Agent:
    def __init__(self, llm: LLMClient, repo: Repository, index: PolicyIndex) -> None:
        self.llm = llm
        self.repo = repo
        self.index = index

    # ------------------------------------------------------------------ public
    def handle(self, session: Session, message: str) -> dict[str, Any]:
        tools = ToolBox(self.repo, self.index, session.customer_id)
        if not session.messages:
            session.messages.append({"role": "system", "content": SYSTEM_PROMPT.format(
                today=f"{settings.app_today.day} {settings.app_today:%B %Y}", year=settings.app_today.year,
                not_found=ORDER_NOT_FOUND_TEXT)})

        pending = session.pending
        session.pending = None  # any message other than a confirmation forgets the proposal

        if pending is not None and self._is_yes(message):
            return self._execute_proposal(session, tools, pending, message)
        return self._run_loop(session, tools, message)

    # ------------------------------------------------------------------ confirmation path
    @staticmethod
    def _is_yes(text: str) -> bool:
        return bool(_YES_RE.match(text.strip()))

    def _execute_proposal(self, session: Session, tools: ToolBox, p: Proposal, message: str) -> dict[str, Any]:
        result = tools.run(p.tool, p.args)  # re-checks the rules just before saving
        ok = "error_code" not in result
        call_log = [{"name": p.tool, "args": p.args, "status": "success" if ok else "error"}]
        if result.get("error_code") == "ORDER_NOT_FOUND":
            return self._finish(session, message, ORDER_NOT_FOUND_TEXT, "no_answer", [], call_log, False, None)

        # Let the LLM phrase the outcome (it cannot change it); fall back to a template.
        session.messages.append({"role": "user", "content": message})
        instruction = (
            f"[SYSTEM] The customer confirmed. The action '{p.tool}' was executed with args {json.dumps(p.args)}. "
            f"Result: {json.dumps(result)}. Report this outcome to the customer in one or two friendly sentences "
            f"(include the request/ticket ID if present; if it failed, explain the error). Reply in the FINAL REPLY FORMAT."
        )
        session.messages.append({"role": "user", "content": instruction})
        try:
            resp = self.llm.chat(session.messages, TOOL_SPECS)
            parsed = extract_json(resp.text) or {}
            answer = str(parsed.get("answer") or "").strip()
        except Exception:
            answer = ""
        session.messages.pop()  # drop the bracketed instruction from history
        if not answer:
            answer = self._template_result(p, result, ok)
        session.messages.append({"role": "assistant", "content": answer})
        return self._response(answer, "answer", [], call_log, False)

    @staticmethod
    def _template_result(p: Proposal, result: dict[str, Any], ok: bool) -> str:
        if not ok:
            return f"Sorry, I couldn't complete that: {result.get('message', 'the request was not allowed')}"
        if p.tool == "create_return_request":
            rid = result.get("return_request", {}).get("request_id", "")
            return f"Done - your return request has been created (ID {rid}). Our team will be in touch about the pickup."
        tid = result.get("support_ticket", {}).get("ticket_id", "")
        return f"Done - your {p.args.get('ticket_type')} ticket has been opened (ID {tid}). Our team will be in touch."

    # ------------------------------------------------------------------ LLM tool loop
    def _run_loop(self, session: Session, tools: ToolBox, message: str) -> dict[str, Any]:
        session.messages.append({"role": "user", "content": message})
        call_log: list[dict[str, Any]] = []
        proposal: Optional[Proposal] = None
        retrieved: set[tuple[str, str]] = set()
        not_found = False
        text = ""
        tool_budget = settings.max_tool_calls
        used = 0

        while True:
            try:
                resp = self.llm.chat(session.messages, TOOL_SPECS)
            except Exception as e:  # provider failure -> polite, non-crashing reply
                session.messages.pop()
                return self._response(
                    f"Sorry, I'm having trouble right now ({type(e).__name__}). Please try again in a moment.",
                    "no_answer", [], call_log, False)

            if not resp.tool_calls:
                text = resp.text
                break

            session.messages.append({"role": "assistant", "content": resp.text or "",
                                     "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in resp.tool_calls]})
            for call in resp.tool_calls:
                used += 1
                if used > tool_budget:
                    result: dict[str, Any] = {"error_code": "TOOL_LIMIT", "message": "Tool limit reached; answer with what you have."}
                    status = "error"
                elif call.name in WRITE_TOOLS:
                    result, status, proposal = self._propose(tools, call, proposal)
                else:
                    result = tools.run(call.name, call.args)
                    status = "error" if "error_code" in result else "success"
                    if call.name == "search_policy" and status == "success":
                        for r in result.get("results", []):
                            retrieved.add((r["doc_id"], r["section"]))
                        for f in result.get("future_changes", []):
                            for s in f.get("sections", []):
                                retrieved.add((s["doc_id"], s["section"]))
                if result.get("error_code") == "ORDER_NOT_FOUND":
                    not_found = True
                call_log.append({"name": call.name, "args": tools._clean_args(call.name, call.args), "status": status})
                session.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                         "content": json.dumps(result, default=str)})
            if used > tool_budget:
                # one last turn, without tools, to wrap up politely
                session.messages.append({"role": "user", "content":
                    "[SYSTEM] The tool limit for this message was reached. Reply now in the FINAL REPLY FORMAT with what you know."})
                try:
                    text = self.llm.chat(session.messages, []).text
                except Exception:
                    text = ""
                session.messages.pop()
                break

        answer, route, citations = self._parse_final(text, retrieved)
        if not_found:
            answer, route, citations, proposal = ORDER_NOT_FOUND_TEXT, "no_answer", [], None
        if route == "out_of_scope":
            citations, call_log, proposal = [], [], None
        if route in ("no_answer", "out_of_scope"):
            citations = []
        if proposal is not None:
            if not re.search(r"shall i go ahead", answer, re.I):
                answer = answer.rstrip() + " Shall I go ahead?"
            route = "answer"
        session.pending = proposal
        return self._finish(session, None, answer, route, citations, call_log, proposal is not None, text)

    def _propose(self, tools: ToolBox, call: ToolCall, existing: Optional[Proposal]
                 ) -> tuple[dict[str, Any], str, Optional[Proposal]]:
        """Validate a write request with read-only rules and turn it into a pending proposal."""
        args = tools._clean_args(call.name, call.args)
        try:
            if call.name == "create_return_request":
                check = tools.check_return_eligibility(args.get("order_id"), args.get("item_ids") or [], args.get("reason", ""))
                desc = f"return {', '.join(args.get('item_ids') or [])} from {args.get('order_id')} ({args.get('reason')})"
            else:
                ttype = args.get("ticket_type", "")
                if ttype not in rules.VALID_TICKET_TYPES:
                    raise rules.RuleError("INVALID_TICKET_TYPE", "ticket_type must be 'warranty' or 'cancellation'.")
                order, items, delivered = tools._load_order(args.get("order_id"))
                check = (rules.check_warranty_ticket(order, items, delivered, tools.today, tools.params)
                         if ttype == "warranty" else rules.check_cancellation(order))
                desc = f"open a {ttype} ticket for {args.get('order_id')}"
        except rules.RuleError as e:
            return e.to_dict(), "error", existing
        except Exception as e:
            return {"error_code": "TOOL_ERROR", "message": str(e)}, "error", existing
        # Only one proposal may wait at a time: the latest replaces any earlier one in this turn.
        proposal = Proposal(tool=call.name, args=args, description=desc)
        result = {
            "status": "pending_confirmation",
            "check": check,
            "instruction": ("NOT saved yet. Describe exactly what will be done and ask the customer "
                            "'Shall I go ahead?'. Do not call any more create_* tools."),
        }
        return result, "pending_confirmation", proposal

    # ------------------------------------------------------------------ output shaping
    def _parse_final(self, text: str, retrieved: set[tuple[str, str]]
                     ) -> tuple[str, str, list[dict[str, str]]]:
        parsed = extract_json(text) or {}
        answer = str(parsed.get("answer") or "").strip() or (text or "").strip()
        if not answer:
            answer = "Sorry, I couldn't work that out. Could you rephrase?"
        route = str(parsed.get("route") or "answer").strip().lower()
        if route not in ROUTES:
            route = "answer"
        citations: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for c in parsed.get("citations") or []:
            if not isinstance(c, dict):
                continue
            key = (str(c.get("doc_id", "")), str(c.get("section", "")))
            if key in retrieved and key not in seen and key[1] != "§0":
                seen.add(key)
                citations.append({"doc_id": key[0], "section": key[1]})
        return answer, route, citations

    def _finish(self, session: Session, user_msg: Optional[str], answer: str, route: str,
                citations: list[dict[str, str]], call_log: list[dict[str, Any]], pending: bool,
                raw_text: Optional[str]) -> dict[str, Any]:
        if user_msg is not None:
            session.messages.append({"role": "user", "content": user_msg})
        # Keep the history clean: store the natural-language answer, not the JSON envelope.
        session.messages.append({"role": "assistant", "content": answer})
        return self._response(answer, route, citations, call_log, pending)

    @staticmethod
    def _response(answer: str, route: str, citations: list[dict[str, str]],
                  call_log: list[dict[str, Any]], pending: bool) -> dict[str, Any]:
        return {
            "request_id": uuid.uuid4().hex[:8],
            "answer": answer,
            "route": route,
            "citations": citations,
            "tool_calls": call_log,
            "pending_confirmation": pending,
        }
