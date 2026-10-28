"""Deterministic rule-based stand-in for an LLM (LLM_PROVIDER=mock).

It only *drives tools and phrases text*; every allow/deny decision still comes from the
Python tools, exactly as with a real model. It covers the intents in the assessment
(track, return, warranty, cancel, policy question, off-topic, clarification) and is
meant for local testing when no API key is available.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date
from typing import Any, Optional

from .llm import LLMResponse, ToolCall

ORDER_RE = re.compile(r"\bSS-\d{3,}\b", re.I)
ITEM_RE = re.compile(r"\bSS-\d{3,}-\d+\b", re.I)

INTENT_WORDS = {
    "return": ["return", "send back", "send it back", "give back", "refund"],
    "cancel": ["cancel"],
    "warranty": ["warranty", "stopped working", "not working", "won't turn on", "wont turn on", "stopped charging",
                 "not charging", "dead", "faulty", "covered"],
    "track": ["where is", "track", "status", "delivered", "delivery", "arrive", "eta", "when will"],
    "policy": ["policy", "how many days", "how long", "window", "fee", "shipping", "cost", "refund", "exchange",
               "payment", "upi", "plus", "international", "outside india", "address", "warranty", "cancel",
               "gift card", "software", "earbuds", "opened", "damaged"],
}
STORE_WORDS = ["order", "item", "return", "refund", "warranty", "cancel", "ship", "deliver", "policy", "track",
               "speaker", "phone", "laptop", "charger", "earbud", "gift card", "shopsphere", "payment", "plus",
               "exchange", "damaged", "defective", "broken", "address", "ss-"]
DEFECT_WORDS = ["defective", "broken", "damaged", "faulty", "not working", "doesn't work", "does not work",
                "wrong item", "wrong product", "stopped", "dead", "cracked"]
MIND_WORDS = ["changed my mind", "change of mind", "don't want", "dont want", "do not want", "no longer need",
              "don't need", "not needed", "changed mind"]
ACTION_WORDS = ["i want to", "want to", "please", "i'd like to", "i would like to", "can you", "could you",
                "start", "create", "open", "do it", "go ahead", "initiate", "raise", "file", "need to", "just do"]
QUESTION_ONLY = re.compile(r"^(can|could|am i able to|is it possible|may i|is (it|this|the|my)|does|do i|will|what|how|when|are)\b", re.I)


def _fmt(d: Optional[str]) -> str:
    if not d:
        return "unknown"
    try:
        dd = date.fromisoformat(str(d)[:10])
        return f"{dd.day} {dd:%B %Y}"
    except ValueError:
        return str(d)


class MockBrain:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        # locate the current turn: everything after the last real user message
        self.turn_start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        self.user = messages[self.turn_start]["content"]
        self.results: list[tuple[str, dict[str, Any], dict[str, Any]]] = []  # (tool, args, result)
        for m in messages[self.turn_start:]:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls", []):
                    self._pending_call = tc
            if m["role"] == "tool":
                args = next((tc["args"] for mm in messages[self.turn_start:] if mm["role"] == "assistant"
                             for tc in mm.get("tool_calls", []) if tc["id"] == m["tool_call_id"]), {})
                self.results.append((m["name"], args, json.loads(m["content"])))
        # previous assistant message (for clarification follow-ups)
        prev_assist = [m for m in messages[:self.turn_start] if m["role"] == "assistant" and m.get("content")]
        self.prev_assistant = prev_assist[-1]["content"] if prev_assist else ""
        prev_user = [m for m in messages[:self.turn_start] if m["role"] == "user" and not m["content"].startswith("[SYSTEM]")]
        self.prev_user = prev_user[-1]["content"] if prev_user else ""
        # If the previous assistant message asked a question, merge the previous request for context.
        self.text = self.user
        if self.prev_assistant.rstrip().endswith("?") and not self.user.startswith("[SYSTEM]") \
                and "shall i go ahead" not in self.prev_assistant.lower():
            self.text = f"{self.prev_user} {self.user}"
        self.low = self.text.lower()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def final(answer: str, route: str = "answer", citations: Optional[list[dict[str, str]]] = None) -> LLMResponse:
        return LLMResponse(text=json.dumps({"answer": answer, "route": route, "citations": citations or []}))

    @staticmethod
    def call(name: str, **args: Any) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(id=str(uuid.uuid4()), name=name, args=args)])

    def got(self, tool: str) -> Optional[dict[str, Any]]:
        for t, _, r in self.results:
            if t == tool:
                return r
        return None

    def got_all(self, tool: str) -> list[dict[str, Any]]:
        return [r for t, _, r in self.results if t == tool]

    def _intent(self) -> str:
        if any(w in self.low for w in INTENT_WORDS["cancel"]) and "cancellation policy" not in self.low:
            return "cancel"
        if any(w in self.low for w in INTENT_WORDS["return"]) and (ORDER_RE.search(self.text) or
                                                                     any(w in self.low for w in ["my ", "the "])):
            if any(w in self.low for w in ["how many days", "how long", "policy", "window"]) and not ORDER_RE.search(self.text):
                return "policy"
            return "return"
        if any(w in self.low for w in INTENT_WORDS["warranty"]) and (ORDER_RE.search(self.text) or " my " in f" {self.low} "):
            return "warranty"
        if ORDER_RE.search(self.text) or any(w in self.low for w in INTENT_WORDS["track"]):
            if any(w in self.low for w in ["how many days", "how long", "policy", "fee", "cost"]) and not ORDER_RE.search(self.text):
                return "policy"
            return "track"
        if any(w in self.low for w in INTENT_WORDS["policy"]):
            return "policy"
        if any(w in self.low for w in STORE_WORDS):
            return "policy"
        return "out_of_scope"

    def _cites(self, res: dict[str, Any], n: int = 2) -> list[dict[str, str]]:
        out = []
        for r in res.get("results", [])[:n]:
            if r["section"] != "§0":
                out.append({"doc_id": r["doc_id"], "section": r["section"]})
        return out

    def _policy_text(self, res: dict[str, Any], n: int = 2) -> str:
        parts = []
        for r in res.get("results", [])[:n]:
            parts.append(f"{r['text'].strip()} ({r['doc_id']} {r['section']})")
        for f in res.get("future_changes", []):
            if f.get("sections"):
                parts.append(f"Note: from {_fmt(f['effective_date'])} this changes: {f['sections'][0]['text'].strip()} "
                             f"({f['sections'][0]['doc_id']} {f['sections'][0]['section']})")
        return " ".join(parts)

    def _wants_action(self) -> bool:
        if QUESTION_ONLY.match(self.user.strip()) and not any(w in self.low for w in ["please", "want to", "go ahead"]):
            return False
        return any(w in self.low for w in ACTION_WORDS) or self.low.startswith(("return", "cancel", "open", "create", "raise"))

    # ------------------------------------------------------------ main step
    def step(self) -> LLMResponse:
        if self.user.startswith("[SYSTEM] The customer confirmed"):
            return self._report_confirmed()
        if self.user.startswith("[SYSTEM]"):
            return self.final("Sorry, I ran out of steps for that request. Could you ask again in smaller parts?")
        if re.match(r"^\W*(no|nope|no thanks|no thank you|never mind|nevermind|cancel that|don't|do not|thanks|thank you|ok thanks)\W*$", self.user.strip(), re.I):
            return self.final("No problem - nothing has been changed. Let me know if there's anything else I can help with.")
        intent = self._intent()
        if intent == "out_of_scope":
            return self.final("I can only help with ShopSphere orders and policies - tracking, returns, warranty, "
                              "cancellations and policy questions.", "out_of_scope")
        return getattr(self, f"_do_{intent}")()

    def _report_confirmed(self) -> LLMResponse:
        m = re.search(r"Result: (\{.*\})\. Report", self.user, re.S)
        res = json.loads(m.group(1)) if m else {}
        if "error_code" in res:
            return self.final(f"Sorry, I couldn't complete that: {res.get('message')}")
        if "return_request" in res:
            return self.final(f"Done - your return request is created (ID {res['return_request']['request_id']}). "
                              f"Our team will be in touch about the pickup.")
        t = res.get("support_ticket", {})
        return self.final(f"Done - your {t.get('ticket_type')} ticket is open (ID {t.get('ticket_id')}). Our team will contact you.")

    # ------------------------------------------------------------ orders
    def _orders(self) -> Optional[list[dict[str, Any]]]:
        res = self.got("get_orders")
        if res is None:
            return None
        return res.get("orders", [])

    def _fetch_orders(self) -> LLMResponse:
        m = ORDER_RE.search(self.text)
        return self.call("get_orders", order_id=m.group(0).upper()) if m else self.call("get_orders")

    def _match_items(self, orders: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Items mentioned by ID or product-name keyword."""
        ids = {i.upper() for i in ITEM_RE.findall(self.text)}
        hits = []
        for o in orders:
            for it in o["items"]:
                if it["item_id"].upper() in ids:
                    hits.append((o, it))
                    continue
                words = [w for w in re.findall(r"[a-z]+", it["product_name"].lower()) if len(w) > 3]
                cat = str(it.get("category") or "").lower()
                if any(w in self.low for w in words) or (cat and cat.rstrip("s") in self.low):
                    hits.append((o, it))
        return hits

    # ------------------------------------------------------------ track
    def _do_track(self) -> LLMResponse:
        orders = self._orders()
        if orders is None:
            return self._fetch_orders()
        if not orders:
            return self.final("I couldn't find any orders on your account.", "no_answer")
        lines = []
        for o in orders[:5]:
            s = o["shipment"]
            if o["status"] == "delivered":
                lines.append(f"Order {o['order_id']} was delivered on {_fmt(o['delivery_date'])} "
                             f"({o['days_since_delivery']} days ago).")
            elif o["status"] == "cancelled":
                lines.append(f"Order {o['order_id']} is cancelled.")
            elif o["status"] == "processing":
                lines.append(f"Order {o['order_id']} is still being prepared and has not shipped yet.")
            else:
                lines.append(f"Order {o['order_id']} is {str(s.get('status') or o['status']).replace('_', ' ')} with "
                             f"{s.get('carrier') or 'our carrier'}; expected delivery {_fmt(s.get('eta'))}.")
        return self.final(" ".join(lines))

    # ------------------------------------------------------------ policy
    def _do_policy(self) -> LLMResponse:
        res = self.got("search_policy")
        if res is None:
            return self.call("search_policy", query=self.text)
        if not res.get("results"):
            return self.final("Our current policies don't cover that. Please email support@shopsphere.example.com "
                              "and the team will help.", "no_answer")
        return self.final(self._policy_text(res), "answer", self._cites(res))

    # ------------------------------------------------------------ return
    def _reason(self) -> Optional[str]:
        if any(w in self.low for w in DEFECT_WORDS):
            return "defective"
        if any(w in self.low for w in MIND_WORDS):
            return "changed_mind"
        return None

    def _do_return(self) -> LLMResponse:
        orders = self._orders()
        if orders is None:
            return self._fetch_orders()
        if not orders:
            return self.final("I couldn't find any orders on your account.", "no_answer")
        hits = self._match_items(orders)
        m = ORDER_RE.search(self.text)
        if m:
            hits = [(o, i) for o, i in hits if o["order_id"] == m.group(0).upper()]
        if not hits and len(orders) == 1 and len(orders[0]["items"]) == 1:
            hits = [(orders[0], orders[0]["items"][0])]
        reason = self._reason()
        missing = []
        if not hits:
            opts = "; ".join(f"{o['order_id']}: " + ", ".join(f"{i['product_name']} ({i['item_id']})" for i in o["items"])
                             for o in orders[:5])
            missing.append(f"which item you'd like to return ({opts})")
        elif len({o["order_id"] for o, _ in hits}) > 1:
            opts = ", ".join(f"{o['order_id']} ({i['product_name']}, delivered {_fmt(o['delivery_date'])})" for o, i in hits)
            missing.append(f"which order you mean - I found that item in {opts}")
        if reason is None:
            missing.append("the reason: did you change your mind, or is the item defective, damaged or wrong?")
        if missing:
            return self.final("Happy to help with a return. Could you tell me " + " and ".join(missing) + "?", "clarify")

        order = hits[0][0]
        item_ids = [i["item_id"] for o, i in hits if o["order_id"] == order["order_id"]]
        pol = self.got("search_policy")
        if pol is None:
            return self.call("search_policy", query="return window days")
        chk = self.got("check_return_eligibility")
        if chk is None:
            return self.call("check_return_eligibility", order_id=order["order_id"], item_ids=item_ids, reason=reason)
        names = ", ".join(i["product_name"] for o, i in hits if o["order_id"] == order["order_id"])
        cites = self._cites(pol, 1)
        if "error_code" in chk:
            return self.final(f"Unfortunately the {names} from {order['order_id']} can't be returned: {chk['message']} "
                              f"(see returns policy).", "answer", cites)
        base = (f"The {names} from {order['order_id']} can be returned. It was delivered {chk['days_since_delivery']} days ago, "
                f"within the {chk['return_window_days']}-day return window.")
        if not self._wants_action():
            return self.final(base + " Would you like me to create the return request?", "answer", cites)
        prop = self.got("create_return_request")
        if prop is None:
            return self.call("create_return_request", order_id=order["order_id"], item_ids=item_ids, reason=reason)
        return self.final(base + f" I'll create a return request for {names} ({', '.join(item_ids)}) with reason "
                          f"'{reason.replace('_', ' ')}'. Shall I go ahead?", "answer", cites)

    # ------------------------------------------------------------ cancel
    def _do_cancel(self) -> LLMResponse:
        orders = self._orders()
        if orders is None:
            return self._fetch_orders()
        if not ORDER_RE.search(self.text):
            if len(orders) == 1:
                pass
            else:
                opts = ", ".join(f"{o['order_id']} ({o['status']})" for o in orders[:6])
                return self.final(f"Which order would you like to cancel? Your orders: {opts}.", "clarify")
        if not orders:
            return self.final("I couldn't find any orders on your account.", "no_answer")
        order = orders[0]
        pol = self.got("search_policy")
        if pol is None:
            return self.call("search_policy", query="cancel order before shipment")
        cites = self._cites(pol, 1)
        if order["status"] != "processing":
            return self.final(f"Order {order['order_id']} has already {('been ' + order['status']) if order['status'] != 'cancelled' else 'been cancelled'}, "
                              f"so it can't be cancelled now. Once delivered, you can return it instead.", "answer", cites)
        prop = self.got("create_support_ticket")
        if prop is None:
            return self.call("create_support_ticket", order_id=order["order_id"], ticket_type="cancellation",
                             summary=f"Customer requests cancellation of {order['order_id']}")
        if "error_code" in prop:
            return self.final(f"Sorry, order {order['order_id']} can't be cancelled: {prop['message']}", "answer", cites)
        return self.final(f"Order {order['order_id']} is still processing, so it can be cancelled. I'll open a cancellation "
                          f"ticket and our team will confirm by email. Shall I go ahead?", "answer", cites)

    # ------------------------------------------------------------ warranty
    def _do_warranty(self) -> LLMResponse:
        orders = self._orders()
        if orders is None:
            return self._fetch_orders()
        if not orders:
            return self.final("I couldn't find any orders on your account.", "no_answer")
        hits = self._match_items([o for o in orders if o["status"] == "delivered"]) or self._match_items(orders)
        if not hits:
            if ORDER_RE.search(self.text) and len(orders) == 1:
                hits = [(orders[0], it) for it in orders[0]["items"]]
            else:
                opts = "; ".join(f"{o['order_id']}: " + ", ".join(i["product_name"] for i in o["items"]) for o in orders[:5])
                return self.final(f"Which product is having the problem? Your orders: {opts}.", "clarify")
        if len({o["order_id"] for o, _ in hits}) > 1:
            opts = ", ".join(f"{o['order_id']} ({i['product_name']}, delivered {_fmt(o['delivery_date'])})" for o, i in hits)
            return self.final(f"I found that product in more than one order: {opts}. Which one do you mean?", "clarify")
        order, item = hits[0]
        pol = self.got("search_policy")
        if pol is None:
            return self.call("search_policy", query="warranty period covered faults claim")
        cites = self._cites(pol, 1)
        if not item["has_warranty"]:
            return self.final(f"The {item['product_name']} has no warranty, so this isn't covered.", "answer", cites)
        if order["status"] != "delivered":
            return self.final(f"Order {order['order_id']} hasn't been delivered yet, so the warranty hasn't started.", "answer", cites)
        if not item["warranty_active"]:
            return self.final(f"The {item['product_name']}'s {item['warranty_months']}-month warranty ended on "
                              f"{_fmt(item['warranty_end_date'])}, so it is no longer covered.", "answer", cites)
        base = (f"The {item['product_name']} from {order['order_id']} is under its {item['warranty_months']}-month warranty "
                f"until {_fmt(item['warranty_end_date'])}. " + self._policy_text(pol, 1))
        if not self._wants_action() and not any(w in self.low for w in ["ticket", "claim", "repair", "fix"]):
            return self.final(base + " Would you like me to open a warranty ticket?", "answer", cites)
        prop = self.got("create_support_ticket")
        if prop is None:
            return self.call("create_support_ticket", order_id=order["order_id"], ticket_type="warranty",
                             summary=f"{item['product_name']}: {self.user[:120]}")
        if "error_code" in prop:
            return self.final(f"Sorry, I can't open a warranty ticket: {prop['message']}", "answer", cites)
        return self.final(base + " I'll open a warranty ticket for it. Shall I go ahead?", "answer", cites)
