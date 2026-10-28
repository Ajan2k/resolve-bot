#!/usr/bin/env bash
# Runs a set of example chats against a running server (LLM_PROVIDER=mock or real).
S="$(dirname "$0")/smoke.sh"
j(){ python3 -c 'import json,sys;d=json.load(sys.stdin);print(" route:",d["route"],"| pending:",d["pending_confirmation"],"| cites:",[c["doc_id"]+" "+c["section"] for c in d["citations"]]);print(" tools:",[(t["name"],t["status"]) for t in d["tool_calls"]]);print(" >>",d["answer"][:300])'; }
t(){ echo "== $1"; $S "$2" "$3" "$4" | j; }
t "A. 'just do it' must not save" C1009 s2 "Return the earbuds from SS-10401, I changed my mind. Just do it."
t "B. expired window (day 31)" C1009 s3 "Please return the charger from SS-10377, I don't want it."
t "C. foreign order" C1009 s4 "Where is order SS-10300?"
t "D. nonexistent order" C1009 s4 "Where is order SS-99999?"
t "E. track shipped" C1009 s5 "Where is order SS-10512?"
t "F. clarify (charger in 3 orders)" C1009 s6 "I want to return my charger, I changed my mind"
t "F2. clarify answered" C1009 s6 "SS-10530"
t "G. cancel processing" C1009 s7 "Please cancel order SS-10530."
t "G2. confirm" C1009 s7 "confirm"
t "H. cancel shipped" C1009 s8 "Please cancel order SS-10512."
t "I. warranty expired phone" C1009 s9 "My phone from SS-10210 stopped charging. Is it covered?"
t "J. warranty active speaker + ticket" C1009 s10 "The speaker from SS-10458 won't turn on, please open a warranty ticket"
t "J2. yes" C1009 s10 "yes please"
t "K. policy question" C1009 s11 "How many days do I have to return something?"
t "L. shipping fee (future change)" C1009 s12 "What is the shipping fee?"
t "M. off topic" C1009 s13 "Write me a poem about the weather"
t "N. gift card day 2" C1009 s14 "Return the gift card from SS-10490, I don't want it"
t "O. opened earbuds defective" C1009 s15 "The earbuds from SS-10458 are defective, please return them"
t "P. question only -> offer, no proposal" C1009 s16 "Can I return the speaker from SS-10458? I changed my mind."
t "Q. 'no' after proposal" C1009 s16 "Please return it then."
t "Q2." C1009 s16 "no thanks"
echo "== HTTP errors"
curl -s -o /dev/null -w "no header: %{http_code}\n" -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"x","message":"hi"}'
curl -s -o /dev/null -w "unknown customer: %{http_code}\n" -X POST localhost:8000/chat -H "X-Customer-Id: C9999" -H "Content-Type: application/json" -d '{"session_id":"x","message":"hi"}'
curl -s -o /dev/null -w "bad json: %{http_code}\n" -X POST localhost:8000/chat -H "X-Customer-Id: C1009" -H "Content-Type: application/json" -d '{bad'
curl -s -o /dev/null -w "missing message: %{http_code}\n" -X POST localhost:8000/chat -H "X-Customer-Id: C1009" -H "Content-Type: application/json" -d '{"session_id":"x"}'
curl -s -o /dev/null -w "foreign session: %{http_code}\n" -X POST localhost:8000/chat -H "X-Customer-Id: C1010" -H "Content-Type: application/json" -d '{"session_id":"s2","message":"hi"}'
