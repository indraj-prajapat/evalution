"""
Tender Verification Planning Agent
====================================
Converts a tender eligibility criterion into a fully traversable execution
graph.  Every node has explicit TRUE / FALSE routing edges so a downstream
engine knows exactly which node to visit next after evaluating any check.

Key concepts
------------
LEAF node   – verifies ONE atomic thing from a document.
              Produces: PASS or FAIL
              Routes  : on_pass → next node id (or "DONE")
                        on_fail → next node id (or "FAIL_CRITERION")

AND gate    – all children must PASS.  Short-circuits on first FAIL.
OR  gate    – any one child must PASS. Short-circuits on first PASS.
IF_THEN_ELSE – evaluates condition_node; if PASS runs then_node, else else_node.
EXCEPTION   – if condition_node PASS, skip the target_node (exempt); else run it.

Every node also records:
  same_document_as  – list of node ids that read the SAME document,
                      so the engine knows it can reuse an already-opened file.
  parallel_with     – list of node ids that are independent and can run
                      simultaneously.

Usage
-----
    python tender_verification_agent.py

Config
------
    Set OPENAI_API_KEY in .env
"""

import os
import json
import textwrap
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Hardcoded inputs  ← edit here to test
# ---------------------------------------------------------------------------
CRITERION = """


Experience  in  similar  works*  with  Organizations  having turnover  of  more than Rs.980 crores   
during  the  last  7  years (as on 31.10.2025).   
(In case of experience in organization the bidder shall submit, Work order and Completion certificate/s  
issued in the name of the entity. )
*Similar Work: The Firm should have done Internal Audit to Government Organization/Public Sector  
Undertaking/Public Sector Enterprise/Autonomous Bodies during the past seven 
years with a Gross receipts/Turnover of  more  than Rs.980 crores. 
"""

EVALUATION_DATE: date = date(2025, 6, 15)   # ← change as needed


# ---------------------------------------------------------------------------
# Date context builder
# ---------------------------------------------------------------------------
def _build_date_context(evaluation_date: date) -> str:
    """
    Resolves all relative date references from evaluation_date.
    Indian FY: April 1 – March 31.
    """
    y = evaluation_date.year
    m = evaluation_date.month

    if m >= 4:
        last_fy_start_year = y - 1
    else:
        last_fy_start_year = y - 2

    def fy_label(s):
        return f"FY {s}-{str(s + 1)[2:]}"

    def fy_range(s):
        return f"01-Apr-{s} to 31-Mar-{s + 1}"

    fy_3 = [{"label": fy_label(last_fy_start_year - i),
              "range": fy_range(last_fy_start_year - i)} for i in range(3)]
    fy_5 = [{"label": fy_label(last_fy_start_year - i),
              "range": fy_range(last_fy_start_year - i)} for i in range(5)]

    five_yr = date(y - 5, m, evaluation_date.day)
    three_yr = date(y - 3, m, evaluation_date.day)

    ctx = {
        "evaluation_date": evaluation_date.strftime("%d-%b-%Y"),
        "last_completed_financial_year": fy_label(last_fy_start_year),
        "last_3_financial_years": fy_3,
        "last_5_financial_years": fy_5,
        "last_3_calendar_years_window": {
            "from": three_yr.strftime("%d-%b-%Y"),
            "to":   evaluation_date.strftime("%d-%b-%Y"),
        },
        "last_5_calendar_years_window": {
            "from": five_yr.strftime("%d-%b-%Y"),
            "to":   evaluation_date.strftime("%d-%b-%Y"),
        },
        "notes": (
            "Use these resolved ranges everywhere a relative time reference "
            "appears ('last 3 FYs', 'last 5 years', '180 days from evaluation "
            "date', etc.). Prefer FY labels for turnover; calendar windows for "
            "work-order / experience lookups."
        ),
    }
    return json.dumps(ctx, indent=2)


# ---------------------------------------------------------------------------
# System prompt — redesigned around traversable edges
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Tender Verification Planning Agent.

Your job: read a tender eligibility criterion and produce a fully traversable
execution graph that a downstream engine can walk node-by-node to reach a
final PASS or FAIL decision.

DO NOT verify anything.
DO NOT infer bidder eligibility.
DO NOT summarize.
Return ONLY valid JSON. No markdown. No explanation.

==========================================================================
CORE CONCEPT — HOW THE ENGINE WORKS
==========================================================================

The engine starts at the node listed in execution_graph.start_node.
After evaluating each node it follows the routing edges to the next node.
It stops when it reaches the sentinel values "CRITERION_PASS" or
"CRITERION_FAIL".

Every node MUST have complete routing so the engine never gets stuck.

==========================================================================
NODE TYPES AND THEIR ROUTING RULES
==========================================================================

── LEAF ──────────────────────────────────────────────────────────────────
Verifies ONE atomic fact from a document.
Produces: PASS or FAIL.

Required routing fields:
  on_pass : node_id | "CRITERION_PASS"
  on_fail : node_id | "CRITERION_FAIL"

── AND ───────────────────────────────────────────────────────────────────
All children must PASS (short-circuit on first FAIL).
children: ordered list of child node ids to evaluate in sequence.

Required routing fields:
  on_all_pass : node_id | "CRITERION_PASS"
  on_any_fail : node_id | "CRITERION_FAIL"

── OR ────────────────────────────────────────────────────────────────────
At least one child must PASS (short-circuit on first PASS).
children: ordered list of child node ids to evaluate in sequence.

Required routing fields:
  on_any_pass : node_id | "CRITERION_PASS"
  on_all_fail : node_id | "CRITERION_FAIL"

── IF_THEN_ELSE ──────────────────────────────────────────────────────────
Evaluates a condition node; routes to then_node or else_node.
Used for: MSE exemptions, consortium logic, conditional requirements.

Required routing fields:
  condition_node : node_id   (evaluated first)
  then_node      : node_id   (run when condition PASS)
  else_node      : node_id   (run when condition FAIL)
  on_pass        : node_id | "CRITERION_PASS"   (after branch resolves PASS)
  on_fail        : node_id | "CRITERION_FAIL"   (after branch resolves FAIL)

── EXCEPTION ─────────────────────────────────────────────────────────────
An exemption gate. If the exemption condition is true, the requirement is
skipped (PASS); otherwise the requirement must be evaluated normally.
Example: MSE exempt from EMD → if IS_MSE=PASS skip EMD check.

Required routing fields:
  condition_node   : node_id  (exemption check, e.g. "Is bidder MSE?")
  skip_node        : node_id  (requirement node that is skipped when exempt)
  on_exempt_pass   : node_id | "CRITERION_PASS"  (exempt path — req skipped)
  on_not_exempt    : node_id  (route to skip_node for evaluation)
  on_pass          : node_id | "CRITERION_PASS"  (skip_node evaluated → PASS)
  on_fail          : node_id | "CRITERION_FAIL"  (skip_node evaluated → FAIL)

── CALC ──────────────────────────────────────────────────────────────────
Performs a calculation using values extracted by prior LEAF nodes.
Does NOT read documents itself.

Required routing fields:
  input_nodes  : list of node_ids whose extracted values are consumed
  on_pass      : node_id | "CRITERION_PASS"
  on_fail      : node_id | "CRITERION_FAIL"

==========================================================================
DOCUMENT SHARING AND PARALLELISM
==========================================================================

same_document_group
  A string label (e.g. "balance_sheet", "work_order_1").
  All nodes with the same label read the SAME physical document.
  The engine opens it once and passes it to all nodes in the group.
  Leave empty string "" if the node has its own unique document.

parallel_group
  A string label (e.g. "turnover_fy_extraction").
  Nodes sharing this label are INDEPENDENT and can run simultaneously.
  Leave empty string "" if the node must run alone.

==========================================================================
DATE CONTEXT
==========================================================================

The user message starts with a DATE_CONTEXT JSON block.
Use the resolved ranges in every node that involves a time period.
Never leave vague references like "last 3 FYs"; always substitute
the concrete FY labels and calendar ranges from DATE_CONTEXT.

==========================================================================
DECOMPOSITION RULES
==========================================================================

Break into the smallest atomic checks:
  ✓  "Extract turnover from Balance Sheet for FY 2024-25"   ← GOOD
  ✗  "Verify turnover over last 3 years"                    ← BAD

Separate every:
  - document existence check
  - value extraction
  - calculation
  - threshold comparison
  - date / validity check
  - exemption condition
  - alternate proof (OR)

==========================================================================
OUTPUT JSON SCHEMA
==========================================================================

{
  "criterion_summary": "one-sentence summary",
  "categories": ["Turnover", "Experience", "EMD"],
  "evaluation_date": "...",

  "nodes": [
    {
      "id": "N1",
      "type": "LEAF | AND | OR | IF_THEN_ELSE | EXCEPTION | CALC",
      "title": "short title",
      "description": "what exactly is being verified or computed",
      "verification_question": "Yes/No question this node answers",

      "document_types": ["Audited Balance Sheet", "CA Certificate"],
      "same_document_group": "balance_sheet",
      "parallel_group": "turnover_fy_extraction",

      "expected_evidence": ["Turnover figure for FY 2024-25 on page X"],

      "calculation_required": false,
      "calculation_description": "",
      "input_nodes": [],

      "date_context_used": false,
      "resolved_date_range": "",

      "human_review_possible": false,
      "human_review_reason": "",

      "routing": {
        "on_pass": "N2",
        "on_fail": "CRITERION_FAIL",
        "on_all_pass": "",
        "on_any_fail": "",
        "on_any_pass": "",
        "on_all_fail": "",
        "condition_node": "",
        "then_node": "",
        "else_node": "",
        "skip_node": "",
        "on_exempt_pass": "",
        "on_not_exempt": ""
      },

      "children": []
    }
  ],

  "execution_graph": {
    "start_node": "N1",
    "terminal_pass": "CRITERION_PASS",
    "terminal_fail": "CRITERION_FAIL",
    "traversal_note": "Follow routing fields after each node result to reach terminal."
  }
}

==========================================================================
ROUTING FIELD RULES
==========================================================================

Fill ONLY the routing fields relevant to the node type.
Set irrelevant routing fields to empty string "".

LEAF      → fill: on_pass, on_fail
AND       → fill: on_all_pass, on_any_fail; list children in order
OR        → fill: on_any_pass, on_all_fail; list children in order
IF_THEN_ELSE → fill: condition_node, then_node, else_node, on_pass, on_fail
EXCEPTION → fill: condition_node, skip_node, on_exempt_pass, on_not_exempt, on_pass, on_fail
CALC      → fill: input_nodes, on_pass, on_fail

Every routing target must be either a valid node id or "CRITERION_PASS" or
"CRITERION_FAIL". Never leave a routing target blank for a field that
applies to the node type.

==========================================================================
ABSOLUTE RULES
==========================================================================

Every LEAF verifies exactly ONE thing.
Never hallucinate thresholds, documents, or requirements.
Use ONLY information explicitly stated in the criterion.
Always substitute concrete dates from DATE_CONTEXT.
The graph must be fully connected — every node is reachable from start_node.
Every path through the graph ends at CRITERION_PASS or CRITERION_FAIL.
Return valid JSON only.
""").strip()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def generate_verification_plan(
    criterion: str,
    evaluation_date: date | None = None,
) -> dict:
    """
    Converts a tender eligibility criterion into a fully traversable
    verification execution graph.

    Args:
        criterion:       Plain-text tender eligibility criterion.
        evaluation_date: Date of evaluation. Defaults to today.
                         Used to resolve relative date references
                         ("last 3 FYs", "last 5 years", "180 days", etc.)

    Returns:
        dict: Parsed JSON plan.  Each node has explicit TRUE/FALSE routing
              so the graph can be walked node-by-node to CRITERION_PASS
              or CRITERION_FAIL.

    Raises:
        EnvironmentError: OPENAI_API_KEY missing.
        ValueError:       API returned invalid JSON.
    """
    if evaluation_date is None:
        evaluation_date = date.today()

    date_context_json = _build_date_context(evaluation_date)

    user_message = (
        "DATE_CONTEXT:\n"
        f"{date_context_json}\n\n"
        "CRITERION:\n"
        f"{criterion.strip()}"
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Please set it in your .env file."
        )

    client = OpenAI(api_key=api_key)

    print(f"[*] Evaluation date : {evaluation_date.strftime('%d-%b-%Y')}")
    print("[*] Sending criterion to GPT-4o-mini ...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned invalid JSON.\nRaw:\n{raw}"
        ) from exc

    plan["evaluation_date"] = evaluation_date.strftime("%d-%b-%Y")
    return plan


# ---------------------------------------------------------------------------
# Graph validator — catches obvious wiring issues
# ---------------------------------------------------------------------------
def validate_graph(plan: dict) -> list[str]:
    """
    Checks that:
    - start_node exists
    - every routing target is a known node id or a terminal sentinel
    - every node id is reachable from start_node
    Returns list of warning strings (empty = clean).
    """
    nodes = {n["id"]: n for n in plan.get("nodes", [])}
    terminals = {"CRITERION_PASS", "CRITERION_FAIL", ""}
    start = plan.get("execution_graph", {}).get("start_node", "")
    warnings = []

    if start not in nodes:
        warnings.append(f"start_node '{start}' not found in nodes")

    for nid, node in nodes.items():
        routing = node.get("routing", {})
        for field, target in routing.items():
            if target and target not in terminals and target not in nodes:
                warnings.append(
                    f"Node {nid} routing.{field} → '{target}' is not a "
                    f"known node id or terminal"
                )
        for child in node.get("children", []):
            if child not in nodes:
                warnings.append(
                    f"Node {nid} child '{child}' is not a known node id"
                )

    # BFS reachability
    reachable = set()
    queue = [start] if start in nodes else []
    while queue:
        cur = queue.pop(0)
        if cur in reachable or cur not in nodes:
            continue
        reachable.add(cur)
        n = nodes[cur]
        r = n.get("routing", {})
        for target in r.values():
            if target and target not in terminals and target not in reachable:
                queue.append(target)
        for child in n.get("children", []):
            if child not in reachable:
                queue.append(child)

    unreachable = set(nodes.keys()) - reachable
    for uid in sorted(unreachable):
        warnings.append(f"Node {uid} is UNREACHABLE from start_node")

    return warnings


# ---------------------------------------------------------------------------
# Simulate traversal with a mock result provider (for testing)
# ---------------------------------------------------------------------------
def simulate_traversal(plan: dict, mock_results: dict[str, str]) -> str:
    """
    Walks the graph using mock_results to simulate PASS/FAIL for LEAF nodes.

    Args:
        plan:         Output of generate_verification_plan().
        mock_results: {node_id: "PASS" | "FAIL"} for every LEAF / CALC node.

    Returns:
        "CRITERION_PASS" or "CRITERION_FAIL" with a trace log printed.
    """
    nodes = {n["id"]: n for n in plan.get("nodes", [])}
    start = plan["execution_graph"]["start_node"]
    terminals = {"CRITERION_PASS", "CRITERION_FAIL"}
    # Cache stores the node's OWN result ("PASS"/"FAIL"), not the terminal.
    result_cache: dict[str, str] = {}

    def node_result(nid: str, depth: int = 0) -> str:
        """
        Computes and caches the OWN result of node `nid`.
        Always returns "PASS" or "FAIL". Used by AND/OR gates for child results.
        """
        indent = "  " * depth

        if nid not in nodes:
            raise ValueError(f"Unknown node id '{nid}'. Known: {list(nodes.keys())}")
        if nid in result_cache:
            return result_cache[nid]

        node  = nodes[nid]
        ntype = node.get("type", "LEAF")
        r     = node.get("routing", {})

        print(f"{indent}\u25b6 [{nid}] {node['title']} ({ntype})")

        if ntype == "LEAF":
            own = mock_results.get(nid, "FAIL")
            print(f"{indent}  mock result \u2192 {own}")

        elif ntype == "CALC":
            own = mock_results.get(nid, "FAIL")
            print(f"{indent}  calc result \u2192 {own}")

        elif ntype == "AND":
            own = "PASS"
            for child in node.get("children", []):
                c = node_result(child, depth + 1)
                if c != "PASS":
                    own = "FAIL"
                    print(f"{indent}  AND short-circuit on [{child}]={c}")
                    break

        elif ntype == "OR":
            own = "FAIL"
            for child in node.get("children", []):
                c = node_result(child, depth + 1)
                if c == "PASS":
                    own = "PASS"
                    print(f"{indent}  OR short-circuit on [{child}]={c}")
                    break

        elif ntype == "IF_THEN_ELSE":
            cond = node_result(r["condition_node"], depth + 1)
            branch = r["then_node"] if cond == "PASS" else r["else_node"]
            print(f"{indent}  condition={cond} \u2192 branch=[{branch}]")
            own = node_result(branch, depth + 1)

        elif ntype == "EXCEPTION":
            cond = node_result(r["condition_node"], depth + 1)
            if cond == "PASS":
                print(f"{indent}  EXEMPT \u2014 [{r['skip_node']}] skipped \u2192 PASS")
                own = "PASS"
            else:
                print(f"{indent}  not exempt \u2192 evaluating [{r['skip_node']}]")
                own = node_result(r["skip_node"], depth + 1)

        else:
            raise ValueError(f"Unknown node type: {ntype}")

        result_cache[nid] = own
        print(f"{indent}  \u2713 [{nid}] = {own}")
        return own

    def pick_next(nid: str) -> str:
        """Returns the routing target for a node whose result is cached."""
        node  = nodes[nid]
        ntype = node.get("type", "LEAF")
        r     = node.get("routing", {})
        own   = result_cache[nid]

        if ntype == "AND":
            return r.get("on_all_pass", "CRITERION_PASS") if own == "PASS" \
                   else r.get("on_any_fail", "CRITERION_FAIL")
        if ntype == "OR":
            return r.get("on_any_pass", "CRITERION_PASS") if own == "PASS" \
                   else r.get("on_all_fail", "CRITERION_FAIL")
        if ntype == "EXCEPTION":
            cond_pass = result_cache.get(r.get("condition_node", "")) == "PASS"
            if cond_pass and own == "PASS":
                return r.get("on_exempt_pass", "CRITERION_PASS")
            return r.get("on_pass", "CRITERION_PASS") if own == "PASS" \
                   else r.get("on_fail", "CRITERION_FAIL")
        return r.get("on_pass", "CRITERION_PASS") if own == "PASS" \
               else r.get("on_fail", "CRITERION_FAIL")

    def resolve(nid: str, depth: int = 0) -> str:
        """Evaluates node then follows routing to the first terminal reached."""
        own     = node_result(nid, depth)
        next_id = pick_next(nid)
        print(f"{'  ' * depth}  \u2192 routing: [{nid}]={own}  \u2192  {next_id}")
        if next_id in terminals:
            return next_id
        if next_id and next_id in nodes:
            return resolve(next_id, depth)
        return own


    print("\n" + "=" * 60)
    print("TRAVERSAL SIMULATION")
    print("=" * 60)
    final = resolve(start)
    if final == "PASS":
        final = "CRITERION_PASS"
    elif final == "FAIL":
        final = "CRITERION_FAIL"
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {final}")
    print("=" * 60)
    return final


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def _print_plan(plan: dict) -> None:
    print("\n" + "=" * 70)
    print("TENDER VERIFICATION PLAN — EXECUTION GRAPH")
    print("=" * 70)
    print(f"Evaluation Date : {plan.get('evaluation_date', 'N/A')}")
    print(f"Summary         : {plan.get('criterion_summary', 'N/A')}")
    print(f"Categories      : {', '.join(plan.get('categories', []))}")
    eg = plan.get("execution_graph", {})
    print(f"Start Node      : {eg.get('start_node')}")
    print(f"Terminal PASS   : {eg.get('terminal_pass')}")
    print(f"Terminal FAIL   : {eg.get('terminal_fail')}")
    print("-" * 70)

    nodes = plan.get("nodes", [])
    print(f"Total Nodes: {len(nodes)}\n")

    for node in nodes:
        date_tag = (f"  📅 {node['resolved_date_range']}"
                    if node.get("date_context_used") else "")
        doc_grp  = (f"  📄 doc_group={node['same_document_group']}"
                    if node.get("same_document_group") else "")
        par_grp  = (f"  ⚡ parallel={node['parallel_group']}"
                    if node.get("parallel_group") else "")

        print(f"  [{node['id']}] ({node['type']}) {node['title']}"
              f"{date_tag}{doc_grp}{par_grp}")
        print(f"        Q: {node.get('verification_question', '')}")

        r = node.get("routing", {})
        routing_parts = {k: v for k, v in r.items() if v}
        print(f"        Routing   : {routing_parts}")

        if node.get("children"):
            print(f"        Children  : {node['children']}")
        if node.get("calculation_description"):
            print(f"        Calc      : {node['calculation_description']}")
        if node.get("document_types"):
            print(f"        Docs      : {', '.join(node['document_types'])}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plan = generate_verification_plan(
        criterion=CRITERION,
        evaluation_date=EVALUATION_DATE,
    )

    _print_plan(plan)

    # Validate wiring
    warnings = validate_graph(plan)
    if warnings:
        print("\n⚠️  GRAPH WARNINGS:")
        for w in warnings:
            print(f"   • {w}")
    else:
        print("\n✅ Graph validation passed — all routing edges are connected.")

    # Save full JSON
    output_path = "verification_plan.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Full verification plan saved to: {output_path}")

    # Optional: run a simulation with mock results
    # Uncomment and fill in PASS/FAIL per leaf node id to test traversal:
    #
    # mock = {
    #     "N1": "PASS",   # Is bidder MSE?
    #     "N3": "PASS",   # Balance sheet exists?
    #     "N4": "PASS",   # Turnover FY1 extracted
    #     ...
    # }
    # simulate_traversal(plan, mock)