"""
Retail AI Assistant
Personal Shopper + Customer Support Agent
Powered by OpenAI function calling
"""

import os
import csv
import ast
import json
import logging
from datetime import datetime, date
from typing import Optional
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich.markdown import Markdown

# ── setup ─────────────────────────────────────────────────────────────────────

console = Console()
LOGGER = logging.getLogger("retail_ai_assistant")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

# Simulation date — fixed so return windows work correctly relative to
# order dates in orders.csv (Jan 17 – Feb 24 2026).
# Swap to date.today() when running against live order data.
SIMULATION_DATE = date(2026, 2, 10)
DEFAULT_MIN_STOCK = 1
DEFAULT_SEARCH_LIMIT = 5
NOCTURNE_RETURN_WINDOW_DAYS = 21
SALE_RETURN_WINDOW_DAYS = 7
STANDARD_RETURN_WINDOW_DAYS = 14
OPENAI_MODEL = "gpt-4o-mini"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── data loading ──────────────────────────────────────────────────────────────

def log_event(level: str, event: str, **context) -> None:
    """Write structured log entries for key runtime events."""
    payload = {"event": event, **context}
    LOGGER.log(getattr(logging, level.upper(), logging.INFO), json.dumps(payload, default=str))


def load_products() -> dict:
    """Load and normalize product inventory rows from CSV."""
    products = {}
    path = os.path.join(DATA_DIR, "product_inventory.csv")
    if not os.path.exists(path):
        console.print(f"[red]Missing file: {path}\nPlease ensure product_inventory.csv is in the same folder as agent.py[/red]")
        raise SystemExit(1)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row_num, row in enumerate(csv.DictReader(f), start=2):
                try:
                    row["price"] = float(row["price"])
                    row["compare_at_price"] = float(row["compare_at_price"])
                    row["bestseller_score"] = int(row["bestseller_score"])
                    row["is_sale"] = row["is_sale"].strip().lower() == "true"
                    row["is_clearance"] = row["is_clearance"].strip().lower() == "true"
                    row["sizes_available"] = [s.strip() for s in row["sizes_available"].split("|")]
                    row["stock_per_size"] = ast.literal_eval(row["stock_per_size"])
                    row["tags"] = [t.strip().lower() for t in row["tags"].split(",")]
                    products[row["product_id"]] = row
                except (KeyError, ValueError, SyntaxError) as exc:
                    raise ValueError(f"Invalid product row {row_num}: {exc}") from exc
    except (OSError, csv.Error, ValueError) as exc:
        log_event("error", "load_products_failed", path=path, error=str(exc))
        console.print(f"[red]Failed to parse product_inventory.csv: {exc}[/red]")
        raise SystemExit(1)
    return products

def load_orders() -> dict:
    """Load and normalize order rows from CSV."""
    orders = {}
    path = os.path.join(DATA_DIR, "orders.csv")
    if not os.path.exists(path):
        console.print(f"[red]Missing file: {path}\nPlease ensure orders.csv is in the same folder as agent.py[/red]")
        raise SystemExit(1)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row_num, row in enumerate(csv.DictReader(f), start=2):
                try:
                    row["price_paid"] = float(row["price_paid"])
                    orders[row["order_id"]] = row
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid order row {row_num}: {exc}") from exc
    except (OSError, csv.Error, ValueError) as exc:
        log_event("error", "load_orders_failed", path=path, error=str(exc))
        console.print(f"[red]Failed to parse orders.csv: {exc}[/red]")
        raise SystemExit(1)
    return orders

def load_policy() -> str:
    """Load the return-policy text used in the system prompt."""
    path = os.path.join(DATA_DIR, "policy.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        log_event("error", "load_policy_failed", path=path, error=str(exc))
        console.print(f"[red]Failed to load policy.txt: {exc}[/red]")
        raise SystemExit(1)

PRODUCTS = load_products()
ORDERS   = load_orders()
POLICY   = load_policy()
log_event("info", "data_loaded", products=len(PRODUCTS), orders=len(ORDERS))

# ── tools ─────────────────────────────────────────────────────────────────────

def search_products(
    tags: Optional[list] = None,
    max_price: Optional[float] = None,
    size: Optional[str] = None,
    is_sale: Optional[bool] = None,
    is_clearance: Optional[bool] = None,
    min_stock: int = DEFAULT_MIN_STOCK,
    limit: int = DEFAULT_SEARCH_LIMIT
) -> list:
    """Filter products and return ranked matches.

    Args:
        tags: Optional list of desired product tags.
        max_price: Optional maximum price in USD.
        size: Optional requested size string.
        is_sale: Optional sale filter.
        is_clearance: Optional clearance filter.
        min_stock: Minimum stock required for the requested size.
        limit: Maximum number of products to return.

    Returns:
        A ranked list of product dictionaries.
    """
    results = []
    size_str = str(size) if size else None

    for p in PRODUCTS.values():
        # price filter
        if max_price and p["price"] > max_price:
            continue
        # sale filter
        if is_sale is True and not p["is_sale"]:
            continue
        # clearance filter
        if is_clearance is False and p["is_clearance"]:
            continue
        # size + stock filter
        if size_str:
            if size_str not in p["sizes_available"]:
                continue
            stock = p["stock_per_size"].get(size_str, 0)
            if stock < min_stock:
                continue
        # tag filter
        if tags:
            if not any(t.lower() in p["tags"] for t in tags):
                continue
        results.append(p)

    # rank: sale first, then by bestseller_score descending
    results.sort(key=lambda x: (not x["is_sale"], -x["bestseller_score"]))
    return results[:limit]


def get_product(product_id: str) -> Optional[dict]:
    """Return full product details or None if not found."""
    return PRODUCTS.get(product_id)


def get_order(order_id: str) -> Optional[dict]:
    """Return order details or None if not found."""
    return ORDERS.get(order_id)


def evaluate_return(order_id: str) -> dict:
    """
    Evaluate return eligibility for an order according to store policy.

    Args:
        order_id: The order identifier to evaluate.

    Returns:
        A dictionary with fields: eligible, verdict, reason, and policy_applied.
    """
    order = get_order(order_id)
    if not order:
        return {
            "eligible": False,
            "verdict": "ORDER NOT FOUND",
            "reason": f"No order with ID '{order_id}' exists in the system.",
            "policy_applied": "N/A"
        }

    product = get_product(order["product_id"])
    if not product:
        return {
            "eligible": False,
            "verdict": "PRODUCT NOT FOUND",
            "reason": f"Product '{order['product_id']}' associated with order '{order_id}' does not exist.",
            "policy_applied": "N/A"
        }

    # calculate days since order
    order_date = datetime.strptime(order["order_date"], "%Y-%m-%d").date()
    today = SIMULATION_DATE
    days_since = (today - order_date).days

    vendor  = product["vendor"].strip().lower()
    is_sale = product["is_sale"]
    is_clr  = product["is_clearance"]

    # ── policy rules ──────────────────────────────────────────────────────────

    # Rule 1: Clearance — always final sale
    if is_clr:
        return {
            "eligible": False,
            "verdict": "NOT ELIGIBLE",
            "reason": "This is a clearance item. All clearance items are final sale and cannot be returned or exchanged.",
            "policy_applied": "Clearance Items: Final sale. Not eligible for return or exchange."
        }

    # Rule 2: Aurelia Couture — exchange only
    if vendor == "aurelia couture":
        return {
            "eligible": True,
            "verdict": "EXCHANGE ONLY",
            "reason": f"Aurelia Couture items are eligible for exchange only — no refunds. Order placed {days_since} days ago.",
            "policy_applied": "Vendor Exception — Aurelia Couture: Exchanges only, no refunds."
        }

    # Rule 3: Nocturne — extended 21-day window
    if vendor == "nocturne":
        window = NOCTURNE_RETURN_WINDOW_DAYS
        if days_since <= window:
            refund_type = "store credit only" if is_sale else "full refund"
            return {
                "eligible": True,
                "verdict": "ELIGIBLE",
                "reason": f"Nocturne has an extended {window}-day return window. Order placed {days_since} days ago. Eligible for {refund_type}.",
                "policy_applied": f"Vendor Exception — Nocturne: Extended return window of {window} days."
            }
        else:
            return {
                "eligible": False,
                "verdict": "NOT ELIGIBLE",
                "reason": f"Nocturne's extended {window}-day return window has passed. Order placed {days_since} days ago.",
                "policy_applied": f"Vendor Exception — Nocturne: Extended return window of {window} days."
            }

    # Rule 4: Sale items — 7 days, store credit only
    if is_sale:
        window = SALE_RETURN_WINDOW_DAYS
        if days_since <= window:
            return {
                "eligible": True,
                "verdict": "ELIGIBLE — STORE CREDIT ONLY",
                "reason": f"Sale item returned within {days_since} days (within {window}-day window). Store credit only — no cash refund.",
                "policy_applied": f"Sale Items: Returnable within {window} days. Store credit only."
            }
        else:
            return {
                "eligible": False,
                "verdict": "NOT ELIGIBLE",
                "reason": f"Sale item return window is {window} days. Order placed {days_since} days ago — window has passed.",
                "policy_applied": f"Sale Items: Returnable within {window} days. Store credit only."
            }

    # Rule 5: Normal items — 14 days, full refund
    window = STANDARD_RETURN_WINDOW_DAYS
    if days_since <= window:
        return {
            "eligible": True,
            "verdict": "ELIGIBLE — FULL REFUND",
            "reason": f"Normal item returned within {days_since} days (within {window}-day window). Eligible for full refund.",
            "policy_applied": f"Normal Items: Returns accepted within {window} days for a full refund."
        }
    else:
        return {
            "eligible": False,
            "verdict": "NOT ELIGIBLE",
            "reason": f"Normal item return window is {window} days. Order placed {days_since} days ago — window has passed.",
            "policy_applied": f"Normal Items: Returns accepted within {window} days for a full refund."
        }

# ── tool definitions for OpenAI ───────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search and filter products by tags, price, size, sale status. Always use this when the user asks for product recommendations. Never guess products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags":         {"type": "array", "items": {"type": "string"}, "description": "List of style tags to match e.g. ['evening', 'modest', 'lace']"},
                    "max_price":    {"type": "number", "description": "Maximum price in USD"},
                    "size":         {"type": "string", "description": "Clothing size e.g. '8', '10', '14'"},
                    "is_sale":      {"type": "boolean", "description": "Set true to filter sale items only"},
                    "is_clearance": {"type": "boolean", "description": "Set false to exclude clearance items"},
                    "min_stock":    {"type": "integer", "description": "Minimum stock units required for the size (default 1)"},
                    "limit":        {"type": "integer", "description": "Max number of results to return (default 5)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Get full details of a single product by its product_id. Use this to verify product details before making a recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product ID e.g. 'P0001'"}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Fetch order details by order ID. Use this when a customer references an order. Returns null if not found — never assume an order exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID e.g. 'O0043'"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_return",
            "description": "Evaluate whether an order is eligible for return based on store policy. Always use this for any return or refund question — never reason about returns without calling this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID to evaluate e.g. 'O0043'"}
                },
                "required": ["order_id"]
            }
        }
    }
]

# ── tool dispatcher ────────────────────────────────────────────────────────────

def dispatch_tool(name: str, args: dict) -> str:
    """Dispatch tool calls from the model and return JSON-encoded results."""
    log_event("info", "tool_dispatch_start", tool=name, args=args)
    if name == "search_products":
        results = search_products(**args)
        if not results:
            return json.dumps({"results": [], "message": "No products found matching the given filters."})
        return json.dumps({"results": results})

    elif name == "get_product":
        product = get_product(args["product_id"])
        if not product:
            return json.dumps({"error": f"Product '{args['product_id']}' not found."})
        return json.dumps(product)

    elif name == "get_order":
        order = get_order(args["order_id"])
        if not order:
            return json.dumps({"error": f"Order '{args['order_id']}' not found."})
        return json.dumps(order)

    elif name == "evaluate_return":
        result = evaluate_return(args["order_id"])
        return json.dumps(result)

    log_event("warning", "unknown_tool", tool=name)
    return json.dumps({"error": f"Unknown tool: {name}"})

# ── rich display helpers ───────────────────────────────────────────────────────

def print_welcome():
    console.print()
    console.print(Panel.fit(
        "[bold white]Retail AI Assistant[/bold white]\n"
        "[dim]Personal Shopper  ·  Customer Support[/dim]\n"
        "[dim]Type your message or 'exit' to quit[/dim]",
        border_style="bright_blue",
        padding=(1, 4)
    ))
    console.print()

def print_user(msg: str):
    console.print(f"\n[bold cyan]You:[/bold cyan] {msg}")

def print_thinking():
    console.print("[dim italic]  thinking...[/dim italic]")

def print_tool_call(name: str, args: dict):
    console.print(f"  [dim]⚙  calling [yellow]{name}[/yellow]({', '.join(f'{k}={v}' for k,v in args.items())})[/dim]")

def print_agent_response(text: str):
    console.print()
    console.print(Panel(
        Markdown(text),
        title="[bold bright_blue]Assistant[/bold bright_blue]",
        border_style="bright_blue",
        padding=(1, 2)
    ))

def print_error(msg: str):
    console.print(Panel(f"[red]{msg}[/red]", border_style="red"))

def print_rule():
    console.print(Rule(style="dim"))

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a Retail AI Assistant with two roles:

1. PERSONAL SHOPPER — Help customers find the perfect product.
   - Always call search_products() with all relevant filters before recommending anything.
   - Never mention a product you haven't retrieved from a tool.
   - Explain WHY each recommendation fits the customer's constraints (size, price, stock, style).
   - Prioritise sale items and high bestseller_score.
   - Always confirm stock availability for the requested size.

2. CUSTOMER SUPPORT — Handle return and refund requests.
   - Always call evaluate_return(order_id) for any return question. Never reason about returns without it.
   - If an order ID is not found, clearly refuse and ask for the correct ID.
   - If a product does not exist, say so clearly.
   - Apply policy exactly as returned by the tool.

ANTI-HALLUCINATION RULES:
- Never invent product names, prices, order details, or policy rules.
- Never answer a product or return question without first calling the appropriate tool.
- If a tool returns no results, say so honestly.

STORE RETURN POLICY (for context only — always use evaluate_return() for decisions):
{POLICY}

Respond in a warm, professional tone. Be concise but justify your reasoning clearly.
"""

# ── agent loop ────────────────────────────────────────────────────────────────

def run_agent():
    """Run the interactive assistant loop with OpenAI tool-calling."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY environment variable not set.\nRun: export OPENAI_API_KEY=your_key_here")
        return

    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    log_event("info", "agent_started")

    print_welcome()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("exit", "quit", "bye"):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        print_thinking()

        # agentic loop — keep calling until no more tool calls
        while True:
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,  # swap to gpt-4o for maximum reasoning depth
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )
            except Exception as exc:
                log_event("error", "openai_chat_completion_failed", error=str(exc))
                print_error("I ran into an issue contacting the AI service. Please try again.")
                break

            msg = response.choices[0].message

            # no tool calls — final response
            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content})
                print_agent_response(msg.content)
                break

            # handle tool calls
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    log_event("error", "tool_argument_parse_failed", tool=name, arguments=tc.function.arguments, error=str(exc))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unable to parse arguments for tool '{name}'."})
                    })
                    continue
                if not isinstance(args, dict):
                    log_event("error", "tool_argument_invalid_type", tool=name, arg_type=type(args).__name__)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Arguments for tool '{name}' must be an object."})
                    })
                    continue
                print_tool_call(name, args)

                result = dispatch_tool(name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        print_rule()

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()
