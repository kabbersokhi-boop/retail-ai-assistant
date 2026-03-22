# Retail AI Assistant
CLI agent for a retail fashion store. Handles product recommendations and return requests using OpenAI function calling. 
Tools are called dynamically based on context, no hardcoded logic.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/kabbersokhi-boop/retail-ai-assistant.git
cd retail-ai-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the assistant
```bash
python agent.py
```

---

## Usage

Just type naturally. The agent handles both shopping and support:

**Personal Shopper:**
```
You: Find me a lace bridal dress under $400 in size 10
```

**Customer Support:**
```
You: I want to return order O0043
You: Can I return order O0099?  (invalid order - edge case)
```

Type `exit` or `quit` to stop.

---

## Tools

| Tool | Description |
|------|-------------|
| `search_products(filters)` | Filter products by tags, price, size, sale status, stock |
| `get_product(product_id)` | Retrieve full details of a single product |
| `get_order(order_id)` | Look up an order by ID |
| `evaluate_return(order_id)` | Apply store policy to determine return eligibility |

The model calls these tools dynamically - it never hardcodes answers.

---

## Data

- `product_inventory.csv` - 100 products with tags, sizes, stock, sale/clearance flags
- `orders.csv` - 100 orders with order date, product, size, price paid
- `policy.txt` - store return and exchange policy

---

## Requirements

- Python 3.9+
- OpenAI API key (GPT-4o-mini)
