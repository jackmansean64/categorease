"""Mock LLM response for testing without incurring API costs."""

MOCK_LLM_RESPONSE = """Let's solve this step by step:

1. Transaction Details:
- Description: "Taylor swift toronto on"
- Amount: -$73.68 (negative, so it's an expense)
- Account: Scotia Momentum VISA Infinite

2. Analyzing the Description:
- "Taylor swift" suggests this is related to a concert or entertainment event
- "toronto on" indicates the location of the event

3. Reviewing Existing Categories:
- I see relevant categories in the Entertainment group:
  - "Entertainment"
  - "Concerts and Shows"

4. Examining Past Transactions:
- I see other entertainment-related transactions like concerts and shows
- The negative amount and event-related description strongly suggest this is a concert expense

5. Confidence Assessment:
- The description clearly indicates a concert
- The category "Concerts and Shows" is a perfect match
- I am >90% confident in this categorization

6. Reasoning:
- The transaction is a negative expense for a Taylor Swift concert
- "Concerts and Shows" is the most precise and appropriate category

<assigned_category>Concerts and Shows</assigned_category>"""
