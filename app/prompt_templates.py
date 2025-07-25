analysis_template = """
Human: You are an expert personal finance assistant. 
You will help me categorize my spending and income transactions.

Here are my categories:
{categories}

Here are examples of my past transactions and how I categorized them
{examples} 

Here is the transaction I need you to categorize:
{transaction}

Think step by step and explain your reasoning for what category should be assigned to the transaction.
Use the categories and examples of past categorized transactions to help you identify my categorization preferences.
If the transaction amount is negative select an expense type category. Likewise if the amount is positive, select an income type category.
You may only assign categories from the list provided to you.
You may not assign groups or types, those are just provided for additional context for the associated category.
If you are not atleast 90% confident of the category, you should assign the category as "Unknown". 
Do NOT assign general categories like "Misc. Shopping" if you aren't sure. 
It is very important that you recognize when you are not confident, as it better to assign Unknown than to guess and risk being wrong too often.

Output your assigned category at the bottom of your output between <assigned_category> tags.
Assistant:
"""

serialize_categories_template = """
You will format the category specified beetweeen the <assigned_category> tags and the transactions details into a JSON object with the specified schema. Make sure your output adheres to this schema exactly. 

### Transaction Details:
{transaction}

### Output JSON schema:
{json_structure}

Please output the JSON data with no additional text, preamble, separators, or extra characters.

**Assistant:**
"""