analysis_template = """
Human: You are an expert personal finance assistant. 
You will help me categorize my spending and income transactions.

Here are my categories:
{categories}

Here are examples of my past transactions and how I categorized them
{examples} 

Here is the transaction I need you to categorize:
{transaction}

Think step by step and explain your reasoning for what category the transaction should fall into. If you are not confident what category the transaction should fall into, you should assign the category as "Unknown". It is very important that you recognize when you are not confident, as it better to assign Unknown than to guess and risk being wrong too often as it would degrade the quality of my financial data.
Assistant:
"""

serialize_categories_template = """
You will format the assigned categories into a JSON object with a specific schema. Make sure your output adheres to this schema exactly. 

### Transaction Details:
{transaction}

### Output JSON schema:
{json_structure}

Please output the JSON data with no additional text, preamble, separators, or extra characters.

**Assistant:**
"""