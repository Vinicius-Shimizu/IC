import json
import re

# Simulated output string from your program
output_text = """
```json
{
  "label": "anniversary",
  "date": "01/05/2018"
}
```

```json
{
  "goal": "10-days-ago"
}
```

```json
{
  "earlier_date": "10-days-ago",
  "later_date": "today",
  "diff": "10 days"
}
```
"""

# Split the input by the '```json' separator and parse each JSON block
json_strings = [block.strip() for block in output_text.strip().split('```') if block.strip()]
keys_dict = {}

for json_str in json_strings: 
    try: # Remove "json" if it exists at the start of the string 
        if json_str.startswith("json"): json_str = json_str[4:].strip() # Remove the 'json' and any leading spaces
        # Parse the cleaned JSON string
        parsed_json = json.loads(json_str)

        # Update the dictionary with keys and their corresponding values
        keys_dict.update(parsed_json)
    except json.JSONDecodeError:
        pass


# print("json_strings:")
# print(json_strings[0])
# print(json_strings[1])
# print(json_strings[2])

print(keys_dict)
args = ["label", "date"]
response_json = {}
for key in args:
    response_json[key] = keys_dict[key]
print(response_json)