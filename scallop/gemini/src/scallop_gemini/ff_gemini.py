from google import genai
import scallopy
import os

from . import ScallopGeminiPlugin

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def get_gemini(plugin: ScallopGeminiPlugin):
  # For memoization
  STORAGE = {}

  @scallopy.foreign_function
  def gemini(prompt: str) -> str:
    if prompt in STORAGE:
      return STORAGE[prompt]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Add performed requests
      plugin.increment_num_performed_request()
      # response = openai.ChatCompletion.create(
      #   model=plugin.model(),
      #   messages=[{"role": "user", "content": prompt}],
      #   temperature=plugin.temperature(),
      # )
      # print("Inside ff")
      # print("prompt :", prompt)
      
      response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
      )

      result = response.candidates[0].content.parts[0].text
      # print("result :", result, "\n\n\n")
      # Store in the storage
      STORAGE[prompt] = result
      return result

  return gemini