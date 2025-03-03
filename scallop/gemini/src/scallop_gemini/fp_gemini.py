from google import genai
import scallopy
import os

from . import ScallopGeminiPlugin

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def get_gemini(plugin: ScallopGeminiPlugin):
  # A storage for @gpt foreign predicate
  STORAGE = {}

  @scallopy.foreign_predicate
  def gemini(s: str) -> scallopy.Facts[None, str]:
    # Check if the storage already contains the response
    if s in STORAGE:
      response = STORAGE[s]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Memoize the response
      plugin.increment_num_performed_request()
      # response = openai.ChatCompletion.create(
      #   model=plugin.model(),
      #   messages=[{"role": "user", "content": s}],
      #   temperature=plugin.temperature(),
      # )
      response = client.models.generate_content(
        model=plugin.model(),
        contents=[{"role": "user", "content": s}],
      )
      STORAGE[s] = response

    # Iterate through all the choices
    for choice in response["choices"]:
      result = choice["message"]["content"].strip()
      yield (result,)

  return gemini
