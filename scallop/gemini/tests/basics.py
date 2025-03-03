import unittest

import scallopy
import scallop_gemini 

class TestBasics(unittest.TestCase):
  def test_ff_gemini(self):

    ctx = scallopy.Context()

    plugin_registry = scallopy.PluginRegistry(load_stdlib=True)
    plugin = scallop_gemini.ScallopGeminiPlugin()
    plugin_registry.load_plugin(plugin)
    plugin_registry.configure()
    plugin_registry.load_into_ctx(ctx)

    ctx.add_program("""
      rel questions = {
        (1, "what is the height of highest mountain in the world?"),
        (2, "are cats larger than dogs?"),
      }

      rel answer(id, $gemini(x)) = questions(id, x)

      query answer
    """)
    ctx.run()

    result = list(ctx.relation("answer"))
    print("Result: ", result)
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0][0], 1)
    self.assertEqual(result[1][0], 2)
