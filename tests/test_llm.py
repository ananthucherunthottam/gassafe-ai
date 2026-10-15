
import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock


LLM_PATH = Path(__file__).resolve().parents[1] / "llm.py"

NOT_FOUND = (
    "I could not find this information "
    "in the provided documents."
)


def load_generate_answer():
    """
    Load generate_answer directly from llm.py
    without importing llama-cpp-python or
    downloading the Qwen model.
    """
    source = LLM_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_answer"
    ]

    if len(functions) != 1:
        raise RuntimeError(
            "Expected exactly one generate_answer "
            "function in llm.py."
        )

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    return compile(
        module,
        str(LLM_PATH),
        "exec",
    )


class TestLLMGeneration(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()

        self.model.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Inspect the pipeline for corrosion."
                        )
                    }
                }
            ]
        }

        self.load_model = MagicMock(
            return_value=self.model
        )

        self.namespace = {
            "load_model": self.load_model,
            "MAX_NEW_TOKENS": 300,
        }

        exec(
            load_generate_answer(),
            self.namespace,
        )

        self.generate_answer = self.namespace[
            "generate_answer"
        ]

    def test_empty_context(self):
        """Empty context must not invoke Qwen."""
        result = self.generate_answer(
            "What is the inspection procedure?",
            "",
        )

        self.assertEqual(result, NOT_FOUND)
        self.load_model.assert_not_called()

    def test_whitespace_context(self):
        """Whitespace-only context must not invoke Qwen."""
        result = self.generate_answer(
            "What is the inspection procedure?",
            "   \n   ",
        )

        self.assertEqual(result, NOT_FOUND)
        self.load_model.assert_not_called()

    def test_valid_context_generates_answer(self):
        """Valid context should invoke the model."""
        result = self.generate_answer(
            "What should be inspected?",
            "Inspect gas pipelines for corrosion.",
        )

        self.assertEqual(
            result,
            "Inspect the pipeline for corrosion.",
        )

        self.load_model.assert_called_once()
        self.model.create_chat_completion.assert_called_once()

    def test_prompt_contains_question_and_context(self):
        """The model prompt must contain retrieved evidence."""
        question = "What is the inspection procedure?"
        context = "Inspect gas pipelines for corrosion."

        self.generate_answer(question, context)

        call_kwargs = (
            self.model.create_chat_completion.call_args.kwargs
        )

        messages = call_kwargs["messages"]

        self.assertEqual(
            messages[0]["role"],
            "system",
        )

        self.assertEqual(
            messages[1]["role"],
            "user",
        )

        self.assertIn(
            question,
            messages[1]["content"],
        )

        self.assertIn(
            context,
            messages[1]["content"],
        )

        self.assertIn(
            "Document context:",
            messages[1]["content"],
        )

    def test_safety_instructions_present(self):
        """The system prompt must contain safety restrictions."""
        self.generate_answer(
            "What are the safety requirements?",
            "Follow the approved safety manual.",
        )

        messages = (
            self.model.create_chat_completion
            .call_args.kwargs["messages"]
        )

        system_prompt = messages[0]["content"].lower()

        self.assertIn(
            "only using the provided document context",
            system_prompt,
        )

        self.assertIn(
            "do not invent safety procedures",
            system_prompt,
        )

        self.assertIn(
            "active gas leak",
            system_prompt,
        )

        self.assertIn(
            "emergency services",
            system_prompt,
        )

    def test_model_generation_settings(self):
        """Generation must use configured token settings."""
        self.generate_answer(
            "What should be inspected?",
            "Inspect the pipeline for corrosion.",
        )

        call_kwargs = (
            self.model.create_chat_completion.call_args.kwargs
        )

        self.assertEqual(
            call_kwargs["max_tokens"],
            300,
        )

        self.assertEqual(
            call_kwargs["temperature"],
            0.1,
        )

    def test_empty_model_response(self):
        """Empty model output should return an empty string."""
        self.model.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None
                    }
                }
            ]
        }

        result = self.generate_answer(
            "What should be inspected?",
            "Inspect gas pipelines for corrosion.",
        )

        self.assertEqual(result, "")

    def test_model_generation_failure(self):
        """
        Model errors should propagate to rag.py,
        which handles generation failures.
        """
        self.model.create_chat_completion.side_effect = (
            RuntimeError("Simulated Qwen failure")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Simulated Qwen failure",
        ):
            self.generate_answer(
                "What should be inspected?",
                "Inspect gas pipelines for corrosion.",
            )


if __name__ == "__main__":
    unittest.main()
  
