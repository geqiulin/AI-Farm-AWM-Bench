import unittest

from benchmark import make_logiq_item, parse_choice, response_text


class BenchmarkTests(unittest.TestCase):
    def test_choice_parser(self):
        self.assertEqual(parse_choice("B"), "B")
        self.assertEqual(parse_choice("Answer: C"), "C")
        self.assertEqual(parse_choice("(D)."), "D")
        self.assertIsNone(parse_choice("I do not know"))

    def test_deterministic_shuffle_and_hidden_gold(self):
        row = {
            "problem_id": "p1",
            "category": "test",
            "subcategory": "test",
            "question": "Which option?",
            "knowledge_context": "(a) -[REL]-> (b)",
            "answer_truth": "truth",
            "answer_option_A": "wrong one",
            "answer_option_B": "wrong two",
            "answer_option_C": "wrong three",
        }
        first = make_logiq_item(row, 7)
        second = make_logiq_item(row, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["options"][first["gold"]], "truth")
        self.assertNotIn("gold", first["prompt"].lower())

    def test_responses_api_text_extraction(self):
        payload = {"output": [{"content": [{"type": "output_text", "text": "A"}]}]}
        self.assertEqual(response_text(payload), "A")


if __name__ == "__main__":
    unittest.main()
