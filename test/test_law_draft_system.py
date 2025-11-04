import unittest
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from law_analyzer import LawDraftSystem

class TestLawDraftSystem(unittest.TestCase):
    def setUp(self):
        self.system = LawDraftSystem()
        self.input_dir = Path("input_drafts")

    def test_import_and_topic_article_mapping(self):
        # Test importing a draft and verify topic count equals article count
        draft_files = list(self.input_dir.glob("*.txt"))
        self.assertTrue(len(draft_files) > 0, "No draft files found in input_drafts")

        draft_id = self.system.import_draft(str(draft_files[0]))
        draft = self.system.drafts[draft_id]

        # Check number of articles
        article_count = len(draft['articles'])
        self.assertGreater(article_count, 0, "Draft has no articles")

        # Analyze topics
        topics = self.system._analyze_topics(draft_id)
        self.assertEqual(len(topics), article_count, "Number of topics does not match number of articles")

        # Check that each topic's raw_content matches the article content
        for topic in topics:
            article_title = topic['article']
            self.assertIn(article_title, draft['articles'], f"Article title {article_title} missing in draft")
            self.assertEqual(topic['raw_content'], draft['articles'][article_title], "Topic raw_content mismatch")

    def test_no_overgeneralization_in_topics(self):
        # Import draft and analyze topics
        draft_files = list(self.input_dir.glob("*.txt"))
        draft_id = self.system.import_draft(str(draft_files[0]))
        topics = self.system._analyze_topics(draft_id)

        # Check that keywords are not empty and are relevant (basic check)
        for topic in topics:
            self.assertTrue(len(topic['keywords']) > 0, "Topic keywords are empty")
            # Keywords should be strings
            for kw in topic['keywords']:
                self.assertIsInstance(kw, str)

if __name__ == "__main__":
    unittest.main()
