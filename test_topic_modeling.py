import unittest

from prep_data import Data
from topic_modeling import (
    tf_idf
)
from CONSTANTS import (
    MAX_DF,
    MIN_DF,
    NO_OF_CLUSTERS,
    NO_OF_MEMBERS
)


class Test_topic_modeling(unittest.TestCase):

    def setUp(self):
        self.data = Data()

    def test_clean_data(self):
        print("\n\nCleaning data...")
        cleaned_propositions = self.data.clean()
        print(f"\n\
            Original proposition:\n\
            -->    {self.data.propositions[-1]}\n\
            Cleaned proposition:\n\
            -->    {cleaned_propositions[-1]}"
        )

    def test_cluster(self):
        print("\n\nExecuting full TF-IDF analysis...")
        cleaned_propositions = self.data.clean()
        result = tf_idf(cleaned_propositions, MAX_DF, MIN_DF, NO_OF_CLUSTERS, NO_OF_MEMBERS)
        self.assertEqual(len(result), NO_OF_CLUSTERS)
        

if __name__ == "__main__":
    unittest.main()
