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
        print("\n\nSetting up the data...")
        self.configure_data = Data()

        print(self.configure_data.dataset[-1])

    def test_clean_data(self):
        print("\n\nCleaning data...")
        original_propositions = [d["Proposition"] for d in self.configure_data.dataset]
        cleaned_propositions = [d["Cleaned Proposition"] for d in self.configure_data.clean()]
        print(f"\n\
            Original proposition:\n\
            -->    {original_propositions[-1]}\n\
            Cleaned proposition:\n\
            -->    {cleaned_propositions[-1]}"
        )

    def test_cluster(self):
        print("\n\nExecuting full TF-IDF analysis...")
        cleaned_propositions = [d["Cleaned Proposition"] for d in self.configure_data.clean()]
        result = tf_idf(cleaned_propositions, MAX_DF, MIN_DF, NO_OF_CLUSTERS, NO_OF_MEMBERS)
        self.assertEqual(len(result), NO_OF_CLUSTERS)
        print(f"\n\
            Cluster 1:\n\
            -->    {result[0]}\n\
            "
        )
        

if __name__ == "__main__":
    unittest.main()
