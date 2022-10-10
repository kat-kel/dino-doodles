import csv
import re
import string

from nltk.corpus import stopwords

from CONSTANTS import DATA_FILE


class Data:
    
    def __init__(self):
        self.propositions = self.parse_propositions(DATA_FILE)

    def parse_propositions(self, data_file):
        with open(data_file, "r") as f:
            reader = csv.DictReader(f)
            return [row["Proposition"] for row in reader]
    
    def clean(self):
        result = []
        for proposition in self.propositions:
            proposition = proposition.lower()
            bag_of_words = [word for word in re.split(r"(\s|'|’)", proposition) if word!=" "]
            bag_of_words = self.remove_stops(bag_of_words)
            #bag_of_words = self.remove_ilfaut(bag_of_words)
            bag_of_words = self.remove_punctuation(bag_of_words)
            cleaned_proposition = " ".join(bag_of_words)
            result.append(cleaned_proposition)
        return result

    def remove_stops(self, bag_of_words):
        stops = stopwords.words("french")
        return [word for word in bag_of_words if word not in stops]
    
    def remove_punctuation(self, bag_of_words):
        result = []
        punctuation = string.punctuation+"’"
        for word in bag_of_words:
            if len(word)>1 and word[-1] in punctuation:
                result.append(word[:-1])
            elif not any(word == p for p in punctuation):
                result.append(word)
        return result

    def remove_ilfaut(self, bag_of_words):
        if bag_of_words[0] == "il" and bag_of_words[1] == "faut":
            return bag_of_words[2:]
        else:
            return bag_of_words



