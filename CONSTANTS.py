import os


controlled_data = os.path.join("data","private.csv")
public_data = os.path.join("data","example.csv")

if os.path.isfile(controlled_data):
    DATA_FILE = controlled_data
else:
    DATA_FILE = public_data


NUMBER_OF_CLUSTERS = 7