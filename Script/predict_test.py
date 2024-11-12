import json, random
import requests

test_data_file = "test_players.json"

url = 'http://localhost:9696/predict'

with open(test_data_file, "r") as infile:
    test_players = json.load(infile)

r = random.randint(0, len(test_players)-1)
response = requests.post(url, json=test_players[r]).json()
print("player:")
print(test_players[r])
print("")
print("result:")
print(response)
