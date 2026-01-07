import requests, os
key = open('eleven_key.txt').read().strip()
url = 'https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL'
headers = {'xi-api-key': key, 'Content-Type': 'application/json'}
text = 'Arthur was just a lonely baker looking for a sign. He didn\'t expect the sign to come with raisins and a buttery attitude. From the moment he pulled it from the oven, he knew this scone was different. It didn\'t just smell good; it had opinions on his dating life. This Christmas, fall in love with something you can also eat. The Sentient Scone.'
data = {'text': text, 'model_id': 'eleven_multilingual_v2'}
res = requests.post(url, json=data, headers=headers)
print(res.status_code)
print(res.text)
