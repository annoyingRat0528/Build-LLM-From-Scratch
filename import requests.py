import requests
response = requests.get('https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe', verify=False)
print(response.status_code)