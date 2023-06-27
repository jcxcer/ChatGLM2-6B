import os
from transformers import AutoTokenizer, AutoModel
import sys
import json
import requests
import urllib.parse

tokenizer = AutoTokenizer.from_pretrained("/var/pyproj/ChatGLM2-6B/models/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/var/pyproj/ChatGLM2-6B/models/chatglm2-6b", trust_remote_code=True, device='cuda')
model = model.eval()


def loadHistory(url):
    # print(url)
    result = requests.post(url, {"test": "ok"})

    # print(result.text)
    result = json.loads(result.text)

    history = result["data"]
    # print(history)

    return history


def main():
    query = sys.argv[1]
    url = urllib.parse.unquote(sys.argv[2])
    history = loadHistory(url)

    response, history = model.chat(tokenizer, query, history=history, max_length=8192, top_p=0.8, temperature=0.8)

    # print(history)

    # os.system("clear")
    print("output:" + response)


if __name__ == "__main__":
    main()
