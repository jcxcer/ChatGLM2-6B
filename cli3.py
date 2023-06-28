import os
from transformers import AutoTokenizer, AutoModel
import sys
import json
import requests
import urllib.parse

tokenizer = AutoTokenizer.from_pretrained("/var/pyproj/ChatGLM2-6B/models/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/var/pyproj/ChatGLM2-6B/models/chatglm2-6b", trust_remote_code=True).cuda()
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
    # url = urllib.parse.unquote(sys.argv[2])
    url = "http://192.168.240.53:9999/api/bigdata/oms?record_ids=126347,126348,126349,126350,126351,126352,126353,126354,126355,126356,126357,126358,126359,126360,126361,126362,126363&module_id=1347"
    # url = "http://192.168.240.53:9999/api/bigdata/oms?record_ids=5032071&module_id=831"
    history = loadHistory(url)
    print(history)
    real_query = ""
    for h in history:
        real_query += h[0] + "\n\n"

    real_query += query

    print(real_query)
    response, history = model.chat(tokenizer, real_query, history=[], max_length=8192, top_p=0.8, temperature=0.8)

    # print(history)

    # os.system("clear")
    print("output:" + response)


if __name__ == "__main__":
    main()
