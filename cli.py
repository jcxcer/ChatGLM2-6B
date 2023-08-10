import os
from transformers import AutoTokenizer, AutoModel
import sys
import json

tokenizer = AutoTokenizer.from_pretrained("/var/pyproj/ChatGLM2-6B/models/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/var/pyproj/ChatGLM2-6B/models/chatglm2-6b", trust_remote_code=True, device='cuda')
model = model.eval()

def saveHistory(conversation_key,history):
    with open(f"/var/pyproj/ChatGLM2-6B/history/{conversation_key}.txt","w",encoding="utf-8") as f:
        f.write(json.dumps(history,ensure_ascii=False))

def loadHistory(conversation_key):
    history = []
    if os.path.exists(f"/var/pyproj/ChatGLM2-6B/history/{conversation_key}.txt"):
        with open(f"/var/pyproj/ChatGLM2-6B/history/{conversation_key}.txt","r",encoding="utf-8") as f:
            history = f.read()
            history = json.loads(history)
    return history

def main():
    history = []
    query_file = sys.argv[1]

    if os.path.exists(f"/var/pyproj/ChatGLM2-6B/history/tmp/" + query_file):
        with open(f"/var/pyproj/ChatGLM2-6B/history/tmp/" + query_file, "r", encoding="utf-8") as f:
            query = f.read()
            query = json.loads(query)

            conversation_key = ""
            if 'key' in query:
                conversation_key = query['key']
                history = loadHistory(conversation_key)

            query_text = query['text']

            response, history = model.chat(tokenizer, query_text, history=history)
            # print(history)

            if conversation_key != "":
                saveHistory(conversation_key, history)

            # os.system("clear")
            print("output:" + response)


if __name__ == "__main__":
    main()
