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
    query = sys.argv[1]

    conversation_key = ""
    if len(sys.argv)>=3:
        conversation_key = sys.argv[2]
        history = loadHistory(conversation_key)

    response, history = model.stream_chat(tokenizer, query, history=history)
    # print(history)

    if conversation_key!="":
        saveHistory(conversation_key,history)

    # os.system("clear")
    print("output:"+response)


if __name__ == "__main__":
    main()
