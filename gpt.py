from prompts_generic import MAIN_PROMPT
import sys
from datetime import date, timedelta
import os
import nltk
import tiktoken

def generate_code_from_gpt(gpt_model, client, prompt, step, config_id, count, role, messages=[]):
    message = {"role":role, "content":prompt}
    messages.append(message)
    check_token_length(messages, gpt_model, role, prompt)
    response = client.chat.completions.create(
                                    model=gpt_model,
                                    messages=messages,
                                    temperature=0,
                                    max_tokens=769,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0
                                )
    
    new_output = ""
    date_today = date.today()
    os.makedirs("/home/rajeshgayathri2003/GPT-fabric-folding/logs_code/"+str(date_today), exist_ok=True)
    file = "/home/rajeshgayathri2003/GPT-fabric-folding/logs_code/{}/log_{}_{}.txt".format(str(date_today), config_id, step)
    content = response.choices[0].message.content
    
    mode = 'w' if count == 0 else 'a'
    sys.stdout = open(file, mode)
    print("Printing for config {} step {} {}".format(config_id, step, str(date_today)))
    print(content)
    new_output+=content
    messages.append({"role":"assistant", "content":new_output})
    
    return content

def check_token_length(messages, gpt_model, role, prompt, chunk_size = 8000):
    tokenizer = tiktoken.encoding_for_model(gpt_model)
    
    length = 0
    
    for message in messages:
        prompt = message["content"]
        token = tokenizer.encode(prompt)
        length+=len(token)
    
    if length > chunk_size:
        print(length)
        messages = [{"role":role, "content":prompt}]
    
        
        
        
        