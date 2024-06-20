from prompts_generic import MAIN_PROMPT
import sys

def generate_code_from_gpt(gpt_model, client, prompt, step, config_id, count, role, messages=[]):
    message = {"role":role, "content":prompt}
    messages.append(message)
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
    file = "/home/rajeshgayathri2003/GPT-fabric-folding/log_{}_{}.txt".format(step, config_id)
    content = response.choices[0].message.content
    
    mode = 'w' if count == 0 else 'a'
    sys.stdout = open(file, mode)
    print("Printing for config {} step {}".format(config_id, step))
    print(content)
    new_output+=content
    messages.append({"role":"assistant", "content":new_output})
    
    return content