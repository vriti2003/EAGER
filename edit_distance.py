# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ### Data Preparation

# %% [markdown]
# #### Hindi

# %%
df=pd.read_csv('data.csv')

# %%
df

# %%
def editDistRec(s1, s2, m, n):

    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # If last characters of two strings are same, nothing
    # much to do. Get the count for
    # remaining strings.
    if s1[m - 1] == s2[n - 1]:
        return editDistRec(s1, s2, m - 1, n - 1)

    # If last characters are not same, consider all three
    # operations on last character of first string,
    # recursively compute minimum cost for all three
    # operations and take minimum of three values.
    return 1 + min(editDistRec(s1, s2, m, n - 1),
                   editDistRec(s1, s2, m - 1, n),
                   editDistRec(s1, s2, m - 1, n - 1))

# Wrapper function to initiate
# the recursive calculation
def editDistance(s1, s2):
    return editDistRec(s1, s2, len(s1), len(s2))


# if __name__ == "__main__":
#     s1 = "abcd"
#     s2 = "bcfe"

#     print(editDistance(s1, s2))

# %%
len(df)

# %%
dic={}
for i in range(len(df)):
    dist=[]
    for j in range(len(df)):
        s1=df['Hindi'].iloc[i]
        s2=df['Sanskrit'].iloc[j]
        d=editDistance(s1,s2)
        dist.append(d)
    dic.update({df['Hindi'].iloc[i]:dist})
    

# %%
dic

# %%
final_dic={}
i=0
for keys,value in dic.items():
    lst=[]
    one=np.inf
    two=np.inf
    three=np.inf
    four=np.inf
    onei=-1
    twoi=-1
    threei=-1
    fouri=-1
    for j in range(len(value)):
        if value[j] <= one:
            four=three
            fouri=threei
            three=two
            threei=twoi
            two=one
            twoi=onei
            one=value[j]
            onei=j
        elif value[j] > one and value[j]<=two:
            four=three
            fouri=threei
            three=two
            threei=twoi
            two=value[j]
            twoi=j
        elif value[j] > two and value[j]<=three:
            four=three
            fouri=threei
            three=value[j]
            threei=j
        elif value[j] > three and value[j]<=four:
            four=value[j]
            fouri=j
    lst.append(df['Sanskrit'].iloc[onei])
    lst.append(df['Sanskrit'].iloc[twoi])
    lst.append(df['Sanskrit'].iloc[threei])
    lst.append(df['Sanskrit'].iloc[i])
    final_dic.update({"मान लो तुम्हें संस्कृत नहीं आती, दिए गए विकल्पों में से कौन सा शब्द "+keys+" के सबसे समान है?":lst})
    i+=1

# %%
final_dic

# %%
import csv
# Specify the CSV file name
csv_file = 'file.csv'

# Writing to CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
	
    # Write data
    for key, value in final_dic.items():
        writer.writerow([key, value])

print(f"Dictionary saved to {csv_file}")

# %%


# %% [markdown]
# #### Sanskrit

# %%
#### Hindi
df=pd.read_csv('data.csv')
df
def editDistRec(s1, s2, m, n):

    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # If last characters of two strings are same, nothing
    # much to do. Get the count for
    # remaining strings.
    if s1[m - 1] == s2[n - 1]:
        return editDistRec(s1, s2, m - 1, n - 1)

    # If last characters are not same, consider all three
    # operations on last character of first string,
    # recursively compute minimum cost for all three
    # operations and take minimum of three values.
    return 1 + min(editDistRec(s1, s2, m, n - 1),
                   editDistRec(s1, s2, m - 1, n),
                   editDistRec(s1, s2, m - 1, n - 1))

# Wrapper function to initiate
# the recursive calculation
def editDistance(s1, s2):
    return editDistRec(s1, s2, len(s1), len(s2))


# %%

len(df)
dic_={}
for i in range(len(df)):
    dist=[]
    for j in range(len(df)):
        s1=df['Hindi'].iloc[i]
        s2=df['Sanskrit'].iloc[j]
        d=editDistance(s1,s2)
        dist.append(d)
    dic_.update({df['Sanskrit'].iloc[i]:dist})
dic_


# %%
final_dic_={}
i=0
for keys,value in dic_.items():
    lst=[]
    one=np.inf
    two=np.inf
    three=np.inf
    four=np.inf
    onei=-1
    twoi=-1
    threei=-1
    fouri=-1
    for j in range(len(value)):
        if value[j] <= one:
            four=three
            fouri=threei
            three=two
            threei=twoi
            two=one
            twoi=onei
            one=value[j]
            onei=j
        elif value[j] > one and value[j]<=two:
            four=three
            fouri=threei
            three=two
            threei=twoi
            two=value[j]
            twoi=j
        elif value[j] > two and value[j]<=three:
            four=three
            fouri=threei
            three=value[j]
            threei=j
        elif value[j] > three and value[j]<=four:
            four=value[j]
            fouri=j
    lst.append(df['Hindi'].iloc[onei])
    lst.append(df['Hindi'].iloc[twoi])
    lst.append(df['Hindi'].iloc[threei])
    lst.append(df['Hindi'].iloc[i])
    final_dic_.update({"कल्पयतु यत् भवन्तः कस्यापि हिन्दीभाषां न जानन्ति, दत्तविकल्पेषु कः शब्दः "+keys:lst})
    i+=1

# %%
final_dic_

# %%
import csv
# Specify the CSV file name
csv_file = 'file_.csv'

# Writing to CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
	
    # Write data
    for key, value in final_dic_.items():
        writer.writerow([key, value])

print(f"Dictionary saved to {csv_file}")

# %% [markdown]
# ### Mistral inferencing

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# %%

messages = [
    {"role": "user", "content": "What is your favourite condiment?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=10, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


# %%
# from transformers import pipeline
# import torch

# messages = [
#     {"role": "user", "content": "Give me 5 non-formal ways to say 'See you later' in French."},
# ]

# chatbot = pipeline("text-generation", model="mistralai/Mistral-Small-24B-Instruct-2501", max_new_tokens=256, torch_dtype=torch.bfloat16)
# chatbot(messages)


# %% [markdown]
# ### Llama inferencing

# %%
hindi = pd.read_csv('file.csv')

# %%
prompt = list(hindi['Key'].astype('str'))
options = list(hindi['Value'].astype('str'))

# %%
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

# %%
result = []
inp = []
for p, o in zip(prompt, options):
    in_ = p + o
    inp.append(in_)
    result.append(pipeline(
    in_, 
    max_new_tokens=50, 
    return_full_text=False 
)[0]['generated_text'])

# %%
percentile_list = pd.DataFrame(np.column_stack([inp,result]), 
                               columns=['Input', 'Result'])
percentile_list.to_csv("hindi_res.csv")

# %% [markdown]
# ### Llama inferencing Sanskrit

# %%
sans = pd.read_csv('file_.csv')
prompt = list(sans['Key'].astype('str'))
options = list(sans['Value'].astype('str'))

result_ = []
inp_ = []
for p, o in zip(prompt, options):
    in_ = p + o
    inp_.append(in_)
    result_.append(pipeline(
    in_, 
    max_new_tokens=50, 
    return_full_text=False 
)[0]['generated_text'])

# %%
percentile_list_ = pd.DataFrame(np.column_stack([inp_,result_]), 
                               columns=['Input', 'Result'])
percentile_list_.to_csv("sanskrit_res.csv")

# %% [markdown]
# ### ChatGPT Inferencing

# %%
from openai import OpenAI
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-3.5-turbo-0125",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)

# %%



