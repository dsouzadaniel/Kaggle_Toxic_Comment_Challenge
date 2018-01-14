########################################
## Author : Daniel Dsouza
## Title : Kaggle Toxic Comment Classification Challenge
########################################

## Libraries
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
from tqdm import trange

## File Paths
train_data_path = './train.csv'
test_data_path = './test.csv'

# Use Pandas to read it as DataFrames
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)


# Split the comments up into their individual categories 

#Toxic Comments
toxic_yes = train_data.loc[train_data['toxic']==1,:]
toxic_no = train_data.loc[train_data['toxic']==0,:]
#Severe_Toxic Comments
sev_toxic_yes = train_data.loc[train_data['severe_toxic']==1,:]
sev_toxic_no = train_data.loc[train_data['severe_toxic']==0,:]
#Obscene Comments
obscene_yes = train_data.loc[train_data['obscene']==1,:]
obscene_no = train_data.loc[train_data['obscene']==0,:]
#Threat Comments
threat_yes = train_data.loc[train_data['threat']==1,:]
threat_no = train_data.loc[train_data['threat']==0,:]
#Insult Comments
insult_yes = train_data.loc[train_data['insult']==1,:]
insult_no = train_data.loc[train_data['insult']==0,:]
#Identity Hate Comments
id_hate_yes = train_data.loc[train_data['identity_hate']==1,:]
id_hate_no = train_data.loc[train_data['identity_hate']==0,:]


# List the number of comments in each category 

#Toxic Comments
print("Total Number of Toxic Comments Data : \n\t Toxic : {0}\t Non-Toxic : {1} \n".format(len(toxic_yes),len(toxic_no)))
#Severe_Toxic Comments
print("Total Number of Severe_Toxic Comments Data : \n\t Severe_Toxic : {0}\t Non-Severe_Toxic : {1} \n".format(len(sev_toxic_yes),len(sev_toxic_no)))
#Obscene Comments
print("Total Number of Obscene Comments Data : \n\t Obscene : {0}\t Non-Obscene : {1} \n".format(len(obscene_yes),len(obscene_no)))
#Threat Comments
print("Total Number of Threat Comments Data : \n\t Threat : {0}\t Non-Threat : {1} \n".format(len(threat_yes),len(threat_no)))
#Insult Comments
print("Total Number of Insult Comments Data : \n\t Insult : {0}\t Non-Insult : {1} \n".format(len(insult_yes),len(insult_no)))
#Identity_Hate Comments
print("Total Number of Identity_Hate Comments Data : \n\t Identity_Hate : {0}\t Non-Identity_Hate : {1} \n".format(len(id_hate_yes),len(id_hate_no)))


# Regex Code to clean up internet comments
regex_cleanup = {r'(.)\1{3,}':r'\1',
        r'!!+':'! ',
        r'https?([^ ]+)':'',
        r'[a-zA-Z]{15,}':'',
        r'\n':' ',
        r'""+':'"',
        r'\\+':' ',
#         r'\.\.+':'.',
        r'(\.){2}':' ',
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}':'',
        r'\.\d{,2}':'',
        r'([a-zA-Z]{1,})+([0-9]{1,})':r'\1',
        r'(..)\1{2}':'$$$',
#         r'[\s]{2,}':' ',
#         r'\d{1,}':''
          }

# Function to remove punctuations in text
def remove_puncts(tex):
    for i in tex:
        for j in string.punctuation:
            if j in tex:
                tex = tex.replace(j,'')
    return tex

# Main cleaning function.
def clean_up(text):
    for i,j in regex_cleanup.items():
#         print(i,j)
#         print(text)
        temp = re.sub(i,j,text)
        text = temp
    clean_text = remove_puncts(text)
    return clean_text.lower()

# List of texts extension for clean_up
def clean_up_all(texts):
#     return [clean_up(text) for text in texts]
    cleaned_texts = []
    for i in trange(len(texts)):
#         print(i)
        uniq_words = set(nltk.word_tokenize(texts[i]))
        if len(uniq_words) < 10 :
            current_text = ' '.join(list(uniq_words))
        else:
            current_text = texts[i]
        cleaned_texts.append(clean_up(current_text))
    return cleaned_texts


    
