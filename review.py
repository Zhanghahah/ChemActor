import pandas as pd
import os
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import os
import requests
import base64
import pickle
import os
from tqdm import tqdm

def badcase_extraction(path):
    data = pd.read_csv(path)
    methods = ['ours', 'mistral', 't5', 'chemtrans']
    total_badcase_idxes = set()
    for method in methods:
        badcase = data[data[f'EM_{method}'] < 40]
        badcase_idx = list(badcase.index)
        total_badcase_idxes = total_badcase_idxes.union(set(badcase_idx))
        print(f'{method}: {len(badcase)} bad cases')

    print(f'Total: {len(total_badcase_idxes)} bad cases')
    badcase = data.iloc[sorted(list(total_badcase_idxes))]
    save_path = path.replace('.csv', '_badcase.csv')
    badcase.to_csv(save_path)
    return save_path
    
def create_cand_ref_pair(data, method):
    cand_ref_pairs = []
    for i in range(len(data)):
        cand = data.iloc[i][f'action_pred_{method}']
        ref = data.iloc[i]['action_labels']
        cand_ref_pairs.append([cand, ref])
    return cand_ref_pairs

def prompt_func(template):
    return lambda args: template.format(*args)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm(model_name, cand_ref_pairs):
    if 'llama3' in model_name:
        model_path = 'llama3_8b_instruct'
    elif 'mistral' in model_name:
        model_path = 'Mistral-7B-Instruct-v0.2'
    else:
        raise ValueError(f'not support loading model {model_name}')
    print(f'loading model from {model_path}')
    llm = LLM(model = model_path, trust_remote_code=True, tensor_parallel_size=1, 
            max_context_len_to_capture=1024, gpu_memory_utilization=0.8, max_num_seqs=1024)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens = 300, stop = ['!!!'])

    scoring_template = '''Please rate the following two experimental operations' similarity from the perspective of chemical experiments. Use a scoring range from 0 to 1, with intervals of 0.1. A higher score indicates smaller differences between them (1 meaning the experimental operations are identical). A lower score indicates larger differences (0 meaning there is no similarity between the experimental operations at all).\n\
    The first operation: {}\n\
    The second oepration: {}\n\
    Only output the score and the explanation with a tuple format of "('explanation',score)" without any other information and descriptions.
    '''

    scoring_func =  prompt_func(scoring_template)
    scoring_prompts = []
    for cand, ref in cand_ref_pairs:
        scoring_prompt = scoring_func([cand,ref])
        scoring_prompts.append(scoring_prompt)

    outputs = llm.generate(scoring_prompts, sampling_params)
    llm_responses = []
    for output in outputs:
        llm_responses.append(output.outputs[0].text.split('<|eot_id|>')[0].split('\n')[0])
    return llm_responses

def get_score(x):
    try:
        return float(eval(x.split("\n")[-1])[1])
    except:
        try:
            x = x.split('.')
            for i in range(len(x)-2, -1, -1):
                if x[i][-1] in ['0', '1'] and x[i+1][0] in [str(j) for j in range(10)]:
                    score = x[i][-1]+'.'+x[i+1][0]
                    assert float(score) <= 1.0 and float(score) >= 0.0
                    assert float(score) is not np.nan
                    return float(score)
                # 0
                if x[i][-1] in ['0', '1'] and x[i+1][0] not in [str(j) for j in range(10)]:
                    return float(x[i][-1])
                if '0)' in x[i+1]:
                    return 0.0
                if '1)' in x[i+1]:
                    return 1.0
            return 'Invalid'
        except:
            return 'Invalid'   

def response_process(x):
    x = x.split('```')[0]
    x = x.split('\"""')[0]
    x = x.split('<|eot_id|>')[0]
    x = x.split('\n\n')[0]
    x = x.split('#')[0]
    
    return x.strip("\n")

def get_llm_scores(path):
    results = pd.read_csv(path)
    methods = ['ours', 'mistral', 't5', 'chemtrans']
    model_names = [path.split('.')[0].split('_')[-1]]
    if model_names == ['gpt']:
        model_names = ['gpt35', 'gpt4o']

    for model_name in model_names:
        print(f'---------------------------')
        print(f'model name: {model_name}')

        invalid_idxes = set()
        for method in methods:
            results[f'{model_name}_{method}_score'] = results[f'{model_name}_{method}_response'].apply(get_score)
            invalid_idx = results[results[f'{model_name}_{method}_score'] == 'Invalid'].index
            invalid_idxes = invalid_idxes.union(set(invalid_idx))

        results = results.drop(invalid_idxes)
        print(f'{len(invalid_idxes)} responses INVALID')

        for method in methods:
            mean_score = np.mean(results[f'{model_name}_{method}_score'])
            print(f'{method} average score: {mean_score:.4f}')

    return results
    
def check_score_validity(path):
    results = get_llm_scores(path)
    methods = ['ours', 'mistral', 't5', 'chemtrans']
    model_names = [path.split('.')[0].split('_')[-1]]
    if model_names == ['gpt']:
        model_names = ['gpt35', 'gpt4o']

    for model_name in model_names:
        for method in methods:
            mean_scores = []
            print(f'--------------------{method}')
            for model_name in model_names:
                print(results[f'{model_name}_{method}_score'].unique())
    return results

def run_gpt4o(prompt, image_path = None):
    # input your openai key
    api_key = ''
    if image_path:
        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {api_key}"
        }

        payload = {
          "model": "gpt-4o",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                }
              ]
            }
          ],
          "max_tokens": 300
        }
    else:
        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {api_key}"
        }

        payload = {
          "model": "gpt-4o",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                }
              ]
            }
          ],
          "max_tokens": 500
        }


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
      return response.json()["choices"][0]["message"]["content"]
    except:
       print('Invalid response')
       return 'Invalid'
    
def run_gpt35(prompt):
    # input your openai key
    api_key = ''
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            }
          ]
        }
      ],
      "max_tokens": 500
    }


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # print(response.json())
    try:
      return response.json()["choices"][0]["message"]["content"]
    except:
       print('Invalid response')
       return 'Invalid'

def merge_results(model_names, methods, prefix_path, round_idx):
    total_results = None
    for model_name in model_names:
        path = os.path.join(prefix_path, f'bad_case_peer_review_scores_round_{round_idx}_{model_name}.csv')
        results = pd.read_csv(path)
        if total_results is None:
            total_results = results
        else:
            for method in methods:
                total_results[f'{model_name}_{method}_response'] = results[f'{model_name}_{method}_response']
                total_results[f'{model_name}_{method}_score'] = results[f'{model_name}_{method}_score']

    print(f'final result columns: {total_results.columns}')
    return total_results

def create_prompts(total_results, main_model_name, method):
    debating_template = '''The first operation: {}\n\
    The second oepration: {}\n\
    You gave the following score and reasons, where the reasons are given with a tuple format of "('explanation',score)":\n\
    The score is {} because {}.
    Other reviewers gave the following scores and reasons, where the reasons are given with a tuple format of "('explanation',score)":\n\
    The remaining first reviewer's score is {}, whose reasons are  {}\n\
    The remaining second reviewer's score is {}, whose reasons are that {}\n\
    The remaining third reviewer's score is {}, whose reasons are that {}\n\
    Based on the feedback from other reviewers, please reconsider your score and rationale. If necessary, provide the revised score and rationale; otherwise, provide the original score and explain why no revision is needed.\n\
    Only output the final score and the explanation with a tuple format of "('explanation',score)" without any other information and descriptions.
    '''
    debating_func =  prompt_func(debating_template)
    model_names = ['gpt35', 'gpt4o', 'llama3', 'mistral']
    prompts = []

    for i in tqdm(range(len(total_results))):
        label = total_results.iloc[i]['action_labels']
        pred = total_results.iloc[i][f'action_pred_{method}']
        main_response = total_results.iloc[i][f'{main_model_name}_{method}_response'].strip('`')
        main_score = total_results.iloc[i][f'{main_model_name}_{method}_score']
        other_responses = []
        other_scores = []
        for model_name in model_names:
            if model_name != main_model_name:
                other_responses.append(total_results.iloc[i][f'{model_name}_{method}_response'].strip('`'))
                other_scores.append(total_results.iloc[i][f'{model_name}_{method}_score'])
        assert len(other_responses) == 3
        args = [label, pred, main_score, main_response, 
                other_scores[0], other_responses[0], other_scores[1], other_responses[1], other_scores[2], other_responses[2]]
        prompt = debating_func(args)
        prompts.append(prompt)
    
    return prompts

def peer_review_debate_gpt(main_model_name, round_idx, data_path=None):
    model_names = ['gpt35', 'gpt4o', 'llama3', 'mistral']
    methods = ['mistral', 't5', 'ours', 'chemtrans']
    prefix_path = './peer_review_scores'
    if data_path is None:
        total_results = merge_results(model_names, methods, prefix_path, round_idx)
    else:
        total_results = pd.read_csv(data_path)

    for method in methods:
        print(f'creating prompts about METHOD {method} with main LLM {main_model_name}')
        prompts = create_prompts(total_results, main_model_name, method)
        responses = []
        for i in tqdm(range(len(prompts))):
            if main_model_name == 'gpt35':
                response = run_gpt35(prompts[i])
            elif main_model_name == 'gpt4o':
                response = run_gpt4o(prompts[i])
            responses.append(response)
            # print(response)
            # break
        
        with open(f'{prefix_path}/peer_review_scores_round_{round_idx+1}_{method}_{main_model_name}.pkl', 
                  'wb') as f:
            pickle.dump(responses, f)
        

def peer_review_debate_llm(main_model_name, round_idx, data_path=None):
    model_names = ['gpt35', 'gpt4o', 'llama3', 'mistral']
    methods = ['mistral', 't5', 'ours', 'chemtrans']
    prefix_path = './peer_review_scores'
    if data_path is None:
        total_results = merge_results(model_names, methods, prefix_path, round_idx)
    else:
        total_results = pd.read_csv(data_path)

    if 'llama3' in main_model_name:
        model_path = 'llama3_8b_instruct'
    elif 'mistral' in main_model_name:
        model_path = 'Mistral-7B-Instruct-v0.2'
    else:
        raise ValueError(f'not support loading model {main_model_name}')

    print(f'loading model from {model_path}')
    llm = LLM(model = model_path, trust_remote_code=True, tensor_parallel_size=1, 
            max_context_len_to_capture=1024, gpu_memory_utilization=0.8, max_num_seqs=1024)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens = 300, stop = ['!!!'])

    for method in methods:
        print(f'creating prompts about METHOD {method} with main LLM {main_model_name}')
        prompts = create_prompts(total_results, main_model_name, method)
        outputs = llm.generate(prompts, sampling_params)
        responses = []
        for i in range(len(outputs)):
            responses.append(outputs[i].outputs[0].text)
        with open(f'{prefix_path}/peer_review_scores_round_{round_idx+1}_{method}_{main_model_name}.pkl', 
                  'wb') as f:
            pickle.dump(responses, f)


def peer_review_debate(main_model_name, round_idx, data_path=None):
    if 'gpt' in main_model_name:
        peer_review_debate_gpt(main_model_name, round_idx, data_path)
    else:
        peer_review_debate_llm(main_model_name, round_idx, data_path)

def peer_review_combine_results(round_idx):
    model_names = ['gpt35', 'gpt4o', 'llama3', 'mistral']
    methods = ['mistral', 't5', 'ours', 'chemtrans']
    prefix_path = './peer_review_scores'

    total_results = pd.DataFrame([])
    invalid_idxes = set()
    for method in methods:
        for model_name in model_names:
            path = os.path.join(prefix_path,
                                f'peer_review_scores_round_{round_idx}_{method}_{model_name}.pkl')
            with open(path, 'rb') as f:
                result = pickle.load(f)
            total_results[f'{model_name}_{method}_response'] = result
            total_results[f'{model_name}_{method}_response'] = total_results[f'{model_name}_{method}_response'].apply(lambda x: response_process(x))
            total_results[f'{model_name}_{method}_score'] = total_results[f'{model_name}_{method}_response'].apply(get_score)
            invalid_idx = total_results[total_results[f'{model_name}_{method}_score'] == 'Invalid'].index
            invalid_idxes = invalid_idxes.union(set(invalid_idx))


    total_results = total_results.drop(invalid_idxes)

    path = os.path.join(prefix_path,
                                f'bad_case_peer_review_scores_round_0_mistral.csv')
    prediction = pd.read_csv(path)
    prediction = prediction.drop(invalid_idxes)
    
    for method in methods:
        total_results[f'action_pred_{method}'] = prediction[f'action_pred_{method}']
    total_results['action_labels'] = prediction['action_labels']
        
    print(f'{len(invalid_idxes)} responses INVALID, {len(total_results)} {len(prediction)}')
    for method in methods:
        mean_scores = []
        print(f'--------------------{method}')
        for model_name in model_names:
            assert max(total_results[f'{model_name}_{method}_score']) <= 1.0
            mean_score = np.mean(total_results[f'{model_name}_{method}_score'])
            mean_scores.append(round(mean_score, 4))
        print(mean_scores)

    print(f'saving data to peer_review_scores_round_{round_idx}.csv')
    total_results.to_csv(os.path.join(prefix_path,
                                f'peer_review_scores_round_{round_idx}.csv'))
    return total_results

def peer_review_main():
    # input result path
    path = ''
    bad_case_path = badcase_extraction(path)
    
    peer_review_debate('gpt35', 0)
    peer_review_debate('gpt4o', 0)
    peer_review_debate('mistral', 0)
    peer_review_debate('llama3', 0)
    total_results = peer_review_combine_results(1)
    peer_review_debate('gpt35', 1, data_path='./peer_review_scores/peer_review_scores_round_1.csv')
    peer_review_debate('gpt4o', 1, data_path='./peer_review_scores/peer_review_scores_round_1.csv')
    peer_review_debate('mistral', 1, data_path='./peer_review_scores/peer_review_scores_round_1.csv')
    peer_review_debate('llama3', 1, data_path='./peer_review_scores/peer_review_scores_round_1.csv')
    total_results = peer_review_combine_results(2)

if __name__ == '__main__':
    peer_review_main()
