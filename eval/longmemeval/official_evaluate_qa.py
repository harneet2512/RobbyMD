import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError,
                                    openai.APITimeoutError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python evaluate_qa.py metric_model hyp_file ref_file')
        exit()

    metric_model_short = sys.argv[1]
    hyp_file = sys.argv[2]
    ref_file = sys.argv[3]
    verbose = os.getenv("LME_JUDGE_VERBOSE", "0") == "1"
    workers = int(os.getenv("LME_JUDGE_WORKERS", "1") or "1")
    
    result_file = hyp_file + '.eval-results-{}'.format(metric_model_short)

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        exit()
    metric_model, metric_model_source = model_zoo[metric_model_short]
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        from eval._openai_client import make_openai_client
        metric_client, metric_model = make_openai_client("longmemeval_judge")
        metric_model_source = "azure"
        label_model = model_zoo[metric_model_short][0]
    elif metric_model_source == 'openai':
        openai.organization = os.getenv('OPENAI_ORGANIZATION')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_api_base = None
        label_model = metric_model
    else:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"
        label_model = metric_model

    if metric_model_source != "azure":
        metric_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    try:
        hypotheses = [json.loads(line) for line in open(hyp_file, encoding="utf-8").readlines()]
    except:
        hypotheses = json.load(open(hyp_file, encoding="utf-8"))
    try:
        references = json.load(open(ref_file, encoding="utf-8"))
    except:
        references = [json.loads(line) for line in open(ref_file, encoding="utf-8").readlines()]
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}

    existing_logs = []
    existing_qids = set()
    if os.path.exists(result_file):
        for line in open(result_file, encoding="utf-8").readlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "autoeval_label" not in entry:
                continue
            existing_logs.append(entry)
            existing_qids.add(entry.get("question_id"))
            if entry.get("question_id") in qid2qtype:
                qtype2acc[qid2qtype[entry["question_id"]]].append(
                    1 if entry["autoeval_label"]["label"] else 0
                )

    pending = []
    for entry in hypotheses:
        if entry['question_id'] not in qid2qtype:
            print('Warning: skipping {} as it is not in reference data.'.format(entry['question_id']))
            continue
        if entry['question_id'] in existing_qids:
            continue
        pending.append(entry)

    def judge(entry):
        qtype = qid2qtype[entry['question_id']]
        q = qid2qdata[entry['question_id']]['question']
        ans = qid2qdata[entry['question_id']]['answer']
        hyp = entry['hypothesis']

        prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
        kwargs = {
            'model': metric_model,
            'messages': [
                {"role": "user", "content": prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 10,
            'timeout': 60,
        }
        completion = chat_completions_with_backoff(metric_client, **kwargs)
        eval_response = completion.choices[0].message.content.strip()
        label = 'yes' in eval_response.lower()
        entry = dict(entry)
        entry['autoeval_label'] = {
            'model': label_model,
            'label': label
        }
        if verbose:
            print(json.dumps({
                'question': q,
                'answer': ans,
                'hypothesis': hyp,
                'autoeval_label': label
            }, indent=4), flush=True)
        return entry

    mode = 'a' if existing_logs else 'w'
    logs = list(existing_logs)
    print(f"Loaded {len(existing_logs)} existing scored rows; judging {len(pending)} pending with workers={workers}")
    with open(result_file, mode, encoding="utf-8") as out_f:
        if workers <= 1:
            for entry in tqdm(pending):
                judged = judge(entry)
                logs.append(judged)
                print(json.dumps(judged, ensure_ascii=False), file=out_f, flush=True)
                qtype2acc[qid2qtype[judged['question_id']]].append(1 if judged['autoeval_label']['label'] else 0)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(judge, entry) for entry in pending]
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    judged = fut.result()
                    logs.append(judged)
                    print(json.dumps(judged, ensure_ascii=False), file=out_f, flush=True)
                    qtype2acc[qid2qtype[judged['question_id']]].append(1 if judged['autoeval_label']['label'] else 0)

            
    print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
    for k,v in qtype2acc.items():
        print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))

    print('Saved to', result_file)
