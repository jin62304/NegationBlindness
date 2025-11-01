This is the official github repository for the paper: [*Semantic Inversion, Identical Replies*: Revisiting Negation Blindness in Large Language Models](https://drive.google.com/file/d/16cuFO9pDs25bbX4IvKUNWhsTeX0LIWPP/view?usp=sharing) (Accepted at EMNLP 2025 Main).

*Jinsung Kim(\*), Seonmin Koo(\*), and Heuiseok Lim*</br>
🏫 [NLP & AI Lab.](https://nlp.korea.ac.kr/), Korea University, South Korea

→ This paper addresses the problem of *negation blindness* in LLMs—their failure to capture semantic contradictions in negated queries despite correct understanding of positive knowledge—and proposes a systematic verification framework to analyze and evaluate this phenomenon.

## 🛠️ Installation
```bash
$ git clone https://github.com/jin62304/NegationBlindness.git
```
```bash
# python_requires >=3.9
$ cd ./NegationBlindness
$ pip install -r requirements.txt 
```
## 🚀 Usage
### Sample Data
- A subset of data samples constructed through our proposed verification framework is provided.
- For the raw datasets used in the experiments, TriviaQA and Natural Questions (NQ) are adopted as our primary sources.

```bash
# sample data path: TriviaQA
$ ./data/trivia/{task}_trivia-verified-web-dev.json

# sample data path: Natural Questions (NQ)
$ ./data/nq/{task}_nq_raw.json
```
### Framework
- This framework consists of three main components: 1) task design, 2) verification set construction, and 3) measurement of the *negation blindness* phenomenon (BLD score). 
- The task design follows two key criteria—constrainedness and complexity—and systematically covers four distinct task types: (1) Boolean selection, (2) Multiple-choice selection, (3) Cloze-style completion, and (4) Free-form generation.
- The verification set is constructed based on each task design through processes such as query corruption (negation) and option construction.
- The BLD score (↓) indicates the extent to which a model exhibits negation blindness and is computed using Equations (1) and (2) in the paper.
- *The current version is implemented as a prototype for testing selected models and tasks. Additional modules and refined implementations for broader model and task coverage will be released in future updates.*
#### Arguments
```bash
## Not all arguments are mandatory. 
--task: {'corrupt', 'gen', 'eval'} # mandatory, 'corrupt': query corruption (including negation), 'gen': model response generation, 'eval': evaluation of the model's response
--model_type: str # mandatory, model type to be verified, such as chatgpt
--eval_model: str # to set an evaluation model (such as gpt4 or gpt-4) for the response of the model to be verified
--input_path: str # to set the input file path manually
--output_path: str # to set the output file path manually
--prompt: str # prompt engineering type such as vanilla, cot, refine, decom, icl, debate, or voting
--data_type: {'trivia', 'nq'} # the type of dataset
--val_type: {'bool', 'cloze', 'mcqa', 'free'} # verification task type
--cor_type: {'pos', 'neg'} # query type (whether the query was negated or not)
--api_key: str  # to set the api key manually (using "openai_api_key.txt" or the dotenv library is recommended.)
--start_idx: int (>= 0) # start index of instances (for debugging)
--end_idx: int (>= 0) # end index of instances (for debugging)
```

#### 1. Verification Set Construction
- Verification sets are built for each task based on the tasks designed to verify the model in this step.
#### 2. Model Response Generation
- In this step, the model produces responses for both the positive and the negated queries in each verification task.
```bash
$ bash scripts/run_main.sh
```
#### 3. *Negation Blindness* Measurement
- The *negation blindness* score (BLD) is calculated **in a pair-wise manner**, by comparing the model’s responses to each positive query and its corresponding negated query.
```bash
$ bash scripts/eval.sh
```

## 📖 Citation
```
@inproceedings{
    To Be Determined...
}
```

### Misc.
- In addition, our other paper [🦅 HAWK: Highlighting Entity-aware Knowledge for Alleviating Information Sparsity in Long Contexts](https://drive.google.com/file/d/1uEyWcNESD7hZDU853yVN-9wv5UNwngRo/view?usp=sharing), which was accepted to EMNLP 2025 Findings, is also a recommended LLM-related paper. 
- This paper addresses the challenge of information sparsity in long-context question answering by proposing an entity-aware framework (HAWK) that highlights and structures key information to enhance LLM performance.