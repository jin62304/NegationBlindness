import numpy as np
import json
import os
import sys
import argparse
from minutes_writer import MinuteWriter
from setproctitle import setproctitle
from utils import *
from tasks import *
from prompt_template import PromptTemplate
from datetime import datetime
from tqdm import tqdm
import re
import random

if __name__ == "__main__":
    # >>> args setup
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="gen", type=str, required=True)  # gen, corruption, eval
    parser.add_argument("--model_type", default="chatgpt", type=str, required=True)
    parser.add_argument("--eval_model", default="gpt4", type=str)
    parser.add_argument("--input_path", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--prompt", default="vanilla", type=str)  # vanilla, cot, refine, decom
    parser.add_argument("--data_type", default="trivia", type=str)  # trivia, nq
    parser.add_argument("--val_type", default="bool", type=str)  # bool, cloze, mcqa, free (verification type)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--cor_type", default="", type=str)  # pos / neg : query type
    parser.add_argument("--proctitle", default="", type=str)
    # to debug
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=0, type=int)
    # <<< args setup

    args = parser.parse_args()
    model_type = args.model_type

    if args.proctitle != "":
        setproctitle(str(args.proctitle))
    else:
        setproctitle("default")

    # >>> api key setup
    if "eval" in args.task:
        api_key = api_setup(args.eval_model, args.api_key)
    else:
        api_key = api_setup(model_type, args.api_key)
    # <<< api key setup

    # >>> file path setup
    if 'trivia' in args.data_type:
        if not os.path.exists('./results/trivia/'):
            os.makedirs('./results/trivia/')
        orig_data_file = 'trivia-verified-web-dev.json'
        # >>> input path
        if args.input_path == "":
            if 'corrupt' in args.task:  # corruption
                input_path = os.path.join('./data/trivia', orig_data_file)
            elif 'gen' in args.task:
                temp = str(args.val_type) + "_" + orig_data_file
                input_path = os.path.join('./data/trivia/', temp)
            else:  # eval
                if "neg" in args.cor_type:
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_neg_" + orig_data_file
                else:
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_pos_" + orig_data_file
                input_path = os.path.join('./results/trivia/', str(model_type), temp)
        else:
            input_path = args.input_path
        # <<<

        # >>> output path
        if args.output_path == "":
            if 'corrupt' in args.task:  # corruption
                temp = str(args.cor_type) + "_" + orig_data_file
                output_path = os.path.join('./data/trivia/', temp)
            elif "gen" in args.task:
                if "neg" in args.cor_type:
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_neg_" + orig_data_file
                else:  # pos
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_pos_" + orig_data_file
                output_dir = os.path.join('./results/trivia/', str(model_type))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, temp)
            else:  # automated metrics eval (acc)
                if "neg" in args.cor_type:
                    temp = "eval_" + str(args.prompt) + "_" + str(args.val_type) + "_neg_" + orig_data_file
                else:
                    temp = "eval_" + str(args.prompt) + "_" + str(args.val_type) + "_pos_" + orig_data_file
                output_path = os.path.join('./results/trivia/', str(model_type), temp)
        else:
            output_dir, output_file = "/".join(args.output_path.split("/")[:-1]), args.output_path.split("/")[-1]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, output_file)
        # <<<
    elif 'nq' in args.data_type:
        if not os.path.exists('./results/nq/'):
            os.makedirs('./results/nq/')
        orig_data_file = 'nq_raw.json'
        # >>> input path
        if args.input_path == "":
            if 'corrupt' in args.task:  # corruption
                if 'bool' in args.cor_type or 'cloze' in args.cor_type:
                    orig_data_file = 'proposition_nq_raw.json'
                    input_path = os.path.join('./data/nq', orig_data_file)
                else:  # free, mc, proposition
                    input_path = os.path.join('./data/nq', orig_data_file)
            elif 'gen' in args.task:
                temp = str(args.val_type) + "_" + orig_data_file
                input_path = os.path.join('./data/nq/', temp)
            else:  # eval
                if "neg" in args.cor_type:
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_neg_" + orig_data_file
                else:
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_pos_" + orig_data_file
                input_path = os.path.join('./results/nq/', str(model_type), temp)
        else:
            input_path = args.input_path
        # <<<

        # >>> output path
        if args.output_path == "":
            if 'corrupt' in args.task:  # corruption
                temp = str(args.cor_type) + "_" + orig_data_file
                output_path = os.path.join('./data/nq/', temp)
            elif "gen" in args.task:
                if "neg" in args.cor_type:
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_neg_" + orig_data_file
                else:  # pos
                    temp = str(args.prompt) + "_" + str(args.val_type) + "_pos_" + orig_data_file
                output_dir = os.path.join('./results/nq/', str(model_type))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, temp)
            else:  # using automated metrics (acc)
                if "neg" in args.cor_type:
                    temp = "eval_" + str(args.prompt) + "_" + str(args.val_type) + "_neg_" + orig_data_file
                else:
                    temp = "eval_" + str(args.prompt) + "_" + str(args.val_type) + "_pos_" + orig_data_file
                output_path = os.path.join('./results/nq/', str(model_type), temp)
        else:
            output_dir, output_file = "/".join(args.output_path.split("/")[:-1]), args.output_path.split("/")[-1]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, output_file)
        # <<<
    else:
        raise ValueError("Check the data_type: trivia, nq.")

    data = load_data(input_path)
    # <<< file path setup

    # >>> (to debug) slicing examples by indices
    if args.start_idx != 0 and args.end_idx != 0:
        data = data[args.start_idx:args.end_idx]
    else:
        if args.start_idx != 0:
            data = data[args.start_idx:]
        if args.end_idx != 0:
            data = data[:args.end_idx]
    # <<<

    # >>> model + prompt template setup
    if 'gen' in args.task:
        # writer setup
        Writer = MinuteWriter(api_key, model_type)
        prompt_template = PromptTemplate(args.prompt, args.val_type, args.task)
        pred_model = Generator(Writer, prompt_template, args.prompt, args.data_type)

        new_data = []
        for idx, datum in tqdm(enumerate(data)):
            contents = pred_model.parse_data(datum, args.cor_type, args.val_type)
            if "vanilla" in args.prompt:
                response = pred_model.response_generate(contents, model_type)  # response -> list
                datum["pred"] = response[0]
            else:
                raise ValueError("Other methods like cot, decom, refine will be...")

            print()
            print(f"# Response: {response}")
            print()

            new_data.append(datum)

        save_json(new_data, output_path)
    elif 'corrupt' in args.task:
        prompt_template = PromptTemplate(args.prompt, args.val_type, args.task)

        model_type = "gpt-4.1-mini-2025-04-14"
        Writer = MinuteWriter(api_key, model_type)
        tagger = Corrupter(Writer, prompt_template, args.cor_type)

        saved_data = []
        if "bool" in args.cor_type:  # task 1 (Boolean-style)
            # bool: processing twice (different tagging obj and prompt template)
            # 1) proposition -> interrogative sentence
            # 2) pos interro -> neg q
            if 'trivia' in args.data_type:
                for datum in tqdm(data[:]):
                    question = datum['question']
                    gold = datum['gold']
                    answers = datum['answers']
                    context = datum['context']
                    file_name = datum['file_name']
                    id = datum['id']

                    declarative_sent, q_pos, q_neg = tagger.bool_r_generate(datum, model_type)
                    # contents = tagger.parse_data(data)
                    # llm_tag = tagger.tag(contents)
                    obj = {
                        "plain_sentence": declarative_sent,
                        "q_pos": q_pos,
                        "q_neg": q_neg,
                        "context": context,
                        "file_name": file_name,
                        "id": id,
                        "raw_data": {
                            "question": question,
                            "gold": gold,
                            "answers": answers
                        }
                    }
                    saved_data.append(obj)
                    print("================================")
            else:  # nq
                for datum in tqdm(data[:]):
                    id = datum['id']
                    q_pos = datum['q_pos']
                    plain_sentence = datum['plain_sentence']
                    gold = datum['gold']
                    context = datum['context']

                    # step 1) from proposition (plain sentence) into bool question
                    contents = tagger.parse_data(datum)
                    bool_q_pos = tagger.tag(contents)

                    print()
                    print(f"bool_q_pos: {bool_q_pos}")
                    print("================================")

                    # step 2) from bool_q_pos into bool_q_neg
                    contents = tagger.bool_parse_data(bool_q_pos, gold)
                    q_neg = tagger.tag(contents)

                    print()
                    print(f"q_neg: {q_neg}")
                    print("================================")

                    obj = {
                        "id": id,
                        "q_pos": q_pos,
                        "plain_sentence": plain_sentence,
                        "bool_q_pos": bool_q_pos,
                        "q_neg": q_neg,
                        "gold": gold,
                        "answers": "",
                        "context": context,
                        "file_name": "",
                    }
                    saved_data.append(obj)

        elif "mcqa" in args.cor_type:  # task 2 (MCQA): option construction process
            import spacy
            nlp = spacy.load("en_core_web_sm")

            if "trivia" in args.data_type:
                for datum in tqdm(data[:]):
                    question = datum['question']
                    gold = datum['gold']
                    answers = datum['answers']
                    context = datum['context']
                    file_name = datum['file_name']
                    id = datum['id']

                    # 1) getting "question" and "gold" (label)
                    # 2) to build the other three options
                    # 	2-1) NER tagging in gold (w/ Spacy)
                    pp_gold = nlp(gold)

                    ner_gold = []
                    for ent in pp_gold.ents:
                        ent_pair = {
                            "text": ent.text,
                            "label": ent.label_
                        }
                        ner_gold.append(ent_pair)

                    # 	2-2) NER tagging in "context" without duplicates (not be duplicated w/ "answers")
                    pp_context = nlp(context)
                    check_duplicates = set()
                    ner_context = []
                    for ent in pp_context.ents:
                        if ent.text not in check_duplicates:
                            check_duplicates.add(ent.text)

                            ent_pair = {
                                "text": ent.text,
                                "label": ent.label_
                            }
                            ner_context.append(ent_pair)

                    # removing duplicates from "gold" and "answers (alias)"
                    temp = [item["text"].lower() for item in ner_gold]
                    text_golds = temp + [item.lower() for item in answers]

                    filtered_ner_context_text = []
                    for ctx_item in ner_context:
                        flag = True
                        for gold_item in text_golds:
                            if ctx_item["text"].lower() in gold_item:
                                flag = False

                        if flag:
                         filtered_ner_context_text.append(ctx_item)

                    # 	2-3) constructing subset with the entities which have same entity type with "gold"
                    labels_in_ner_gold = {item["label"] for item in ner_gold}
                    filtered_ner_context_label = [item for item in filtered_ner_context_text if item["label"] in labels_in_ner_gold]

                    # 	2-4) randomly extracting 3 entities (w/o duplicates) from the filtered subset
                    #   if the len() of subset is less than three

                    if len(filtered_ner_context_label) < 3:
                        total_subset = 3
                        add_num = total_subset - len(filtered_ner_context_label)

                        selected_samples = [item['text'].lower() for item in filtered_ner_context_label]
                        candidate_subset = [item["text"].lower() for item in filtered_ner_context_text if item["text"] not in selected_samples]
                        candidate_subset = list(set(candidate_subset))

                        if len(candidate_subset) < add_num:
                            n = add_num - len(candidate_subset)  # num to extract
                            ctx = [word.lower() for word in context.split() if
                                   len(word) >= 4]  # word length limit: more than 4 chars

                            check_duplicates_choice = selected_samples + candidate_subset
                            filtered_ctx = []
                            for ctx_item in ctx:
                                flag = True
                                for choice in check_duplicates_choice:
                                    if ctx_item.lower() in choice:
                                        flag = False
                                if flag:
                                    filtered_ctx.append(ctx_item)

                            ctx_samples = random.sample(filtered_ctx, n)

                            choice_list = selected_samples + candidate_subset + ctx_samples
                        else:
                            add_samples = random.sample(candidate_subset, add_num)
                            choice_list = selected_samples + add_samples
                    else:
                        temp = [item['text'].lower() for item in filtered_ner_context_label]
                        filtered_ner_context_label = list(set(temp))
                        choice_list = random.sample(filtered_ner_context_label, 3)

                    # 3) randomly shuffling the position of gold
                    choices, answerKey = generate_choices_format(gold, choice_list)

                    # 4) making instance into MCQA format
                    obj = {
                        "id": id,
                        "context": context,
                        "question": question,
                        "choices": choices,
                        "answerKey": answerKey,
                        "gold": gold,
                        "answers": answers,
                        "file_name": file_name,
                    }
                    saved_data.append(obj)
            else:  # nq dataset
                # corruption + option construction at once --> saving choices + q_neg
                free_file = './data/nq/free_allneg_nq_raw.json'
                free_dict = dict()
                with open(free_file, 'r', encoding='utf-8') as f:
                    free_data = json.load(f)
                    for line in free_data:
                        free_dict[line['id']] = line['q_neg']

                for datum in tqdm(data[:]):
                    id = datum['idx']
                    q_pos = datum['q_pos']
                    gold = datum['gold']
                    context = datum['context']
                    q_neg = free_dict[id]

                    # 1) gold answer NER
                    pp_gold = nlp(gold)
                    ner_gold = []
                    for ent in pp_gold.ents:
                        ent_pair = {
                            "text": ent.text.lower(),
                            "label": ent.label_
                        }
                        ner_gold.append(ent_pair)

                    # 2) context NER
                    pp_context = nlp(context)
                    check_duplicates = set()
                    ner_context = []
                    for ent in pp_context.ents:
                        if ent.text.lower() not in check_duplicates:
                            check_duplicates.add(ent.text.lower())

                            ent_pair = {
                                "text": ent.text.lower(),
                                "label": ent.label_
                            }
                            ner_context.append(ent_pair)

                    # 3) remove duplicates
                    filtered_ner_context_text = []
                    for ctx_item in ner_context:
                        flag = True
                        for gold_item in ner_gold:
                            if ctx_item["text"].lower() in gold_item["text"]:
                                flag = False
                        if flag:
                            filtered_ner_context_text.append(ctx_item)

                    labels_in_ner_gold = {item["label"] for item in ner_gold}
                    filtered_ner_context_label = [item for item in filtered_ner_context_text if item["label"] in labels_in_ner_gold]

                    # 4) if # of entities with same entity type < 3
                    if len(filtered_ner_context_label) < 3:
                        total_subset = 3
                        add_num = total_subset - len(filtered_ner_context_label)

                        selected_samples = [item['text'].lower() for item in filtered_ner_context_label]
                        candidate_subset = [item["text"].lower() for item in filtered_ner_context_text if
                                            item["text"] not in selected_samples]
                        candidate_subset = list(set(candidate_subset))

                        if len(candidate_subset) < add_num:
                            # filling in the missing number in context
                            n = add_num - len(candidate_subset)  # num to extract
                            ctx = [word.lower() for word in context.split() if len(word) >= 4]  # word length limit: more than 4 chars

                            check_duplicates_choice = selected_samples + candidate_subset
                            filtered_ctx = []
                            for ctx_item in ctx:
                                flag = True
                                for choice in check_duplicates_choice:
                                    if ctx_item.lower() in choice:
                                        flag = False
                                if flag:
                                    filtered_ctx.append(ctx_item)

                            ctx_samples = random.sample(filtered_ctx, n)

                            choice_list = selected_samples + candidate_subset + ctx_samples
                        else:
                            add_samples = random.sample(candidate_subset, add_num)
                            # add_samples = [item.lower() for item in random_samples]
                            choice_list = selected_samples + add_samples
                    else:
                        temp = [item['text'].lower() for item in filtered_ner_context_label]
                        filtered_ner_context_label = list(set(temp))
                        choice_list = random.sample(filtered_ner_context_label, 3)

                    # 5) changing option ordering
                    choices, answerKey = generate_choices_format(gold, choice_list)

                    obj = {
                        "id": id,
                        "q_pos": q_pos,
                        "q_neg": q_neg,
                        "choices": choices,
                        "answerKey": answerKey,
                        "gold": gold,
                        "answers": "",
                        "context": context,
                        "file_name": "",
                    }

                    saved_data.append(obj)
        elif "cloze" in args.cor_type:  # task 3 (Cloze-style Completion)
            if "trivia" in args.data_type:
                for datum in tqdm(data[:]):
                    plain_sentence = datum['plain_sentence']
                    context = datum['context']
                    file_name = datum['file_name']
                    id = datum['id']
                    gold = datum["raw_data"]["gold"]
                    answers = datum["raw_data"]["answers"]

                    contents = tagger.parse_data(datum)
                    p_neg = tagger.tag(contents)

                    # changing the "gold" part into [MASK] token
                    masked_plain_sentence = re.sub(rf"\b{gold}\b", "[MASK]", plain_sentence, flags=re.IGNORECASE)
                    masked_p_neg = re.sub(rf"\b{gold}\b", "[MASK]", p_neg, flags=re.IGNORECASE)

                    print(f"p_neg: {p_neg}")
                    print("================================")

                    obj = {
                        "plain_sentence": {
                            "pos": plain_sentence,
                            "neg": p_neg
                        },
                        "masked_plain_sentence": {
                            "pos": masked_plain_sentence,
                            "neg": masked_p_neg
                        },
                        "gold": gold,
                        "answers": answers,
                        "context": context,
                        "file_name": file_name,
                        "id": id,
                    }

                    saved_data.append(obj)
            elif "nq" in args.data_type:
                for datum in tqdm(data[:]):
                    id = datum['id']
                    q_pos = datum['q_pos']
                    plain_sentence = datum['plain_sentence']
                    gold = datum['gold']
                    context = datum['context']

                    contents = tagger.parse_data(datum)
                    q_neg = tagger.tag(contents)

                    # changing the "gold" part into [MASK] token
                    gold_pattern = re.escape(gold.strip())
                    gold_pattern = re.sub(r'\\\s+', r'\\s*', gold_pattern)  # when having several white spaces

                    # allowing arbitrary spaces before and after gold (e.g., "+ 7", "+7", " +7 ", "+7 ")
                    gold_pattern = r'\s*' + gold_pattern + r'\s*'

                    # Replace gold with case-insensitive and spaces-inclusive
                    masked_plain_sentence = re.sub(gold_pattern, ' [MASK] ', plain_sentence, flags=re.IGNORECASE)
                    masked_q_neg = re.sub(gold_pattern, ' [MASK] ', q_neg, flags=re.IGNORECASE)

                    print(f"masked_plain_sentence: {masked_plain_sentence}")
                    print(f"q_neg: {masked_q_neg}")
                    print("================================")

                    obj = {
                        "id": id,
                        "q_pos": q_pos,
                        "q_neg": masked_q_neg,
                        "plain_sentence": plain_sentence,
                        "masked_plain_sentence": masked_plain_sentence,
                        "gold": gold,
                        "answers": "",
                        "context": context,
                        "file_name": "",
                    }

                    saved_data.append(obj)
            else:
                pass
        elif "free" in args.cor_type:  # task 4 (free-form Gen)
            if "trivia" in args.data_type:
                for datum in tqdm(data[:]):
                    question = datum['question']
                    gold = datum['gold']
                    answers = datum['answers']
                    context = datum['context']
                    file_name = datum['file_name']
                    id = datum['id']

                    contents = tagger.parse_data(datum)
                    q_neg = tagger.tag(contents)

                    print(f"q_neg: {q_neg}")
                    print("================================")

                    obj = {
                        "question": question,
                        "question_neg": q_neg,
                        "gold": gold,
                        "answers": answers,
                        "context": context,
                        "file_name": file_name,
                        "id": id,
                    }
                    saved_data.append(obj)

                    print(saved_data)
            elif "nq" in args.data_type:
                for datum in tqdm(data[:]):
                    id = datum['idx']
                    q_pos = datum['q_pos']
                    gold = datum['gold']
                    context = datum['context']

                    contents = tagger.parse_data(datum)
                    q_neg = tagger.tag(contents)

                    print(f"q_neg: {q_neg}")
                    print("================================")

                    obj = {
                        "id": id,
                        "q_pos": q_pos,
                        "q_neg": q_neg,
                        "gold": gold,
                        "answers": "",
                        "context": context,
                        "file_name": "",
                    }
                    saved_data.append(obj)

                    # exit(0)
            else:
                pass
        elif "proposition" in args.cor_type:
            if "trivia" in args.data_type:
                for datum in tqdm(data[:]):
                    question = datum['question']
                    gold = datum['gold']
                    answers = datum['answers']
                    context = datum['context']
                    file_name = datum['file_name']
                    id = datum['id']

                    contents = tagger.parse_data(datum)
                    q_neg = tagger.tag(contents)

                    print(f"q_neg: {q_neg}")
                    print("================================")

                    obj = {
                        "question": question,
                        "question_neg": q_neg,
                        "gold": gold,
                        "answers": answers,
                        "context": context,
                        "file_name": file_name,
                        "id": id,
                    }
                    saved_data.append(obj)
            elif "nq" in args.data_type:
                for datum in tqdm(data[:]):
                    id = datum['idx']
                    q_pos = datum['q_pos']
                    gold = datum['gold']
                    context = datum['context']

                    contents = tagger.parse_data(datum)
                    q_neg = tagger.tag(contents)

                    print(f"plain_sentence: {q_neg}")
                    print("================================")

                    obj = {
                        "id": id,
                        "q_pos": q_pos,
                        "plain_sentence": q_neg,
                        "gold": gold,
                        "answers": "",
                        "context": context,
                        "file_name": "",
                    }
                    saved_data.append(obj)

                    # exit(0)
                else:
                    pass
        else:
            raise ValueError("Please check the corruption type: proposition, bool, mcqa, cloze, or free.")

        # saving file
        output_path = args.output_path

        output_dir = "/".join(output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(saved_data, file, ensure_ascii=False, indent=2)

        print(f"SAVED DONE: {output_path}")
    else:  # eval
        if "bool" in args.val_type:  # appending answerKey into full data
            # >>> generator setup
            Writer = MinuteWriter(api_key, args.eval_model)
            prompt_template = PromptTemplate(args.prompt, args.val_type, args.task)
            pred_model = Generator(Writer, prompt_template, args.prompt, args.data_type)
            # <<< generator setup

            # to get gt answerKey
            new_data = []
            for idx, datum in enumerate(tqdm(data)):
                new_dict = dict()
                contents = pred_model.nli_parse_data(datum, args.cor_type, args.val_type)
                response = pred_model.judge_generate(contents, model_type)  # response -> list

                if "neg" in args.cor_type:
                    question = datum["q_neg"]
                else:  # pos
                    question = datum["q_pos"]
                new_dict["question"] = question
                new_dict["pred"] = datum["pred"].replace("#", "").replace("Your Answer", "").replace("Rationale", "").replace(":", "").strip()
                new_dict["van_pred"] = "Yes. " + datum['plain_sentence']
                new_dict["nli"] = response[0]

                print(f"{args.val_type} NLI result between pred and gt: {response}")

                new_data.append(new_dict)

            save_json(new_data, output_path)
        elif "mcqa" in args.val_type:
            # >>> generator setup
            Writer = MinuteWriter(api_key, args.eval_model)
            prompt_template = PromptTemplate(args.prompt, args.val_type, args.task)
            pred_model = Generator(Writer, prompt_template, args.prompt, args.data_type)
            # <<< generator setup

            new_data = []
            total_num = len(data)
            for datum in tqdm(data):
                new_dict = dict()
                contents = pred_model.nli_parse_data(datum, args.cor_type, args.val_type)
                response = pred_model.judge_generate(contents, model_type)  # response -> list

                if "neg" in args.cor_type:
                    question = datum["q_neg"]
                    # score = 1 if score == 0 else 0  # if ==, 0 or !=, 1 (contrary to pos case)
                else:  # pos
                    question = datum["q_pos"]

                new_dict["question"] = question
                new_dict["pred"] = datum["pred"]
                new_dict["gold"] = datum["answerKey"]
                new_dict["nli"] = response[0]

                print(f"{args.val_type} NLI between pred and gt: {response}")

                new_data.append(new_dict)

            save_json(new_data, output_path)
        else:  # free, cloze
            # >>> generator setup
            Writer = MinuteWriter(api_key, args.eval_model)
            prompt_template = PromptTemplate(args.prompt, args.val_type, args.task)
            pred_model = Generator(Writer, prompt_template, args.prompt, args.data_type)
            # <<< generator setup

            new_data = []
            for datum in tqdm(data):
                new_dict = dict()
                contents = pred_model.nli_parse_data(datum, args.cor_type, args.val_type)
                response = pred_model.judge_generate(contents, model_type)  # response -> list

                if "neg" in args.cor_type:
                    question = datum["q_neg"]
                else:  # pos
                    question = datum["q_pos"]
                new_dict["question"] = question
                new_dict["pred"] = datum["pred"]
                new_dict["van_pred"] = [datum["gold"]] + [datum["answers"]]
                new_dict["nli"] = response[0]

                print(f"{args.val_type} NLI result between pred and gt: {response}")

                new_data.append(new_dict)

            save_json(new_data, output_path)

    print("=================END==================")
