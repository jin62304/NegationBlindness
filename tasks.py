import os
import math
from collections import Counter
from utils import *


class Generator:
    def __init__(self, writer, prompt_template, prompt_type, data_type):
        # LLM message setup
        self.writer = writer
        print(self.writer.model)
        self.prompt = prompt_template
        self.prompt_type = prompt_type
        self.data_type = data_type

    def parse_data(self, datum, cor_type, val_type):
        ctx = datum['context']

        # >>> model context length (window size) limit (under 16k token)
        # calculating the word:token ratio to be approximately 1.5 times
        ctx_words = ctx.split(" ")
        threshold_token_len = 14000
        threshold_word_len = int(threshold_token_len/1.5)
        token_len = 1.5 * len(ctx_words)
        if token_len > threshold_word_len:
            ctx = " ".join(ctx_words[:threshold_word_len])
        # <<<

        query = datum['q_pos']
        corrupted = datum['q_neg']

        # replacing the corrupted ("{no X}" format)
        if "bool" in val_type and "trivia" in self.data_type:
            def normalize(text):
                return re.sub(r'\s+', ' ', text.strip().lower())

            q_neg = datum['q_neg']
            gold = datum['gold']

            # default value: keeping original sentence
            corrupted = q_neg

            # comparison with normalized string
            normalized_q = normalize(q_neg)
            normalized_gold = normalize(gold)
            normalized_full = f"not {normalized_gold}"

            if normalized_full in normalized_q:
                # matching in the original sentence w/regular expression
                pattern = re.compile(r'(not\s+' + re.escape(gold) + r')', re.IGNORECASE)
                match = pattern.search(q_neg)

                if match:
                    corrupted = q_neg[:match.start()] + '{' + match.group(1) + '}' + q_neg[match.end():]

            print()
            print(f"[corrupted]: {corrupted}, [gold] {gold}")
            print()

        if "mcqa" in val_type:
            choices = datum["choices"]["text"]  # list [,,,]
            # answers = datum['answerKey']  # A, B, C, or D
            choices_str = f"A: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[-1]}"

            if cor_type != "":  # neg gen
                prompt = self.prompt.prompting(**{'context': ctx,
                                                  'question': corrupted,
                                                  'choices': choices_str, })
            else:  # positive gen
                prompt = self.prompt.prompting(**{'context': ctx,
                                                  'question': query,
                                                  'choices': choices_str, })
        else:
            # gold = datum['gold']
            # answers = datum['answers']  # list, for triviaQA

            if cor_type != "":  # neg gen
                prompt = self.prompt.prompting(**{'context': ctx,
                                                  'question': corrupted, })
            else:  # positive gen
                prompt = self.prompt.prompting(**{'context': ctx,
                                                  'question': query, })

        system_prompt, user_prompt = prompt.split("---")[0], prompt.split("---")[-1]

        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = user_prompt

        return contents

    def response_generate(self, contents, model_type):
        messages = [
            {"role": "system",
             "content": contents["system"]},
            {"role": "user",
             "content": contents["user"]}
        ]
        if "vanilla" in self.prompt_type:
            if "gpt" in model_type:
                response = self.writer.write(messages)
            else:  # llama, mistral, gemma, qwen (hf)
                response = self.writer.hf_write(messages)

            return response
        else:  # prompt engineering: cot, decom, self-refine
            # return response, rsn_path
            raise ValueError("Other methods like cot, decom, refine will be...")

    def judge_parse_data(self, datum, van_datum, cor_type, val_type, answer_key=None):
        van_pred = van_datum["pred"]
        pred = datum["pred"]

        if "neg" in cor_type:
            question = van_datum["q_neg"]

            if "mcqa" in val_type:
                choices = "# Possible Choices: \n"
                for l, c in zip(van_datum["choices"]["label"], van_datum["choices"]["text"]):
                    choices += f'{l}: {c}\n'
                van_datum["choices"]["label"].remove(van_datum["answerKey"])

                gold = f'{van_datum["choices"]["label"][0]} or {van_datum["choices"]["label"][1]} or {van_datum["choices"]["label"][-1]} (Except {van_datum["answerKey"]}: {van_datum["gold"]})'

                prompt = self.prompt.prompting(**{'choices': choices,
                                                  'question': question,
                                                  'gold': gold,
                                                  'answer_a': pred,  # cot, decom ...
                                                  'answer_b': van_pred, })  # vanilla
            elif "bool" in val_type:
                gold = answer_key
                prompt = self.prompt.prompting(**{'question': question,
                                                  'gold': gold,
                                                  'answer_a': pred,  # cot, decom ...
                                                  'answer_b': van_pred, })  # vanilla
            else:  # cloze, free
                gold = f'Anything except for the following list: {[van_datum["gold"]] + van_datum["answers"]}'
                prompt = self.prompt.prompting(**{'question': question,
                                                  'gold': gold,
                                                  'answer_a': pred,  # cot, decom ...
                                                  'answer_b': van_pred, })  # vanilla
        else:  # pos
            question = van_datum["q_pos"]

            if "mcqa" in val_type:
                choices = "# Possible Choices: \n"
                for l, c in zip(van_datum["choices"]["label"], van_datum["choices"]["text"]):
                    choices += f'{l}: {c}\n'

                gold = f'{van_datum["answerKey"]} ({van_datum["gold"]})'
                prompt = self.prompt.prompting(**{'choices': choices,
                                                  'question': question,
                                                  'gold': gold,
                                                  'answer_a': pred,  # cot, decom ...
                                                  'answer_b': van_pred, })  # vanilla
            elif "bool" in val_type:
                gold = answer_key
                prompt = self.prompt.prompting(**{'question': question,
                                                  'gold': gold,
                                                  'answer_a': pred,  # cot, decom ...
                                                  'answer_b': van_pred, })  # vanilla
            else:  # cloze, free
                gold = [van_datum["gold"]] + van_datum["answers"]
                prompt = self.prompt.prompting(**{'question': question,
                                                  'gold': gold,
                                                  'answer_a': pred,  # cot, decom ...
                                                  'answer_b': van_pred, })  # vanilla

        system_prompt, user_prompt = prompt.split("---")[0], prompt.split("---")[-1]

        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = user_prompt

        return contents

    def judge_generate(self, contents, model_type):
        messages = [
            {"role": "system",
             "content": contents["system"]},
            {"role": "user",
             "content": contents["user"]}
        ]

        response = self.writer.write(messages)

        return response

    def nli_parse_data(self, datum, cor_type, val_type, answer_key=None):  # compare w/ GT
        if "neg" in cor_type:
            question = datum["q_neg"]
        else:  # pos
            question = datum["q_pos"]

        if "bool" in val_type:
            pred = datum["pred"].replace("#", "").replace("Your Answer", "").replace("Rationale", "").replace(":", "").strip()
        else:
            pred = datum["pred"].strip()

        if "mcqa" in val_type:
            # merging with
            gold = f'{datum["answerKey"]}: {datum["gold"]}'
            choices = ""
            for label, choice in zip(datum["choices"]["label"], datum["choices"]["text"]):
                temp = f"{label}: {choice}\n"
                choices += temp

            prompt = self.prompt.prompting(**{'question': question,
                                              'choices': choices,
                                              'answer_a': pred,  # predictions
                                              'answer_b': gold, })  # gt
        elif "bool" in val_type:
            if "trivia" in self.data_type:
                gold = "Yes. " + datum["plain_sentence"]
            else:  # nq
                gold = "Yes. " + datum["plain_sentence"]


            prompt = self.prompt.prompting(**{'question': question,
                                              'answer_a': pred,  # predictions
                                              'answer_b': gold, })  # gt
        else:  # cloze, free
            if datum["answers"] == "":  # nq
                gold = datum["gold"]
            else:  # trivia
                gold = [datum["gold"]] + datum["answers"]

            prompt = self.prompt.prompting(**{'question': question,
                                              'answer_a': pred,  # predictions
                                              'answer_b': gold, })  # gt

        system_prompt, user_prompt = prompt.split("---")[0], prompt.split("---")[-1]

        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = user_prompt

        return contents


class Corrupter:
    def __init__(self, writer, prompt_template, target):
        self.writer = writer
        print(self.writer.model)
        self.prompt = prompt_template
        self.tagging_obj = target  # str

    def parse_data(self, datum):
        if 'cloze' in self.tagging_obj or 'bool' in self.tagging_obj:
            input_sentence = datum["plain_sentence"]
            prompt = self.prompt.prompting(**{'input_sentence': input_sentence,
                                              })
        elif 'free' in self.tagging_obj:
            input_sentence = datum["q_pos"]
            prompt = self.prompt.prompting(**{'input_sentence': input_sentence,
                                              })
        elif 'proposition' in self.tagging_obj:
            input_sentence = datum["q_pos"]
            gold = datum['gold']
            prompt = self.prompt.prompting(**{'input_sentence': input_sentence,
                                              'gold': gold,
                                              })
        else:
            print("Wrong tagging object.")
            input_sentence = ""
            prompt = self.prompt.prompting(**{'input_sentence': input_sentence,
                                              })

        system_prompt, user_prompt = prompt.split("---")[0], prompt.split("---")[-1]

        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = user_prompt

        return contents

    def tag(self, contents):
        messages = [
            {"role": "system",
             "content": contents["system"]},
            {"role": "user",
             "content": contents["user"]}
        ]

        response = self.writer.write(messages)

        # print("#################################")
        # print(response)
        # print("#################################")

        try:
            final_answer = response[0].split(":")[1].strip()
        except:
            final_answer = response[0].strip()

        return final_answer

    def bool_parse_data(self, bool_q_pos, gold):  # bool_q_pos -> proposition into bool format positive query
        prompt = self.prompt.cot_prompting(**{'bool_q_pos': bool_q_pos,
                                              'gold': gold, })

        system_prompt, user_prompt = prompt.split("---")[0], prompt.split("---")[-1]

        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = user_prompt

        return contents

    def bool_r_generate(self, datum, model_type):
        query = datum['question']
        gold_answer = datum['gold']

        prompt = self.prompt.prompting(**{'question': query,
                                          'answer': gold_answer,
                                          })

        messages = prompt_split_msg(prompt)

        if "gpt" in model_type:
            # 1) given question and "gold", converting into the form of proposition (p)
            response = self.writer.write(messages)
            first_response = response[0].split(":")[1]
            print(f"STEP1: {first_response}")

            # 2) Using p, convert to a positive question with Did or was/were added at the very beginning (q with Yes/No answer)
            interrogative_prompt = self.prompt.cot_prompting(**{'input_sentence': first_response})
            messages = prompt_split_msg(interrogative_prompt)
            response = self.writer.write(messages)
            second_response = response[0].split(": ")[1]
            print(f"STEP2: {second_response}")

            # 3) Maintaining the question form of pos, add "not" to make it grammatically correct to convert it to a neg question.
            convert_neg_prompt = self.prompt.neg_prompting(**{'input_sentence': second_response})
            messages = prompt_split_msg(convert_neg_prompt)
            response = self.writer.write(messages)
            # print(response)
            final_response = response[0].split(": ")[1]
            print(f"STEP3: {final_response}")

            return first_response, second_response, final_response