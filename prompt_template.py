import os
from jinja2 import Template


class PromptTemplate:
    def __init__(self, prompt_type, val_type, task):
        if "eval" in task:
            if "mcqa" in val_type:  # mcqa eval
                template_path = "prompts/mcqa_eval.jinja"
            else:  # nli (free, cloze)
                template_path = "prompts/llm_nli.jinja"
        elif "corrupt" in task:
            if "bool" in val_type:
                template_path = "./prompts/corruption_bool.jinja"
            elif "mcqa" in val_type:
                template_path = "./prompts/corruption_free_allneg.jinja"
            elif "cloze" in val_type:
                template_path = "./prompts/corruption_cloze_allneg.jinja"
            elif "free" in val_type:
                template_path = "./prompts/corruption_free_allneg.jinja"
            elif "proposition" in val_type:
                template_path = "./prompts/corruption_proposition.jinja"
            else:
                raise ValueError("prompt type is not valid.")
        else:  # gen
            if "bool" in val_type:
                if "vanilla" in prompt_type:
                    template_path = "prompts/gen_bool.jinja"
                else:
                    raise ValueError("prompt type is not valid.")
            elif "mcqa" in val_type:
                if "vanilla" in prompt_type:
                    template_path = "prompts/gen_mcqa.jinja"
                else:
                    raise ValueError("prompt type is not valid.")
            elif "cloze" in val_type:
                if "vanilla" in prompt_type:
                    template_path = "prompts/gen_cloze.jinja"
                else:
                    raise ValueError("prompt type is not valid.")
            elif "free" in val_type:
                if "vanilla" in prompt_type:
                    template_path = "prompts/gen_free.jinja"
                else:
                    raise ValueError("prompt type is not valid.")

        with open(template_path, 'r') as fp:
            self.template = Template(fp.read())
        self.prompt = self.template.blocks['prompt']
    
    def prompting(self, **kwargs):
        context = self.template.new_context(kwargs)
        return ''.join(self.prompt(context)).strip()

    def cot_prompting(self, **kwargs):
        cot_prompt = self.template.blocks['cot']
        context = self.template.new_context(kwargs)
        return ''.join(cot_prompt(context)).strip()

    def neg_prompting(self, **kwargs):
        cot_prompt = self.template.blocks['convert_neg']
        context = self.template.new_context(kwargs)
        return ''.join(cot_prompt(context)).strip()


if __name__ == '__main__':
    prompt_type = "vanilla"
