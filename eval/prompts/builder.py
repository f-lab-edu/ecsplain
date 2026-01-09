from langchain_core.prompts import PromptTemplate


class ExplanationPromptBuilder: 
    def __init__(self, prompt_path):
        self.prompt_format = prompt_path.read_text()
        self.prompt_template = PromptTemplate(
            template = self.prompt_format,
            input_variables=["input", "context"],
            template_format="jinja2"
        )

    def get_template(self):
        return self.prompt_template 
    
    def get_variables(self, context):
        variables = {"input": context.sample.input}
        
        variables["context"] = context.artifacts.get("retrieved_docs", [])
        variables["context"] = "\n\n".join([
           doc.page_content for doc in variables["context"]
        ])

        return variables


class ArticulationPromptBuilder:
    def __init__(self, prompt_path):
        self.prompt_format = prompt_path.read_text() 
        self.prompt_template = PromptTemplate.from_template(self.prompt_format) 

    def get_template(self):
        return self.prompt_template

    def get_variables(self, context):
        return {
            "Explanation": context.output
        }


class ContextualityPromptBuilder:
    def __init__(self, prompt_path):
        self.prompt_format = prompt_path.read_text() 
        self.prompt_template = PromptTemplate.from_template(self.prompt_format) 

    def get_template(self):
        return self.prompt_template

    def get_variables(self, context):
        return {
            "Document": context.sample.input,
            "Explanation": context.output
        }

class LucidityPromptBuilder:
    def __init__(self, prompt_path):
        self.prompt_format = prompt_path.read_text() 
        self.prompt_template = PromptTemplate.from_template(self.prompt_format) 

    def get_template(self):
        return self.prompt_template

    def get_variables(self, context):
        return {
            "Document": context.sample.input,
            "Explanation": context.output
        }


def get_prompt_builder(prompt_path, prompt_format=None):
    str_path = str(prompt_path)

    if 'explanation' in str_path: 
        return ExplanationPromptBuilder(prompt_path) 
    elif 'articulation' in str_path:
        return ArticulationPromptBuilder(prompt_path) 
    elif 'contextuality' in str_path: 
        return ContextualityPromptBuilder(prompt_path) 
    elif 'lucidity' in str_path: 
        return LucidityPromptBuilder(prompt_path) 
    
    raise ValueError(f'{prompt_path}를 위한 PromptBuilder가 지원되지 않습니다.')