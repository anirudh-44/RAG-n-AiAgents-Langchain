from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm  = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of {superhero}. \n {format_instructions}',
    input_variables=['superhero'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

#'''     without chains
prompt = template.format(superhero='ironman')

result = model.invoke(prompt)
final_result = parser.parse(result.content)

print(final_result)
print(type(final_result))

'''
chain = template | model | parser
result = chain.invoke({'superhero':'spiderman'})
print(result)'''