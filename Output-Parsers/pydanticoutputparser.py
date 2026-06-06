from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
   

load_dotenv()

llm  = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='City of the person')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a superhero character {superhero} \n {format_instruction}',
    input_variables=['superhero'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'superhero':'omni-man'})

''' without chain
print(prompt)
print('-------------------------------')

result = model.invoke(prompt)
print(result)
print('-------------------------------')
final_result = parser.parse(result.content)
print(final_result)
'''

chain = template | model | parser
result = chain.invoke({'superhero':'shazam'})
print(result)