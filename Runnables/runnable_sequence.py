from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm1  = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation"
)

'''
llm2  = HuggingFaceEndpoint(
        repo_id = "deepseek-ai/DeepSeek-V4-Flash",
        task = "text-generation"
)
'''

model1 = ChatHuggingFace(llm=llm1)
#model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Write a joke on {topic}',
    input_variables=['topic']
)

chain = RunnableSequence(prompt,model1,parser)
print(chain.invoke({'topic':'cricket'}))