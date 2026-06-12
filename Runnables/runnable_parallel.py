from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm1  = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation"
)

llm2  = HuggingFaceEndpoint(
        repo_id = "deepseek-ai/DeepSeek-V4-Flash",
        task = "text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a LinkedIn post on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a X(twitter) post on {topic}',
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'linkedIn':RunnableSequence(prompt1,model1,parser),
    'tweet':RunnableSequence(prompt2,model2,parser),
})

result = parallel_chain.invoke({'topic':'Agentic AI Engineering'}) 
print(type(result))
print(result)