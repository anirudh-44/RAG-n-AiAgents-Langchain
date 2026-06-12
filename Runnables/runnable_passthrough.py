from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

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
    template = 'Write a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the joke - {joke}',
    input_variables=['joke']
)

joke_generator_chain = RunnableSequence(prompt1,model1,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model2,parser)
})

final_chain = RunnableSequence(joke_generator_chain,parallel_chain)

result = final_chain.invoke({'topic':'Agentic AI Engineering'}) 
print(result)