from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

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
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template = 'Summarize the following text \n - {text}',
    input_variables=['text']
)


def word_count(text):
    return len(text.split()) > 100

report_generator_chain = RunnableSequence(prompt1,model1,parser)

branch_chain = RunnableBranch(
    (RunnableLambda(word_count), RunnableSequence(prompt2,model2,parser)), #(condition, runnable),
    RunnablePassthrough()                                                        #default
)

final_chain = RunnableSequence(report_generator_chain, branch_chain)

result = final_chain.invoke({'topic':'cricket'}) 
print(result)