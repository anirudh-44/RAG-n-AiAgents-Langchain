from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import HumanMessage,AIMessage

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}') # today's latest query
])

chat_history = []
# load chat history
with open('chat_history.txt') as file:
    #chat_history.extend(file.readlines())
    for line in file:
        line = line.strip()

        if line.startswith("HumanMessage(content="):
            content = line[len('HumanMessage(content="'):-2]
            chat_history.append(HumanMessage(content=content))

        elif line.startswith("AIMessage(content="):
            content = line[len('AIMessage(content="'):-2]
            chat_history.append(AIMessage(content=content))
#print(chat_history)

# create prompt 
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})
print(prompt)