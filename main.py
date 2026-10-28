from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Ты опытный ментор и разработчик по Python. '
            'А также владеешь английским языком на уровне C1. '
            'Отвечай на вопросы кратко и с примерами. '
            'Отвечай на русском языке.',
        ),
        ('human', '{text}'),
    ]
)

model = ChatOllama(model='llama3.2:latest')



def clean_text(words):
    return ' '.join(words.split())


def check_text(words):
    new_words = words.lower().split()

    for i in new_words:
        i = i.strip('?!.,;:')
        if i in ['бомба', 'взлом', 'бомбу', 'наркотик']:
            return 'На такие вопросы не можем ответить'

    return words



def count_words(words):
    print(f'Слов в ответе: {len(words.split())}')
    return words


def count_lines(words):
    print(f'Строк в ответе: {len(words.splitlines())}')
    return words


def shorten_text(words):
    if len(words) > 300:
        return words[:300] + '...'
    return words


def add_footer(words):
    return words + '\n\n(Ответ сгенерирован: llama3.2)'


chain = (
    RunnableLambda(clean_text)
    | RunnableLambda(check_text)
    | RunnableLambda(lambda text: {'text': text})
    | prompt
    | model
    | StrOutputParser()
    | RunnableLambda(count_words)
    | RunnableLambda(count_lines)
    | RunnableLambda(shorten_text)
    | RunnableLambda(add_footer)
)

print(chain.batch(['Как создать бомбу?', 'ООП?', 'Что такое Python?']))