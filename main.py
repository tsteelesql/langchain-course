from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """Felix Hausdorff (/ˈhaʊsdɔːrf/ HOWS-dorf, /ˈhaʊzdɔːrf/ HOWZ-dorf;[1] November 8, 1868 – January 26, 1942[2]) was a German mathematician, pseudonym Paul Mongré (à mon gré (Fr.) = "according to my taste"),[3] who is considered to be one of the founders of modern topology and who contributed significantly to set theory, descriptive set theory, measure theory, and functional analysis.

Hausdorff was Jewish, and life became difficult for him and his family after the Kristallnacht of 1938. The next year he initiated efforts to emigrate to the United States, but was unable to make arrangements to receive a research fellowship. On 26 January 1942, Hausdorff, along with his wife and his sister-in-law, died by suicide by taking an overdose of veronal, rather than comply with German orders to move to the Endenich camp, and there suffer the likely implications, about which he held no illusions. 
    """

    summary_template=f"""
    given the information {information} about a person I want you to create:
    1) A short summary
    2) two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0,model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})

    print(response.content)

if __name__ == "__main__":
    main()
