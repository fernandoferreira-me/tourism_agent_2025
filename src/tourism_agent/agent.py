from langchain.llms import OpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

import logging

logging.basicConfig(level=logging.INFO)


class TravelTemplate:
    def __init__(self):
        self.system_template = """
        You are a Brazilian travel agent that give nice and cheerful advices
        for your customers about travels around the world. Your main goal is
        to plan in little details every step of your customer trip.

        The customer request will be denoted by four hashtags.

        Give your answer as list of places that they should visit. Initially,
        your customer will ask for informations in Portuguese about a place to
        go and you will answer with a list.

        For example:
        ++++
        #### O que fazer no Rio de Janeiro?

          - No primeiro dia, faça checking no hotel
          - Ainda no primeiro dia, visite a praia de Copacabana e experimente
          agua de coco.
        ++++
        """

        self.human_template = """
        #### {request}
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt,
                                                             self.human_message_prompt])
        
class Agent:
    def __init__(self, open_ai_key, model="gpt-4-turbo", temperature=0.1):
        self.open_ai_key = open_ai_key
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.chat_model = ChatOpenAI(model=self.model,
                                     temperature=self.temperature,
                                     openai_api_key=self.open_ai_key)

    def get_tips(self, request):
        travel_prompt = TravelTemplate()
        parser = LLMChain(
            llm=self.chat_model,
            prompt=travel_prompt.chat_prompt,
            output_key="travel_tips"
        )

        chain = SequentialChain(
            chains=[parser],
            input_variables=["request"],
            output_variables=["travel_tips"],
            verbose=True
        )
        return chain(
            {"request": request},
            return_only_outputs=True
        )