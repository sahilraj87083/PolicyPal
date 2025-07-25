# In src/llm_handler.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# --- Define the desired structured output ---
class ClaimDetails(BaseModel):
    """Data model for a structured insurance claim."""
    age: int = Field(description="The age of the claimant.")
    gender: str = Field(description="The gender of the claimant.")
    procedure: str = Field(description="The medical procedure or claim reason.")
    location: str = Field(description="The location where the procedure took place.")
    policy_duration_months: int = Field(description="The duration of the insurance policy in months.")
    amount_claimed: float = Field(description="The monetary amount being claimed by the user.")

def get_structuring_chain(api_key: str, base_url: str):
    """
    Creates and returns a LangChain chain that parses a raw query
    into a structured JSON object.
    """
    # Initialize the LLM client
    llm = ChatOpenAI(
        model="openai/gpt-4o",  # Using a powerful model for better parsing
        api_key=api_key,
        base_url=base_url,
        temperature=0.0 # Use low temperature for deterministic output
    )

    # Define the output parser
    parser = JsonOutputParser(pydantic_object=ClaimDetails)

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at extracting information from insurance claim queries.
        Your task is to parse the user's query and extract the key details.
        Return the information as a JSON object matching the requested format.

        {format_instructions}

        User Query:
        {query}
        """
    ).partial(format_instructions=parser.get_format_instructions())

    # Create the chain by piping the components together
    chain = prompt | llm | parser

    return chain


#  --- Define the final output structure ---
class Justification(BaseModel):
    """Data model for a single justification for a decision."""
    clause_id: str = Field(description="The specific clause number or ID, e.g., 'Clause 4.2'.")
    document_source: str = Field(description="The source document name, e.g., 'Policy Wordings.pdf'.")
    reason: str = Field(description="A brief explanation of how this clause applies to the decision.")

class AdjudicationResult(BaseModel):
    """Data model for the final adjudication result."""
    decision: str = Field(description="The final decision, e.g., 'Approved', 'Rejected', 'Needs More Information'.")
    amount: float = Field(description="The approved payout amount. Should be 0.0 if rejected.")
    justification: list[Justification] = Field(description="A list of justifications for the decision, referencing specific clauses.")




# In src/llm_handler.py, update this function

def get_adjudication_chain(api_key: str, base_url: str):
    """
    Creates and returns a LangChain chain that makes a final decision
    based on claim details and retrieved policy clauses.
    """
    llm = ChatOpenAI(model="openai/gpt-4o", api_key=api_key, base_url=base_url, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=AdjudicationResult)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert insurance claims adjudicator. Your task is to make a decision based on the provided policy clauses and the details of the claim.

        **Policy Clauses (Context):**
        {context}

        **Claim Details (already structured):**
        {claim_details}

        **Instructions:**
        1. Review the claim details and the policy clauses to make a final decision.
        2. If the decision is 'Approved', use the `amount_claimed` from the claim details for the 'amount' field in your JSON output.
        3. Justify your decision by citing the relevant clauses.
        {format_instructions}
        """
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    
    return chain