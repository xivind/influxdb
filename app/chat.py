#!/usr/bin/env python3

from typing import AsyncGenerator
import chromadb
from openai import AsyncOpenAI
import tiktoken
from icecream import ic
from dotenv import load_dotenv
import json
from .utils import read_json_file

CONFIG = read_json_file('app/config.json')
SECRETS = read_json_file('secrets.json')

# Initialize OpenAI API client
client = AsyncOpenAI(api_key=SECRETS['openai_api_key'])

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CONFIG['chroma_db_path'])
collection = chroma_client.get_collection(CONFIG['collection_name'])
high_level_collection = chroma_client.get_collection("high_level_embeddings")  # New high-level collection

async def get_relevant_context(query: str) -> str:
    """
    Retrieve relevant context from the vector database with hierarchical retrieval.
    First, query the high-level collection, then refine the search by querying the low-level collection.
    """
    # Get embeddings for the query
    response = await client.embeddings.create(
        model=CONFIG['embedding_model_name'],
        input=query)
    
    query_embedding = response.data[0].embedding

    # First, query the high-level collection to get relevant categories
    high_level_results = high_level_collection.query(
        query_embeddings=[query_embedding],
        n_results=CONFIG['top_k'])

    # Combine relevant high-level contexts
    relevant_high_level = high_level_results['documents']
    
    # Debug: Print the structure of the context
    print(f"High level results structure: {relevant_high_level}")
    
    # Check if the result contains a 'category' field (assuming the results have this field)
    low_level_contexts = []
    for context in relevant_high_level:
        if isinstance(context, dict) and 'category' in context:
            # Query low-level collection based on each high-level result
            low_level_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=CONFIG['top_k'],
                where={"category": context['category']}  # Filter by high-level category
            )
            low_level_contexts.extend(low_level_results['documents'])
        else:
            # If no category field, handle accordingly (e.g., skip or use the entire content)
            low_level_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=CONFIG['top_k']
            )
            low_level_contexts.extend(low_level_results['documents'])

    # Ensure low_level_contexts is a list of strings
    low_level_contexts = [str(item) for item in low_level_contexts]

    # Combine all relevant contexts (both high-level and low-level)
    full_context = "\n\n---\n\n".join(low_level_contexts)
    return full_context



async def generate_response(query: str, context: str) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response using the OpenAI API

    """
    prompt = f"""Oppfør deg som en ekspert på digitalisering innen helse- og omsorgssektoren i Norge.
    Din oppgave er å veilede om krav og anbefalinger som gjelder digitalisering i helse- og omsorgssektoren i Norge.
    Du skal legge spesielt vekt på datagrunnlaget som inngår i denne prompten, men du kan også støtte deg på informasjon fra internett.
    I så fall skal du legge særlig vekt på informasjon fra ehelse.no, hdir.no og lovdata.no. Du skal gi så fullstendige svar som mulig.
    Pass på å ikke utelate noe. For eksempel må du huske å nevne både lover, forskrifter og fortolkninger. 

    Når du lister opp elementer som kodeverk, standarder, eller andre krav, må du alltid: 
    - sjekke datagrunnlaget systematisk og grundig for å finne alle relevante elementer
    - liste opp alle elementene du finner, ikke bare de mest åpenbare
    - gruppere elementene på en logisk måte
    - forklare hvis det er relasjoner mellom elementene

    VIKTIG OM FORMATTERING:
    Du skal svare med HTML-formattering. Bruk følgende HTML-elementer:
    - <h1> for hovedoverskrift
    - <h2> for underoverskrifter
    - <p> for tekstavsnitt
    - <ul> og <li> for punktlister
    - <ol> og <li> for nummererte lister
    - <a href="url"> for lenker
    - <strong> for uthevet tekst
    - <br> for linjeskift der det trengs

    VIKTIG: 
    - IKKE start svaret med ```html
    - IKKE avslutt svaret med ``
    - Bruk komplette HTML-tags (<ul><li>punkt</li></ul>, ikke bare 'ul>')
    - IKKE skriv 'ul>' separat
    - IKKE skriv 'ol>' separat

    Eksempel på formattering:
    <h1>Hovedtittel</h1>
    <p>Et avsnitt med tekst som kan inneholde <strong>uthevet tekst</strong> og <a href="https://ehelse.no">lenker</a>.</p>
    <h2>Undertittel</h2>
    <ul>
        <li>Punkt 1</li>
        <li>Punkt 2</li>
    </ul>

    Om informasjonsmodellen / metamodellen
Reguleringsplanen inneholder krav og anbefalinger som er strukturert i henhold til en informasjonsmodell. Modellen er delt inn i henhold til rammeverk for digital samhandling i juridiske, organisatoriske, semantiske og tekniske krav og anbefalinger. Informasjonsmodellen inneholder også elementer som reguleringsplanen kan utvides med senere etter behov. Samlet sett beskriver metamodellen de ulike nivåene av samhandling i eller på tvers av virksomheter, fra juridiske rammer til teknisk implementering. Metamodellen beskriver ulike typer samhandlingsevne, delt inn i fire hovedkategorier: 

Innenfor hvert av disse områdene er det noen hovedgrupper av informasjonselementer:
Juridisk: lover, forskrifter, faktaark og veiledere til normen. Dette er rettslige rammer.
Organisatorisk: informasjonstjenester, organisatoriske krav og prinsipper, nasjonale e-helseløsninger, organisatoriske samhandlingsformer.
Semantisk: informasjonsmodeller, kodeverk og terminologi, semantiske krav og prinsipper. Handler om felles forståelse og tolkning av informasjon, semantikk, inkludert informasjonsmodeller, kodeverk og terminologi samt tilhørende krav og prinsipper.
Teknisk: samhandlingskomponenter, datalager, tekniske grensesnitt, teknisk samhandlingsform, tekniske krav og prinsipper.

Nærmere beskrivelse av det enkelte element i informasjonsmodellen:
Her følger beskrivelse av informasjonselement, interoperabilitetsnivå, beskrivelse og eksempel skilt med semikolon:
Lov og forskrift; Juridisk; Norsk lov og forskrift; Pasientjournalloven, Kjernejournalforskriften.
Avtale; Juridisk; Avtale mellom to eller flere parter; Bruksvilkår for bruk av kjernejournal.
Rundskriv, veiledninger og tolkninger; Juridisk; Føringer og anbefalinger fra et departement eller direktoratet om forståelse eller anvendelse av regelverk mv.; Digitaliseringsrundskrivet.
Bransjenorm; Juridisk; Bransjens / sektorens egne retningslinjer, ofte basert på lovverk. Utvikles av bransjen / sektoren selv, i noen tilfeller i samarbeid med myndigheter og andre; Norm for informasjonssikkerhet og personvern i helse- og omsorgssektoren (Normen).
Nasjonal e-helseløsning; Organisatorisk; Samhandlingskomponent og / eller informasjonslager som er definert som en nasjonal e-helseløsning i pasientjournalloven § 8; Helsenorge, kjernejournal, e-resept og helsenettet (som inkluderer nasjonal infrastruktur, felles tjenester og felleskomponenter for utveksling av opplysninger med virksomheter i helse- og omsorgstjenesten).
Samhandlingstjeneste; Organisatorisk; En samhandlingstjeneste gjør det mulig å dele informasjon mellom helsepersonell og med innbygger; Pasientens journaldokumenter.
Krav og prinsipper; Organisatorisk; Krav og prinsipper som retter seg mot elementer på det organisatoriske laget; Forretningsprinsipper for dokumentdeling; f.eks. 'Det skal sentralt føres oversikt over hvem som har hentet ned et dokument'.
Arbeidsprosess; Organisatorisk; Navn på en beskrivelse av en overordnet arbeidsprosess. En arbeidsprosess kan omfatte organisatoriske samhandlingsformer; Generisk pasientforløp for spesialisthelsetjenesten.
Aktørtype; Organisatorisk; En type virksomhet eller gruppering av virksomheter / personer som spiller en aktiv rolle på et bestemt område; Nasjonal tjenestetilbyder, regionalt helseforetak, kommuner, fastleger.
Organisatorisk samhandlingsform; Organisatorisk; Helsepersonell trenger tjenester
Context: {context}

Question: {query}

Answer:"""

    response = await client.chat.completions.create(
        model=CONFIG['completion_model_name'],
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens'],
        stream=True)

    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

async def get_streaming_response(query: str) -> AsyncGenerator[str, None]:
    """
    Main function to handle the chat workflow
    """
    context = await get_relevant_context(query)
    async for token in generate_response(query, context):
        yield token 
