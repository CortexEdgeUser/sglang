import argparse
import dataclasses
from typing import List, Dict, Any

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

import sglang as sgl
from sglang.srt.server_args import ServerArgs

class GenerateRequest(BaseModel):
    prompts: List[str]
    sampling_params: Dict[str, Any] = {"temperature": 0.8, "top_p": 0.95}

class GenerateResponse(BaseModel):
    outputs: List[Dict[str, Any]]

# Global variable to hold the LLM engine
llm_engine: sgl.Engine = None

async def lifespan(app: FastAPI):
    """
    Lifespan event handler to manage application startup and shutdown.
    """
    global llm_engine

    # Initialize the LLM engine during startup
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    # Parse known args to allow uvicorn to pass its own arguments
    args, _ = parser.parse_known_args()
    server_args = ServerArgs.from_cli_args(args)

    try:
        print("Initializing LLM Engine...")
        llm_engine = sgl.Engine(**dataclasses.asdict(server_args))
        print("LLM Engine initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM Engine: {e}")
        raise e  # This will prevent the application from starting

    yield  # Control returns to FastAPI to handle requests

    # Optional: Cleanup actions during shutdown
    try:
        print("Shutting down LLM Engine...")
        if hasattr(llm_engine, 'shutdown'):
            llm_engine.shutdown()
            print("LLM Engine shut down successfully.")
        else:
            print("LLM Engine does not have a shutdown method.")
    except Exception as e:
        print(f"Error during LLM Engine shutdown: {e}")

# Initialize FastAPI with the lifespan event handler
app = FastAPI(
    default_response_class=ORJSONResponse,
    lifespan=lifespan
)

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if not llm_engine:
        raise HTTPException(status_code=500, detail="LLM Engine not initialized.")
    
    try:
        outputs = llm_engine.generate(request.prompts, request.sampling_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return GenerateResponse(outputs=outputs)

# Ajout du bloc principal pour démarrer uvicorn automatiquement
if __name__ == "__main__":
    import uvicorn
    import sys

    # Vous pouvez personnaliser les paramètres de l'hôte et du port ici
    uvicorn.run(
        "test:app",  # Remplacez "main" par le nom de votre fichier si différent
        host="0.0.0.0",
        port=8001,
        reload=True,  # Active le rechargement automatique en développement
        log_level="info",
    )
