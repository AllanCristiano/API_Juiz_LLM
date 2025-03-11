from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ia_constitucional import ComplianceAnalyzer
from relevancia import RelevanciaController

app = FastAPI()

# Modelos para a funcionalidade de relevância
class Consulta(BaseModel):
    pergunta: str = Field(..., example="Qual é a capital da França?")
    resposta: str = Field(..., example="A capital da França é Paris.")

    class Config:
        schema_extra = {
            "example": {
                "pergunta": "Quem descobriu o Brasil?",
                "resposta": "Pedro Álvares Cabral descobriu o Brasil."
            }
        }

class AnaliseRelevanciaResponse(BaseModel):
    pergunta: str = Field(..., example="Quem descobriu o Brasil?")
    resposta: str = Field(..., example="Pedro Álvares Cabral descobriu o Brasil.")
    similaridade: float = Field(..., example=0.95)
    limiar: float = Field(..., example=0.8)
    relevante: bool = Field(..., example=True)
    detalhes: dict = Field(..., example={
        "valorSimilaridade": 0.95,
        "limiarUtilizado": 0.8,
        "status": "Relevante"
    })
    analise: dict = Field(..., example={
        "nivel": "Excelente correspondência",
        "observacoes": ["Relação temática muito forte", "Contextos perfeitamente alinhados"]
    })
    recomendacoes: List[str] = Field(..., example=["Nenhuma recomendação necessária."])

# Modelo para ajuste do limiar
class AjusteLimiarResponse(BaseModel):
    novo_limiar: float = Field(..., example=0.85)

# Modelos para a funcionalidade de conformidade
class TextRequest(BaseModel):
    texto: str = Field(..., example="Monitoramento não declarado de áreas de descanso dos funcionários.")

class AnaliseConformidadeResponse(BaseModel):
    texto_analisado: str = Field(..., example="Monitoramento não declarado de áreas de descanso dos funcionários.")
    versao_sistema: str = Field(..., example="3.2.1")
    conformidade_geral: bool = Field(..., example=True)
    violacoes: dict = Field(..., example={
        "privacidade": {
            "reasons": ["similaridade_alta"],
            "risk_score": 0.7,
            "details": {"termos_detectados": {"exatos": [], "semanticos": []}, "metricas_similaridade": {"valor": 0.65, "threshold": 0.65}}
        }
    })
    detalhes_analise: dict = Field(..., example={
        "privacidade": {
            "similarity_score": 0.65,
            "similarity_threshold": 0.65,
            "detected_terms": {},
            "violation_triggered": False
        }
    })
    erro: Optional[dict] = Field(None, example=None)

    class Config:
        schema_extra = {
            "example": {
                "texto_analisado": "Monitoramento não declarado de áreas de descanso dos funcionários.",
                "versao_sistema": "3.2.1",
                "conformidade_geral": True,
                "violacoes": None,
                "detalhes_analise": {
                    "privacidade": {
                        "similarity_score": 0.65,
                        "similarity_threshold": 0.65,
                        "detected_terms": {},
                        "violation_triggered": False
                    }
                },
                "erro": None
            }
        }

class VersaoResponse(BaseModel):
    versao: str = Field(..., example="3.2.1")

# Instancia dos controllers
relevancia_controller = RelevanciaController()
compliance_analyzer = ComplianceAnalyzer()

# Rotas para a funcionalidade de relevância
@app.post("/avaliar", response_model=AnaliseRelevanciaResponse)
def avaliar_consulta(consulta: Consulta):
    """
    Recebe "pergunta" e "resposta", processa a avaliação de relevância e retorna o relatório detalhado.
    """
    relevancia_controller.processar_consulta(consulta.pergunta, consulta.resposta)
    relatorio = relevancia_controller.exibir_relatorio(-1)
    return relatorio

@app.get("/historico")
def obter_historico():
    """Retorna o histórico completo de avaliações de relevância."""
    return {"historico": relevancia_controller.obter_historico()}

@app.get("/ajustar-limiar", response_model=AjusteLimiarResponse)
def ajustar_limiar():
    """Ajusta automaticamente o limiar com base no histórico de relevância e retorna o novo valor."""
    novo_limiar = relevancia_controller.ajustar_limiar()
    return {"novo_limiar": novo_limiar}

# Rotas para a funcionalidade de conformidade (Compliance)
@app.post("/analisar", response_model=AnaliseConformidadeResponse)
def analisar_texto(request: TextRequest):
    """
    Recebe um JSON com o campo "texto", executa a análise de conformidade e retorna o resultado.
    """
    resultado = compliance_analyzer.analisar(request.texto)
    return resultado.to_dict()

@app.get("/version", response_model=VersaoResponse)
def get_version():
    """Retorna a versão do sistema de análise de conformidade."""
    return {"versao": compliance_analyzer.version}
