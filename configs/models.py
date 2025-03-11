from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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
