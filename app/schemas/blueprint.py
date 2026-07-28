"""API so'rov va javob sxemalari."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Lang = Literal["uz", "ru", "en"]
Region = Literal["uz", "eu", "us", "global"]


class AnalyzeRequest(BaseModel):
    description: str = Field(min_length=20, max_length=6000, description="Startup g'oyasi erkin matnda")
    monthly_users: int = Field(default=1000, ge=10, le=50_000_000, description="Kutilayotgan oylik faol foydalanuvchi")
    region: Region = "uz"
    lang: Lang = "uz"
    brand_hint: str = Field(default="", max_length=40, description="Brend nomi (domen taklifi uchun)")
    check_domains: bool = True


class ClarifyRequest(BaseModel):
    """Aniqlashtiruvchi savollar so'rovi — generatsiyadan oldingi qadam."""

    description: str = Field(min_length=20, max_length=6000, description="Startup g'oyasi erkin matnda")
    lang: Lang = "uz"
    count: int = Field(default=5, ge=3, le=6, description="Nechta savol berilsin")


class QuestionOut(BaseModel):
    id: str
    question: str
    hint: str = ""
    examples: list[str] = Field(default_factory=list)


class ClarifyResponse(BaseModel):
    project_type: str
    project_title: str
    questions: list[QuestionOut]
    generated_by_llm: bool
    warning: Optional[str] = None


class AnswerIn(BaseModel):
    """Savolga berilgan javob — promptga alohida bo'lim bo'lib kiradi."""

    question: str = Field(max_length=500)
    answer: str = Field(default="", max_length=2000)


class GenerateRequest(AnalyzeRequest):
    target_model: str = Field(default="anthropic/claude-opus-5", description="OpenRouter model id")
    target_model_name: str = Field(default="", max_length=120)
    save: bool = True
    # Bo'sh javoblar ham kelishi mumkin — filtrlash server tomonda.
    answers: list[AnswerIn] = Field(
        default_factory=list,
        max_length=10,
        description="Aniqlashtiruvchi savollarga javoblar",
    )


class ChoiceOut(BaseModel):
    category: str
    pick: str
    why: str
    alternatives: list[str]


class AnalyzeResponse(BaseModel):
    project_type: str
    project_title: str
    confidence: float
    alternatives: list[dict]
    signals: dict[str, bool]
    detected_users: Optional[int] = None
    monthly_users: int
    stack: list[ChoiceOut]
    anti_patterns: list[str]
    requirements: dict
    server_options: list[dict]
    scaling: list[str]
    prices_updated: str
    uiux: dict
    domains: list[dict]
    registrars: list[dict]
    payment: Optional[dict]
    needs_clarification: bool
    clarify_question: Optional[str] = None


class GenerateResponse(BaseModel):
    id: Optional[str]
    prompt: str
    skeleton: str
    used_llm: bool
    target_model: str
    sources: list[dict]
    warning: Optional[str] = None
    analysis: AnalyzeResponse


class ModelOut(BaseModel):
    id: str
    name: str
    family: str
    context_length: int
    prompt_usd_per_1m: Optional[float]
    completion_usd_per_1m: Optional[float]
    description: str


class PaymentCreateRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)
    amount_uzs: int = Field(ge=1000, le=100_000_000)
    return_url: str = Field(default="http://localhost:3000/payment/done", max_length=500)
    provider: Optional[Literal["mock", "payme", "click"]] = None
