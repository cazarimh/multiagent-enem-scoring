import json
import re
import unicodedata
from typing import Optional, TypedDict

import spacy
from langchain_core.tools import tool
from language_tool_python import LanguageTool

language_tool = LanguageTool("pt-BR")
nlp = spacy.load("pt_core_news_sm")

def _morph(token: spacy.tokens.Token, key: str) -> Optional[str]:
  values = token.morph.get(key)
  return values[0] if values else None

# Competência 1

class SpellingMatch(TypedDict):
  message: str
  replacements: list[str]
  context: str

@tool
def check_text_spelling(text: str) -> list[SpellingMatch]:
  """Verifica erros de ortografia, gramática e pontuação em português (norma culta)."""
  matches = language_tool.check(text)

  if not matches:
    return [{
      "message": "Nenhum erro encontrado",
      "replacements": [],
      "context": text
    }]

  return [
    {
      "message": match.message,
      "replacements": match.replacements[:3],
      "context": match.context
    }
    for match in matches
  ]

class AgreementIssue(TypedDict):
  kind: str
  snippet: str
  message: str

def _span_text(doc: spacy.tokens.Doc, *tokens: spacy.tokens.Token) -> str:
  start = min(token.i for token in tokens)
  end = max(token.i for token in tokens)
  return doc[start:end + 1].text

@tool
def analyze_agreement(text: str) -> list[AgreementIssue]:
  """Detecta possíveis erros de concordância verbal e nominal (Competência I)."""
  doc = nlp(text)
  issues: dict[tuple[str, str], AgreementIssue] = {}

  for token in doc:
    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubj:pass")]
    if not subjects:
      continue

    if token.pos_ == "VERB" and "Fin" in token.morph.get("VerbForm"):
      verb = token
    else:
      copulas = [
        child for child in token.children
        if child.dep_ == "cop" and "Fin" in child.morph.get("VerbForm")
      ]
      verb = copulas[0] if copulas else None

    if verb is None:
      continue

    for subject in subjects:
      subject_number, verb_number = _morph(subject, "Number"), _morph(verb, "Number")
      disagreement = (
        subject_number and verb_number and subject_number != verb_number
      ) or (
        subject.pos_ == "PRON"
        and _morph(subject, "Person") and _morph(verb, "Person")
        and _morph(subject, "Person") != _morph(verb, "Person")
      )
      if disagreement:
        snippet = _span_text(doc, subject, verb)
        issues.setdefault(("verbal", snippet), {
          "kind": "verbal",
          "snippet": snippet,
          "message": (
            f"O sujeito '{subject.text}' e o verbo '{verb.text}' "
            f"não concordam em número/pessoa."
          )
        })

  for token in doc:
    if token.pos_ != "NOUN":
      continue
    noun_gender, noun_number = _morph(token, "Gender"), _morph(token, "Number")
    for child in token.children:
      if child.dep_ not in ("amod", "det"):
        continue
      gender, number = _morph(child, "Gender"), _morph(child, "Number")
      if (gender and noun_gender and gender != noun_gender) or (
        number and noun_number and number != noun_number
      ):
        snippet = _span_text(doc, token, child)
        issues.setdefault(("nominal", snippet), {
          "kind": "nominal",
          "snippet": snippet,
          "message": (
            f"'{child.text}' e '{token.text}' não concordam em gênero/número."
          )
        })

  return list(issues.values())

class CapitalizationIssue(TypedDict):
  snippet: str
  message: str

@tool
def analyze_capitalization(text: str) -> list[CapitalizationIssue]:
  """Detecta frases que iniciam com letra minúscula (Competência I)."""
  doc = nlp(text)
  issues: list[CapitalizationIssue] = []

  for sent in doc.sents:
    first = next((token for token in sent if token.is_alpha), None)
    if first is None or not first.text[0].islower():
      continue
    issues.append({
      "snippet": sent.text.strip()[:60],
      "message": f"A frase inicia com letra minúscula: '{first.text}'."
    })

  return issues

# Competência 2

class Citation(TypedDict):
  author: str
  marker: str
  sentence: str

_CITATION_MARKERS = [
  "segundo", "conforme", "de acordo com", "na visão de", "na perspectiva de",
  "nas palavras de", "segundo o", "segundo a", "como afirma", "como defende",
  "como diz", "afirma que", "defende que"
]

_MARKER_PATTERN = "|".join(
  sorted((re.escape(marker) for marker in _CITATION_MARKERS), key=len, reverse=True)
)

_NAME_PATTERN = r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ]+)*"

@tool
def extract_citations(text: str) -> list[Citation]:
  """Extrai citações e menções atribuídas a autores/documentos para validação (Competência 2)."""
  sentences = re.split(r"(?<=[.!?])\s+", text)

  found: dict[tuple[str, str, str], Citation] = {}

  for sentence in sentences:
    for match in re.finditer(
      rf"(?:{_MARKER_PATTERN})\s+(?:o\s+|a\s+|os\s+|as\s+)?", sentence, re.IGNORECASE
    ):
      marker = match.group(0).strip()
      author_match = re.search(_NAME_PATTERN, sentence[match.end():])
      author = author_match.group(0) if author_match else ""

      key = (author, marker.lower(), sentence.strip())
      found.setdefault(key, {
        "author": author,
        "marker": marker,
        "sentence": sentence.strip()
      })

  return list(found.values())

# Competência 4

class ConnectiveUsage(TypedDict):
  connective: str
  count: int
  paragraphs: list[int]

def _strip_accents(value: str) -> str:
  return "".join(
    char for char in unicodedata.normalize("NFD", value)
    if unicodedata.category(char) != "Mn"
  )

_CONNECTIVES = [
  "além disso", "ademais", "outrossim", "bem como", "também", "ainda",
  "mas", "porém", "contudo", "todavia", "entretanto", "no entanto",
  "por outro lado", "em contrapartida", "ao passo que", "embora",
  "apesar de", "ainda que",
  "porque", "pois", "já que", "uma vez que", "visto que", "dado que",
  "por isso",
  "portanto", "assim", "logo", "dessa forma", "desse modo", "em suma",
  "em síntese", "por fim", "finalmente",
  "isto é", "ou seja", "a saber", "por exemplo", "com efeito",
  "assim como", "tal qual", "enquanto",
  "para que", "a fim de", "com o objetivo de",
  "depois", "antes", "logo que", "assim que"
]

@tool
def extract_connectives(text: str) -> list[ConnectiveUsage]:
  """Mapeia conectivos/operadores argumentativos por quantidade e parágrafo (Competência 4)."""
  paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

  usages: dict[str, ConnectiveUsage] = {}

  for index, paragraph in enumerate(paragraphs, start=1):
    lowered = _strip_accents(paragraph.lower())
    for connective in _CONNECTIVES:
      pattern = rf"\b{re.escape(_strip_accents(connective))}\b"
      occurrences = len(re.findall(pattern, lowered))
      if occurrences:
        usage = usages.setdefault(
          connective, {"connective": connective, "count": 0, "paragraphs": []}
        )
        usage["count"] += occurrences
        if index not in usage["paragraphs"]:
          usage["paragraphs"].append(index)

  return sorted(usages.values(), key=lambda usage: (-usage["count"], usage["connective"]))

# Orquestrador

class JsonValidationResult(TypedDict):
  valid: bool
  missing_keys: list[str]
  error: Optional[str]

@tool
def validate_json_format(
  raw_json: str, required_keys: Optional[list[str]] = None
) -> JsonValidationResult:
  """Valida se uma saída é JSON sintaticamente correto e contém as chaves exigidas."""
  try:
    data = json.loads(raw_json)
  except json.JSONDecodeError as exc:
    return {
      "valid": False,
      "missing_keys": [],
      "error": f"{exc.msg} (linha {exc.lineno}, coluna {exc.colno})"
    }

  if not isinstance(data, dict):
    return {
      "valid": False,
      "missing_keys": [],
      "error": "A raiz do JSON não é um objeto."
    }

  missing = [key for key in (required_keys or []) if key not in data]
  return {"valid": not missing, "missing_keys": missing, "error": None}
