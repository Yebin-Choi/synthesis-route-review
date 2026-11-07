"""Shared presentation: theme, structure and a concise evidence-first material card."""
from io import BytesIO
import json

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

import ui_kit
from molecule_catalog import MOLECULES
from toxicology import canonicalize_smiles

STATUS = {
    "known_concern": "ICH M7 목록 물질",
    "structural_alert": "경고구조",
    "no_alert_detected": "경고구조 없음",
    "invalid": "구조를 읽지 못함",
}
# Short hover explanations for terms a reader cannot guess from the label.
GLOSSARY = {
    "ICH M7 목록 물질": "의약품 불순물 지침 ICH M7(R2)이 변이원으로 이름을 올린 물질입니다.",
    "경고구조": "변이원과 자주 함께 나타나는 부분 구조가 분자 안에서 발견됐습니다.",
    "경고구조 없음": "이 목록과 구조 규칙에서 걸리는 항목을 찾지 못했습니다.",
    "구조를 읽지 못함": "SMILES 문자열을 분자 구조로 읽지 못했습니다.",
    "모델 양성": "선별 정책에서 고른 모델이 Ames 시험 양성으로 예측했습니다. "
    "기본 설정은 ADMET-AI이며 기준선은 0.50입니다.",
    "모델 미평가": "모델이 이 물질에 결과를 내지 못했습니다. 음성이라는 뜻이 아닙니다.",
    "모델 음성": "선별 정책에서 고른 모델이 Ames 시험 음성으로 예측했습니다. "
    "안전하다는 뜻이 아니라 이 모델이 양성 신호를 찾지 못했다는 뜻입니다.",
    "반응 조건 미기재": "온도, 용매, 시간 같은 실제 반응 조건이 경로 자료에 없습니다.",
    "부산물 미평가": "이 단계에서 어떤 부산물이 나오는지 아직 계산하지 않았습니다.",
    "빠진 항목": "판단에 필요한 항목이 경로 자료에 비어 있습니다.",
    "반응 조건 없음": "온도, 용매, 시간 같은 실제 반응 조건이 경로 자료에 없습니다.",
    "자료 없음": "이 항목이 경로 자료에 비어 있습니다.",
    "검토 필요": "전문가가 직접 확인할 신호가 경로에 남아 있습니다.",
    "정보 부족": "판정에 필요한 항목이 비어 있어 결론을 내지 않았습니다.",
    "후보": "연구 정책 기준으로 제외되지 않고 남은 경로입니다.",
    "우려 신호로 제외": "연구 정책 기준에 걸려 기본 목록에서 빠진 경로입니다.",
    "AMES 점수": "ADMET-AI가 낸 0에서 1 사이 숫자입니다. 0.50 이상을 양성으로 봅니다. "
    "이 0.50은 우리가 같은 자료에서 재어 보니 이 도구에 맞는 값이었습니다"
    "(균형 정확도 0.50에서 0.749, 0.25에서 0.700). 확률이 아니라 물질 사이의 "
    "순위를 매기는 점수입니다.",
    "경고구조 없음": "경고구조가 걸리지 않았다는 뜻이고 음성 예측이 아닙니다. "
    "Toxtree에서 파생된 ISS 문서도 '경고구조 없음'은 음성 예측과 같지 않다고 적습니다. "
    "우리가 잰 규칙 화면 민감도는 0.552로, 실제 양성의 절반 가까이에서 경보가 "
    "걸리지 않습니다.",
    "예측 결과 없음": "도구가 스스로 판정을 내지 않았거나 실행이 실패했습니다. "
    "음성이 아닙니다. ICH M7 (Q)SAR 분석 지침도 적용 범위 밖과 상충된 결과를 "
    "음성으로 세지 말라고 합니다.",
    "상충된 결과": "적용 범위 안이지만 결론을 내지 못한 경우입니다. 음성이 아닙니다.",
    "적용 범위 밖": "이 물질을 판단할 근거가 모델에 없습니다. 음성이 아니고, "
    "위험하다는 뜻도 아닙니다.",
    "판정 기준선": "점수가 이 값 이상이면 양성으로 표시합니다. 점수를 내는 도구는 "
    "모두 0.50을 씁니다. 자체 모델은 자료가 양성 30%로 치우친 것을 학습 가중치로 "
    "보정했기 때문에 기준선을 옮기지 않아도 됩니다. 규칙 도구에는 기준선이 없습니다.",
    "모델 적용 범위": "이 물질이 학습 구조와 얼마나 가까운지, 그리고 그 구간에서 "
    "모델이 실제로 얼마나 맞았는지를 보여줍니다. 모델에 대한 설명이고 물질이 "
    "위험한지에 대한 설명이 아닙니다.",
    "applicability": "모델 학습 범위와 이 물질이 얼마나 가까운지를 가리키는 표시입니다.",
    "AUC": "양성과 음성을 하나씩 뽑았을 때 양성 쪽에 더 높은 점수가 가는 비율입니다. "
    "0.5면 동전 던지기와 같습니다.",
    "균형 정확도": "양성 쪽 적중률과 음성 쪽 적중률의 평균입니다. 한쪽이 많아도 수치가 "
    "부풀지 않습니다.",
    "TTC": "독성 자료가 없는 불순물에 쓰는 하루 섭취 참고 한도입니다.",
    "특별 우려 구조": "ICH M7이 TTC보다 낮은 한도를 따로 두는 구조군입니다.",
    "UV면적%": "HPLC 자외선 검출기 피크 면적의 비율입니다.",
}


# A tool that declines says why, and the reason reaches the screen. The
# services write those reasons in English because they also go into the saved
# record, the provenance and the tests, where the exact wording is checked.
# Translating them there would change what is saved; translating them here
# changes only what is read. The match is on a fragment rather than the whole
# sentence, so a service wording that gains a clause still maps.
REASON_TEXT = (
    ("could not read this structure", "모델이 이 구조를 읽지 못했습니다."),
    ("rejected its input", "작업 프로그램이 입력을 받지 않았습니다."),
    ("missing a dependency", "모델 환경에 필요한 패키지가 없습니다."),
    ("not installed", "모델 환경에 필요한 패키지가 설치되어 있지 않습니다."),
    ("model files are missing", "모델 파일이 없습니다."),
    ("model resources are missing", "모델 파일이 없습니다."),
    ("could not complete prediction", "모델이 예측을 끝내지 못했습니다."),
    ("did not return the AMES endpoint", "도구가 Ames 결과를 돌려주지 않았습니다."),
    ("out-of-range", "도구가 쓸 수 없는 점수를 돌려줬습니다."),
    ("non-numeric", "도구가 쓸 수 없는 점수를 돌려줬습니다."),
    ("structure identity", "도구가 요청한 구조와 다른 행을 돌려줬습니다."),
    ("different number of rows", "도구가 요청과 다른 개수의 행을 돌려줬습니다."),
    ("duplicate", "도구가 같은 구조에 중복된 행을 돌려줬습니다."),
    ("timed out", "시간 안에 끝나지 않았습니다."),
    # Route generation. Same approach: the service keeps its English wording,
    # because the saved record and the tests check it, and the screen says the
    # same thing in the language it is written in. Each line says what to do,
    # because a reader who meets one of these is stuck until they act.
    ("escaped the owned request boundary",
     "작업 폴더 밖의 파일이 감지되어 중간에 멈췄습니다. 실행 중 만들어지는 임시 "
     "파일 때문에 드물게 생깁니다. '다시 확인'을 누르면 대개 넘어갑니다."),
    ("could not start the isolated worker",
     "설정된 인터프리터가 작업 프로세스를 띄우지 못했습니다. 그 환경에 필요한 "
     "패키지가 갖춰져 있는지 확인하세요."),
    ("cannot import a required original backend",
     "설정된 인터프리터에 필요한 패키지가 없습니다. 그 환경에서 따로 설치해야 합니다."),
    ("interpreter is missing",
     "설정 파일에 적힌 Python 실행 파일이 없습니다. generation.local.json의 경로를 "
     "확인하세요."),
    ("Set an explicit interpreter",
     "generation.local.json에 쓸 Python 실행 파일을 직접 적어야 합니다. 기본 환경이나 "
     "저장된 경로를 대신 쓰지 않습니다."),
    ("configuration is invalid",
     "생성 설정이 올바르지 않습니다. 절대 경로만 쓰고 자격 증명이나 환경 변수는 "
     "넣지 마세요."),
    ("configured inputs changed",
     "요청 도중 설정이 바뀌었습니다. 설정을 확인한 뒤 다시 실행하세요."),
    ("Required local models or stocks are missing",
     "필요한 모델이나 재고 파일이 없습니다. 내려받기를 대신 시도하지 않았습니다."),
    ("No route containing reaction steps",
     "주어진 탐색 범위 안에서 반응 단계가 있는 경로를 찾지 못했습니다. 목표물이 이미 "
     "재고에 있을 수 있고, 합성이 불가능하다는 뜻은 아닙니다."),
    ("exceeded its declared wall-clock limit",
     "정해진 시간 안에 끝나지 않아 멈췄습니다."),
    ("did not complete the request",
     "백엔드가 요청을 끝내지 못했습니다. 자동으로 다시 시도하거나 다른 방법으로 "
     "대체하지 않았습니다."),
    ("requires a separately supplied personal API key",
     "SynthEx는 개인 API 키와 이 요청에 대한 동의가 있어야 돌아갑니다."),
    ("valid complete target SMILES",
     "4096자 이하의 완전한 목표 SMILES가 필요합니다."),
    ("outside_applicability_domain", "이 도구의 적용 범위 밖입니다."),
    ("unreadable_structure", "구조를 해석하지 못했습니다."),
    ("tool_not_installed", "도구가 설치되어 있지 않습니다."),
)


def reason_text(reason):
    """A tool's reason for not answering, in the screen's language.

    An unmatched reason is shown as it came rather than hidden, because a
    blank where a reason belongs is the thing this is here to avoid.
    """
    if not reason:
        return reason
    text = str(reason)
    lowered = text.casefold()
    for marker, korean in REASON_TEXT:
        if marker.casefold() in lowered:
            return korean
    return text


# The model services write their limitations in English, because the same
# strings travel in the saved envelope and are checked there. The screen is
# Korean, so the same statements are said in Korean here. These are safety
# statements, so each one keeps its exact claim; anything unmapped is shown
# as it came rather than dropped.
LIMITATION_TEXT = (
    ("not a safety determination or an ICH M7 classification",
     "연구용 Ames 선별 결과입니다. 안전성 판정도 ICH M7 분류도 아닙니다."),
    ("not clinically calibrated probabilities",
     "점수는 보정된 확률이 아닙니다. 적용 범위와 학습 자료 중복 여부를 알 수 없고, "
     "음성 예측은 안전하다는 뜻이 아닙니다."),
    ("Only AMES is reported",
     "Ames만 봅니다. 다른 독성, 노출량, 잔류 농도, 정제 제거는 평가하지 않습니다."),
    ("map-free canonicalization preserves",
     "구조를 비교할 때 원자 번호만 떼어냅니다. 입체화학·전하·분리된 성분은 그대로 "
     "두므로 염이나 양성자화 상태가 다르면 다른 구조로 봅니다."),
    ("remain abstentions, never negative predictions",
     "읽지 못한 입력과 실행 실패는 기권으로 남습니다. 음성으로 바꾸지 않습니다."),
    ("uncalibrated positive-class score, not a chance",
     "보정되지 않은 양성 점수이고 변이원일 확률이 아닙니다. 순위를 매기는 값으로 "
     "읽어야 합니다. 묶음 교차검증에서 평균 0.288이 표시된 구조들의 실제 활성 "
     "비율은 0.330이었고 Brier 점수는 0.111입니다."),
    ("uncalibrated positive-class score from a gradient boosted tree",
     "부스팅 트리 모델이 낸 보정되지 않은 양성 점수입니다. 변이원일 확률이 아니라 "
     "순위를 매기는 값으로 읽어야 합니다. 묶음 교차검증에서 0.287쯤으로 표시된 "
     "구조들은 실제로 35.0%가 양성이었습니다. 낮은 쪽이 실제보다 낮게 표시됩니다."),
    ("Measured on 933 independent structures it had not seen",
     "모델이 처음 보는 독립 구조 933개에서 ROC-AUC 0.902, 균형 정확도 0.847이고 "
     "0.5 기준선에서 양성의 77.4%를 찾았습니다."),
    ("The collection is 30.2% active and that imbalance is carried",
     "학습 자료는 양성이 30.2%로 치우쳐 있습니다. 그 치우침을 기준선이 아니라 학습 "
     "가중치로 보정했기 때문에 0.5가 이 모델의 균형점 0.43과 가깝습니다. 0.40과 "
     "0.50이 똑같이 0.847이고 균형점에서도 0.857밖에 되지 않습니다. 기준선은 바꿀 "
     "수 있고, 옮기면 민감도와 특이도가 맞바뀝니다."),
    ("come from the same curated Ames collection",
     "학습 자료와 평가 자료가 같은 Ames 모음에서 나왔습니다. 그 모음과 다른 종류의 "
     "화학 구조에서는 성능을 알 수 없습니다."),
    ("This model too carries the collection's 30.2% active rate",
     "이 모델도 자료가 양성 30.2%로 치우친 것을 기준선이 아니라 학습 가중치로 "
     "보정했으므로 0.50에서 돕니다. 독립 구조 1,201개에서 균형 정확도 0.789이고, "
     "자체 균형점 0.43에서도 0.797입니다."),
    ("Measured on the same independent structures, this model reached ROC-AUC",
     "같은 독립 구조에서 이 모델은 ROC-AUC 0.833, 지문 모델은 0.902였습니다. "
     "대신 이 모델은 모든 구조에 답하고, 지문 모델은 22%에서 기권합니다."),
    ("agree on 89.8% of calls",
     "이 모델과 지문 모델의 판정은 89.8%가 일치합니다. 엇갈릴 때 어느 쪽이 맞는지는 "
     "이 소프트웨어가 정하지 않습니다."),
    ("describes this model's output in log-odds",
     "기여도는 이 모델의 출력을 log-odds로 설명한 것입니다. 작용 기전이 아니고 "
     "화학에 대한 설명도 아닙니다."),
)


def limitation_text(value):
    """One model limitation in the screen's language, or as it came."""
    if not value:
        return value
    text = str(value)
    lowered = text.casefold()
    for marker, korean in LIMITATION_TEXT:
        if marker.casefold() in lowered:
            return korean
    return text


STATUS_KINDS = {
    "known_concern": ui_kit.CONCERN,
    "structural_alert": ui_kit.REVIEW,
    # Not a negative. A rule screen that raises nothing has not said the
    # structure is clean, and our own rule sensitivity is 0.552.
    "no_alert_detected": ui_kit.ABSENT,
    "invalid": ui_kit.UNKNOWN,
}


def status_chip(status):
    """One material's rule verdict, in the shared chip vocabulary."""
    label = STATUS[status]
    return ui_kit.chip(label, STATUS_KINDS[status], hint=GLOSSARY.get(label))


def term(label):
    """Label plus a hover mark carrying its glossary line."""
    note = GLOSSARY.get(label)
    if not note:
        return label
    safe = note.replace('"', "&quot;")
    return f'{label}<span class="review-term" title="{safe}">?</span>'
ROLES = {
    "reactants": "반응물", "products": "생성물·중간체", "reagents": "시약",
    "solvents": "용매", "catalysts": "촉매", "ligands": "리간드",
    "bases": "염기", "coproducts": "공동생성물", "byproducts": "부산물 후보",
}
# How a material's evidence state reads on screen. The services write the
# English enum because it is also what the saved file and the tests check.
EVIDENCE_TEXT = {"observed": "관측", "proposed": "제안", "unknown": "자료 없음"}
ALERTS = {
    "N-nitroso": "N-니트로소기", "Azoxy (special-concern review)": "아족시기",
    "Aziridine": "아지리딘 고리", "Epoxide": "에폭사이드 고리",
    "Sulfonate ester": "설폰산 에스터", "Aromatic nitro": "방향족 니트로기",
    "Aromatic amine": "방향족 아민", "Alkyl halide (Cl/Br/I)": "알킬 할라이드",
    "Hydrazine": "히드라진 구조", "Azo": "아조기",
    "Michael acceptor": "마이클 수용체", "Aldehyde": "알데하이드기",
}
NAMES = {canonicalize_smiles(m["smiles"]): m["name_en"] for m in MOLECULES}

# Common synthesis reagents and solvents, so routes do not render as unnamed
# SMILES. These are ordinary chemical identities, not a safety or supply list.
_REAGENT_NAMES = {
    "CC(C)N": "Isopropylamine",
    "OC1=CC=CC2=CC=CC=C12": "1-Naphthol",
    "C=CCCl": "Allyl chloride",
    "OC(=O)OOC1=CC=CC(Cl)=C1": "m-Chloroperoxybenzoic acid",
    "OC(CCl)CCl": "1,3-Dichloro-2-propanol",
    "ClCC1CO1": "Epichlorohydrin",
    "OCC1CO1": "Glycidol",
    "NN": "Hydrazine",
    "CCBr": "Ethyl bromide",
    "CCCl": "Ethyl chloride",
    "CCl": "Methyl chloride",
    "C=O": "Formaldehyde",
    "CC=O": "Acetaldehyde",
    "NC1=CC=CC=C1": "Aniline",
    "C=CC1=CC=CC=C1": "Styrene",
    "ClCc1ccccc1": "Benzyl chloride",
    "C=CC#N": "Acrylonitrile",
    "BrCCBr": "1,2-Dibromoethane",
    "ClCOCCl": "Bis(chloromethyl)ether",
    "CN(C)C(=O)Cl": "Dimethylcarbamyl chloride",
    "C=COC(C)=O": "Vinyl acetate",
}
REAGENT_NAMES = {}
for _smiles, _label in _REAGENT_NAMES.items():
    _key = canonicalize_smiles(_smiles)
    if _key:
        REAGENT_NAMES[_key] = _label


def material_name(identity, given=None):
    """Readable label for a structure, falling back to its molecular formula.

    The local Korean names come first so one material does not appear under an
    English rule-list name in one place and a Korean name in another.
    """
    label = NAMES.get(identity) or REAGENT_NAMES.get(identity) or given
    if label:
        return label
    if identity:
        molecule = Chem.MolFromSmiles(identity)
        if molecule is not None:
            return rdMolDescriptors.CalcMolFormula(molecule)
    return "구조 미해석"


# Two palettes rather than one with a few swapped values. Light mode was a
# dark palette with the background inverted, so its borders were slate-500
# against white, which reads as a drawn box rather than a divider, and its
# surfaces were the same grey as its text's own container.
#
# Both are anchored on the product's two blues: the navy of the mark and the
# brighter blue of its orbit. A border is a divider and stays close to the
# surface it sits on; an accent is for one thing at a time.
PALETTES = {
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fb", "raised": "#eef2f7",
        "line": "#dde3ec", "text": "#16223a", "muted": "#5a6b86",
        "accent": "#1b4fa8", "action": "#1b3a7a",
    },
    "dark": {
        "bg": "#0b1422", "panel": "#15223a", "raised": "#1b2b47",
        "line": "#243550", "text": "#eef3fa", "muted": "#9fb0c8",
        "accent": "#6cc3f5", "action": "#1d5fa8",
    },
}


def theme_css(light):
    colours = PALETTES["light" if light else "dark"]
    bg, panel, text = colours["bg"], colours["panel"], colours["text"]
    muted, border, accent = colours["muted"], colours["line"], colours["accent"]
    raised, action = colours["raised"], colours["action"]
    return f"""<style>
    :root {{ --review-bg:{bg}; --review-panel:{panel}; --review-raised:{raised};
      --review-line:{border}; --review-text:{text}; --review-muted:{muted};
      --review-accent:{accent}; --review-action:{action}; }}
    html,body,input,textarea,button,select {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",sans-serif;}}
    [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background:{bg};color:{text};}}
    [data-testid="stMainBlockContainer"] {{max-width:min(100%,calc(50vw + 620px));padding:4.5rem 24px 32px;}}
    [data-testid="stVerticalBlock"] {{gap:10px;}}
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"],
    [data-testid="stExpander"] summary, [role="tab"], label, input, textarea, button {{color:{text};}}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:focus-visible,
    [data-testid="stExpander"] summary:focus-within,
    [data-testid="stExpander"] details[open] > summary {{background:{panel}!important;color:{text}!important;}}
    [data-testid="stExpander"] summary :is(p,span,svg) {{color:{text}!important;}}
    [data-testid="stText"], [data-testid="stText"] pre, [data-testid="stText"] span {{background:{bg}!important;color:{text}!important;font-size:17px!important;line-height:1.5!important;white-space:pre-wrap;overflow-wrap:anywhere;}}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p, input, textarea, button p, [role="tab"] p {{font-size:17px!important;line-height:1.5!important;}}
    [data-testid="stCaptionContainer"] {{opacity:1!important;}}
    [data-testid="stCaptionContainer"] p {{font-size:16px;line-height:1.5;color:{muted};}}
    h1 {{font-size:26px!important;padding:0!important;letter-spacing:-0.01em;}}
    h2 {{font-size:19px!important;font-weight:600;margin-top:4px!important;}}
    h3 {{font-size:16px!important;font-weight:600;color:{muted}!important;
      letter-spacing:0.01em;}}
    input, textarea, [data-testid="stSelectbox"] [role="group"], [data-testid="stSelectbox"] button,
    [role="listbox"], [role="option"] {{background:{panel}!important;color:{text}!important;}}
    [role="option"] {{font-size:17px!important;min-height:48px;}}
    button, input:not([type="checkbox"]):not([type="radio"]):not([type="file"]),
    [data-testid="stSelectbox"] [role="group"], [data-testid="stExpander"] summary,
    [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {{min-height:48px!important;}}
    button {{min-width:48px!important;border-radius:10px!important;padding:0 20px!important;}}
    .review-term {{display:inline-flex;align-items:center;justify-content:center;
      width:16px;height:16px;margin-left:4px;border:1px solid {muted};border-radius:50%;
      background:transparent;color:{muted};font-size:11px;line-height:1;font-weight:600;
      vertical-align:middle;cursor:help;user-select:none;}}
    [data-testid="stTooltipHoverTarget"] button {{color:{text}!important;}}
    button, [data-testid="stExpander"] {{border-color:{border}!important;}}
    [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {{background:{panel}!important;color:{text}!important;}}
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{background:{action}!important;color:#fff!important;border-color:{action}!important;}}
    [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p {{color:#fff;}}
    a, [role="tab"][aria-selected="true"] {{color:{accent};}}
    .react-aria-SelectionIndicator {{background:{accent}!important;}}
    [data-testid="stCheckbox"] label:has(input:checked):not(:has(input:disabled)) > div:first-of-type,
    [data-testid="stRadioOption"]:has(input:checked):not(:has(input:disabled)) > div > div:first-child {{background:{action}!important;border-color:{action}!important;}}
    [data-testid="stMain"] :is(button,input,textarea,a,summary,[role="tab"]):focus-visible {{outline:3px solid {accent}!important;outline-offset:2px;}}
    [data-testid="stSelectbox"]:focus-within, [data-testid="stRadio"] label:focus-within {{outline:3px solid {accent};outline-offset:2px;}}
    [role="tablist"] {{overflow-x:auto;gap:4px;border-bottom:1px solid {border};}}
    [role="tab"] {{min-height:44px;flex-shrink:0;padding:0 2px;}}
    [role="tab"][aria-selected="true"] {{box-shadow:inset 0 -2px 0 0 {accent};}}
    [data-testid="stDataFrame"] {{border:1px solid {border};border-radius:8px;overflow:hidden;}}
    [data-testid="stMetric"] {{background:{panel};border:1px solid {border};
      border-radius:10px;padding:10px 12px;}}
    [data-testid="stMetricLabel"] p {{font-size:12px!important;color:{muted};
      letter-spacing:0.02em;}}
    [data-testid="stMetricValue"] {{font-size:24px!important;font-weight:600;}}
    [data-testid="stExpander"] details {{border-radius:10px;}}
    hr, [data-testid="stDivider"] hr {{border-color:{border};}}
    /* The tool picker ships Streamlit's own red tag colour, which is the one
       thing on the screen that means "concern" everywhere else. The tag is a
       plain span carrying data-tag, not the data-baseweb element an earlier
       selector looked for, so that rule never matched and the tags stayed red. */
    [data-testid="stMultiSelectTagsContainer"] span[data-tag] {{
      background:{action}!important;border-radius:6px!important;}}
    [data-testid="stMultiSelectTagsContainer"] span[data-tag] span,
    [data-testid="stMultiSelectTagsContainer"] span[data-tag] svg {{
      color:#fff!important;fill:#fff!important;}}
    [data-testid="stCode"] pre {{white-space:pre-wrap;overflow-wrap:anywhere;background:{panel}!important;color:{text}!important;}}
    [data-testid="stAlert"] {{background:{panel}!important;border:1px solid {border};}}
    [data-testid="stMarkdownContainer"] {{overflow-wrap:anywhere;}}
    @media(max-width:760px) {{[data-testid="stMainBlockContainer"] {{padding:4.5rem 12px 24px;}}}}
    </style>"""


@st.cache_data(show_spinner=False, max_entries=256)
def structure_png(smiles):
    if not isinstance(smiles, str) or len(smiles) > 4096:
        return None
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    try:
        molecule = Chem.MolFromSmiles(canonical)
        if molecule.GetNumAtoms() > 256:
            return None
        image = Draw.MolToImage(molecule, size=(360, 220))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except (ValueError, RuntimeError, OSError):
        return None


def show_structure(smiles):
    image = structure_png(smiles)
    if image:
        st.image(image, width=300)
    else:
        st.caption("구조 그림 없음")


def route_dot(route):
    """Draw only supplied reaction edges; equal valid structures share nodes."""
    lines = ['digraph route { rankdir=LR; node [fontname="sans-serif"];']
    structures = {}
    for index, step in enumerate(route["steps"]):
        reaction = f"reaction_{index}"
        lines.append(f'{reaction} [shape=box,label={json.dumps(step["id"], ensure_ascii=False)}];')
        for role in ("reactants", "products"):
            for position, smiles in enumerate(step[role]):
                canonical = canonicalize_smiles(smiles) if len(smiles) <= 4096 else None
                identity = canonical or (index, role, position)
                if identity not in structures:
                    node = f"material_{len(structures)}"
                    structures[identity] = node
                    label = NAMES.get(canonical, smiles)
                    lines.append(f'{node} [label={json.dumps(label, ensure_ascii=False)}];')
                node = structures[identity]
                source, destination = (node, reaction) if role == "reactants" else (reaction, node)
                lines.append(f"{source} -> {destination};")
    return "\n".join([*lines, "}"])


def material_reason(material):
    alerts = ", ".join(ALERTS.get(value, value) for value in material.get("alerts", []))
    if material["status"] == "known_concern":
        return f"목록 일치: {material.get('name') or '구성 성분'}" + (f" · 검출 구조 {alerts}" if alerts else "")
    if material["status"] == "structural_alert":
        return f"검출 구조: {alerts}"
    return ""


def material_card(material, location="", *, tool=None):
    with st.container(border=True):
        drawing, evidence = st.columns([1, 2], gap="small")
        with drawing:
            show_structure(material["input_smiles"])
        with evidence:
            st.subheader(material_name(material.get("canonical_smiles"), material.get("name")))
            role = ROLES.get(material.get("role"), "물질 하나만 검사")
            st.caption(" · ".join(part for part in (location, role, tool) if part))
            # The same chip vocabulary the other screens use. This card used
            # to set the verdict in bold text, so one screen said a thing with
            # a chip and another said it with a sentence.
            marks = [status_chip(material["status"])]
            if material.get("special_concern"):
                marks.append(ui_kit.chip("특별 우려 구조", ui_kit.CONCERN,
                                         hint=GLOSSARY.get("특별 우려 구조")))
            st.markdown(" ".join(marks), unsafe_allow_html=True)
            # Only when it says more than the chip already did. "해당 없음"
            # followed by "검출된 신호 없음" is the same sentence twice.
            reason = material_reason(material)
            if reason:
                st.write(reason)
        # Facts, not a dump. The raw record used to be printed into every
        # card, under a line that showed the evidence state as the English
        # enum the services write ("unknown"). The whole record is still in
        # the saved file and under the route's own "판정 사유와 원문".
        with st.expander("구조·출처"):
            st.caption("입력 SMILES")
            st.code(material["input_smiles"], language=None)
            if material.get("canonical_smiles") != material["input_smiles"] and material.get("canonical_smiles"):
                st.caption("정규화 SMILES")
                st.code(material["canonical_smiles"], language=None)
            facts = [("근거", EVIDENCE_TEXT.get(material.get("evidence_status"),
                                              "자료 없음"))]
            if material.get("origin"):
                facts.append(("출처", ui_kit.text(material["origin"])))
            if material.get("source_url"):
                facts.append(("주소", ui_kit.text(material["source_url"])))
            # "없음" stays on the card. A missing intake figure is a fact
            # about the record, and dropping the row would let a reader take
            # the silence for a number that was simply not worth showing.
            # A plain label: ui_kit.facts escapes labels by contract, so the
            # glossary marker would have printed as its own tag.
            intake = material.get("intake_ug_day")
            facts.append(("TTC 참고 섭취량",
                          f"{intake:g} µg/day" if intake is not None else "없음"))
            ui_kit.show_facts(facts)
