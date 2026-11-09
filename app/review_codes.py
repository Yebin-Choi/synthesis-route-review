"""The reason-code vocabulary, in one place.

Two screens read the same reasons: the route cards and the model-evidence
fold. They were naming the codes separately, so a code mapped on one screen
printed as itself on the other. Keeping the tables here also breaks the
import cycle, since review_views imports the evidence view.

Both screens show every code. What these tables decide is only how a code is
worded and whether it reads as a finding or as a gap.
"""

# Codes that name an actual concern signal, versus routine gaps that repeat on
# every material and every step. Both stay in the saved evidence; only the
# screen collapses the routine ones into a single line.
DECIDING_CODES = {
    # Same wording as presentation.STATUS, so one material does not carry two
    # different names for the same finding on one screen.
    "known_concern": "ICH M7 목록 물질",
    "structural_alert": "경고구조",
    "admet_positive": "모델 양성",
    "current_model_positive_exclusion": "모델 양성",
    "rule_admet_joint_positive": "규칙과 모델 모두 양성",
    "byproduct_concern": "부산물 우려 (제외 아님)",
    "prohibited_material": "제외하기로 한 물질",
    "prohibited_component": "제외하기로 한 성분",
    "invalid_material": "구조를 읽지 못함",
}
# The deciding list gained the two prohibition codes and the unreadable-
# structure code. All three were falling through to the routine line, where
# an unmapped code printed as itself: a route dropped for a prohibited
# material was reporting "prohibited_material" on screen.
# Declaration order is display order, so the line reads the same every time.
# "자료 없음" sat next to "조건 자료 없음" and the two could not be told apart.
ROUTINE_CODES = {
    "missing_information": "빠진 항목",
    "conditions_not_observed": "반응 조건 없음",
    "byproducts_incomplete": "부산물 미평가",
    "admet_unassessed": "모델 미평가",
    "conditional_warning": "조건부 경고",
    "no_materials": "검사할 물질 없음",
}
_ROUTINE_ORDER = {code: index for index, code in enumerate(ROUTINE_CODES)}
# Shown when a code has no label here. The code itself stays in the saved
# file, so nothing is lost; it just does not go on the screen as itself.
UNKNOWN_CODE = "그 밖의 신호"


# Codes that annotate another reason rather than being a finding of their
# own. They are shown, because a reader needs to know a concern came from a
# predicted byproduct, but they are not counted as separate signals: the
# alert and the note beside it are one finding.
NOTE_CODES = {"byproduct_concern"}


# Codes whose full message belongs in the saved file rather than on screen.
# A negative model score is not a reason this route was excluded.
QUIET_CODES = {"admet_negative", "candidate_within_scope"}


# Reasons that are findings about the route even though they name no single
# structure. A conditional warning is two reagents that can react with each
# other, so it reads with the findings rather than with the gaps.
FINDING_CODES = {"conditional_warning"}


# What the engine writes when a material has no recorded source. Matched
# rather than changed there: the string is part of the saved record. The
# screen and the exported report both drop it, because a column that says
# "not recorded" on every row carries no more than a blank one.
ORIGIN_UNSET = "물질별 출처 미지정"
