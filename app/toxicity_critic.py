"""A SynthEx Critic that also refuses a step whose precursors raise our alerts.

SynthEx reviews each proposed route with an LLM Critic and, where that Critic
raises a blocking finding, hands the route to an Editor that tries to repair
that step. The route therefore survives in an altered form instead of being
dropped. That is the one thing this pipeline does that a template search
cannot, and it is the reason the engine is offered at all.

Out of the box the Critic judges chemistry, not genotoxicity. This subclass
runs the same local rule screen the rest of the software uses over the
precursors of every step and adds a blocking finding for each one the screen
names. The LLM is told what was found; it is never asked to decide it. The
verdict stays with `toxicology.assess_material`, which reads a published list
and a fixed set of alerts and does not call anything.

What this does not claim: a route that comes back with fewer findings has not
been shown to be safer. The Editor rewrites a step to avoid a flagged
precursor, and whatever it produces is screened again from scratch by the
same tools as any other route. Nothing here shortens that review.
"""

from __future__ import annotations

if __package__:
    from .toxicology import assess_material
else:
    from toxicology import assess_material


# The statuses the rule screen uses for a material it has something against.
# "no_alert_detected" is not among them, and neither is "invalid": an alert
# that was never evaluated is not an alert that was not found.
CONCERN_STATUSES = ("known_concern", "structural_alert")

BLOCKING_TEMPLATE = (
    "Precursor {smiles} is flagged by the project's ICH M7 rule screen "
    "({finding}). Rework this step to avoid it, or replace it with a "
    "precursor that the screen does not flag."
)
UNREADABLE_TEMPLATE = (
    "Precursor {smiles} could not be parsed, so it was not screened for "
    "mutagenicity. This is not a clean result."
)


def _finding_text(assessment):
    """What the screen found, in a few words the Editor can act on."""
    alerts = assessment.get("alerts") or []
    if assessment["status"] == "known_concern":
        name = assessment.get("name") or "a named entry"
        return f"named on the ICH M7 list as {name}"
    return "structural alert: " + (", ".join(alerts) if alerts else "unnamed")


def precursors_of(rendered):
    """Every step's precursors, as (step_id, smiles) pairs.

    `render_plan` writes one `product>>precursors` entry per step, the same
    shape the stock Critic reads for its atom accounting. Precursors are
    dot-separated; each component is screened on its own, because a route
    charges them separately.
    """
    pairs = []
    for entry in rendered.get("route") or []:
        reaction = entry.get("smiles") or ""
        _, separator, precursors = reaction.partition(">>")
        if not separator:
            continue
        for part in precursors.split("."):
            part = part.strip()
            if part:
                pairs.append((entry.get("id"), part))
    return pairs


def screen_precursors(rendered):
    """Findings for one rendered route, worst first, one per flagged precursor.

    Returns (blocking, unreadable) as lists of (step_id, message). A structure
    is reported once per step even when it appears twice in the same step.
    """
    blocking, unreadable, seen = [], [], set()
    for step_id, smiles in precursors_of(rendered):
        if (step_id, smiles) in seen:
            continue
        seen.add((step_id, smiles))
        try:
            assessment = assess_material(smiles)
        except Exception:  # noqa: BLE001 - a screen failure is not a clean result
            unreadable.append((step_id, UNREADABLE_TEMPLATE.format(smiles=smiles)))
            continue
        if assessment["status"] in CONCERN_STATUSES:
            blocking.append((step_id, BLOCKING_TEMPLATE.format(
                smiles=smiles, finding=_finding_text(assessment))))
        elif assessment["status"] == "invalid":
            unreadable.append((step_id, UNREADABLE_TEMPLATE.format(smiles=smiles)))
    return blocking, unreadable


def build(base_class, step_finding_class, render_plan):
    """Make the subclass, given the pieces SynthEx supplies.

    They are passed in rather than imported because SynthEx is installed only
    inside the generation worker's own environment. This module stays
    importable, and testable, without it.
    """

    class ToxicityAwareCritic(base_class):
        name = "critic"
        prompt_version = base_class.prompt_version + "+ich-m7-rule-screen"

        async def run(self, request, context):
            critique = await super().run(request, context)
            rendered = render_plan(request.route.to_op_plan())
            blocking, unreadable = screen_precursors(rendered)
            if not blocking and not unreadable:
                return critique
            findings = list(critique.problematic_steps)
            findings += [step_finding_class(step_id=step_id, severity="blocking",
                                            feedback=message)
                         for step_id, message in blocking]
            findings += [step_finding_class(step_id=step_id, severity="suggestion",
                                            feedback=message)
                         for step_id, message in unreadable]
            # A blocking finding must also clear `is_satisfactory`, because
            # `Critique.is_resolved` reads both and the loop would otherwise
            # stop on the LLM's own verdict while our screen was still
            # objecting. An unreadable precursor is only a suggestion: we
            # cannot call a structure we failed to parse dangerous, and
            # forcing an edit on one would ask the Editor to fix nothing.
            satisfactory = critique.is_satisfactory and not blocking
            return critique.model_copy(update={
                "problematic_steps": findings,
                "is_satisfactory": satisfactory,
            })

    return ToxicityAwareCritic


__all__ = ["CONCERN_STATUSES", "build", "precursors_of", "screen_precursors"]
