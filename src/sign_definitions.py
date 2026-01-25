"""BSL Data Extraction - Sign definitions for all target signs.

This module contains definitions for all 76 BSL signs to be captured,
organized by category with metadata and descriptions.

BSL (British Sign Language) uses a two-handed fingerspelling alphabet,
unlike ASL which is one-handed.
"""

from typing import NamedTuple

from src.types import SignCategory, SignDefinition


# =============================================================================
# Category Metadata
# =============================================================================


class CategoryInfo(NamedTuple):
    """Metadata for a sign category."""

    id: SignCategory
    display_name: str
    description: str
    icon_emoji: str


CATEGORIES: list[CategoryInfo] = [
    CategoryInfo(
        id="alphabet",
        display_name="Alphabet",
        description="BSL fingerspelling alphabet (A-Z)",
        icon_emoji="🔤",
    ),
    CategoryInfo(
        id="numbers",
        display_name="Numbers",
        description="Numbers 0-10 in BSL",
        icon_emoji="🔢",
    ),
    CategoryInfo(
        id="greetings",
        display_name="Greetings & Responses",
        description="Common greetings and polite responses",
        icon_emoji="👋",
    ),
    CategoryInfo(
        id="people",
        display_name="Family & People",
        description="Family members and relationships",
        icon_emoji="👨‍👩‍👧‍👦",
    ),
    CategoryInfo(
        id="food_drink",
        display_name="Food & Drink",
        description="Food, drinks, and eating-related signs",
        icon_emoji="🍽️",
    ),
    CategoryInfo(
        id="colors",
        display_name="Colours",
        description="Basic colour signs",
        icon_emoji="🎨",
    ),
    CategoryInfo(
        id="time",
        display_name="Time",
        description="Time-related concepts",
        icon_emoji="⏰",
    ),
    CategoryInfo(
        id="emotions",
        display_name="Feelings & States",
        description="Emotions, feelings, and states",
        icon_emoji="😊",
    ),
]


# =============================================================================
# Alphabet Signs (A-Z) - BSL uses two-handed fingerspelling
# =============================================================================

_ALPHABET_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_alphabet_a",
        name="A",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Point index finger of dominant hand at open palm of non-dominant hand",
    ),
    SignDefinition(
        id="bsl_alphabet_b",
        name="B",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Point index finger at extended fingers of flat non-dominant hand",
    ),
    SignDefinition(
        id="bsl_alphabet_c",
        name="C",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Curved hand forming C shape against non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_d",
        name="D",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index finger touches tip of curved non-dominant index finger forming D",
    ),
    SignDefinition(
        id="bsl_alphabet_e",
        name="E",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index finger touches bent fingers of non-dominant hand",
    ),
    SignDefinition(
        id="bsl_alphabet_f",
        name="F",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index and middle finger of dominant hand on index of non-dominant",
    ),
    SignDefinition(
        id="bsl_alphabet_g",
        name="G",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Point at fist with extended index finger of non-dominant hand",
    ),
    SignDefinition(
        id="bsl_alphabet_h",
        name="H",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index and middle fingers extended across non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_i",
        name="I",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Little finger of dominant hand on palm of non-dominant hand",
    ),
    SignDefinition(
        id="bsl_alphabet_j",
        name="J",
        category="alphabet",
        difficulty=2,
        two_handed=True,
        description="Little finger traces J shape on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_k",
        name="K",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index and middle fingers form V on non-dominant index finger",
    ),
    SignDefinition(
        id="bsl_alphabet_l",
        name="L",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="L-shape (thumb and index) with thumb on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_m",
        name="M",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Three fingers (index, middle, ring) on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_n",
        name="N",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Two fingers (index, middle) on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_o",
        name="O",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Finger and thumb form O shape touching non-dominant hand",
    ),
    SignDefinition(
        id="bsl_alphabet_p",
        name="P",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index and middle point down between non-dominant index and middle",
    ),
    SignDefinition(
        id="bsl_alphabet_q",
        name="Q",
        category="alphabet",
        difficulty=2,
        two_handed=True,
        description="Bent index touches base of non-dominant palm, curling motion",
    ),
    SignDefinition(
        id="bsl_alphabet_r",
        name="R",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Crossed index and middle fingers on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_s",
        name="S",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Fist with thumb across, on non-dominant open palm",
    ),
    SignDefinition(
        id="bsl_alphabet_t",
        name="T",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index finger tucked under thumb on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_u",
        name="U",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Index and middle together pointing at non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_v",
        name="V",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="V-shape (index and middle spread) on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_w",
        name="W",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Three fingers spread (index, middle, ring) on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_x",
        name="X",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Hooked index finger on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_y",
        name="Y",
        category="alphabet",
        difficulty=1,
        two_handed=True,
        description="Extended thumb and little finger on non-dominant palm",
    ),
    SignDefinition(
        id="bsl_alphabet_z",
        name="Z",
        category="alphabet",
        difficulty=2,
        two_handed=True,
        description="Index finger traces Z shape on non-dominant palm",
    ),
]


# =============================================================================
# Number Signs (0-10)
# =============================================================================

_NUMBER_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_numbers_0",
        name="0",
        category="numbers",
        difficulty=1,
        two_handed=False,
        description="Closed fist with fingers facing viewer",
    ),
    SignDefinition(
        id="bsl_numbers_1",
        name="1",
        category="numbers",
        difficulty=1,
        two_handed=False,
        description="Index finger extended upward, other fingers closed",
    ),
    SignDefinition(
        id="bsl_numbers_2",
        name="2",
        category="numbers",
        difficulty=1,
        two_handed=False,
        description="Index and middle fingers extended in V-shape",
    ),
    SignDefinition(
        id="bsl_numbers_3",
        name="3",
        category="numbers",
        difficulty=1,
        two_handed=False,
        description="Index, middle, and ring fingers extended",
    ),
    SignDefinition(
        id="bsl_numbers_4",
        name="4",
        category="numbers",
        difficulty=1,
        two_handed=False,
        description="Four fingers extended, thumb tucked",
    ),
    SignDefinition(
        id="bsl_numbers_5",
        name="5",
        category="numbers",
        difficulty=1,
        two_handed=False,
        description="All five fingers extended, palm facing viewer",
    ),
    SignDefinition(
        id="bsl_numbers_6",
        name="6",
        category="numbers",
        difficulty=1,
        two_handed=True,
        description="Five on one hand plus one finger on other hand",
    ),
    SignDefinition(
        id="bsl_numbers_7",
        name="7",
        category="numbers",
        difficulty=1,
        two_handed=True,
        description="Five on one hand plus two fingers on other hand",
    ),
    SignDefinition(
        id="bsl_numbers_8",
        name="8",
        category="numbers",
        difficulty=1,
        two_handed=True,
        description="Five on one hand plus three fingers on other hand",
    ),
    SignDefinition(
        id="bsl_numbers_9",
        name="9",
        category="numbers",
        difficulty=1,
        two_handed=True,
        description="Five on one hand plus four fingers on other hand",
    ),
    SignDefinition(
        id="bsl_numbers_10",
        name="10",
        category="numbers",
        difficulty=1,
        two_handed=True,
        description="Both hands showing five, or two fists with thumbs up",
    ),
]


# =============================================================================
# Greeting Signs
# =============================================================================

_GREETING_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_greetings_hello",
        name="Hello",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Wave hand with palm facing outward, fingers together",
    ),
    SignDefinition(
        id="bsl_greetings_goodbye",
        name="Goodbye",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Wave hand side to side, palm facing outward",
    ),
    SignDefinition(
        id="bsl_greetings_thank_you",
        name="Thank You",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Flat hand touches chin then moves forward and down",
    ),
    SignDefinition(
        id="bsl_greetings_please",
        name="Please",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Flat hand circles on chest",
    ),
    SignDefinition(
        id="bsl_greetings_sorry",
        name="Sorry",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Fist circles on chest in apologetic motion",
    ),
    SignDefinition(
        id="bsl_greetings_yes",
        name="Yes",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Fist nods up and down like a nodding head",
    ),
    SignDefinition(
        id="bsl_greetings_no",
        name="No",
        category="greetings",
        difficulty=1,
        two_handed=False,
        description="Index and middle finger close onto thumb repeatedly",
    ),
]


# =============================================================================
# Family/People Signs
# =============================================================================

_FAMILY_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_people_mother",
        name="Mother",
        category="people",
        difficulty=1,
        two_handed=False,
        description="Tap side of chin twice with fingers together",
    ),
    SignDefinition(
        id="bsl_people_father",
        name="Father",
        category="people",
        difficulty=1,
        two_handed=False,
        description="Tap forehead twice with fingers together",
    ),
    SignDefinition(
        id="bsl_people_brother",
        name="Brother",
        category="people",
        difficulty=2,
        two_handed=True,
        description="Two fists together, moving apart then together",
    ),
    SignDefinition(
        id="bsl_people_sister",
        name="Sister",
        category="people",
        difficulty=2,
        two_handed=True,
        description="Crossed index fingers, moving apart",
    ),
    SignDefinition(
        id="bsl_people_baby",
        name="Baby",
        category="people",
        difficulty=1,
        two_handed=True,
        description="Cradling motion with both arms",
    ),
    SignDefinition(
        id="bsl_people_family",
        name="Family",
        category="people",
        difficulty=2,
        two_handed=True,
        description="Both hands form F-shape, circle around each other",
    ),
]


# =============================================================================
# Food & Drink Signs
# =============================================================================

_FOOD_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_food_eat",
        name="Eat",
        category="food_drink",
        difficulty=1,
        two_handed=False,
        description="Bunched fingers move to mouth repeatedly",
    ),
    SignDefinition(
        id="bsl_food_drink",
        name="Drink",
        category="food_drink",
        difficulty=1,
        two_handed=False,
        description="C-shaped hand tips toward mouth like holding cup",
    ),
    SignDefinition(
        id="bsl_food_water",
        name="Water",
        category="food_drink",
        difficulty=1,
        two_handed=False,
        description="W-handshape taps chin twice",
    ),
    SignDefinition(
        id="bsl_food_tea",
        name="Tea",
        category="food_drink",
        difficulty=1,
        two_handed=True,
        description="Mime dipping tea bag with pinched fingers into cupped hand",
    ),
    SignDefinition(
        id="bsl_food_coffee",
        name="Coffee",
        category="food_drink",
        difficulty=1,
        two_handed=True,
        description="Two fists stacked, grinding motion like coffee grinder",
    ),
    SignDefinition(
        id="bsl_food_hungry",
        name="Hungry",
        category="food_drink",
        difficulty=1,
        two_handed=False,
        description="Flat hand moves down chest, indicating empty stomach",
    ),
]


# =============================================================================
# Colour Signs
# =============================================================================

_COLOUR_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_colors_red",
        name="Red",
        category="colors",
        difficulty=1,
        two_handed=False,
        description="Index finger strokes down from lower lip",
    ),
    SignDefinition(
        id="bsl_colors_blue",
        name="Blue",
        category="colors",
        difficulty=1,
        two_handed=False,
        description="B-handshape twists at wrist",
    ),
    SignDefinition(
        id="bsl_colors_green",
        name="Green",
        category="colors",
        difficulty=1,
        two_handed=False,
        description="G-handshape twists at wrist",
    ),
    SignDefinition(
        id="bsl_colors_yellow",
        name="Yellow",
        category="colors",
        difficulty=1,
        two_handed=False,
        description="Y-handshape (thumb and pinky) shakes side to side",
    ),
    SignDefinition(
        id="bsl_colors_black",
        name="Black",
        category="colors",
        difficulty=1,
        two_handed=False,
        description="Index finger draws across eyebrow",
    ),
    SignDefinition(
        id="bsl_colors_white",
        name="White",
        category="colors",
        difficulty=1,
        two_handed=False,
        description="Flat hand brushes down chest, palm inward",
    ),
]


# =============================================================================
# Time Signs
# =============================================================================

_TIME_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_time_today",
        name="Today",
        category="time",
        difficulty=1,
        two_handed=True,
        description="Both flat hands move down in front of body",
    ),
    SignDefinition(
        id="bsl_time_tomorrow",
        name="Tomorrow",
        category="time",
        difficulty=1,
        two_handed=False,
        description="Thumb on cheek, hand arcs forward",
    ),
    SignDefinition(
        id="bsl_time_yesterday",
        name="Yesterday",
        category="time",
        difficulty=1,
        two_handed=False,
        description="Thumb touches cheek, then moves backward over shoulder",
    ),
    SignDefinition(
        id="bsl_time_morning",
        name="Morning",
        category="time",
        difficulty=1,
        two_handed=True,
        description="Flat hand rises from elbow of other arm like rising sun",
    ),
    SignDefinition(
        id="bsl_time_night",
        name="Night",
        category="time",
        difficulty=1,
        two_handed=True,
        description="Flat hand descends past elbow of other arm like setting sun",
    ),
    SignDefinition(
        id="bsl_time_week",
        name="Week",
        category="time",
        difficulty=1,
        two_handed=True,
        description="Index finger slides along flat palm of other hand",
    ),
]


# =============================================================================
# Feelings/Emotions Signs
# =============================================================================

_FEELING_SIGNS: list[SignDefinition] = [
    SignDefinition(
        id="bsl_emotions_happy",
        name="Happy",
        category="emotions",
        difficulty=1,
        two_handed=False,
        description="Flat hand circles on chest with pleased expression",
    ),
    SignDefinition(
        id="bsl_emotions_sad",
        name="Sad",
        category="emotions",
        difficulty=1,
        two_handed=False,
        description="Flat hand moves down face, fingers trailing",
    ),
    SignDefinition(
        id="bsl_emotions_tired",
        name="Tired",
        category="emotions",
        difficulty=1,
        two_handed=True,
        description="Both flat hands on chest, drop down with exhale",
    ),
    SignDefinition(
        id="bsl_emotions_good",
        name="Good",
        category="emotions",
        difficulty=1,
        two_handed=False,
        description="Thumbs up, or flat hand moves away from chin",
    ),
    SignDefinition(
        id="bsl_emotions_bad",
        name="Bad",
        category="emotions",
        difficulty=1,
        two_handed=False,
        description="Flat hand moves away from chin with negative expression",
    ),
    SignDefinition(
        id="bsl_emotions_help",
        name="Help",
        category="emotions",
        difficulty=1,
        two_handed=True,
        description="Fist on open palm, both rise together",
    ),
]


# =============================================================================
# Combined Sign List
# =============================================================================

SIGN_DEFINITIONS: list[SignDefinition] = (
    _ALPHABET_SIGNS
    + _NUMBER_SIGNS
    + _GREETING_SIGNS
    + _FAMILY_SIGNS
    + _FOOD_SIGNS
    + _COLOUR_SIGNS
    + _TIME_SIGNS
    + _FEELING_SIGNS
)
"""All 74 BSL sign definitions."""


# Build lookup dictionaries for fast access
_SIGNS_BY_ID: dict[str, SignDefinition] = {s.id: s for s in SIGN_DEFINITIONS}
_SIGNS_BY_CATEGORY: dict[SignCategory, list[SignDefinition]] = {}
for _sign in SIGN_DEFINITIONS:
    if _sign.category not in _SIGNS_BY_CATEGORY:
        _SIGNS_BY_CATEGORY[_sign.category] = []
    _SIGNS_BY_CATEGORY[_sign.category].append(_sign)


# =============================================================================
# Helper Functions
# =============================================================================


def get_signs_by_category(category: SignCategory) -> list[SignDefinition]:
    """Get all sign definitions for a given category.

    Args:
        category: The SignCategory to filter by.

    Returns:
        List of SignDefinition objects in that category.
        Empty list if category has no signs.
    """
    return _SIGNS_BY_CATEGORY.get(category, [])


def get_sign_by_id(sign_id: str) -> SignDefinition | None:
    """Get a sign definition by its unique ID.

    Args:
        sign_id: The unique sign identifier (e.g., 'bsl_alphabet_a').

    Returns:
        The SignDefinition if found, None otherwise.
    """
    return _SIGNS_BY_ID.get(sign_id)


def get_sign_count() -> dict[str, int | dict[str, int]]:
    """Get counts of signs total and by category.

    Returns:
        Dictionary with 'total' count and 'by_category' breakdown.
    """
    by_category = {cat: len(signs) for cat, signs in _SIGNS_BY_CATEGORY.items()}
    return {
        "total": len(SIGN_DEFINITIONS),
        "by_category": by_category,
    }


def get_all_sign_ids() -> list[str]:
    """Get list of all sign IDs.

    Returns:
        List of all sign ID strings in definition order.
    """
    return [s.id for s in SIGN_DEFINITIONS]


def get_two_handed_signs() -> list[SignDefinition]:
    """Get all signs that require both hands.

    Returns:
        List of SignDefinition objects where two_handed is True.
    """
    return [s for s in SIGN_DEFINITIONS if s.two_handed]


def get_one_handed_signs() -> list[SignDefinition]:
    """Get all signs that only require one hand.

    Returns:
        List of SignDefinition objects where two_handed is False.
    """
    return [s for s in SIGN_DEFINITIONS if not s.two_handed]


def get_category_info(category: SignCategory) -> CategoryInfo | None:
    """Get category metadata by category ID.

    Args:
        category: The SignCategory to look up.

    Returns:
        CategoryInfo if found, None otherwise.
    """
    for cat in CATEGORIES:
        if cat.id == category:
            return cat
    return None
