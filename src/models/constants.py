"""Class constants for volleyball action detection."""

ACTION_CLASS_NAMES = {
    0: "serve",
    1: "receive",
    2: "set",
    3: "attack",
}

NOACTION_CLASS_ID = 4

ACTION_WITH_NOACTION_CLASS_NAMES = {
    **ACTION_CLASS_NAMES,
    NOACTION_CLASS_ID: "noaction",
}

EXPORT_CLASS_NAMES = {
    0: "serve",
    1: "receive",
    2: "set",
    3: "attack",
    4: "player",
    5: "ball",
}
