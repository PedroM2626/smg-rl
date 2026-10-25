"""Discrete held-input combinations, including long-jump and ground-pound inputs."""

DIRECTIONS = ((), ("w",), ("w", "a"), ("a",), ("a", "s"), ("s",), ("s", "d"), ("d",), ("d", "w"))
MODIFIERS = ((), ("jump",), ("spin",), ("crouch",), ("crouch", "jump"), ("jump", "spin"))
ACTIONS = tuple(direction + modifier for modifier in MODIFIERS for direction in DIRECTIONS)
MOTOR_ACTIONS = DIRECTIONS + (("jump",), ("w", "jump"), ("a", "jump"),
                              ("s", "jump"), ("d", "jump"), ("spin",), ("crouch",))
EIGHT_ACTIONS = ((), ("d",), ("a",), ("w",), ("s",), ("jump",), ("spin",), ("crouch",))


def action_set(config=None):
    profile = (config or {}).get("actions", {}).get("profile", "legacy54")
    if profile not in ("legacy54", "motor16-v1", "motor8-v1"):
        raise ValueError(f"Unknown action profile: {profile}")
    if profile == "motor8-v1":
        return EIGHT_ACTIONS
    return MOTOR_ACTIONS if profile == "motor16-v1" else ACTIONS


def action_name(index: int, config=None) -> str:
    if (config or {}).get("actions", {}).get("profile") == "motor8-v1" and index == 0:
        return "none"
    return "+".join(action_set(config)[index]) or "idle"


def advance_action(backend, names, config):
    """Motor v1: fixed 12-frame decisions, explicit 8/4-frame button release."""
    if config.get("actions", {}).get("profile", "legacy54") not in ("motor16-v1", "motor8-v1"):
        backend.advance(names)
        return
    if (config["parallel"].get("backend") != "native" or config["parallel"]["frame_skip"] != 12
            or not hasattr(backend, 'bridge')):
        raise ValueError("Motor profiles require native twelve-frame decisions.")
    pulse = set(names) & {"jump", "spin"}
    if not pulse:
        backend.advance(names)
        return
    # Backends with video support implement precise subdecision frame capture.
    if hasattr(backend, "advance_pulse"):
        backend.advance_pulse(names, 8, 4)
        return
    previous = backend.config["parallel"]["frame_skip"]
    try:
        backend.config["parallel"]["frame_skip"] = 8
        backend.advance(names)
        backend.config["parallel"]["frame_skip"] = 4
        backend.advance(tuple(name for name in names if name not in pulse))
        backend.last_frame_count = 12
    finally:
        backend.config["parallel"]["frame_skip"] = previous
