"""Discrete held-input combinations, including long-jump and ground-pound inputs."""

DIRECTIONS = ((), ("w",), ("w", "a"), ("a",), ("a", "s"), ("s",), ("s", "d"), ("d",), ("d", "w"))
MODIFIERS = ((), ("jump",), ("spin",), ("crouch",), ("crouch", "jump"), ("jump", "spin"))
ACTIONS = tuple(direction + modifier for modifier in MODIFIERS for direction in DIRECTIONS)


def action_name(index: int) -> str:
    return "+".join(ACTIONS[index]) or "idle"
