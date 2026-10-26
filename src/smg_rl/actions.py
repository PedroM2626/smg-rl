"""Discrete held-input combinations, including long-jump and ground-pound inputs."""

DIRECTIONS = ((), ("w",), ("w", "a"), ("a",), ("a", "s"), ("s",), ("s", "d"), ("d",), ("d", "w"))
MODIFIERS = ((), ("jump",), ("spin",), ("crouch",), ("crouch", "jump"), ("jump", "spin"))
ACTIONS = tuple(direction + modifier for modifier in MODIFIERS for direction in DIRECTIONS)
MOTOR_ACTIONS = DIRECTIONS + (("jump",), ("w", "jump"), ("a", "jump"),
                              ("s", "jump"), ("d", "jump"), ("spin",), ("crouch",))
EIGHT_ACTIONS = ((), ("d",), ("a",), ("w",), ("s",), ("jump",), ("spin",), ("crouch",))


def action_set(config=None):
    profile = (config or {}).get("actions", {}).get("profile", "legacy54")
    if profile not in ("legacy54", "motor16-v1", "motor8-v1", "motor8-timed-v2"):
        raise ValueError(f"Unknown action profile: {profile}")
    if profile in ("motor8-v1", "motor8-timed-v2"):
        return EIGHT_ACTIONS
    return MOTOR_ACTIONS if profile == "motor16-v1" else ACTIONS


def action_name(index: int, config=None) -> str:
    if (config or {}).get("actions", {}).get("profile") in ("motor8-v1", "motor8-timed-v2") and index == 0:
        return "none"
    return "+".join(action_set(config)[index]) or "idle"


def action_timing(config):
    profile = config.get('actions', {}).get('profile', 'legacy54')
    if profile == 'legacy54':
        return None
    if profile in ('motor8-v1', 'motor16-v1'):
        frames, pressed = 12, 8
    elif profile == 'motor8-timed-v2':
        frames = config['actions'].get('decision_frames')
        pressed = config['actions'].get('pulse_frames')
        if frames not in (4, 12) or pressed not in (2, frames):
            raise ValueError('Timed motor8 requires4/12 frames and explicit two-frame/full-hold input.')
    else:
        raise ValueError('Unknown motor timing profile.')
    if config['parallel'].get('backend') != 'native' or config['parallel']['frame_skip'] != frames:
        raise ValueError('Motor timing must match native frame stepping.')
    return frames, pressed, frames - pressed


def execution_signature(config):
    profile = config.get('actions', {}).get('profile', 'legacy54')
    if profile == 'legacy54':
        return None
    if profile in ('motor8-v1', 'motor16-v1'):
        return profile + ':12-frames:jump-spin-8-on-4-off'
    frames, pressed, released = action_timing(config)
    mode = config['actions'].get('jump_direction', 'inertia')
    if mode not in ('inertia', 'carry'):
        raise ValueError('Unknown timed jump direction mode.')
    return f'{profile}:{frames}-frames:jump-spin-{pressed}-on-{released}-off:jump-{mode}'


def advance_action(backend, names, config):
    """Apply signed motor timing; retain legacy pulses and explicit timed variants."""
    timing = action_timing(config)
    if timing is None:
        backend.advance(names)
        return
    frames, pressed, released = timing
    if not hasattr(backend, 'bridge'):
        raise ValueError('Motor profiles require a native bridge.')
    backend.logical_action = '+'.join(names) or 'none'
    if config.get('actions', {}).get('profile') == 'motor8-timed-v2':
        mode = config['actions'].get('jump_direction', 'inertia')
        if mode not in ('inertia', 'carry'):
            raise ValueError('Unknown timed jump direction mode.')
        if len(names) == 1 and names[0] in ('w', 'a', 's', 'd'):
            backend._motor_direction = names
        elif not names or names == ('crouch',):
            backend._motor_direction = ()
        elif names == ('jump',) and mode == 'carry':
            names = tuple(getattr(backend, '_motor_direction', ())) + names
    pulse = set(names) & {"jump", "spin"}
    if not pulse or released == 0:
        backend.advance(names)
        return
    # Backends with video support implement precise subdecision frame capture.
    if hasattr(backend, "advance_pulse"):
        backend.advance_pulse(names, pressed, released)
        return
    previous = backend.config["parallel"]["frame_skip"]
    try:
        backend.config["parallel"]["frame_skip"] = pressed
        backend.advance(names)
        backend.config["parallel"]["frame_skip"] = released
        backend.advance(tuple(name for name in names if name not in pulse))
        backend.last_frame_count = frames
    finally:
        backend.config["parallel"]["frame_skip"] = previous
