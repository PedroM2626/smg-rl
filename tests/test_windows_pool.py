import struct

import pytest

from smg_rl.windows_pool import parse_pool_tags


def test_pool_tag_decoder_checks_bounds_and_reports_counters():
    entry = struct.pack('<4sII4xQIIQ', b'EtwD', 100, 20, 4096, 9, 4, 8192)
    values = parse_pool_tags(struct.pack('<I4x', 1) + entry)
    assert values == [{'tag': 'EtwD', 'paged_bytes': 4096, 'nonpaged_bytes': 8192,
                       'paged_outstanding': 80, 'nonpaged_outstanding': 5}]
    with pytest.raises(ValueError, match='count'):
        parse_pool_tags(struct.pack('<I4x', 2) + entry)
    with pytest.raises(ValueError, match='header'):
        parse_pool_tags(b'')
