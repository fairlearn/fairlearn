# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import fairlearn


def test_smoke(capsys):
    fairlearn.show_versions()
    captured = capsys.readouterr()
    # The following are not comprehensive
    assert "System:" in captured.out
    assert "    python:" in captured.out
    assert "executable:" in captured.out
    assert "   machine:" in captured.out
    assert "Python dependencies:" in captured.out
    assert "   sklearn:" in captured.out
    assert "     scipy:" in captured.out
    assert "    pandas:" in captured.out
