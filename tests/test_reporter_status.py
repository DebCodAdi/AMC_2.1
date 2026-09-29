from signal9.report.reporter import Signal9Reporter
from signal9.status import Status


def test_detection_promotes_default_no_signal_status(tmp_path):
    reporter = Signal9Reporter(tmp_path / "capture.iq", fs=1_000_000.0)

    reporter.add_detection(
        {"t_lo": 0.0, "t_hi": 0.1, "duration": 0.1},
        {"levels": {}},
    )

    result = reporter.to_dict()
    assert result["total_signals_detected"] == 1
    assert result["status"] == Status.DEMOD_NO_CRC.value


def test_detection_does_not_overwrite_specific_status(tmp_path):
    reporter = Signal9Reporter(tmp_path / "capture.iq", fs=1_000_000.0)
    reporter.set_status(Status.UNKNOWN_MODULATION)

    reporter.add_detection({}, {"levels": {}})

    assert reporter.to_dict()["status"] == Status.UNKNOWN_MODULATION.value
