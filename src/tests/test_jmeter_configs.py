"""Structural checks for Issue #9 JMeter HTTP load-test plans.

These checks do not execute JMeter. They verify the portable configuration
that JMeter 5.6+ loads and executes when a JMeter runtime is available.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


CONFIG_DIR = Path(__file__).parents[1] / "traffic" / "jmeter_configs"

SCENARIOS = {
    "flashsale_e2_10x.jmx": {"threads": "100", "target_rps": "1000"},
    "flashsale_e3_50x.jmx": {"threads": "200", "target_rps": "5000"},
    "flashsale_e4_100x.jmx": {"threads": "400", "target_rps": "10000"},
    "flashsale_e6_noisy.jmx": {"threads": "100", "target_rps": "200"},
}


def _prop(root: ET.Element, name: str) -> str:
    node = root.find(f".//stringProp[@name='{name}']")
    assert node is not None, f"Missing JMeter property: {name}"
    return node.text or ""


@pytest.mark.parametrize("filename, expected", SCENARIOS.items())
def test_flashsale_plan_matches_issue_9_configuration(filename, expected):
    root = ET.parse(CONFIG_DIR / filename).getroot()

    assert root.tag == "jmeterTestPlan"
    assert root.attrib["jmeter"].startswith("5.6")
    assert _prop(root, "ThreadGroup.num_threads") == expected["threads"]
    assert _prop(root, "ThreadGroup.ramp_time") == "10"
    assert _prop(root, "ThreadGroup.duration") == "310"
    assert _prop(root, "HTTPSampler.domain") == "${ALB_DNS}"
    assert _prop(root, "HTTPSampler.path") == "/request"
    assert _prop(root, "HTTPSampler.method") == "POST"
    assert _prop(root, "Argument.value") == "${__P(ALB_DNS,localhost)}"

    throughput = root.find(".//ConstantThroughputTimer/doubleProp[@name='throughput']/value")
    assert throughput is not None
    assert throughput.text == "${__javaScript(${TARGET_RPS}*60)}"
    assert expected["target_rps"] in [
        node.text for node in root.findall(".//elementProp[@name='TARGET_RPS']/stringProp")
    ]

    collector_names = {
        node.attrib["testname"] for node in root.findall(".//ResultCollector")
    }
    assert "Aggregate Report - P95/P99 and throughput" in collector_names
    assert "Summary Report - throughput" in collector_names


def test_noisy_plan_has_arrival_jitter():
    root = ET.parse(CONFIG_DIR / "flashsale_e6_noisy.jmx").getroot()
    timer = root.find(".//GaussianRandomTimer")
    assert timer is not None
    assert _prop(timer, "RandomTimer.range") == "2"
