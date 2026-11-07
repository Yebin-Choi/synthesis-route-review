"""Offline tests for the QSAR Toolbox reading rules. Nothing here calls the API."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'evaluation' / 'scripts'))

import qsar_toolbox_service as toolbox  # noqa: E402


class AlertReadingTests(unittest.TestCase):
    def test_alert_names_are_positive(self):
        verdict, alerts, reason = toolbox._read_alerts(
            ['SN2', 'SN2 >> Alkylation, direct acting epoxides and related'])
        self.assertEqual(verdict, 'positive')
        self.assertEqual(len(alerts), 2)
        self.assertIsNone(reason)

    def test_only_no_alert_is_negative(self):
        verdict, alerts, reason = toolbox._read_alerts([toolbox.NO_ALERT])
        self.assertEqual(verdict, 'negative')
        self.assertEqual(alerts, [])
        self.assertIsNone(reason)

    def test_mixture_carrying_both_is_positive(self):
        # A mixture record answers with one constituent's alerts and the
        # literal "No alert found" for another, in the same list. Measured on
        # lou-1058 (butylhydrazine oxalate) against the 4.9 install.
        verdict, alerts, _ = toolbox._read_alerts(
            ['AN2', 'AN2 >> Nucleophilic addition reaction with cycloisomerization',
             toolbox.NO_ALERT, 'Radical'])
        self.assertEqual(verdict, 'positive')
        self.assertNotIn(toolbox.NO_ALERT, alerts)
        self.assertEqual(len(alerts), 3)

    def test_not_applicable_abstains_and_is_not_negative(self):
        verdict, alerts, reason = toolbox._read_alerts([toolbox.NOT_APPLICABLE])
        self.assertIsNone(verdict)
        self.assertEqual(alerts, [])
        self.assertEqual(reason, 'not_applicable_record')

    def test_empty_answer_abstains(self):
        for value in ([], None):
            verdict, _, reason = toolbox._read_alerts(value)
            self.assertIsNone(verdict)
            self.assertEqual(reason, 'no_record')


class RecordPickingTests(unittest.TestCase):
    """Search order is reshuffled per call, so the pick must not depend on it."""

    def _hits(self):
        return [{'ChemId': 'cccc', 'SubstanceType': 'Polymer'},
                {'ChemId': 'aaaa', 'SubstanceType': 'MonoConstituent'},
                {'ChemId': 'bbbb', 'SubstanceType': 'MonoConstituent'}]

    def test_same_pick_whatever_the_order(self):
        hits = self._hits()
        picked = {toolbox._pick_record(hits[i:] + hits[:i])['ChemId']
                  for i in range(len(hits))}
        self.assertEqual(picked, {'aaaa'})

    def test_polymer_and_uvcb_set_aside(self):
        for kind in toolbox.UNPROFILABLE_TYPES:
            hits = [{'ChemId': 'aaaa', 'SubstanceType': kind},
                    {'ChemId': 'zzzz', 'SubstanceType': 'MultiConstituent'}]
            self.assertEqual(toolbox._pick_record(hits)['ChemId'], 'zzzz')

    def test_multiconstituent_is_kept(self):
        hits = [{'ChemId': 'aaaa', 'SubstanceType': 'MultiConstituent'}]
        self.assertEqual(toolbox._pick_record(hits)['ChemId'], 'aaaa')

    def test_unprofilable_only_is_still_returned(self):
        # Returned rather than dropped, so the caller reports an abstention
        # with the tool's own "(N/A)" reason instead of a silent miss.
        hits = [{'ChemId': 'aaaa', 'SubstanceType': 'Polymer'}]
        self.assertEqual(toolbox._pick_record(hits)['ChemId'], 'aaaa')

    def test_no_hits_is_none(self):
        self.assertIsNone(toolbox._pick_record([]))


class LocalOnlyTests(unittest.TestCase):
    """Structures must not leave this machine."""

    def test_loopback_accepted(self):
        for url in ('http://127.0.0.1:5000', 'http://localhost:5000',
                    'http://[::1]:5000'):
            toolbox._check_local(url)

    def test_remote_refused(self):
        for url in ('http://example.com:5000', 'https://10.0.0.5',
                    'http://qsartoolbox.org'):
            with self.assertRaises(ValueError):
                toolbox._check_local(url)


class ScoringTests(unittest.TestCase):
    def test_abstentions_are_left_out_not_counted_negative(self):
        import run_qsar_toolbox as runner
        records = [{'label': '1', 'prediction': '1'},
                   {'label': '1', 'prediction': 'abstain'},
                   {'label': '0', 'prediction': '0'},
                   {'label': '0', 'prediction': 'abstain'}]
        result = runner.score(records)
        self.assertEqual(result['n_total'], 4)
        self.assertEqual(result['n_answered'], 2)
        self.assertEqual(result['sensitivity'], 1.0)
        self.assertEqual(result['specificity'], 1.0)
        self.assertEqual(result['balanced'], 1.0)
        self.assertEqual(result['false_negative'], 0)


class WebApiStartTests(unittest.TestCase):
    """Starting the local API. Nothing here launches a process or calls out."""

    def setUp(self):
        self.calls = []
        self.popen = mock.patch.object(
            toolbox.subprocess, 'Popen',
            side_effect=lambda *a, **k: self.calls.append(a) or mock.MagicMock())
        self.popen.start()
        self.addCleanup(self.popen.stop)

    def test_a_port_that_already_answers_starts_nothing(self):
        with mock.patch.object(toolbox, '_answering', return_value=True):
            self.assertTrue(toolbox.ensure_webapi())
        self.assertEqual(self.calls, [])

    def test_a_remote_base_url_is_never_started_or_used(self):
        """Structures stay on this machine, so a non-loopback host is refused."""
        with mock.patch.dict(toolbox.os.environ,
                             {'QSAR_TOOLBOX_WEBAPI': 'http://10.0.0.5:5000'}):
            self.assertFalse(toolbox.ensure_webapi())
        self.assertEqual(self.calls, [])

    def test_another_local_port_is_probed_but_left_alone(self):
        """A reader who moved the port is running it themselves."""
        with mock.patch.dict(toolbox.os.environ,
                             {'QSAR_TOOLBOX_WEBAPI': 'http://127.0.0.1:5050'}),                 mock.patch.object(toolbox, '_answering', return_value=False):
            self.assertFalse(toolbox.ensure_webapi())
        self.assertEqual(self.calls, [])

    def test_a_missing_install_is_reported_rather_than_guessed(self):
        with mock.patch.object(toolbox, '_answering', return_value=False),                 mock.patch.object(toolbox, 'webapi_executable', return_value=None):
            self.assertFalse(toolbox.ensure_webapi())
        self.assertEqual(self.calls, [])

    def test_it_starts_the_installed_binary_and_waits_for_it(self):
        answers = iter([False, False, True])
        with mock.patch.object(toolbox, '_answering',
                               side_effect=lambda **k: next(answers)),                 mock.patch.object(toolbox, 'webapi_executable',
                                  return_value=Path('C:/x/Bin/LMC.Toolbox.WebAPI.exe')),                 mock.patch.object(toolbox.time, 'sleep'):
            self.assertTrue(toolbox.ensure_webapi(timeout=30))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0],
                         [str(Path('C:/x/Bin/LMC.Toolbox.WebAPI.exe'))])

    def test_a_binary_that_never_answers_gives_up_instead_of_hanging(self):
        with mock.patch.object(toolbox, '_answering', return_value=False),                 mock.patch.object(toolbox, 'webapi_executable',
                                  return_value=Path('C:/x/Bin/LMC.Toolbox.WebAPI.exe')),                 mock.patch.object(toolbox.time, 'sleep'):
            self.assertFalse(toolbox.ensure_webapi(timeout=0.01))

    def test_an_override_path_that_does_not_exist_is_not_used(self):
        with mock.patch.dict(toolbox.os.environ,
                             {toolbox.WEBAPI_EXE_ENV: 'C:/nope/missing.exe'}):
            self.assertIsNone(toolbox.webapi_executable())


if __name__ == '__main__':
    unittest.main()
