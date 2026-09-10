import unittest

from spatialforge.i18n import (
    DEFAULT_LANG,
    get_dictionary,
    ls,
    supported_languages,
    t,
    tab_label,
)


class I18nTests(unittest.TestCase):
    def test_supported_languages(self):
        langs = supported_languages()
        self.assertIn("zh-CN", langs)
        self.assertIn("en", langs)
        self.assertIn("bilingual", langs)

    def test_default_is_bilingual(self):
        self.assertEqual(DEFAULT_LANG, "bilingual")

    def test_chinese_short_label(self):
        self.assertIn("观察", ls("tab.observation", "zh-CN"))
        self.assertIn("智能体", ls("tab.agent", "zh-CN"))

    def test_english_short_label(self):
        self.assertEqual(ls("tab.observation", "en"), "Observation")
        self.assertEqual(t("tab.observation", "en"), "Agent first-person observation")

    def test_bilingual_composes_short_labels(self):
        val = ls("tab.debug", "bilingual")
        self.assertIn("调试", val)
        self.assertIn("Debug", val)
        # short labels, not duplicated long paragraphs
        self.assertNotIn("PRIVILEGED", val)

    def test_missing_key_fallback(self):
        self.assertIn("nope.missing", t("nope.missing", "en"))
        self.assertIn("nope.missing", ls("nope.missing", "bilingual"))

    def test_dictionary_has_zh_and_en(self):
        d = get_dictionary()
        self.assertIn("zh", d)
        self.assertIn("en", d)
        self.assertEqual(set(d["zh"].keys()), set(d["en"].keys()))

    def test_tab_label_bilingual(self):
        self.assertIn("环境", tab_label("tab.environment", "bilingual"))
        self.assertIn("Environment", tab_label("tab.environment", "bilingual"))

    def test_core_keys_present(self):
        for key in ("tab.observation", "tab.agent", "tab.task", "tab.environment",
                    "tab.training", "tab.debug", "tab.settings", "debug.warn"):
            self.assertNotIn("[", t(key, "zh-CN"))
            self.assertNotIn("[", t(key, "en"))

    def test_workbench_keys_present(self):
        for key in ("mode.live", "mode.replay", "wb.godview", "wb.timeline",
                    "wb.decision", "wb.privileged", "dec.raw", "dec.parsed",
                    "dec.executed", "dec.teacher", "dec.visible", "dec.verifier",
                    "dec.terminal", "metrics.actions", "live.poll", "overlay.target",
                    "player.play", "player.pause", "player.prev", "player.next",
                    "player.speed"):
            self.assertNotIn("[", t(key, "zh-CN"))
            self.assertNotIn("[", t(key, "en"))
            self.assertNotIn("[", ls(key, "bilingual"))


if __name__ == "__main__":
    unittest.main()
