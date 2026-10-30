"""Essential filesystem resolution tests for launch inputs."""

from __future__ import annotations

import unittest
from dataclasses import replace

from orchestration.configured_eval_run import EvalConfigError
from orchestration.eval_run_path_resolution import resolve_eval_run
from phase1_resolution_test_support import ResolutionFixture


class EvalRunPathResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ResolutionFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def resolve(self, config=None):
        return resolve_eval_run(
            config or self.fixture.config(),
            self.fixture.workspace,
            repository_root=self.fixture.runtime,
            validate_destination=False,
        )

    def test_resolves_exact_iteration_and_pinned_tokenizer(self) -> None:
        resolved = self.resolve()
        self.assertEqual(resolved.paths.checkpoint_root, str(self.fixture.checkpoint))
        self.assertEqual(resolved.paths.checkpoint_metadata, str(self.fixture.metadata))
        cli = dict(resolved.resolved_cli_overrides)
        self.assertEqual(cli["tokenizer-model"], "nvidia/n4-v6-262k")
        self.assertEqual(
            cli["tokenizer-hf-revision"],
            "8f5dd326777cbdd5c018d92ba35f473d971e121d",
        )
        self.assertEqual(cli["expert-model-parallel-size"], 8)
        self.assertNotIn("inference-context-from-checkpoint", cli)
        self.assertEqual(resolved.paths.chat_template, str(self.fixture.template))
        self.assertEqual(cli["chat-template"], str(self.fixture.template))

    def test_checkpoint_owned_context_is_an_explicit_server_opt_in(self) -> None:
        config = self.fixture.config()
        config = replace(
            config,
            model=replace(
                config.model,
                server=replace(config.model.server, context_length=None),
            ),
        )

        cli = dict(self.resolve(config).resolved_cli_overrides)

        self.assertIs(cli["inference-context-from-checkpoint"], True)

    def test_tokenizer_policy_emits_no_explicit_chat_template(self) -> None:
        config = self.fixture.config()
        config = replace(
            config,
            benchmark=replace(
                config.benchmark,
                generation=replace(
                    config.benchmark.generation,
                    chat_template="tokenizer",
                ),
            ),
        )
        resolved = self.resolve(config)
        self.assertIsNone(resolved.paths.chat_template)
        self.assertNotIn("chat-template", dict(resolved.resolved_cli_overrides))

    def test_custom_template_is_resolved_beneath_nvbench(self) -> None:
        custom = self.fixture.template.with_name("custom.jinja")
        custom.write_text("{{ messages }}\n", encoding="utf-8")
        config = self.fixture.config()
        config = replace(
            config,
            benchmark=replace(
                config.benchmark,
                generation=replace(
                    config.benchmark.generation,
                    chat_template="assets/custom.jinja",
                ),
            ),
        )
        resolved = self.resolve(config)
        self.assertEqual(resolved.paths.chat_template, str(custom))
        self.assertEqual(dict(resolved.resolved_cli_overrides)["chat-template"], str(custom))

    def test_does_not_consult_latest_checkpoint_tracker(self) -> None:
        self.fixture.tracker.write_text("1\n", encoding="utf-8")
        self.assertEqual(self.resolve().configured.checkpoint.iteration, 49250)

    def test_missing_explicit_iteration_fails(self) -> None:
        config = self.fixture.config()
        config = replace(config, checkpoint=replace(config.checkpoint, iteration=49249))
        with self.assertRaisesRegex(EvalConfigError, "iteration directory"):
            self.resolve(config)

    def test_tokenizer_requires_a_pinned_hugging_face_revision(self) -> None:
        config = self.fixture.config()
        config = replace(
            config,
            model=replace(
                config.model,
                tokenizer=replace(config.model.tokenizer, revision="main"),
            ),
        )
        with self.assertRaisesRegex(EvalConfigError, "full lowercase commit SHA"):
            self.resolve(config)

    def test_output_must_remain_under_workspace(self) -> None:
        config = self.fixture.config()
        config = replace(
            config,
            output=replace(config.output, configured_root="/tmp/outside-nvbench-test"),
        )
        with self.assertRaisesRegex(EvalConfigError, "beneath"):
            self.resolve(config)


if __name__ == "__main__":
    unittest.main()
