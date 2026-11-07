import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from video_classification_claim_audit import audit as a
from video_classification_claim_audit.phrases import VIDEOCLASSIFICATION, NOVIDEOCLASSIFICATION


class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(VIDEOCLASSIFICATION, (
            'video classification', 'video-classification', 'videoclassification',
            'video classification model', 'video classification models',
            'video classification system', 'video classification systems',
            'video classification feature', 'video classification capability',
            'video classification support', 'supports video classification',
            'video classification tool', 'video classification api',
            'video classification pipeline', 'video classification algorithm',
            'video classification library', 'video classification framework',
            'video classification service', 'video classification function',
            'video classification engine', 'video classification engines',
            'video classifier', 'video classifiers', 'video clip classification',
            'video clip classifier', 'classify video', 'classify videos',
            'classifies video', 'classifies videos', 'classifying video',
            'classifying videos', 'video categorization', 'video categorisation',
            'categorize video', 'categorise video', 'video category classification',
            'video genre classification', 'video genre classifier',
            'video content classification', 'clip classification', 'clip classifier',
            'classify video clips', 'video scene classification', 'video tagging',
            'video tagger', 'video label classification', 'video class prediction',
            'predict video class', 'video classification task',
            'video classification benchmark'))
        self.assertEqual(NOVIDEOCLASSIFICATION, (
            'does not support video classification', 'do not support video classification',
            "doesn't support video classification", "don't support video classification",
            'no video classification', 'without video classification',
            'never performs video classification', 'does not perform video classification',
            'video classification not supported', 'video classification unsupported',
            'no video classification support', 'lacks video classification',
            'missing video classification', 'video classification disabled',
            'no video classification capability', 'without video classification support',
            'does not provide video classification', 'no video classification model',
            'video classification not available', 'does not use video classification',
            'no video classifier', 'without video classifier',
            'does not support video classifier', 'no clip classification'))

    def test_generic_names(self):
        self.assertEqual(a.GENERIC_NAMES, (
            'classify_video', 'video_classify', 'video_classification',
            'classify_clip', 'clip_classification', 'video_classifier',
            'classify_videos', 'predict_video', 'video_predict',
            'categorize_video', 'video_categorize', 'label_video',
            'video_label', 'tag_video', 'video_tag', 'video_category',
            'categorize_clip', 'clip_categorize', 'classify_video_clip',
            'video_clip_classification', 'predict_video_label',
            'video_label_predict', 'video_genre_classify', 'classify_video_genre'))

    def test_dedicated_names(self):
        self.assertEqual(a.DEDICATED_NAMES, ('VideoMAEForVideoClassification',
            'VivitForVideoClassification', 'TimesformerForVideoClassification',
            'AutoModelForVideoClassification'))
        self.assertEqual(a.BACKBONE_NAMES, ('r3d_18', 'mc3_18', 'r2plus1d_18',
            's3d', 'swin3d_t', 'swin3d_s', 'swin3d_b', 'create_slowfast', 'create_x3d'))

    def test_pipeline_pairs(self):
        for prefix, name in [('import transformers', 'transformers.pipeline'),
                             ('import transformers as t', 't.pipeline'),
                             ('from transformers import pipeline', 'pipeline'),
                             ('from transformers import pipeline as p', 'p')]:
            for task in ["'video-classification'", "task='video-classification'"]:
                for body in [f'm={name}({task}); m(clip)', f'{name}({task})(clip)']:
                    with self.subTest(body=body, prefix=prefix):
                        self.assertTrue(a.evidence(prefix + '\n' + body))

    def test_dedicated_pairs(self):
        for name in a.DEDICATED_NAMES:
            for ctor in [name, name + '.from_pretrained', 'transformers.' + name,
                         'transformers.' + name + '.from_pretrained']:
                for body in [f'm={ctor}(x); m(clip)', f'{ctor}(x)(clip)']:
                    with self.subTest(body=body):
                        self.assertTrue(a.evidence(body))
            self.assertTrue(a.evidence(f'from transformers import {name} as M\nm=M(x); m(clip)'))

    def test_backbone_pairs(self):
        for name in a.BACKBONE_NAMES:
            module = 'pytorchvideo.models.slowfast' if name == 'create_slowfast' else 'pytorchvideo.models.x3d' if name == 'create_x3d' else 'torchvision.models.video'
            for prefix, ctor in [('', name), ('', module + '.' + name),
                                  (f'import {module} as v\n', 'v.' + name),
                                  (f'from {module} import {name} as M\n', 'M')]:
                for body in [f'm={ctor}(); m(clip)', f'{ctor}()(clip)']:
                    with self.subTest(body=body, prefix=prefix):
                        self.assertTrue(a.evidence(prefix + body))

    def test_generic_calls(self):
        for name in a.GENERIC_NAMES:
            for body in [f'obj.{name}(clip)', f'factory().{name}(clip)',
                         f'objects[0].{name}(clip)', f'(1).{name}(clip)',
                         f'from lib import {name}\n{name}(clip)',
                         f'from lib import {name} as f\nf(clip)']:
                with self.subTest(body=body):
                    self.assertTrue(a.evidence(body))

    def test_rejections(self):
        cases = [
            'import torchvision, transformers, pytorchvideo, torch, cv2, decord',
            "from transformers import pipeline\np=pipeline('image-classification'); p(frame)",
            'm=ResNet(); m(frame)', 'm=EfficientNet(); m(frame)',
            'labels=[image_model(frame) for frame in frames]; vote(labels)',
            'x.recognize_action(clip)', 'x.detect_action(clip)', 'x.action_recognition(clip)',
            'from lib import recognize_action\nrecognize_action(clip)',
            'm=VideoMAEModel(); m(clip)', 'm=VivitModel(); m(clip)',
            'm=TimesformerModel(); m(clip)', 'cv2.VideoCapture(path)',
            'decord.VideoReader(path)', 'extract_frames(clip)',
            "from transformers import pipeline\npipeline('video-classification')",
            'VideoMAEForVideoClassification(x)', 'r3d_18()',
            'm(clip)', 'model(clip)', 'VideoMAEForVideoClassification',
            'x.classify_video', 'classify_video', 'classify_video(clip)',
            "'video classification video classifier'", '# video classification',
            '"""video classifier"""',
            "subprocess.run(['video-classification', 'clip'])",
            "from transformers import pipeline\np=pipeline('video-classification'); other(clip)",
            'm=r3d_18(); other(clip)', 'm=VideoMAEForVideoClassification(); other(clip)',
            "pipeline('video-classification')(clip)",
            'this is invalid python',
        ]
        for body in cases:
            with self.subTest(body=body):
                self.assertFalse(a.evidence(body))

    def test_all_lone_constructions_and_references(self):
        for name in a.DEDICATED_NAMES + a.BACKBONE_NAMES:
            self.assertFalse(a.evidence(name + '()'))
            self.assertFalse(a.evidence(name))
        for name in a.GENERIC_NAMES:
            self.assertFalse(a.evidence('from lib import ' + name))
            self.assertFalse(a.evidence('x.' + name))

    def test_shadowing(self):
        cases = [
            'from lib import classify_video\ndef classify_video(x): pass\nclassify_video(x)',
            'def classify_video(x): pass\nx.classify_video(x)',
            'class VideoMAEForVideoClassification: pass\nVideoMAEForVideoClassification()(x)',
            'from transformers import VideoMAEForVideoClassification as M\nclass M: pass\nM()(x)',
            'from torchvision.models.video import r3d_18\ndef r3d_18(): pass\nr3d_18()(x)',
            'import torchvision.models.video as v\nclass v: pass\nv.r3d_18()(x)',
            "from transformers import pipeline\ndef pipeline(x): pass\npipeline('video-classification')(x)",
            'from lib import classify_video as f\ndef f(x): pass\nf(x)',
            'from lib import classify_video\nclassify_video = other\nclassify_video(x)',
        ]
        for body in cases:
            with self.subTest(body=body):
                self.assertFalse(a.evidence(body))

    def test_span_removal(self):
        self.assertEqual(a.match_phrases('does not support video classification'),
                         ([], ['does not support video classification']))
        for denial in NOVIDEOCLASSIFICATION:
            self.assertEqual(a.match_phrases(denial), ([], [denial]))

    def test_longest_first(self):
        for text in ['video classification model', 'video clip classification']:
            self.assertEqual(a.match_phrases(text), ([text], []))
        self.assertEqual(a.match_phrases('video classification model; video classification'),
                         (['video classification model', 'video classification'], []))

    def test_case_and_collision(self):
        self.assertEqual(a.match_phrases('VIDEO CLASSIFICATION'), (['video classification'], []))
        self.assertEqual(a.match_phrases('clip classification of audio samples'), (['clip classification'], []))
        self.assertEqual(a.match_phrases('video tagging with object labels'), (['video tagging'], []))
        self.assertEqual(a.match_phrases('video category classification for the library index'), (['video category classification'], []))

    def test_fenced_and_inline(self):
        self.assertEqual(a.match_phrases('```text\nvideo classification\n```\n'), ([], []))
        self.assertEqual(a.match_phrases('`video classification`'), (['video classification'], []))
        self.assertEqual(a.match_phrases('video--classification'), ([], []))

    def fixture(self, root, readme='', source=None, extra=None):
        (root / 'README.md').write_text(readme, encoding='utf-8')
        if source is not None:
            (root / 'main.py').write_text(source, encoding='utf-8')
        for name, text in (extra or {}).items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')

    def cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = a.main(args)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_cli_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 'video classification', 'x.classify_video(clip)')
            code, out, _ = self.cli(['audit', tmp])
            self.assertEqual((code, out.strip()), (0, 'PASS'))

    def test_cli_bad_json_and_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), 'VIDEO CLASSIFICATION')
            code, out, _ = self.cli(['audit', tmp, '--json'])
            self.assertEqual(code, 2)
            findings = json.loads(out)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]['verdict'], 'documented-video-classification-claim-false')
            self.assertEqual(findings[0]['severity'], 'HIGH')
            self.assertTrue(findings[0]['message'])

    def test_cli_test_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), extra={'tests/fixture.md': 'video classification'})
            for options, expected in [([], 0), (['--fail-on', 'medium'], 2)]:
                code, out, _ = self.cli(['audit', tmp, '--json'] + options)
                self.assertEqual(code, expected)
                finding = json.loads(out)[0]
                self.assertEqual(finding['severity'], 'MEDIUM')
                self.assertEqual(finding['verdict'], 'documented-video-classification-claim-false')

    def test_cli_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 'no video classification', 'x.classify_video(clip)')
            code, out, _ = self.cli(['audit', tmp, '--json'])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)[0]['verdict'], 'documented-no-video-classification-claim-false')
            self.assertEqual(json.loads(out)[0]['severity'], 'HIGH')
            (root / 'main.py').unlink()
            self.assertEqual(self.cli(['audit', tmp])[0], 0)

    def test_cli_no_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), 'Hello world')
            self.assertEqual(self.cli(['audit', tmp])[:2], (0, 'PASS\n'))

    def test_cli_ignore_before_positional(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), extra={'docs/claims.md': 'video classification', 'docs/other.md': 'video classifier'})
            self.assertEqual(self.cli(['audit', '--ignore', 'docs/', tmp])[0], 0)
            self.assertEqual(self.cli(['audit', tmp, '--ignore', 'docs/claims', '--ignore', 'docs/other'])[0], 0)
            self.assertEqual(self.cli(['audit', tmp])[0], 2)

    def test_ignore_is_literal_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), extra={'docs2/ignored.md': 'Hello',
                                         'tests2/claim.py': 'x.classify_video(clip)',
                                         'docs/claim.md': 'video classification'})
            # A trailing slash must not swallow a sibling with a longer name.
            self.assertEqual(self.cli(['audit', tmp, '--ignore', 'tests/'])[0], 0)
            self.assertEqual(self.cli(['audit', tmp, '--ignore', 'tests2/'])[0], 2)

    def test_instance_aliases_and_rebinding(self):
        self.assertTrue(a.evidence('m=r3d_18(); alias=m; alias(clip)'))
        self.assertFalse(a.evidence('m=r3d_18(); m=other; m(clip)'))
        self.assertFalse(a.evidence('a=b; b=a; a(clip)'))

    def test_cli_ignore_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), 'video classification', 'x.classify_video(clip)')
            self.assertEqual(self.cli(['audit', tmp, '--ignore', 'main.py'])[0], 2)

    def test_cli_version_help_errors(self):
        self.assertEqual(self.cli(['--version'])[:2], (0, '0.1.0\n'))
        self.assertEqual(self.cli(['--help'])[0], 0)
        self.assertEqual(self.cli([])[0], 1)
        self.assertEqual(self.cli(['audit', '--fail-on', 'low'])[0], 1)
        self.assertEqual(self.cli(['audit', '/nonexistent/repository'])[0], 1)

    def test_verbose(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp), 'Hello')
            code, out, err = self.cli(['audit', tmp, '--verbose', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out), [])
            self.assertIn('Scope:', err)
            self.assertIn('README.md:', err)

    def test_scan_scope_and_split_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 'video classification', extra={
                'construct.py': 'm=r3d_18()', 'invoke.py': 'm(clip)',
                '.git/fake.py': 'x.classify_video(clip)',
                '__pycache__/fake.py': 'x.classify_video(clip)',
                '.venv/fake.py': 'x.classify_video(clip)',
                'customenv/pyvenv.cfg': '', 'customenv/fake.py': 'x.classify_video(clip)'})
            self.assertEqual(self.cli(['audit', tmp])[0], 2)
            (root / 'README.md').write_text('Hello')
            (root / 'OTHER.md').write_text('video classification')
            self.assertEqual(self.cli(['audit', tmp])[0], 0)
            self.fixture(root, extra={'docs/deep/claim.md': 'video classification'})
            self.assertEqual(self.cli(['audit', tmp])[0], 2)

    def test_self_audit(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(a.match_phrases((root / 'README.md').read_text()), ([], []))
        self.assertEqual(self.cli(['audit', str(root)])[0], 0)


if __name__ == '__main__':
    unittest.main()
