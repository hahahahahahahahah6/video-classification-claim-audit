"""Deterministic prose scanning and single-module AST evidence checks."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import VIDEOCLASSIFICATION, NOVIDEOCLASSIFICATION

GENERIC_NAMES = ('classify_video', 'video_classify', 'video_classification', 'classify_clip', 'clip_classification', 'video_classifier', 'classify_videos', 'predict_video', 'video_predict', 'categorize_video', 'video_categorize', 'label_video', 'video_label', 'tag_video', 'video_tag', 'video_category', 'categorize_clip', 'clip_categorize', 'classify_video_clip', 'video_clip_classification', 'predict_video_label', 'video_label_predict', 'video_genre_classify', 'classify_video_genre')
DEDICATED_NAMES = ('VideoMAEForVideoClassification', 'VivitForVideoClassification', 'TimesformerForVideoClassification', 'AutoModelForVideoClassification')
BACKBONE_NAMES = ('r3d_18', 'mc3_18', 'r2plus1d_18', 's3d', 'swin3d_t', 'swin3d_s', 'swin3d_b', 'create_slowfast', 'create_x3d')


def evidence(source):
    """Return whether one module contains a complete recognized evidence route."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    nodes = list(ast.walk(tree))
    shadows = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    # Parameters and ordinary rebindings must not turn an imported name into evidence.
    shadows.update(n.arg for n in nodes if isinstance(n, ast.arg))
    imports = {}
    from_names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != '*':
                    local = alias.asname or alias.name
                    imports[local] = (node.module or '') + '.' + alias.name
                    from_names.add(local)
    assignments = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
                    if target.id in imports:
                        shadows.add(target.id)

    def resolved(node):
        if isinstance(node, ast.Name):
            if node.id in shadows:
                return ''
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolved(node.value)
            return base + '.' + node.attr if base and node.attr not in shadows else ''
        return ''

    def construction(node, seen=None):
        seen = set() if seen is None else seen
        if isinstance(node, ast.Name) and node.id not in shadows and node.id not in seen:
            values = assignments.get(node.id, [])
            return len(values) == 1 and construction(values[0], seen | {node.id})
        if not isinstance(node, ast.Call):
            return False
        name = resolved(node.func)
        parts = name.split('.')
        leaf = parts[-1]
        if leaf == 'from_pretrained' and len(parts) > 1:
            leaf = parts[-2]
        if leaf in DEDICATED_NAMES + BACKBONE_NAMES:
            return True
        if leaf == 'pipeline' and name.startswith('transformers.'):
            task = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == 'task'), None)
            return isinstance(task, ast.Constant) and task.value == 'video-classification'
        return False

    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if construction(node.func):
            return True
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in GENERIC_NAMES and func.attr not in shadows:
                # Any base is accepted, except a locally shadowed root.
                base = func.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if not isinstance(base, ast.Name) or base.id not in shadows:
                    return True
        elif isinstance(func, ast.Name) and func.id in from_names:
            if resolved(func).split('.')[-1] in GENERIC_NAMES:
                return True
    return False


def prose(text):
    """Exclude triple-backtick fences; inline code remains prose."""
    lines = []
    fenced = False
    for line in text.splitlines(keepends=True):
        if re.match(r'^\s*```', line):
            fenced = not fenced
            lines.append('\n')
        elif not fenced:
            lines.append(line.replace('`', ''))
    return ''.join(lines)


def _consume(text, phrases):
    hits = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start != -1:
            hits.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return text, hits


def match_phrases(text):
    text, denials = _consume(prose(text).lower(), NOVIDEOCLASSIFICATION)
    _, claims = _consume(text, VIDEOCLASSIFICATION)
    return claims, denials


def audit(repo, ignore=(), verbose=False):
    """Scan documentation, test fixtures, and Python modules without writing files."""
    root = Path(repo)
    if not root.is_dir():
        raise OSError('repository is not a directory: ' + str(root))
    files = []
    def walk_error(error):
        raise error
    excluded = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv'}
    for directory, dirs, names in os.walk(root, onerror=walk_error, followlinks=False):
        relative = Path(directory).relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in excluded
                         and not (Path(directory) / d / 'pyvenv.cfg').exists()
                         and not (Path(directory) / d).is_symlink()
                         and not any(((relative / d).as_posix() + '/').startswith(p) for p in ignore))
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink() or any(rel.startswith(p) for p in ignore):
                continue
            files.append((path, rel))
    has_evidence = False
    documents = []
    fixtures = []
    if verbose:
        print('Scope: README.md, docs/**/*.md, tests/ fixtures; Python sources in ' + str(root), file=sys.stderr)
    for path, rel in files:
        parts = Path(rel).parts
        is_doc = rel == 'README.md' or (parts[0] == 'docs' and path.suffix == '.md')
        is_test = 'tests' in parts[:-1]
        if path.suffix != '.py' and not is_doc and not is_test:
            continue
        text = path.read_text(encoding='utf-8')
        found = path.suffix == '.py' and evidence(text)
        has_evidence |= found
        if is_doc:
            documents.append((rel, match_phrases(text)))
        elif is_test:
            fixtures.append((rel, match_phrases(text)))
        if verbose:
            print(f'{rel}: evidence={bool(found)} documentation={is_doc} fixture={is_test}', file=sys.stderr)
    findings = []
    def add(path, denial, severity):
        verdict = 'documented-' + ('no-' if denial else '') + 'video-classification-claim-false'
        findings.append({'verdict': verdict, 'message': 'Documented denial contradicts source evidence.' if denial else 'Documented capability lacks recognized source evidence.', 'severity': severity, 'path': path})
    for rel, (claims, denials) in documents:
        if claims and not has_evidence:
            add(rel, False, 'HIGH')
        if denials and has_evidence:
            add(rel, True, 'HIGH')
    if not has_evidence and not any(claims for _, (claims, _) in documents):
        for rel, (claims, _) in fixtures:
            if claims:
                add(rel, False, 'MEDIUM')
    return findings


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f'{self.prog}: error: {message}\n')


def main(argv=None):
    parser = Parser(description='Offline documentation claim auditor')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    scan = commands.add_parser('audit')
    scan.add_argument('repo', nargs='?', default='.')
    scan.add_argument('--json', action='store_true')
    scan.add_argument('--verbose', action='store_true')
    scan.add_argument('--ignore', action='append', default=[], metavar='PREFIX')
    scan.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    args = parser.parse_args(argv)
    try:
        findings = audit(args.repo, args.ignore, args.verbose)
    except (OSError, UnicodeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(findings, indent=2))
    elif findings:
        for finding in findings:
            print('{severity} {verdict} {path}: {message}'.format(**finding))
    else:
        print('PASS')
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
