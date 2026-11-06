"""Replay the scalar/if subset used by the actual Pine tracker, not Pine itself.

No TA, market feeds or Pine runtime emulation: fixtures supply completed-bar
candidate packets. This executes the production transition block verbatim.
"""
import math
import re


class ScalarPine:
    def __init__(self, source):
        self.source = source
        self.functions = {}

    def evaluate(self, expression, values):
        expression = expression.strip()
        depth = 0
        quote = False
        question = None
        nested = 0
        for index, char in enumerate(expression):
            if char == '"':
                quote = not quote
            if quote:
                continue
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0 and char == '?':
                if question is None:
                    question = index
                else:
                    nested += 1
            elif depth == 0 and char == ':' and question is not None:
                if nested:
                    nested -= 1
                else:
                    condition = self.evaluate(expression[:question], values)
                    branch = expression[question + 1:index] if condition else expression[index + 1:]
                    return self.evaluate(branch, values)
        expression = expression.replace('math.min', 'min').replace('math.max', 'max')
        expression = expression.replace('na(', 'is_na(').replace('close[1]', 'previous_close')
        expression = re.sub(r'\bna\b', 'NA', expression)
        expression = re.sub(r'\btrue\b', 'True', expression)
        expression = re.sub(r'\bfalse\b', 'False', expression)
        bindings = dict(math=math, min=min, max=max, NA=float('nan'),
                        is_na=lambda v: v is None or isinstance(v, float) and math.isnan(v))
        for name in ('f_setup_level', 'f_candidate_block', 'f_trigger_price'):
            bindings[name] = lambda *args, name=name: self.call(name, *args)
        return eval(expression, {'__builtins__': {}}, bindings | values)

    def parse(self, text):
        lines = []
        for raw in text.splitlines():
            raw = raw.split('//', 1)[0].rstrip()
            if raw.strip():
                lines.append((len(raw) - len(raw.lstrip()), raw.strip()))

        def suite(index, indent):
            nodes = []
            while index < len(lines) and lines[index][0] == indent:
                statement = lines[index][1]
                if statement.startswith('else'):
                    break
                if statement.startswith('if '):
                    branches = []
                    condition = statement[3:]
                    while True:
                        index += 1
                        child, index = suite(index, lines[index][0])
                        branches.append((condition, child))
                        if index >= len(lines) or lines[index][0] != indent or not lines[index][1].startswith('else'):
                            break
                        statement = lines[index][1]
                        if statement == 'else':
                            condition = 'true'
                        elif statement.startswith('else if '):
                            condition = statement[8:]
                        else:
                            raise ValueError(statement)
                    nodes.append(('if', branches))
                else:
                    assignment = re.fullmatch(r'(?:(?:varip|var) )?(?:(?:float|int|bool|string) )?(\w+)\s*(?::=|=)\s*(.+)', statement)
                    if assignment:
                        nodes.append(('assign', *assignment.groups()))
                    else:
                        nodes.append(('expression', statement))
                    index += 1
            return nodes, index

        result, end = suite(0, lines[0][0])
        if end != len(lines):
            raise ValueError(f'Unparsed Pine statements: {lines[end:]}')
        return result

    def execute(self, nodes, values):
        result = None
        for node in nodes:
            if node[0] == 'if':
                for condition, branch in node[1]:
                    if self.evaluate(condition, values):
                        result = self.execute(branch, values)
                        break
            elif node[0] == 'assign':
                values[node[1]] = self.evaluate(node[2], values)
            else:
                result = self.evaluate(node[1], values)
        return result

    def call(self, name, *args):
        if name not in self.functions:
            lines = self.source.splitlines()
            for index, line in enumerate(lines):
                match = re.fullmatch(re.escape(name) + r'\((.*)\) =>', line)
                if match:
                    names = [part.strip().split()[-1] for part in match[1].split(',')]
                    body = []
                    for raw in lines[index + 1:]:
                        if raw and not raw.startswith(' '):
                            break
                        body.append(raw)
                    self.functions[name] = names, self.parse('\n'.join(body))
                    break
            else:
                raise ValueError(f'Function not found: {name}')
        names, nodes = self.functions[name]
        if len(names) != len(args):
            raise ValueError((name, names, args))
        return self.execute(nodes, dict(zip(names, args)))
