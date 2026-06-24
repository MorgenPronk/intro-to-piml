"""Make LaTeX math survive marked's CommonMark escapes.

CommonMark treats `\\X` as a literal X for X in !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~.
That breaks our LaTeX inside $...$ and $$...$$ blocks because:
  - paired `_X_` get parsed as emphasis (we fix this by escaping each `_` to `\\_`)
  - `\\;` (LaTeX thin space) loses its backslash and becomes `;`
  - `\\,` loses its backslash and becomes `,`
  - `\\{` `\\}` similar

Fix: inside math regions, double the backslash before any CommonMark-escapable
punctuation, EXCEPT for the underscore — for that, use the single-backslash
form `\\_` which marked converts back to `_` (the subscript trigger KaTeX needs).
"""
import re

# Punctuation chars that CommonMark treats as escapable. Excluding _ which is
# handled separately because of its subscript semantics.
ESCAPABLE = set(r"""!"#$%&'()*+,-./:;<=>?@[]^`{|}~""")


def fix_math(inner: str) -> str:
    """Walk the math string; double `\\X` and escape lone `_`."""
    out = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        # Double backslashes followed by CommonMark-escapable punctuation
        if ch == "\\" and i + 1 < len(inner) and inner[i + 1] in ESCAPABLE:
            out.append("\\\\")
            out.append(inner[i + 1])
            i += 2
            continue
        # Escape lone underscores so marked doesn't pair them as emphasis
        if ch == "_" and (i == 0 or inner[i - 1] != "\\"):
            out.append("\\_")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


with open("slides.md", encoding="utf-8") as f:
    src = f.read()


def unescape_prior(inner: str) -> str:
    """Undo a prior pass: `\\_` → `_` and `\\X` → `\X` for X in ESCAPABLE."""
    inner = inner.replace("\\_", "_")
    for x in ESCAPABLE:
        inner = inner.replace("\\\\" + x, "\\" + x)
    return inner


# Undo any prior escaping inside math regions, then re-apply cleanly.
def fix_idempotent(inner: str) -> str:
    return fix_math(unescape_prior(inner))


src = re.sub(r"\$\$[^$]+\$\$", lambda m: fix_idempotent(m.group(0)), src)
src = re.sub(r"(?<!\$)\$(?!\$)[^$\n]+?(?<!\$)\$(?!\$)", lambda m: fix_idempotent(m.group(0)), src)

with open("slides.md", "w", encoding="utf-8", newline="\n") as f:
    f.write(src)

print(f"\\_ count: {src.count(chr(92) + '_')}")
print(f"\\\\; count: {src.count(chr(92) + chr(92) + ';')}")
print(f"\\\\, count: {src.count(chr(92) + chr(92) + ',')}")
