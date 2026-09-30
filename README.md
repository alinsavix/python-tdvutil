# tdvutil

A grab bag of little Python helpers I kept copy-pasting between projects, so
I finally stuck them in one place.

**Heads up:** this is mostly here for my own convenience. You're welcome to
use it, but I make no promises about keeping the API stable. Things can and
will get renamed, changed, or ripped out whenever it suits me, even in minor
releases. If you depend on it, pin a version.

## Installing

```sh
pip install tdvutil
```

Needs Python 3.9+.

## What's in here

From `tdvutil` directly:

- `ppretty`: pretty-prints objects
- `alintrospect` / `whatis`: a very janky way to poke at an object's class
  hierarchy and see where its methods and attributes come from
- `now` / `nowf`: current time, with less typing
- `pathfix`: path cleanup
- `sec_to_hms`, `sec_to_timecode`, `sec_to_shortstr`, `hms_to_sec`,
  `timecode_to_sec`: converting between seconds and human-readable times

Plus a couple of submodules:

- `tdvutil.argparse`: `CheckFile` and `NegateAction`, for argparse
- `tdvutil.llmcost`: pricing data for a bunch of LLM models, and a way to
  figure out what a call cost

```python
from tdvutil import sec_to_hms
from tdvutil.llmcost import get_model

sec_to_hms(3725)  # '01:02:05.000'

cost = get_model("gpt-4o").cost(input_tokens=10_000, output_tokens=2_000)
print(cost.total_cost)  # 0.045
```

## Hacking on it

Uses [uv](https://docs.astral.sh/uv/). `make localdev` sets things up,
`make test` runs the tests, and `make dist` builds.

GitHub Actions runs the tests and builds packages for PRs targeting `main`
and pushes to `main`, without publishing. Maintainer approval is required
to run outside contributors' PR workflows.

Pushing a `v*` tag makes GitHub Actions build it and publish to TestPyPI,
then PyPI. Manually running the workflow on a `v*` tag does the same,
including publishing to real PyPI. Manual branch runs just run the checks
and save the built packages as workflow artifacts; they don't publish.

## License

GPLv3, see [LICENSE.txt](LICENSE.txt).
