# storyteller

## Managing Python dependencies (uv)

Python dependencies are declared in [`pyproject.toml`](pyproject.toml) and
locked in [`uv.lock`](uv.lock), which pins every direct and transitive
dependency (with hashes) so that installs are reproducible. Both files are
committed to the repository. Docker builds run `uv sync --locked`, which
**fails the build** if `uv.lock` is out of date with `pyproject.toml` — this
is intentional, so an outdated lock file never silently reaches production.

Install [uv](https://docs.astral.sh/uv/) locally once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Adding a new library

```bash
uv add <package>                # runtime dependency, e.g. uv add httpx
uv add --group dev <package>    # dev-only tool, e.g. uv add --group dev mypy
```

This adds the package to `pyproject.toml` and updates `uv.lock` in one step.
Commit both files.

### Bumping a version

```bash
uv lock --upgrade-package <package>   # bump one package (and whatever it requires)
uv lock --upgrade                     # bump everything to the latest allowed version
```

If you instead edit a version pin directly in `pyproject.toml`, regenerate
the lock file afterwards:

```bash
uv lock
```

Always commit the updated `uv.lock` together with `pyproject.toml`.

### Installing dependencies locally (for your editor, tests, etc.)

```bash
uv sync            # creates/updates ./.venv with runtime + dev dependencies
uv run manage.py check   # run any command inside that environment
```

Point your editor's Python interpreter at `./.venv/bin/python` for linting and
type checking to work against the same versions used in CI and production.

### Keeping dependencies current

[Dependabot](.github/dependabot.yml) opens a weekly pull request for outdated
`uv` dependencies, the Docker base image, and GitHub Actions. Review and merge
these like any other PR; CI/the Docker build will catch a broken upgrade
before it ships.

## Huey recovery after deployment

Video-generation tasks are persisted in Huey's Redis queue. A deployment can
still terminate the worker after it has claimed a task, so `GenVideo` stores
its processing state and the worker checks for stale in-progress videos before
it starts. A periodic Huey task repeats the check every five minutes.

Configure recovery with environment variables:

| Setting | Default | Purpose |
| --- | --- | --- |
| `HUEY_RECOVERY_ENABLED` | `true` | Enables startup and periodic recovery. Set to `false` to disable it temporarily. |
| `HUEY_RECOVERY_STALE_MINUTES` | `10` | A task is considered interrupted only after this many minutes without an update. Set this above the expected duration of normal API calls. |
| `HUEY_RECOVERY_STARTUP_STALE_MINUTES` | `0` | Startup-specific stale threshold. Keep `0` to resume interrupted work immediately after a worker restart; increase it only for a rolling deployment with more than one active worker. |
| `HUEY_RECOVERY_MAX_ATTEMPTS` | `3` | Maximum automatic recovery attempts. A further interruption marks the video `FAILED` with error type `RECOVERY`. |

Run the recovery scan manually when needed:

```bash
docker compose run --rm django python manage.py recover_stuck_videos
```

The command accepts `--stale-minutes` and `--max-attempts` overrides for a
controlled recovery operation.

Recovery resumes from durable artifacts whenever possible. In particular, a
saved voice file and ElevenLabs character alignment are used to regenerate
subtitles without another paid ElevenLabs TTS request. Processing remains
at-least-once: an external API request interrupted before its result is saved
cannot be proven not to have completed, so recovery is intentionally limited.

### Optional production drain

The Kubernetes Huey container defaults to `HUEY_DRAIN_SECONDS=0`, so local/dev
deployments do not wait. To drain a worker in production, set
`HUEY_DRAIN_SECONDS` in [kustomize/deployment.yaml](kustomize/deployment.yaml)
or via your deployment configuration to a value up to `600`. Its `preStop`
hook sends `SIGTERM` to Huey, which stops the consumer; Kubernetes permits up
to ten minutes for the current task to finish. Work that does not finish is
handled by recovery on the next worker start.
