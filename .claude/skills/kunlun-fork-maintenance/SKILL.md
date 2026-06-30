---
name: kunlun-fork-maintenance
description: Maintain the Kunlun downstream vllm-omni fork as a clean patch stack over upstream. Use when working on klx_main, temporary sync validation branches, release/kunlun branches, upstream syncs, cherry-picking upstream commits, backports, or conflict resolution for Kunlun hardware adaptation.
---

# Kunlun Fork Maintenance

Use this skill to keep Kunlun hardware adaptation separate from upstream vllm-omni history, while making upstream syncs, cherry-picks, and user releases predictable.

## First Inspect

Before recommending or changing branches, inspect the actual repository state:

```bash
git status --short --branch
git remote -v
git branch -vv
git log --oneline --decorate --graph --max-count=40
```

If comparing Kunlun to upstream:

```bash
git merge-base klx_main main
git log --oneline main..klx_main
git log --oneline --left-right --cherry-pick klx_main...main
```

Do not assume branch names, SHAs, or remotes are current. Treat untracked or dirty files as user work unless the user explicitly asks to clean them.

## Branch Roles

Use these roles consistently:

| Branch | Role | Rebase? | Force push? | Users depend on it? |
|--------|------|---------|-------------|---------------------|
| `main` | Local upstream mirror branch | Yes, from upstream | Only if mirror policy allows | No |
| `klx_main` | Official Kunlun patch stack over a known upstream base | Yes, after validation | Maintainers only, with `--force-with-lease` if pushed | Prefer no |
| `sync/kunlun-YYYYMMDD` | Temporary validation branch after replaying `klx_main` onto newer upstream | Yes | Do not push unless explicitly requested | No |
| `backport/upstream-<sha>-to-klx` | Temporary branch for selected upstream cherry-picks | No, usually cherry-pick only | No unless reviewed | No |
| `release/kunlun/<base>` | Stable user branch created after validation | No | No | Yes |

Keep `klx_main` as the source of truth for Kunlun-specific changes. It should contain Kunlun adaptation commits, not random upstream backports. Do not maintain `platform/kunlun-next` as a long-lived remote branch by default; if that name appears, treat it as a legacy or temporary validation branch.

## Initial Setup From Current klx_main

When `klx_main` is a small set of Kunlun commits over a known base, preserve an archive marker and use `klx_main` itself as the official patch stack:

```bash
git config rerere.enabled true
git config rerere.autoupdate true

git branch archive/klx_main-<base-sha> klx_main
```

Use the real merge-base SHA in the archive branch name. Do not create `platform/kunlun-patches` unless the team explicitly decides to split the patch-stack role away from `klx_main`.

## Full Upstream Sync

Create a temporary sync branch to test replaying `klx_main` onto a new upstream base:

```bash
# Ensure main is current with the real upstream branch before this step.
git switch -c sync/kunlun-YYYYMMDD klx_main
git rebase --onto main <old-base-sha>
```

Use a dated name so multiple attempts can coexist locally:

```bash
git switch -c sync/kunlun-YYYYMMDD-try2 klx_main
git rebase --onto main <old-base-sha>
```

Resolve conflicts by favoring upstream behavior for generic code and preserving only the Kunlun-specific hooks, device logic, platform registration, worker overrides, requirements, and tests that are still required.

After the rebase:

```bash
git range-diff <old-base-sha>..klx_main main..HEAD
git diff --stat main..HEAD
git log --oneline --left-right --cherry-pick main...HEAD
```

Then run the relevant unit, smoke, hardware, and packaging checks. If validation passes, publish a stable release branch and decide whether to update `klx_main` to the validated result. Do not push the temporary sync branch unless the user explicitly asks for remote review.

## Rebase klx_main

Do not rebase `klx_main` just because upstream has new commits. Rebase it only after a temporary sync branch has proven that the Kunlun patch stack works on the new upstream base.

Use this order:

```bash
# First: trial sync on a candidate branch.
git switch -c sync/kunlun-YYYYMMDD klx_main
git rebase --onto main <old-base-sha>

# Validate the candidate.
git range-diff <old-base-sha>..klx_main main..sync/kunlun-YYYYMMDD

# Only after validation: update the official patch stack.
git switch klx_main
git rebase --onto main <old-base-sha>
```

Rebase `klx_main` when:

- The team has decided to move Kunlun development to a new upstream base.
- A dated `sync/kunlun-YYYYMMDD` branch has resolved conflicts and passed Kunlun validation.
- Future Kunlun development should build on the new upstream base.
- A new `release/kunlun/...` branch or tag is being prepared from the new base.

Do not rebase `klx_main` when:

- The sync is only exploratory.
- Conflicts are unresolved.
- Kunlun hardware validation has not passed.
- The task is only cherry-picking one upstream bugfix.
- Users are depending on `klx_main` as a stable branch and have not been migrated to `release/kunlun/...`.

Treat `sync/kunlun-YYYYMMDD` as the trial branch and `klx_main` as the official patch-stack ledger.

## Publishing A Stable User Branch

Only publish user-facing branches after the candidate branch is validated:

```bash
git switch -c release/kunlun/upstream-<upstream-sha> sync/kunlun-YYYYMMDD
git tag klx-vllm-omni-<upstream-sha>-1
```

Do not rebase or force-push release branches. If a fix is needed after release, add another normal commit or create a new release branch/tag.

## Cherry-Picking Upstream Commits

Treat cherry-picking upstream commits as a backport/hotfix, not as part of the clean Kunlun patch stack.

Preferred flow for a user-facing hotfix:

```bash
git fetch upstream
git switch klx_main
git switch -c backport/upstream-<sha>-to-klx
git cherry-pick -x <upstream-sha>
```

Always use `-x` so later syncs show where the commit came from.

If the commit has dependencies, inspect them before cherry-picking:

```bash
git show --stat <upstream-sha>
git log --oneline --ancestry-path <current-base-sha>..<upstream-sha>
```

Cherry-pick dependency commits in upstream order:

```bash
git cherry-pick -x <dep-sha-1>
git cherry-pick -x <dep-sha-2>
git cherry-pick -x <target-sha>
```

Avoid adding upstream backports to `klx_main`. Exception: if a Kunlun patch cannot function without the backport and the full base upgrade is intentionally delayed, include it temporarily with a message like:

```text
[Backport][Upstream] <subject>

Cherry-picked from upstream <sha>.
Required by Kunlun platform support until the base is updated past <sha>.
```

During a later full upstream sync, expect Git to skip equivalent cherry-picks or ask for conflict resolution. If patch IDs differ because the backport was adjusted, resolve by taking upstream's final version plus the minimum Kunlun-specific adaptation.

## Conflict Rules

When conflicts occur:

1. Identify whether the file is upstream-generic or Kunlun-specific.
2. Before editing conflicted files, summarize the conflict files, the proposed resolution direction, and any risk to Kunlun behavior; ask the user to confirm the resolution plan.
3. For upstream-generic files, prefer the new upstream shape and re-apply only required Kunlun hooks.
4. For `vllm_omni/platforms/kunlun/`, prefer the Kunlun side unless upstream introduced an API contract change.
5. For shared platform files such as `vllm_omni/platforms/__init__.py`, `setup.py`, requirements, worker dispatch, and attention backend selection, preserve upstream changes and add Kunlun registration in the smallest compatible form.
6. Do not silently keep obsolete Kunlun code if upstream added a neutral extension point that can replace it.

After conflict resolution:

```bash
git status --short
git diff --check
git range-diff <old-base-sha>..klx_main main..HEAD
```

## User Prompt Examples

Use these examples to recognize and execute common user requests with this skill.

### Inspect klx_main

```text
使用 kunlun-fork-maintenance 规则，帮我检查当前 klx_main 相对 main 是否还是干净的 Kunlun patch stack。
```

```text
使用该规则，帮我列出 klx_main 相比 main 的 Kunlun 自有提交，并判断有没有混入上游 backport。
```

Expected action: inspect branch state, merge base, Kunlun-only commits, and dirty working tree; report findings without changing branches.

### Sync Upstream

```text
使用该规则帮我同步上游分支。当前 main 是上游镜像，请从 klx_main 临时创建 sync/kunlun-YYYYMMDD 验证分支进行 rebase，遇到冲突先让我确认。
```

```text
按 Kunlun fork maintenance 规则，把 klx_main 试着同步到最新 main，只做本地验证分支，不要推远程。
```

Expected action: create a temporary `sync/kunlun-YYYYMMDD` branch from `klx_main`, rebase onto `main`, stop for user confirmation before resolving conflicts, then run comparison and validation commands.

### Resolve Sync Conflicts

```text
使用该规则帮我处理 sync/kunlun-YYYYMMDD 上的 rebase 冲突。先列出冲突文件、建议解决方向和风险，等我确认后再修改。
```

Expected action: summarize conflict files and proposed resolution first; do not edit conflicted files until the user confirms.

### Cherry-Pick Upstream Commit

```text
使用该规则帮我 cherry-pick 上游 commit <sha> 到 klx_main。请创建 backport/upstream-<sha>-to-klx 分支，使用 -x，并先检查是否有依赖提交。
```

```text
按 Kunlun fork maintenance 规则，评估 upstream 的 <sha> 能不能单独 backport 到 klx_main。如果依赖复杂，先给我结论，不要直接改。
```

Expected action: inspect the upstream commit and dependency path, create a temporary backport branch only when appropriate, use `git cherry-pick -x`, and avoid mixing backports into `klx_main` without user approval.

### Publish Stable User Version

```text
使用该规则帮我基于 sync/kunlun-YYYYMMDD 发布一个稳定用户版本，创建 release/kunlun/upstream-<main-sha> 分支并打 tag。
```

```text
按 Kunlun fork maintenance 规则，帮我把已经验证通过的同步结果发布给用户。release 分支不要 rebase，也不要 force-push。
```

Expected action: create a stable `release/kunlun/...` branch and tag from the validated sync branch; do not rewrite release history.

### Update klx_main After Validation

```text
使用该规则，在 sync/kunlun-YYYYMMDD 验证通过后，把 klx_main 更新到新的 main base。更新前请确认用户分支发布和 force-with-lease 风险。
```

Expected action: confirm validation and user-impact assumptions, then update `klx_main` by replaying the same rebase onto `main`; call out any history rewrite before pushing.

### Report A Sync Or Backport

```text
使用该规则，帮我生成这次 Kunlun 上游同步报告，包括 base 变化、Kunlun patch 数量、冲突、backport 和验证结果。
```

Expected action: use the report template below and include concrete branch names, SHAs, conflict status, validation commands, and release/tag outcome.

## What To Report

For each upstream sync or backport, report:

```text
Base: <old-upstream-sha> -> <new-upstream-sha>
Kunlun patch count: <N>
Branch used: <sync/... | backport/... | klx_main>
Conflicts: <files or none>
Backports included: <sha list or none>
Validation: <commands and pass/fail>
User branch/tag: <release branch/tag or not published>
```

Keep the final recommendation clear: users consume stable `release/kunlun/...` branches or tags; maintainers work on `klx_main` and temporary sync candidates.
