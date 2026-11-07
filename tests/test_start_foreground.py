"""FOREGROUND=1 ./start.sh: attached to rank 0 until the server ends, every rank stopped when any rank ends, on a
failed start and on SIGINT / SIGTERM; attaching to a server that already runs.

Offline, on test_scripts.py's StartRestart fixture (fake ssh, docker, curl, flock, ss, free) with a docker that also
knows a stopped container: `docker stop` marks it exited, `inspect` then reads Running false and the exit code in
$SIM/hosts/<host>/code (0 if none), `docker ps --format {{.State}}` says running or exited (nothing when there is no
such container), and `docker logs -f` follows until the container is no longer running (each follower's arguments are
lines in $SIM/follow). The real flock, where there is one: stop.sh's lock is part of what is tested.
"""
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import time
import unittest

_FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_scripts.py")))
NAME = _FIXTURE["NAME"]
HOSTS = ["head", "w1", "w2", "w3"]

FAKE_DOCKER = r"""#!BASH
h=${SIM_HOST:-head}; d="$SIM/hosts/$h"; mkdir -p "$d"; touch "$d/containers" "$d/exited"
echo "$h docker $*" >> "$SIM/docker.log"
has() { grep -qxF "$1" "$d/containers"; }
up() { has "$1" && ! grep -qxF "$1" "$d/exited"; }
last=${!#}
case "$1" in
  ps) [[ ! -e "$d/ps_fails" ]] || { echo "Cannot connect to the Docker daemon" >&2; exit 1; }
      if [[ "$*" == *'{{.State}}'* ]]; then         # docker ps -a --filter name=^/?N$ --format {{.State}}
        for a in "$@"; do [[ "$a" != name=* ]] || n=${a#'name=^/?'}; done; n=${n%\$}
        if up "$n"; then echo running; elif has "$n"; then echo exited; fi
      else cat "$d/containers"; fi ;;
  inspect) has "$last" || { echo "Error: No such object: $last" >&2; exit 1; }
           case "$*" in *State.Running*) if up "$last"; then echo true; else echo false; fi ;;
                        *ExitCode*) cat "$d/code" 2>/dev/null || echo 0 ;; esac ;;
  stop) if up "$last"; then echo "$last" >> "$d/exited"; fi ;;
  rm) grep -vxF "$last" "$d/containers" > "$d/c.new" || true; mv "$d/c.new" "$d/containers"
      grep -vxF "$last" "$d/exited" > "$d/e.new" || true; mv "$d/e.new" "$d/exited" ;;
  logs) if [[ " $* " == *" -f "* ]]; then
          echo "$h $*" >> "$SIM/follow"; echo "a log line"
          while up "$last"; do sleep 0.1; done
        else echo "a log line"; fi ;;
  image) [[ -e "$SIM/image" ]] || exit 1; echo image-ident ;;
  run) n=""; prev=""; for a in "$@"; do [[ "$prev" != --name || -n "$n" ]] || n=$a; prev=$a; done
       echo "$n" >> "$d/containers"; echo container-id ;;
  *) exit 0 ;;
esac
"""


@unittest.skipUnless(_FIXTURE["BASH"], "needs bash 4 or later")
class StartForeground(unittest.TestCase):
    def setUp(self):
        self.fixture = _FIXTURE["StartRestart"](methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.sim = self.fixture.sim
        docker = self.fixture.tmp / "bin" / "docker"
        docker.write_text(FAKE_DOCKER.replace("BASH", _FIXTURE["BASH"]))
        docker.chmod(0o755)
        if shutil.which("flock", path="/usr/bin:/bin:/usr/local/bin"):
            (self.fixture.tmp / "bin" / "flock").unlink()
        self.proc = None

    def start(self, extra_env=None, args=()):
        env = {**self.fixture.env, "PREPARE": "0", "FOREGROUND": "1", "FOREGROUND_POLL": "1", **(extra_env or {})}
        self.proc = subprocess.Popen([_FIXTURE["BASH"], "start.sh", *args], cwd=self.fixture.repo, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(self.reap)

    def reap(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.communicate()

    def finish(self, timeout=30):
        out, _ = self.proc.communicate(timeout=timeout)
        return self.proc.returncode, out

    def wait_attached(self, timeout=20):
        """Until start.sh follows rank 0's log in FOREGROUND=1's way (--since or --tail, not step 4's follower)."""
        f = self.sim / "follow"
        end = time.time() + timeout
        while time.time() < end:
            if f.exists() and any(l.startswith("head ") and ("--since" in l or "--tail" in l)
                                  for l in f.read_text().splitlines()):
                return
            if self.proc.poll() is not None:
                self.fail("start.sh ended before it attached:\n" + self.proc.communicate()[0])
            time.sleep(0.1)
        self.fail("start.sh did not attach to rank 0")

    def exit_rank(self, host, code):
        d = self.sim / "hosts" / host
        (d / "code").write_text(f"{code}\n")
        with open(d / "exited", "a") as f:
            f.write(NAME + "\n")

    def assertAllStopped(self, out):
        for h in HOSTS:
            self.assertEqual(self.fixture.left(h), [], f"{h}\n{out}")
        self.fixture.assertNoTraceback(out)

    def test_rank_0_ending_ends_start_sh_with_its_code(self):
        self.start()
        self.wait_attached()
        self.exit_rank("head", 3)
        code, out = self.finish()
        self.assertEqual(code, 3, out)
        self.assertIn("is now LIVE", out)
        self.assertIn("OK:", out)                    # the smoke test ran first
        self.assertIn("rank 0 ended (exit code 3)", out)
        self.assertAllStopped(out)
        self.assertTrue(list((self.fixture.home / ".cache" / "dsv41-tensorfold" / "logs").glob("*-rank0.log.gz")), out)

    def test_a_worker_rank_ending_stops_every_rank_and_fails(self):
        self.start()
        self.wait_attached()
        self.exit_rank("w2", 0)                      # rank 0 then shuts down cleanly: still a failure
        code, out = self.finish()
        self.assertEqual(code, 1, out)
        self.assertIn("rank 2 on w2 is not running (seen on two polls 1s apart; docker ps there said: exited rc=0", out)
        self.assertAllStopped(out)

    def test_a_worker_that_cannot_be_asked_does_not_take_the_server_down(self):
        # ssh answers its probe and drops every other command (test_scripts.py's flaky), then docker ps fails there
        self.start()
        self.wait_attached()
        self.fixture.flaky("w1")
        time.sleep(3.5)
        (self.sim / "flaky").unlink()
        self.fixture.ps_fails("w3")
        time.sleep(3.5)
        self.assertIsNone(self.proc.poll())
        for h in HOSTS:
            self.assertEqual(self.fixture.left(h), [NAME], h)
        self.proc.send_signal(signal.SIGTERM)
        code, out = self.finish()
        self.assertEqual(code, 143, out)
        self.assertIn("cannot check rank 1 on w1", out)
        self.assertIn("cannot check rank 3 on w3", out)
        self.assertNotIn("stopping rank 0", out)
        self.assertEqual(out.count("cannot check rank 1 on w1"), 1, out)   # said once while it lasts

    def test_sigterm_stops_every_rank(self):
        self.start()
        self.wait_attached()
        t = time.time()
        self.proc.send_signal(signal.SIGTERM)
        code, out = self.finish()
        self.assertEqual(code, 143, out)
        self.assertLess(time.time() - t, 15, out)   # not after the log follower or a poll
        self.assertIn("SIGTERM: stopping every rank", out)
        self.assertAllStopped(out)

    def test_a_stop_sh_from_outside_ends_it_without_a_second_stop_failing(self):
        # a supervisor's ExecStop runs ./stop.sh while start.sh is attached: start.sh's own clean-up waits for it
        self.start()
        self.wait_attached()
        stop = subprocess.run([_FIXTURE["BASH"], "stop.sh"], cwd=self.fixture.repo, env=self.fixture.env,
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(stop.returncode, 0, stop.stdout + stop.stderr)
        code, out = self.finish()
        # it waits for that stop.sh: the container removed is a stop that was asked for, exit 0 (a crash leaves it)
        if shutil.which("flock", path=self.fixture.env["PATH"]):
            self.assertEqual(code, 0, out)
            self.assertIn("rank 0's container was removed", out)
        self.assertNotIn("could not stop", out)
        self.assertAllStopped(out)

    def test_sigint_stops_every_rank(self):
        self.start()
        self.wait_attached()
        self.proc.send_signal(signal.SIGINT)
        code, out = self.finish()
        self.assertEqual(code, 130, out)
        self.assertAllStopped(out)

    def test_already_running_attaches(self):
        for h in HOSTS:
            self.fixture.containers(h, NAME)
        self.start()
        self.wait_attached()
        self.assertEqual(self.fixture.launched(), [])
        self.proc.send_signal(signal.SIGTERM)
        code, out = self.finish()
        self.assertEqual(code, 143, out)
        self.assertIn("FOREGROUND=1 attaches to it", out)
        self.assertNotIn("nothing to do", out)
        self.assertAllStopped(out)

    def test_failed_smoke_test_stops_every_rank(self):
        (self.sim / "smoke_status").write_text("500")
        self.start()
        code, out = self.finish()
        self.assertEqual(code, 1, out)
        self.assertIn("smoke test request failed", out)
        self.assertIn("every rank was stopped (FOREGROUND=1)", out)
        self.assertNotIn("is now LIVE", out)
        self.assertAllStopped(out)

    def test_state_and_log_dirs_are_the_ones_set(self):
        state, logs = self.fixture.tmp / "state-tp4", self.fixture.tmp / "logs-tp4"
        self.start({"STATE_DIR": str(state), "LOG_DIR": str(logs)})
        self.wait_attached()
        self.exit_rank("head", 0)
        code, out = self.finish()
        self.assertEqual(code, 0, out)
        self.assertTrue((state / "start.lock").exists(), out)
        self.assertTrue((state / "stop.lock").exists(), out)
        self.assertTrue(list(logs.glob("*-rank0.log.gz")), out)
        self.assertFalse((self.fixture.home / ".local" / "state" / "dsv41-tensorfold").exists(), out)

    def test_without_foreground_start_sh_returns_and_the_ranks_run(self):
        self.start({"FOREGROUND": "0"})
        code = self.proc.wait(timeout=30)            # (step 4's log follower, still on stdout, ends with rank 0)
        for h in HOSTS:
            self.assertEqual(self.fixture.left(h), [NAME], h)
        self.assertFalse(any("--since" in l for l in (self.sim / "follow").read_text().splitlines()))
        self.exit_rank("head", 0)
        _, out = self.finish()
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)

    def test_bad_stop_timeout_refused(self):
        self.start({"STOP_TIMEOUT": "30s"})
        code, out = self.finish()
        self.assertEqual(code, 1, out)
        self.assertIn("STOP_TIMEOUT is a count of seconds, not 30s", out)
        self.assertEqual(self.fixture.launched(), [])

    def test_bad_foreground_value_refused(self):
        self.start({"FOREGROUND": "yes"})
        code, out = self.finish()
        self.assertEqual(code, 1, out)
        self.assertIn("FOREGROUND is 0 or 1, not yes", out)
        self.assertEqual(self.fixture.launched(), [])

    def test_dry_run_says_it_would_stay_attached(self):
        self.start({"DRY_RUN": "1"})
        code, out = self.finish()
        self.assertEqual(code, 0, out)
        self.assertIn("FOREGROUND=1 would stay attached", out)
        self.assertEqual(self.fixture.launched(), [])


if __name__ == "__main__":
    unittest.main()
