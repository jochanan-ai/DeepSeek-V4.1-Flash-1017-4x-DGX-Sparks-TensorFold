#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 (deepseek-ai's own FP8 / MXFP4 release) on four DGX Sparks on a switch with TensorFold's
# native engine (four Sparks cabled as a ring without a switch: ./start-switchless.sh, the same script), end to end: runs scripts/prepare.sh when the image, the checkpoint, the weight files, the kit or the workers'
# NFS volumes are not ready yet (the first run, or after the patches change), starts ranks 3, 2 and 1 on the workers and
# rank 0 here, which serves the API on port 8888, waits until the OpenAI API answers, then runs a smoke test. Every
# Spark reads the weights from this one over NFS. Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                  # scripts/config.sh's defaults (16 requests at once, a 1,048,576-token window, drafts,
#                               # images); if the server already runs, says so and leaves it alone
#   ./start.sh restart          # stop every rank (./stop.sh), then start again (e.g. after changing settings)
#   CONTEXT=262144 ./start.sh restart
#   FOREGROUND=1 ./start.sh     # for a supervisor (systemd Type=simple, llama-swap): after the smoke test, stay attached
#                               # to rank 0's log until the server ends, and exit with its code; any rank ending, a
#                               # failed start, SIGINT or SIGTERM stops every rank (./stop.sh); attaches if already running
#   DRY_RUN=1 ./start.sh        # print the links found and every rank's docker command, change nothing
# Extra arguments go to rank 0's `tensorfold-native serve` after the defaults, so they win.
# Setup: WORKER, WORKER2, WORKER3 (ssh targets, key-based) in scripts/local.sh: on a ring in its order.
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their notes in scripts/config.sh):
#   serving  CONTEXT, POOL, PARALLEL, ROUND_ROWS, MAX_TOKENS, THINKING, REASONING_EFFORT, TEMPERATURE, TOP_P, TOP_K, DRAFTS, VISION, SERVED_NAME,
#            HOST, PORT, RDMA, GRAPHS, SIDE, PREFETCH, ENGRAM_AIO, ENGRAM_PACKED, ENGRAM_OWNED, ENGRAM_CACHE_MIB,
#            ENGRAM_INSTREAM, ROUND_MS, DRAFT_MS, SERVED_K
#   nodes    WORKER, WORKER2, WORKER3, FABRIC_PEER*, MASTER_ADDR, MASTER_PORT, SOCKET_IFNAME, ORDER, NCCL_CHANNELS,
#            NCCL_DEBUG
#   files    DERISKED, NVFP4, MODEL_ID, MODEL_REVISION, HF_CACHE, NFS_PATH, NFS_SERVER*, NFS_VOLUME, STATE_DIR, LOG_DIR, LOG_KEEP
#   engine   TF_REPO, TF_REF, KIT_REPO, KIT_REF, ZIG_VERSION, BASE_IMAGE, IMAGE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), WAIT_TIMEOUT (seconds, default 2400), DRY_RUN=1, FOREGROUND=1, FOREGROUND_POLL
#            (seconds between FOREGROUND=1's checks of ranks 1-3, default 30)
#   ranks    every TENSORFOLD_* and TF_DS_* variable you set goes to every rank
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# the fabric this script serves: mesh (every pair of Sparks on one RoCE subnet: a switch); ./start-switchless.sh sets
# ring. scripts/nodes.sh stops when the links are the other kind and names the script that fits.
export FABRIC_EXPECT="${FABRIC_EXPECT:-mesh}"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) usage; exit 0 ;;
esac
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
FG="${FOREGROUND:-0}"; FOREGROUND_POLL="${FOREGROUND_POLL:-30}"
[[ "$FG" =~ ^[01]$ ]] || die "FOREGROUND is 0 or 1, not $FG"
[[ "$FOREGROUND_POLL" =~ ^[1-9][0-9]*$ ]] || die "FOREGROUND_POLL is a positive count of seconds, not $FOREGROUND_POLL"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || die "STOP_TIMEOUT is a count of seconds, not $STOP_TIMEOUT"
for v in CONTEXT POOL PARALLEL ROUND_ROWS MAX_TOKENS; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v is a positive count, not ${!v}"; done
(( CONTEXT <= 1048576 )) || die "CONTEXT is at most 1048576 (the model's window), not $CONTEXT"
(( POOL >= CONTEXT )) || die "POOL ($POOL) holds every stream's positions together: at least CONTEXT ($CONTEXT)"
(( PARALLEL <= 16 )) || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$SERVED_K" =~ ^(solo|all|off)$ ]] || die "SERVED_K is solo, all or off, not $SERVED_K"
for v in THINKING DRAFTS VISION RDMA GRAPHS SIDE PREFETCH ENGRAM_AIO ENGRAM_PACKED ENGRAM_OWNED ENGRAM_INSTREAM RANK_LOCAL; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$ENGRAM_CACHE_MIB" =~ ^[0-9]+$ ]] || die "ENGRAM_CACHE_MIB is a count of MiB, not $ENGRAM_CACHE_MIB"
[[ "$ENGRAM_INSTREAM" == 0 || ( "$ENGRAM_PACKED" == 1 && "$ENGRAM_AIO" == 1 ) ]] || die "ENGRAM_INSTREAM=1 needs ENGRAM_PACKED=1 and ENGRAM_AIO=1"
check_workers

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · 4 x DGX Spark · %s at once · %s-token window · %s · port %s%s\n\n' "$D" "$MODEL_ID" "$PARALLEL" "$CONTEXT" "$MODEL_FORMAT" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"   # an IPv6 address in a URL needs brackets
URL="http://$API_HOST:$PORT"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
all_running() { local i; running_here || return 1; for i in 1 2 3; do running_worker "$i" || return 1; done; }

# FOREGROUND=1: once the smoke test has passed, stay attached to rank 0's log and exit when the server ends, so that a
# supervisor (systemd's Type=simple, llama-swap) sees the server end: with rank 0's exit code; 1 when one of the
# workers' ranks ended first, the watchdog failed or rank 0's code cannot be read; 0 when rank 0's container is gone (a
# ./stop.sh removed it: a stop someone asked for; a crash leaves the container). Any rank ending takes the others down (a
# lone rank waits for its peers forever): a watchdog asks each worker every FOREGROUND_POLL seconds what its container
# is (one ssh, docker ps, bounded by timeout). Only an answer counts: "exited" or "gone" seen on two polls in a row stops
# rank 0; no answer (ssh or docker out of reach) concludes nothing. Every rank left is then stopped with ./stop.sh,
# which saves each rank's log. From the launch on, SIGINT and SIGTERM run ./stop.sh too, and a start that fails (a rank
# exits, WAIT_TIMEOUT, the smoke test) stops every rank: a supervisor's stop or retry never leaves a rank behind. With
# the server already running on all four Sparks it attaches to it (no new smoke test) instead of saying "nothing to
# do". start.lock stays held until it exits.
# (From MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks's and GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold's start.sh,
# for three workers.)
FG_WATCH=""; FG_LOGS=""; SMOKE_PID=""
# a rank's container as docker ps sees it, then "rc=<docker ps's status>": run here or on a worker (over ssh)
PS_STATE="timeout 20 docker ps -a --filter 'name=^/?$CONTAINER_NAME\$' --format '{{.State}}' 2>/dev/null; echo rc=\$?"
# rank_state <that output>: up, down (exited, or no such container) or unknown (no answer from docker or ssh)
rank_state() {
  case "$1" in
    rc=0) echo down ;;
    *$'\n'rc=0) case "${1%%$'\n'*}" in running|restarting|paused) echo up ;; *) echo down ;; esac ;;
    *) echo unknown ;;
  esac
}
here_state() { local s; s=$(eval "$PS_STATE"); rank_state "$s"; }
stop_watch() {   # the watchdog ends at once, or within 5 s of an ssh it is waiting on; then it is killed
  local n
  [[ -n "$FG_WATCH" ]] || return 0
  kill "$FG_WATCH" 2>/dev/null || true
  for (( n = 0; n < 50; n++ )); do kill -0 "$FG_WATCH" 2>/dev/null || break; sleep 0.1; done
  kill -9 "$FG_WATCH" 2>/dev/null || true
  wait "$FG_WATCH" 2>/dev/null || true
}
fg_stop() {   # fg_stop <signal> <exit code>: the INT and TERM trap
  trap '' INT TERM
  [[ -z "$FG_WATCH" ]] || kill "$FG_WATCH" 2>/dev/null || true
  [[ -z "$FG_LOGS" ]] || kill "$FG_LOGS" 2>/dev/null || true
  [[ -z "$SMOKE_PID" ]] || kill "$SMOKE_PID" 2>/dev/null || true
  warn "$1: stopping every rank (./stop.sh)"
  ./stop.sh || warn "./stop.sh could not stop every rank: see above"
  stop_watch
  exit "$2"
}
fg_traps() { trap 'fg_stop SIGINT 130' INT; trap 'fg_stop SIGTERM 143' TERM; }
foreground() {   # foreground <docker logs options>: where rank 0's log starts
  local w code st n
  trap - EXIT; LOGS_PID=""                           # step 4's log follower is long gone: nothing to kill on exit
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    trap 'kill $! 2>/dev/null; exit 0' TERM
    declare -A down=() quiet=()
    while :; do
      sleep "$FOREGROUND_POLL" & wait $!
      for i in 1 2 3; do
        raw=$(worker "$i" "$PS_STATE") || raw+=$'\n'"ssh rc=$?"
        case "$(rank_state "$raw")" in
          up) down[$i]=0; quiet[$i]=0 ;;
          down)
            down[$i]=$(( ${down[$i]:-0} + 1 )); quiet[$i]=0
            (( ${down[$i]} >= 2 )) || continue        # confirmed on the next poll, FOREGROUND_POLL seconds later
            warn "rank $i on $(worker_host "$i") is not running (seen on two polls ${FOREGROUND_POLL}s apart; docker ps there said: $(tr '\n' ' ' <<<"$raw")): stopping rank 0"
            docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
            exit 1 ;;
          *) down[$i]=0                               # a worker out of reach says nothing about its rank
             (( ${quiet[$i]:-0} )) || warn "cannot check rank $i on $(worker_host "$i") ($(tr '\n' ' ' <<<"$raw")): no conclusion, checking again"
             quiet[$i]=1 ;;
        esac
      done
    done ) &
  FG_WATCH=$!
  # (both in the background, checked every second: a trapped signal ends a wait at once, not a command in the
  # foreground, and a watchdog that ended is noticed while rank 0 still runs)
  while :; do
    docker logs -f "$@" "$CONTAINER_NAME" 8>&- 2>&1 &
    FG_LOGS=$!
    while kill -0 "$FG_LOGS" 2>/dev/null && kill -0 "$FG_WATCH" 2>/dev/null; do sleep 1 & wait $! || true; done
    if ! kill -0 "$FG_WATCH" 2>/dev/null; then kill "$FG_LOGS" 2>/dev/null || true; wait "$FG_LOGS" 2>/dev/null || true; break; fi
    wait "$FG_LOGS" || true
    st=$(here_state); n=0
    while [[ "$st" == unknown ]] && (( ++n < 3 )); do sleep 5; st=$(here_state); done
    [[ "$st" == up ]] || break                       # down, or docker here did not answer three times
    sleep 5; set -- --since "$(date +%s)"            # the log follower ended with rank 0 still up: follow it again
  done
  trap '' INT TERM                                   # the server has ended: finish taking it down
  w=0
  if kill -0 "$FG_WATCH" 2>/dev/null; then stop_watch; else wait "$FG_WATCH" || w=$?; fi
  (( w == 0 || w == 1 )) || warn "the watchdog of ranks 1-3 ended (code $w): taking the server down"
  # a ./stop.sh still at work (a supervisor's ExecStop) is waited for: its removal is what tells a stop from a crash
  if command -v flock >/dev/null; then exec 7>"$STATE_DIR/stop.lock"; flock -w 300 7 || true; exec 7>&-; fi
  if code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null); then
    [[ "$code" =~ ^[0-9]+$ ]] || code=1
    warn "rank 0 ended (exit code $code): stopping every rank left (./stop.sh)"
  elif [[ "$(here_state)" == down ]]; then
    code=0; log "rank 0's container was removed (./stop.sh): stopping every rank left (./stop.sh)"
  else
    code=1; warn "cannot read rank 0's exit code: stopping every rank left (./stop.sh)"
  fi
  (( w == 0 || code != 0 )) || code=1                # a worker's rank or the watchdog failed, even if rank 0 shut down cleanly
  ./stop.sh || { warn "./stop.sh could not stop every rank: see above"; (( code )) || code=1; }
  exit "$code"
}

if (( ! DRY )) && [[ "$MODE" == start ]] && all_running; then
  if (( FG )); then
    log "$CONTAINER_NAME is already running on all 4 Sparks (port $PORT): FOREGROUND=1 attaches to it. ./stop.sh (or SIGTERM to this script) stops it."
    fg_traps
    foreground --tail 50
  fi
  log "$CONTAINER_NAME is already running on all 4 Sparks (port $PORT): nothing to do. $(launcher "$FABRIC_EXPECT") restart restarts it, ./stop.sh stops it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image, checkpoint, weight files and kit on spark1; NFS volumes on the workers"
need_prepare=0
case "${PREPARE:-auto}" in
  1) need_prepare=1 ;;
  0) ;;
  *) # (computed on its own: a failure inside $(...) would otherwise read as an empty line equal to an empty marker)
     state=$(prepared_state) || die "cannot tell whether the setup is ready: see the error above"
     [[ "$state" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || need_prepare=1 ;;
esac
if (( need_prepare )); then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    log "Not ready yet: running scripts/prepare.sh (the first time this builds the image, downloads $MODEL_SIZE and writes four ~80 GB weight files)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE, $MODEL_ID @ $(snapshot_rev | cut -c1-8), its weight files and kit"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || (( DRY )) || die "image $IMAGE missing: scripts/prepare.sh did not build it"
MODEL_ARG=$(model_in_container)

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: links, previous server, port, memory"
detect_links
NODE_ORDER=$(node_order)
log "Fabric: $FABRIC; rendezvous $MASTER_ADDR:$MASTER_PORT; 2D nodes by NCCL rank: $NODE_ORDER"
for r in 0 1 2 3; do
  log "  rank $r ($( (( r == 0 )) && echo this node || worker_host "$r")): RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]:-per device})$( (( r == 0 )) || echo ", NFS from ${LINK_HEAD_ADDR[r]:-?}")"
done
if (( ! DRY )); then
  if running_here || running_worker 1 || running_worker 2 || running_worker 3; then ./stop.sh; fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  for i in 1 2 3; do worker "$i" "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1; true"; done
  if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"; fi
  here_gb=$(free -g | awk '/^Mem:/ {print $7}')
  (( here_gb >= 100 )) || warn "only ${here_gb} GiB memory available here: the weights take ~75 GiB a Spark, the caches the rest"
  for i in 1 2 3; do
    g=$(worker "$i" "free -g | awk '/^Mem:/ {print \$7}'" 2>/dev/null || echo 0)
    (( g >= 100 )) || warn "only ${g} GiB memory available on $(worker_host "$i"): stop other GPU work there"
  done
fi

# ---------------------------------------------------------------- 3. launch
# What every rank's engine reads (HF_CACHE at /hf: this Spark's own, the workers' over NFS)
ENGINE_ENV=(TF_DS_2D_GU=parity "TF_DS_2D_ORDER=$NODE_ORDER" "TF_DS_ROUND_ROWS=$ROUND_ROWS" "TF_DS_STREAMS=$PARALLEL" "TF_DS_SERVED_K=$SERVED_K"
            "TF_DS_WINDOW=$POOL" "TF_DS_GRAPHS=$GRAPHS" "TF_DS_HC_SIDE=$SIDE" "TF_DS_L2_PREFETCH=$PREFETCH" "TF_DS_ENGRAM_AIO=$ENGRAM_AIO"
            "TF_DS_ROUND_MS=$ROUND_MS" "TF_DS_DRAFT_MS=$DRAFT_MS" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE")
# Engram's packed shards (ENGRAM_PACKED=1, zig/patches/0003): every rank reads its own at /engram-packed
[[ "$ENGRAM_PACKED" != 1 ]] || ENGINE_ENV+=(TF_DS_ENGRAM_PACKED=/engram-packed "TF_DS_ENGRAM_OWNED=$ENGRAM_OWNED"
                                            "TF_DS_ENGRAM_CACHE_MIB=$ENGRAM_CACHE_MIB" "TF_DS_ENGRAM_INSTREAM=$ENGRAM_INSTREAM")
while IFS='=' read -r name _; do ENGINE_ENV+=("$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_DS|TF_SSE)_[A-Z0-9_]+=' | grep -vE '^TF_DS_(2D_GU|2D_ORDER|ROUND_ROWS|STREAMS|SERVED_K|WINDOW|GRAPHS|HC_SIDE|L2_PREFETCH|ENGRAM_AIO|ENGRAM_PACKED|ENGRAM_OWNED|ENGRAM_CACHE_MIB|ENGRAM_INSTREAM|ROUND_MS|DRAFT_MS)=' || true)
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }
RUN_ARGS="--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
DATA=/hf/$DATA_SUB
KIT=/hf/$KIT_SUB/kit
# the followers take the round's settings as flags only (tf-dsv41-lanes reads no TF_DS_GRAPHS & co.): without them they
# run every round ungraphed, with no side stream, prefetch or AIO, and rank 0 waits on them each round
FOLLOW="--pool $POOL --context $CONTEXT --engram $MODEL_ARG --token-map $DATA/dsv41_token_map.json --drafts $DRAFTS --round-rows $ROUND_ROWS --streams $PARALLEL"
FOLLOW+=" --graphs $GRAPHS --side $SIDE --prefetch $PREFETCH --aio $ENGRAM_AIO"
step 3 "Launch: container $CONTAINER_NAME, ranks 3 to 1 on the workers, then rank 0 here"
(( DRY || ! FG )) || fg_traps                      # FOREGROUND=1: from here on, SIGINT and SIGTERM stop every rank
IFS=, read -r -a ORD <<<"$NODE_ORDER"
LOCAL_RANKS=()   # the workers that read their weight file from their own disk
for i in 3 2 1; do
  # its weight file from its own disk when the copy there (prepare.sh) matches this Spark's, else over NFS
  rank_dir="$DATA/rank-cache"; rank_mount=""
  rf=""; [[ "$RANK_LOCAL" != 1 ]] || rf=$(rank_file "${ORD[i]}")
  if [[ -n "$rf" ]] && worker "$i" "[[ \"\$(stat -c '%s %Y' $WORKER_RANK_DIR/${rf##*/} 2>/dev/null)\" == '$(rank_stamp "$rf")' ]]" 2>/dev/null; then
    rank_dir=/rank-local; rank_mount="-v $WORKER_RANK_DIR:/rank-local:ro"; LOCAL_RANKS=("$i" "${LOCAL_RANKS[@]}")
  elif [[ "$RANK_LOCAL" == 1 ]]; then
    warn "rank $i reads its weight file over NFS: no matching copy in $WORKER_RANK_DIR on $(worker_host "$i") (PREPARE=1 $(launcher "$FABRIC_EXPECT") restart makes one)"
  fi
  # Engram's tables from the worker's own disk (prepare.sh), the rest over NFS; with packed shards the tables' headers
  # come over NFS and the rows from the shards on the worker's disk
  if [[ "$ENGRAM_PACKED" == 1 ]]; then
    cmd="tf-dsv41-lanes $MODEL_ARG $rank_dir $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT $FOLLOW"
    engram_mount="-v $ENGRAM_PACKED_DIR:/engram-packed:ro"
  else
    cmd="tf-dsv41-lanes $MODEL_ARG $rank_dir $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT ${FOLLOW/--engram $MODEL_ARG/--engram /engram}"
    engram_mount="-v $WORKER_ENGRAM_DIR:/engram:ro"
  fi
  [[ "$RDMA" == 0 ]] || cmd+=" --rdma ${NODE_HCAS[i]}"   # this node's RoCE devices
  remote="mkdir -p \$HOME/.cache/dsv41-tensorfold && docker run -d --name $CONTAINER_NAME $RUN_ARGS -v $NFS_VOLUME:/hf:ro $engram_mount $rank_mount $(envs "${ENGINE_ENV[@]}") $(rank_nccl_env "$i") $IMAGE $cmd"
  if (( DRY )); then printf '[dry-run] rank %s on %s:\n  %s\n' "$i" "$(worker_host "$i")" "$remote"; continue; fi
  log "Rank $i on $(worker_host "$i"): $cmd"
  worker "$i" "$remote" >/dev/null || die "could not start rank $i on $(worker_host "$i")"
done
# rank 0 listens for each worker at the address that worker was given (its link's address of this node; a ring has
# no one address every cable reaches)
LISTEN="${LINK_HEAD_ADDR[1]:-$MASTER_ADDR},${LINK_HEAD_ADDR[2]:-$MASTER_ADDR},${LINK_HEAD_ADDR[3]:-$MASTER_ADDR}"
ENV0=(TF_TP_WORLD=4 "TF_TP_MASTER=$MASTER_ADDR" "TF_TP_LISTEN=$LISTEN" "TF_TP_PORT=$MASTER_PORT" "TF_DS_RANK_CACHE=$DATA/rank-cache" "TF_DS_KIT=$KIT"
      "TENSORFOLD_CUDA_KERNELS=$KIT/aot" "TF_DS_ENGRAM=$MODEL_ARG" "TF_DS_TOKEN_MAP=$DATA/dsv41_token_map.json")
SERVE=(tensorfold-native serve "$(served_in_container)" --backend cuda --host "$HOST" --port "$PORT" --name "$SERVED_NAME"
       --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE+=(--thinking); else SERVE+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE+=(--reasoning-effort "$REASONING_EFFORT")
# drafts off on rank 0 as on the followers (a follower without the drafter refuses a drafted round)
[[ "$DRAFTS" == 1 ]] || SERVE+=(--no-drafts)
SERVE+=("$@")
here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HF_CACHE:/hf:ro")
[[ "$ENGRAM_PACKED" != 1 ]] || here+=(-v "$(packed_dir_here):/engram-packed:ro")
[[ "$RDMA" == 0 ]] || ENV0+=("TF_RDMA_DEVICES=${NODE_HCAS[0]}")
for x in "${ENV0[@]}" "${ENGINE_ENV[@]}"; do here+=(-e "$x"); done
read -r -a nenv <<<"$(rank_nccl_env 0)"
here+=("${nenv[@]}" "$IMAGE" "${SERVE[@]}")
if (( DRY )); then
  printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here[@]}")"
  (( ! FG )) || log "DRY_RUN: FOREGROUND=1 would stay attached to rank 0 after the smoke test, checking ranks 1-3 every ${FOREGROUND_POLL}s"
  exit 0
fi
log "Rank 0 here: ${SERVE[*]}"
"${here[@]}" >/dev/null

# ---------------------------------------------------------------- 4. load
# where the workers read their weight file (most of the ~75 GiB): over NFS, unless RANK_LOCAL=1 mounted their own copy
local_n=${#LOCAL_RANKS[@]}; local_list=$(IFS=,; echo "${LOCAL_RANKS[*]}")
if (( local_n == 3 )); then step 4 "Loading: ~75 GiB of weights on each Spark from its own disk, then the caches (a minute or two)"
elif (( local_n == 1 )); then step 4 "Loading: ~75 GiB of weights on each Spark over NFS (rank $local_list from its own disk), then the caches (a few minutes)"
elif (( local_n == 2 )); then step 4 "Loading: ~75 GiB of weights on each Spark over NFS (ranks $local_list from their own disks), then the caches (a few minutes)"
else step 4 "Loading: ~75 GiB of weights on each Spark over NFS, then the caches (a few minutes)"; fi
# Neither log follower may retain the parent launcher lock after it exits.
docker logs -f "$CONTAINER_NAME" 8>&- 2>&1 | sed -u "s/^/  ${D}│${R} /" 8>&- &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  for i in 1 2 3; do printf '%s── rank %s (%s): last log lines ──%s\n' "$D" "$i" "$(worker_host "$i")" "$R"; worker "$i" docker logs --tail 15 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'; done
  if (( FG )); then trap '' INT TERM; ./stop.sh || true; fi   # FOREGROUND=1: a failed start leaves no rank behind
  die "$1"
}
LEFT="the ranks are still running"; (( ! FG )) || LEFT="every rank was stopped (FOREGROUND=1)"
start=$SECONDS
until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
  running_here || fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  for i in 1 2 3; do running_worker "$i" || fail "rank $i on $(worker_host "$i") exited before the server was ready"; done
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); $LEFT"
  sleep 5
done
FG_SINCE=$(date +%s)                               # FOREGROUND=1 follows rank 0's log again from here
kill $LOGS_PID 2>/dev/null || true
log "Server answered after $((SECONDS - start))s"

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion through all 4 ranks"
# a pass needs HTTP 200, no error object and the assistant's reply text (curl -w adds the status as the last line)
# (in the background and waited for: under FOREGROUND=1 a trapped SIGTERM need not wait out curl's 300 s)
SMOKE_OUT="$STATE_DIR/smoke.out"
{ curl -s --max-time 300 -w '\n%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys
def no(why): sys.exit(why)
body, _, code = sys.stdin.read().rpartition("\n")
code == "200" or no("HTTP " + code)
r = json.loads(body)
r.get("error") is None or no("an error reply")
m = r["choices"][0]["message"]
m.get("role", "assistant") == "assistant" or no("not the assistant")
c = (m.get("content") or "").strip()
c or no("no reply text")
len(c) < 8 or len(set(c)) > 2 or no("one repeated character")
print(repr(c[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null; } >"$SMOKE_OUT" 8>&- &
SMOKE_PID=$!
if wait "$SMOKE_PID"; then
  SMOKE_PID=""; smoke=$(<"$SMOKE_OUT"); rm -f "$SMOKE_OUT"
  log "OK: $smoke"
else
  SMOKE_PID=""; rm -f "$SMOKE_OUT"
  fail "the smoke test request failed (an HTTP error, an error reply, no reply text, or one character repeated); $LEFT"
fi
# the address to print: HOST itself, or for a wildcard one of this Spark's own (0.0.0.0 binds IPv4 only: an IPv4 one)
if [[ "$HOST" == 0.0.0.0 ]]; then IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | awk 'NF && !/:/ {print; exit}')
elif [[ "$HOST" == "::" ]]; then IP=$(hostname -I 2>/dev/null | awk '{print $1}')
else IP="$HOST"; fi
[[ "$IP" != *:* ]] || IP="[$IP]"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED_NAME" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED_NAME)
    Window   $CONTEXT tokens · $PARALLEL at once · $MODEL_FORMAT · drafts $( [[ "$DRAFTS" == 1 ]] && echo on || echo off) · images $( [[ "$VISION" == 1 ]] && echo on || echo off)
    Logs     docker logs -f $CONTAINER_NAME$(for i in 1 2 3; do printf '   (rank %s: ssh %s docker logs -f %s)' "$i" "$(worker_host "$i")" "$CONTAINER_NAME"; done)
    Restart  $(launcher "$FABRIC_EXPECT") restart
    Stop     ./stop.sh

EOF
[[ "$FG" != 1 ]] || foreground --since "$FG_SINCE"
