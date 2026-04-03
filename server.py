"""
server.py — Demo 后端 (FastAPI + WebSocket)

提供：
  1. REST API：获取题目列表、已有轨迹
  2. WebSocket：实时推送证明尝试进度
  3. 静态文件：前端 UI

启动: python server.py
"""

import json
import asyncio
import logging
import yaml
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from core.models import BenchmarkProblem, ProofAttempt, ProofTrace, AttemptStatus
from core.lean_checker import LeanChecker
from core.llm_policy import create_provider
from core.retriever import PremiseRetriever
from core.orchestrator import Orchestrator, OrchestratorConfig
from benchmarks.loader import load_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 加载配置 ───────────────────────────────────────────────────
config_path = Path("config/local.yaml")
if not config_path.exists():
    config_path = Path("config/default.yaml")
with open(config_path) as f:
    CONFIG = yaml.safe_load(f) or {}

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(title="AI4Math Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 加载题目 ───────────────────────────────────────────────────
bench_config = CONFIG.get("benchmark", {})
PROBLEMS = load_benchmark(
    benchmark=bench_config.get("name", "builtin"),
    split=bench_config.get("split", "test"),
    path=bench_config.get("path", ""),
)
PROBLEM_MAP = {p.problem_id: p for p in PROBLEMS}

# ── 已保存的轨迹 ──────────────────────────────────────────────
RESULTS_DIR = Path(CONFIG.get("output", {}).get("dir", "results")) / "traces"


# ── REST API ───────────────────────────────────────────────────

class ProblemResponse(BaseModel):
    problem_id: str
    name: str
    theorem_statement: str
    difficulty: str
    source: str
    natural_language: str


class ProveRequest(BaseModel):
    problem_id: str
    max_attempts: Optional[int] = None


@app.get("/api/problems")
async def list_problems():
    """获取所有题目"""
    return [
        ProblemResponse(
            problem_id=p.problem_id,
            name=p.name,
            theorem_statement=p.theorem_statement,
            difficulty=p.difficulty,
            source=p.source,
            natural_language=p.natural_language,
        )
        for p in PROBLEMS
    ]


@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str):
    """获取单个题目"""
    if problem_id not in PROBLEM_MAP:
        raise HTTPException(404, "Problem not found")
    p = PROBLEM_MAP[problem_id]
    return ProblemResponse(
        problem_id=p.problem_id,
        name=p.name,
        theorem_statement=p.theorem_statement,
        difficulty=p.difficulty,
        source=p.source,
        natural_language=p.natural_language,
    )


@app.get("/api/traces/{problem_id}")
async def get_trace(problem_id: str):
    """获取已保存的轨迹"""
    trace_path = RESULTS_DIR / f"{problem_id}.json"
    if not trace_path.exists():
        raise HTTPException(404, "Trace not found")
    with open(trace_path) as f:
        return json.load(f)


@app.get("/api/traces")
async def list_traces():
    """列出所有已保存的轨迹"""
    if not RESULTS_DIR.exists():
        return []
    traces = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        traces.append({
            "problem_id": data.get("problem_id"),
            "problem_name": data.get("problem_name"),
            "solved": data.get("solved"),
            "total_attempts": data.get("total_attempts"),
            "total_tokens": data.get("total_tokens"),
        })
    return traces


# ── WebSocket: 实时证明 ────────────────────────────────────────

@app.websocket("/ws/prove")
async def websocket_prove(websocket: WebSocket):
    """
    WebSocket 接口：实时证明。

    客户端发送: {"problem_id": "...", "max_attempts": 5}
    服务端推送:
      - {"type": "start", "problem": {...}}
      - {"type": "attempt", "attempt": {...}}       (每次尝试)
      - {"type": "done", "trace": {...}}            (完成)
      - {"type": "error", "message": "..."}         (异常)
    """
    await websocket.accept()

    try:
        # 接收请求
        data = await websocket.receive_json()
        problem_id = data.get("problem_id", "")
        max_attempts = data.get("max_attempts", 10)

        if problem_id not in PROBLEM_MAP:
            await websocket.send_json({"type": "error", "message": "Problem not found"})
            return

        problem = PROBLEM_MAP[problem_id]

        # 通知开始
        await websocket.send_json({
            "type": "start",
            "problem": {
                "problem_id": problem.problem_id,
                "name": problem.name,
                "theorem_statement": problem.theorem_statement,
            },
        })

        # 构建组件
        llm = create_provider(CONFIG.get("llm", {}))
        lean = LeanChecker(
            mode=CONFIG.get("lean", {}).get("mode", "docker"),
            docker_image=CONFIG.get("lean", {}).get("docker_image", "ai4math-lean"),
            docker_container=CONFIG.get("lean", {}).get("docker_container", ""),
            timeout_seconds=CONFIG.get("lean", {}).get("timeout_seconds", 120),
        )
        retriever = PremiseRetriever(CONFIG.get("retriever", {}))

        # 回调：每次尝试完成后推送到 WebSocket
        async def on_attempt_sync(attempt: ProofAttempt):
            """同步回调包装"""
            pass  # 实际推送在下面的异步版本中

        attempt_queue: asyncio.Queue = asyncio.Queue()

        def on_attempt(attempt: ProofAttempt):
            attempt_queue.put_nowait(attempt)

        orc = Orchestrator(
            lean_checker=lean,
            llm_provider=llm,
            retriever=retriever,
            config=OrchestratorConfig(max_attempts=max_attempts),
            on_attempt=on_attempt,
        )

        # 在线程中运行 (Lean 编译是阻塞的)
        loop = asyncio.get_event_loop()

        async def run_and_stream():
            # 启动证明任务
            prove_task = loop.run_in_executor(None, orc.prove, problem)

            # 持续读取 attempt 队列并推送
            while True:
                try:
                    attempt = await asyncio.wait_for(attempt_queue.get(), timeout=0.5)
                    await websocket.send_json({
                        "type": "attempt",
                        "attempt": attempt.to_dict(),
                    })
                except asyncio.TimeoutError:
                    if prove_task.done():
                        # 消耗队列中剩余的
                        while not attempt_queue.empty():
                            attempt = attempt_queue.get_nowait()
                            await websocket.send_json({
                                "type": "attempt",
                                "attempt": attempt.to_dict(),
                            })
                        break

            trace = await prove_task
            return trace

        trace = await run_and_stream()

        # 保存并通知完成
        trace_path = RESULTS_DIR / f"{problem.problem_id}.json"
        trace.save(trace_path)

        await websocket.send_json({
            "type": "done",
            "trace": trace.to_dict(),
        })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── 启动 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    server_config = CONFIG.get("server", {})
    uvicorn.run(
        "server:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=True,
    )
