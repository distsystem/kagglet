"""Relay IPython cell execution from the Kaggle kernel to a user-controlled subkernel.

Kaggle notebooks run in a fixed-env kernel you can't touch. `RelaySession` starts
a second Jupyter kernel (typically one you installed from your own pixi/conda env),
hijacks `IPython.run_cell` / `run_cell_async` on the outer kernel, and forwards
every cell to the inner kernel while streaming stdout / stderr / display data /
errors back to the user.

Usage:
    relay = RelaySession("my-kernel-name", cwd=pathlib.Path("/workspace"))
    relay.start()
    # ... notebook cells now execute in the inner kernel ...
    relay.cleanup()        # detach, remove extra paths, kill the kernel
    # or relay.shutdown()  # detach + kill, no extra cleanup
"""

import pathlib

DEFAULT_LOG_PATH = pathlib.Path("/kaggle/working/relay.log")


class RelayExecutionError(Exception):
    """Raised when a relay cell produces a traceback on the inner kernel."""

    def __init__(self, message: str, traceback_lines: list[str]) -> None:
        super().__init__(message)
        self.traceback_lines = traceback_lines

    def _render_traceback_(self) -> list[str]:
        return self.traceback_lines


_RELAY_ERROR_TYPES: dict[str, type[RelayExecutionError]] = {}


def _relay_error_type(name: str) -> type[RelayExecutionError]:
    error_type = _RELAY_ERROR_TYPES.get(name)
    if error_type is None:
        error_type = type(name, (RelayExecutionError,), {})
        _RELAY_ERROR_TYPES[name] = error_type
    return error_type


def _relay_error(content: dict) -> RelayExecutionError:
    error_type = _relay_error_type(content["ename"])
    return error_type(content["evalue"], list(content.get("traceback") or []))


class RelaySession:
    """Bridge the Kaggle notebook to an external Jupyter kernel."""

    def __init__(
        self,
        kernel_name: str,
        *,
        cwd: pathlib.Path | None = None,
        local_name: str = "relay",
        log_path: pathlib.Path = DEFAULT_LOG_PATH,
        cleanup_paths: list[pathlib.Path] | None = None,
    ) -> None:
        """
        Args:
            kernel_name: registered Jupyter kernel spec name (from `jupyter kernelspec list`).
            cwd: working dir for the inner kernel (also exported as `PIXI_PROJECT_ROOT`).
            local_name: variable name users call `.shutdown()` / `.cleanup()` on; cells
                that contain `{local_name}.shutdown()` or `{local_name}.cleanup(` run
                on the outer kernel, not relayed.
            log_path: file for relay-internal events (kernel lifecycle, cell snippets,
                memory usage, cleanup traces).
            cleanup_paths: extra paths to `shutil.rmtree` during `cleanup()` — typical
                use: remove an extracted env directory before Kaggle auto-packages output.
        """
        self.kernel_name = kernel_name
        self.cwd = cwd
        self.local_name = local_name
        self.log_path = log_path
        self.cleanup_paths = list(cleanup_paths or [])
        self.kernel_manager = None
        self.kernel_client = None
        self._ip = None
        self._original_run_cell = None
        self._original_run_cell_async = None
        self._log_file = None
        self._nvml_handle = None

    def _log(self, msg: str) -> None:
        import datetime

        if self._log_file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "a")  # noqa: SIM115
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_file.write(f"[{ts}] {msg}\n")
        self._log_file.flush()

    def _cpu_memory_usage(self) -> str | None:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return f"cpu={int(line.split()[1]) / 1e6:.1f}G"
        except (IndexError, OSError, ValueError):
            return None
        return None

    def _gpu_memory_usage(self) -> str | None:
        try:
            import pynvml
        except ImportError:
            return None

        nvml_error = getattr(pynvml, "NVMLError", OSError)
        try:
            if self._nvml_handle is None:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        except (AttributeError, OSError, ValueError, nvml_error):
            return None
        return f"gpu={info.used / 1e9:.1f}/{info.total / 1e9:.1f}G"

    def _log_memory(self) -> None:
        parts = [part for part in (self._cpu_memory_usage(), self._gpu_memory_usage()) if part]
        if not parts:
            return
        msg = " | ".join(parts)
        self._log(msg)
        print(f"[mem] {msg}", flush=True)

    def _rmtree_configured(self) -> None:
        import shutil

        for path in self.cleanup_paths:
            print(f"cleanup relay path: {path}", flush=True)

            def _on_error(func, path, exc_info):
                self._log(f"rmtree failed: {func.__name__}({path}): {exc_info[1]}")

            shutil.rmtree(path, onexc=_on_error)

    def cleanup(self) -> None:
        """Detach, rmtree `cleanup_paths`, then kill the kernel."""
        kc, km = self.kernel_client, self.kernel_manager
        self._log("detach")
        self._detach()
        if self.cleanup_paths:
            self._log("rmtree_configured")
            self._rmtree_configured()
        self._log("kill_kernel")
        self._kill_kernel(kc, km)
        self._log("done")
        print("shutdown_relay: done")

    def start(self) -> None:
        """Launch the inner kernel and install the run_cell hijacks."""
        import os

        import jupyter_client

        self._log(f"starting kernel '{self.kernel_name}'")
        km = jupyter_client.KernelManager(kernel_name=self.kernel_name)
        if self.cwd:
            km.cwd = str(self.cwd)
            os.environ.setdefault("PIXI_PROJECT_ROOT", str(self.cwd))
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=60)

        self.kernel_manager = km
        self.kernel_client = kc
        self._install_run_cell_hook()
        self._log(f"kernel ready, pid={km.provisioner.pid}")
        print(f"init_relay: kernel '{self.kernel_name}' ready")

    def _detach(self) -> None:
        """Reset state and restore run_cell hook. Never blocks."""
        self.kernel_client = None
        self.kernel_manager = None
        self._restore_run_cell_hook()

    @staticmethod
    def _kill_kernel(kc, km) -> None:
        """Best-effort kernel process cleanup. May block."""
        try:
            if kc:
                kc.stop_channels()
            if km:
                km.shutdown_kernel(now=True)
                km.cleanup_resources()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Detach and kill the inner kernel (no extra cleanup)."""
        kc, km = self.kernel_client, self.kernel_manager
        self._detach()
        self._kill_kernel(kc, km)
        print("shutdown_relay: done")

    def execute_cell(self, code, *, silent: bool = False, store_history: bool = True):
        """Execute code on relay kernel and relay output. Returns False if kernel died."""
        import queue

        poll_timeout = 60
        snippet = code.strip().split("\n", 1)[0][:80]
        self._log(f"cell: {snippet}")
        msg_id = self.kernel_client.execute(code, silent=silent, store_history=store_history)
        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=poll_timeout)
            except queue.Empty:
                self._log(f"waiting for iopub message ({poll_timeout}s silence)")
                if not self.kernel_manager.is_alive():
                    self._log("kernel died (process exited)")
                    return False
                continue
            except Exception:
                if not self.kernel_manager.is_alive():
                    self._log("kernel died (channel error after process exit)")
                    return False
                raise

            msg_type = msg["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                print(content["text"], end="", flush=True)
            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {})
                metadata = content.get("metadata", {})
                try:
                    from IPython.display import publish_display_data

                    publish_display_data(data=data, metadata=metadata)
                except Exception:
                    text = data.get("text/plain", "")
                    if text:
                        print(text)
            elif msg_type == "error":
                self._log(f"cell error: {content['ename']}: {content['evalue']}")
                self._log("traceback:")
                for line in content.get("traceback") or []:
                    self._log(f"  {line}")
                raise _relay_error(content)
            elif msg_type == "status" and content["execution_state"] == "idle":
                if msg["parent_header"].get("msg_id") == msg_id:
                    break
        self._log_memory()
        return True

    def _install_run_cell_hook(self) -> None:
        try:
            ip = get_ipython()  # noqa: F821
        except NameError:
            return
        if ip is None:
            return

        self._ip = ip
        self._original_run_cell = ip.run_cell
        ip.run_cell = self._hijacked_run_cell
        # ipykernel 7.x calls run_cell_async for top-level await cells,
        # bypassing run_cell — patch both to ensure relay intercepts all cells
        if hasattr(ip, "run_cell_async"):
            self._original_run_cell_async = ip.run_cell_async
            ip.run_cell_async = self._hijacked_run_cell_async

    def _restore_run_cell_hook(self) -> None:
        if self._ip is not None:
            if self._original_run_cell is not None:
                self._ip.run_cell = self._original_run_cell
            if self._original_run_cell_async is not None:
                self._ip.run_cell_async = self._original_run_cell_async
        self._ip = None
        self._original_run_cell = None
        self._original_run_cell_async = None

    def _runs_locally(self, raw_cell: str) -> bool:
        return (
            self.kernel_client is None
            or f"{self.local_name}.shutdown()" in raw_cell
            or f"{self.local_name}.cleanup(" in raw_cell
        )

    def _hijacked_run_cell(self, raw_cell, store_history=False, silent=False, shell_futures=True, cell_id=None):
        if self._runs_locally(raw_cell):
            return self._original_run_cell(
                raw_cell,
                store_history=store_history,
                silent=silent,
                shell_futures=shell_futures,
                cell_id=cell_id,
            )
        return self._relay_run_cell(
            raw_cell,
            store_history=store_history,
            silent=silent,
            shell_futures=shell_futures,
            cell_id=cell_id,
        )

    async def _hijacked_run_cell_async(
        self,
        raw_cell,
        store_history=False,
        silent=False,
        shell_futures=True,
        *,
        transformed_cell=None,
        preprocessing_exc_tuple=None,
        cell_id=None,
    ):
        if self._runs_locally(raw_cell):
            return await self._original_run_cell_async(
                raw_cell,
                store_history=store_history,
                silent=silent,
                shell_futures=shell_futures,
                transformed_cell=transformed_cell,
                preprocessing_exc_tuple=preprocessing_exc_tuple,
                cell_id=cell_id,
            )
        return self._relay_run_cell(
            raw_cell,
            store_history=store_history,
            silent=silent,
            shell_futures=shell_futures,
            cell_id=cell_id,
        )

    def _relay_run_cell(self, raw_cell, *, store_history=False, silent=False, shell_futures=True, cell_id=None):
        from IPython.core.interactiveshell import ExecutionInfo, ExecutionResult

        if silent:
            store_history = False

        info = ExecutionInfo(
            raw_cell=raw_cell,
            store_history=store_history,
            silent=silent,
            shell_futures=shell_futures,
            cell_id=cell_id,
        )
        result = ExecutionResult(info)
        if self._ip is not None:
            self._ip._last_traceback = None
            # Mirror IPython prompt numbering locally while execution happens remotely.
            result.execution_count = self._ip.execution_count
            if store_history:
                self._ip.execution_count += 1
        try:
            if not self.execute_cell(raw_cell, silent=silent, store_history=store_history):
                self._record_error(result, RuntimeError("relay kernel died"))
        except Exception as exc:
            self._record_error(result, exc)
        return result

    def _record_error(self, result, exc: Exception) -> None:
        result.error_in_exec = exc
        self._show_error(exc)
        self._fail_and_cleanup(exc)

    def _show_error(self, exc: Exception) -> None:
        if self._ip is None:
            return
        try:
            self._ip.showtraceback(exc_tuple=(type(exc), exc, exc.__traceback__), running_compiled_code=True)
        except Exception as traceback_exc:
            self._log(f"showtraceback failed: {traceback_exc}")

    def _fail_and_cleanup(self, exc: Exception) -> None:
        import os

        self._log(f"FATAL: {type(exc).__name__}: {exc}")
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
            self._log("interactive mode — skipping auto-cleanup")
            return
        try:
            self.cleanup()
        except Exception as cleanup_exc:
            self._log(f"cleanup failed: {cleanup_exc}")
