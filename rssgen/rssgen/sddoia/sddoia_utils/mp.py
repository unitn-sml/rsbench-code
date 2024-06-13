import multiprocessing as mp
import signal
import sys
import os
import time
import ctypes
import itertools


class CustomPool:
    """Custom process pool for multiprocessing"""

    def __init__(self, max_processes):
        self.max_processes = max_processes
        self.processes = []
        self.manager = mp.Manager()
        self.job_queue = []
        self.results = self.manager.Queue()
        self.finished = mp.Array(ctypes.c_bool, max_processes)
        self.busy = mp.Array(ctypes.c_bool, max_processes)
        self.finished_lock = mp.Lock()
        self.id_counter = itertools.count()
        self.available_pids = self.manager.Queue()

        # initialize available pids
        for i in range(max_processes):
            self.available_pids.put(i)
            self.finished[i] = True
            self.busy[i] = False

        self.terminate_flag = False
        self.last_activity_time = time.time()
        signal.signal(signal.SIGINT, self.terminate_on_interrupt)

    def get_results(self):
        results_list = []
        while not self.results.empty():
            results_list.append(self.results.get())
        return results_list

    def get_next_pid(self):
        if self.available_pids:
            return self.available_pids.get()
        return None

    def apply_async(self, func, args):
        self.job_queue.append((func, args))
        self.start_process_if_available()

    def start_process_if_available(self):
        while len(self.processes) < self.max_processes and self.job_queue:
            with self.finished_lock:
                func, args = self.job_queue.pop(0)
                pid = self.get_next_pid()

            if pid is not None:
                process = mp.Process(
                    target=self.wrapper_func,
                    args=(
                        self.results,
                        self.finished,
                        self.busy,
                        self.finished_lock,
                        func,
                        args,
                        pid,
                    ),
                )
                process.start()
                self.processes.append((process, (func, args), pid))
            else:
                print("No available processes")
                break

    def wrapper_func(self, results, finished, busy, finished_lock, func, args, pid):
        try:
            # Initialization
            with finished_lock:
                finished[pid] = False
                busy[pid] = True
                print("Pid", pid, "started")

            result = func(*args)
            results.put(result)

            with finished_lock:
                finished[pid] = True
                busy[pid] = False
                print("Pid", pid, "finished")

        except Exception as e:
            print(f"Process terminated unexpectedly: {e}")
            self.terminate()

    def wait_for_termination(self):
        while self.processes:
            current_time = time.time()
            elapsed_time = current_time - self.last_activity_time

            if (
                elapsed_time > 30
            ):  # If it's been more than 60 seconds without any activity
                print(
                    "Too much time without activity. Stopping everything [press CTR-C]. Try to run the code with the load_status_dict parameter"
                )
                os._exit(1)

            for process_tuple in self.processes[:]:
                process, job, pid = process_tuple
                process.join(timeout=0.1)

                if not process.is_alive():
                    self.processes.remove(process_tuple)
                    with self.finished_lock:
                        print("Pid", pid, "is dead")
                        if not self.finished[pid]:
                            print(pid, "hasn't finished")
                            if job not in self.job_queue:
                                self.job_queue.append(
                                    job
                                )  # Reassign job if not in queue
                                print("Reassigning job of pid", pid)
                        self.last_activity_time = time.time()
                        self.finished[pid] = False
                        self.busy[pid] = False
                        self.available_pids.put(pid)
                        print("Pid is now available", pid)

                    if self.job_queue:
                        self.start_process_if_available()  # Respawn dead process
                else:
                    with self.finished_lock:
                        print(
                            "Pid is",
                            pid,
                            "JOB QUEUE",
                            self.job_queue,
                            self.busy[pid],
                            self.finished[pid],
                        )
                        print("Job", job)
                        if not self.busy[pid]:
                            print("Pid", pid, "has nothing to do, terminating...")
                            self.last_activity_time = time.time()
                            process.terminate()
            time.sleep(0.1)

    def terminate(self):
        self.terminate_flag = True
        for process, _ in self.processes:
            process.terminate()
        self.processes = []  # Clear processes list

    def terminate_on_interrupt(self, signum, frame):
        self.terminate()
        sys.exit(1)

    def __enter__(self):
        self.last_activity_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()
