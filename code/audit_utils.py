"""
IMPORTANT WARNING: Modification of this code is prohibited, any alterations will be detected.
v1.0.0
"""
import os
import threading
import time
import hashlib
import psutil
import logging
import datetime
from pathlib import Path


ALLOWED_SHELL = ['/bin/bash', '/usr/bin/bash',
                 '/bin/zsh', '/usr/bin/zsh',
                 '/bin/sh', '/usr/bin/sh',
                 '/bin/dash', '/usr/bin/dash',
                 '/bin/zsh', '/usr/bin/zsh',
                 '/bin/tcsh', '/usr/bin/tcsh',
                 '/bin/csh', '/usr/bin/csh',
                 '/bin/ksh', '/usr/bin/ksh',
                 '/bin/fish', '/usr/bin/fish']


def get_ancestors_process(pid):
    p = psutil.Process(pid)
    parents = p.parents()
    ret = []
    for p in parents:
        exe = p.exe()
        ret.append(f'{exe}-{p.pid}')
        if exe in ALLOWED_SHELL:
            break
    return ret


def get_proc_tree_total_rss_mb(root_pid) -> tuple[float, list]:
    root = psutil.Process(root_pid)
    all_proc = [root]
    children = root.children(recursive=True)
    all_proc.extend(children)

    total_rss_bytes = 0
    total_vms_bytes = 0
    total_read_bytes = 0
    total_read_count = 0
    total_chars_bytes = 0
    details = []
    for p in all_proc:
        try:
            mem = p.memory_info()
            rss = mem.rss
            vms = mem.vms
            total_rss_bytes += rss
            total_vms_bytes += vms

            counters = p.io_counters()
            read_bytes = counters.read_bytes
            read_count = counters.read_count
            read_chars = counters.read_chars
            total_read_bytes += read_bytes
            total_read_count += read_count
            total_chars_bytes += read_chars
            details.append([p.exe(), p.pid , rss, vms, read_bytes, read_count, read_chars])
        except Exception:
            continue

    total_rss_mb = round(total_rss_bytes / 1024 / 1024, 2)
    total_vms_mb = round(total_vms_bytes / 1024 / 1024, 2)
    return [total_read_count, total_rss_mb + total_vms_mb, total_rss_bytes, total_rss_mb, 
            total_vms_bytes, total_vms_mb, 
            [total_read_bytes, total_chars_bytes], details]

def get_mem_usage():
    pid = os.getpid()
    res = get_proc_tree_total_rss_mb(pid)
    return res[1: 6]


class ChainedHashFileHandler(logging.FileHandler):
    def __init__(self, filename, mode: str = "a", encoding='utf-8', delay = False) -> None:
        self.filename = filename
        super().__init__(filename, mode, encoding, delay)
        content = Path(__file__).read_bytes() + str(time.time()).encode()
        self.f = hashlib.sha256(content).hexdigest()[:16]

    def emit(self, record):
        try:
            msg = self.format(record)
            final_msg = f"{msg} | {self.f}"
            self.stream.write(final_msg + self.terminator)
            self.flush()
            self.f = hashlib.sha256(final_msg.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
                self.handleError(record)
                self.stream.write(str(e))

    def archive(self):
        os.rename(self.filename, self.filename + ".csv")


class SecureLogger:
    __init_lock = threading.Lock()

    def __init__(self) -> None:
        self.log_dir = './audit-logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.last_hash = hashlib.md5(b"LOG_CHAIN_ROOT").hexdigest()[:16]

    def _calc_hash(self, raw: str) -> str:
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_logger(self, logger_name: str, console: bool = False) -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        if logger.handlers:
            return logger
        with self.__init_lock:
            if logger.handlers:
                return logger

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            process = psutil.Process(pid)
            
            filepath = os.path.join(self.log_dir, f'{logger_name}_{pid}_{timestamp}.log')
            file_handler = ChainedHashFileHandler(filepath, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            if console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            def archive_gracefully():
                logger.info("Logs archived gracefully, check completed.")
                file_handler.archive()
            logger.archive_gracefully = archive_gracefully

            logger.info(f"program {process.cmdline()} started at {process.create_time()}, audit logger1 inited, {filepath}")
            counter = process.io_counters()
            logger.info(f"{counter.read_count}, {counter.read_bytes}, {counter.read_chars}")
            return logger 


secure_logger = SecureLogger()
_logger = secure_logger.get_logger('file-loading')
digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_logger.info(f"file-loading v1.0.0 {digest[:16]}")


def get_train_logger(console: bool = True):
    logger = secure_logger.get_logger('training', console)
    return logger


def audit_file_loading(file_path: os.PathLike):
    timestamp = time.time()
    pids = [os.getpid()]
    ppid = os.getppid()
    pids.append(ppid)
    pids += get_ancestors_process(pids[0])
    full_path = os.path.abspath(file_path)
    file = open(full_path, 'rb')
    info = os.fstat(file.fileno())
    file_size = info.st_size
    if file_size > 12288:
        sample_buffer = b""
        step_length = file_size // 4
        for idx in range(4):
            offset = idx * step_length
            file.seek(offset)
            chunk = file.read()
            sample_buffer += chunk
        finger_print = hashlib.md5(sample_buffer).hexdigest()
    else:
        finger_print = hashlib.md5(file.read()).hexdigest()
    mem = get_proc_tree_total_rss_mb(pids[0])
    msg = f'{timestamp} loading {pids} {time.time()}, {full_path}, {file_size}, {finger_print}, {mem}'
    _logger.info(msg)
    file.close()


def audit_dataset_wrapper(file_path: str):
    codes = ["from audit_utils import audit_file_loading, audit_dataset_wrapper",
             "audit_dataset_wrapper(__file__)",
             "# PLEASE DO NOT EDIT THE FIRST THREE LINES"]
    file_bin = Path(file_path).read_bytes()
    lines = file_bin.decode().splitlines()
    for i, c in enumerate(codes):
        same = c == lines[i]
        if not same:
            raise Exception(f"PLEASE DO NOT EDIT THE FIRST THREE LINES OF {file_path}")
    digest = hashlib.sha256(file_bin).hexdigest()
    _logger.info(f'Official dataset, {file_path}, {digest[:16]}')
